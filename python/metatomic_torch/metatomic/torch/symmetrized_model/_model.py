from typing import Dict, List, Optional

import metatensor.torch as mts
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    is_atomistic_model,
)

from ..o3 import O3Transformation, transform_system, transform_tensor
from ._decompose import _decompose_output
from ._gradients import _evaluate_with_gradients
from ._projections import _accumulate_batch, _finalize_system, _wigner_stacks
from ._quadrature import (
    _choose_quadrature,
    _rotations_from_angles,
    get_euler_angles_quadrature,
)
from ._utils import (
    _infer_n_systems,
    _max_spherical_lambda,
    _prepend_system_to_samples,
    _reshape_block_by_local_system,
    _selected_atoms_for_local_systems,
)


def _reduce_weighted_batch_tensor(
    tensor: TensorMap,
    weights: torch.Tensor,
    system_index: int,
    *,
    component_norm: bool = False,
) -> TensorMap:
    """Reduce a batch of rotated-copy outputs to its quadrature contribution.

    With ``component_norm=False``, returns ``sum_g 0.5 w_g x(g)`` over the
    batch, i.e. a partial sum of the O(3) mean of ``x``. With
    ``component_norm=True``, the values are squared and summed over the
    component axes first, giving a partial sum of the mean squared norm (the
    second moment). Weights are halved because the caller sums the two
    inversion passes (+1 and -1) to average over O(3). Samples are re-labeled
    with the global ``system_index``.
    """
    n_local_systems = weights.numel()
    reduced_blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        values, sample_names, sample_values = _reshape_block_by_local_system(
            block, n_local_systems
        )

        components = block.components
        if component_norm:
            component_dims = tuple(range(2, 2 + len(block.components)))
            if len(component_dims) == 0:
                values = values**2
            else:
                values = torch.sum(values**2, dim=component_dims)
            components = []

        weight = weights.to(dtype=values.dtype, device=values.device)
        view = [values.shape[0]] + [1] * (values.ndim - 1)
        reduced_values = torch.sum(0.5 * weight.view(view) * values, dim=0)
        if reduced_values.ndim == 1:
            reduced_values = reduced_values.unsqueeze(0)

        reduced_blocks.append(
            TensorBlock(
                values=reduced_values,
                samples=_prepend_system_to_samples(
                    sample_names,
                    sample_values,
                    system_index,
                    device=block.samples.values.device,
                ),
                components=components,
                properties=block.properties,
            )
        )

    return TensorMap(tensor.keys, reduced_blocks)


def _accumulate_tensormap(
    accumulators: Dict[str, TensorMap], name: str, contribution: TensorMap
) -> None:
    if name in accumulators:
        accumulators[name] = mts.add(accumulators[name], contribution)
    else:
        accumulators[name] = contribution


def _append_tensormap(
    accumulators: Dict[str, List[TensorMap]], name: str, contribution: TensorMap
) -> None:
    accumulators.setdefault(name, []).append(contribution)


def _join_tensormap_list(tensors: List[TensorMap]) -> TensorMap:
    """Join per-system TensorMaps along samples, taking the union of keys."""
    if len(tensors) == 1:
        return tensors[0]
    return mts.join(tensors, "samples", different_keys="union")


def _mean_norm_squared_tensor(tensor: TensorMap) -> TensorMap:
    """Squared values summed over the component axes, per sample and property.

    Applied to a ``_mean`` map, this gives the squared norm of the mean in the
    same (component-free) layout as the second-moment maps.
    """
    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        if block.values.ndim > 2:
            values = torch.sum(
                block.values**2, dim=tuple(range(1, block.values.ndim - 1))
            )
        else:
            values = block.values**2
        if values.ndim == 1:
            values = values.unsqueeze(0)
        blocks.append(
            TensorBlock(
                values=values,
                samples=block.samples,
                components=[],
                properties=block.properties,
            )
        )
    return TensorMap(tensor.keys, blocks)


def _finalize_variance(
    second_moment: TensorMap,
    mean: TensorMap,
) -> TensorMap:
    """Variance over O(3): mean squared norm minus squared norm of the mean."""
    mean_norm_sq = _mean_norm_squared_tensor(mean)
    return mts.subtract(second_moment, mean_norm_sq)


def per_system_equivariance_rmse(
    outputs: Dict[str, TensorMap],
    name: str,
    n_systems: Optional[int] = None,
) -> TensorMap:
    """
    Reduce the ``<name>_var`` entry of a :py:class:`SymmetrizedModel` output
    dictionary to per-system equivariance RMSE values.

    The reduction preserves the block and property structure of the target:
    the returned :py:class:`TensorMap` has the same keys as the output (e.g.
    the ``o3_lambda``/``o3_sigma`` blocks of a spherical target), one
    ``system`` sample per system, and the target's properties (e.g. radial
    channels). Within each block, the variance (already summed over the
    component axes) is divided by the irrep multiplicity (:math:`2\\lambda+1`,
    read from the component axes of the matching ``<name>_mean`` entry) and
    averaged over the samples belonging to each system (e.g. over atoms for
    per-atom outputs), before taking the square root. This matches the
    normalization of an element-wise accuracy RMSE over the same block, so the
    two are directly comparable.

    Aggregating afterwards is up to the caller: to pool per-system values into
    a dataset-level RMSE, combine the squares weighted by the number of
    samples of each system, ``sqrt(sum_A N_A rmse_A^2 / sum_A N_A)``; a plain
    mean of per-system RMSEs is not the pooled RMSE when system sizes differ.
    Similarly, pooling over properties or blocks is a mean of squares. Systems
    that contribute no samples to a block get an RMSE of zero. Extensivity is
    not corrected either: for a total-energy output, dividing by the number of
    atoms per system is up to the caller.

    :param outputs: dictionary returned by :py:class:`SymmetrizedModel`
    :param name: name of the output to reduce, without the ``_var`` suffix
        (e.g. ``"energy_l0"``)
    :param n_systems: number of systems the output was computed for. If ``None``
        (default), inferred from the largest ``system`` sample index; trailing
        systems that contributed no samples are then silently missing from the
        result, so pass it explicitly whenever systems can be empty.
    :return: a :py:class:`TensorMap` with the same keys and properties as the
        output, and values of shape ``(n_systems, n_properties)`` per block
    """
    variance = outputs[name + "_var"]
    mean = outputs[name + "_mean"]

    if n_systems is None:
        n_systems = _infer_n_systems([variance])

    blocks: List[TensorBlock] = []
    for key, block in variance.items():
        multiplicity = 1
        for component in mean.block(key).components:
            multiplicity *= len(component)

        values = block.values
        dtype = values.dtype
        device = values.device
        n_properties = values.shape[-1]
        system = block.samples.column("system").to(dtype=torch.long, device=device)

        total = torch.zeros(n_systems, n_properties, dtype=dtype, device=device)
        count = torch.zeros(n_systems, dtype=dtype, device=device)
        total.index_add_(0, system, values / multiplicity)
        count.index_add_(0, system, torch.ones(len(system), dtype=dtype, device=device))

        total = total / torch.clamp(count, min=1.0).unsqueeze(1)
        blocks.append(
            TensorBlock(
                values=torch.sqrt(torch.clamp(total, min=0.0)),
                samples=Labels(
                    ["system"],
                    torch.arange(n_systems, device=device).unsqueeze(1),
                ),
                components=[],
                properties=block.properties,
            )
        )

    return TensorMap(variance.keys, blocks)


class SymmetrizedModel(torch.nn.Module):
    r"""
    Wrapper around an atomistic model that symmetrizes its outputs over :math:`O(3)`
    and computes equivariance metrics.

    The model is evaluated over a quadrature grid on :math:`O(3)`, constructed from a
    Lebedev grid supplemented by in-plane rotations. For each sampled group element, the
    model outputs are "back-rotated" according to the known :math:`O(3)` action
    appropriate for their tensorial type (scalar, vector, tensor, etc.). Averaging these
    back-rotated predictions over the quadrature grid yields fully
    :math:`O(3)`-symmetrized outputs. In addition, two complementary equivariance
    metrics are computed:

    1. Variance under :math:`O(3)` of the back-rotated outputs.

        For a perfectly equivariant model, the back-rotated output :math:`x(g)` is
        independent of the group element :math:`g`. Deviations from perfect equivariance
        are quantified by the difference between the average squared norm over
        :math:`O(3)` and the squared norm of the :math:`O(3)`-averaged output:

        .. math::

            \mathrm{Var}_{O(3)}[x]
            =
            \left\langle \,\| x(g) \|^{2} \,\right\rangle_{O(3)}
            -
            \left\| \left\langle x(g) \right\rangle_{O(3)} \right\|^{2} .

        Here, :math:`\|\cdot\|` denotes the Euclidean norm over the ``component`` axis,
        and :math:`\langle \cdot \rangle_{O(3)}` denotes averaging over the quadrature
        grid. This quantity is the squared norm of the component orthogonal to the
        perfectly equivariant subspace and therefore provides a scalar measure of the
        deviation from exact equivariance.

    2. Decomposition into isotypical components of :math:`O(3)`.

        Each output component may be viewed as a scalar function on :math:`O(3)`,
        which can be decomposed into isotypical components labeled by the irreducible
        representations :math:`\ell,\sigma` of :math:`O(3)`. The projection onto the
        :math:`(\ell,\sigma)`-th isotypical subspace is computed as a convolution with
        the corresponding character :math:`\chi_{\ell,\sigma}`:

        .. math::

            (P_{\ell,\sigma} x)(g)
            =
            (2\ell+1)
            \int_{O(3)} \chi_{\ell,\sigma}(h^{-1} g)\, x(h)\, \mathrm{d}\mu(h),

        where the prefactor is the dimension of the irreducible representation
        and makes :math:`P_{\ell,\sigma}` idempotent, so that the squared norms
        of the components of a band-limited function sum to :math:`\| x \|^{2}`.

        The character is the trace of the representation matrices: on proper
        rotations :math:`R` it does not depend on :math:`\sigma`,

        .. math::

            \chi_{\ell,\sigma}(R) = \operatorname{tr} D^{\ell}(R),

        with :math:`D^{\ell}` the (real) Wigner-D matrix, while on the improper
        coset (a rotation composed with the inversion :math:`i`) it becomes

        .. math::

            \chi_{\ell,\sigma}(i R)
            =
            \sigma \, (-1)^{\ell} \, \operatorname{tr} D^{\ell}(R).

        Following the metatensor ``o3_sigma`` convention, :math:`\sigma = +1`
        labels proper tensors (inversion parity :math:`(-1)^{\ell}`, e.g.
        vectors at :math:`\ell = 1`) and :math:`\sigma = -1` pseudotensors
        (inversion parity :math:`-(-1)^{\ell}`). These labels appear as
        ``chi_lambda`` and ``chi_sigma`` in the keys of the returned character
        projections.

        The squared :math:`L^{2}` norm of the projection over :math:`O(3)` is

        .. math::

            \| P_{\ell,\sigma} x \|^{2}
            =
            \left\langle \, | (P_{\ell,\sigma} x)(g) |^{2} \, \right\rangle_{O(3)} .

        These quantities describe how the model output is distributed across the
        different :math:`O(3)` irreducible sectors. The complementary component,
        orthogonal to all isotypical subspaces, is given by

        .. math::

            \| x \|^{2}
            -
            \sum_{\ell,\sigma} \| P_{\ell,\sigma} x \|^{2} ,

        and provides a refined measure of the deviation from lying entirely within any
        prescribed set of :math:`O(3)` irreducible representations.

    :param base_model: atomistic model to symmetrize, either a module following
        the :py:class:`ModelInterface` call convention or an exported
        :py:class:`AtomisticModel` (including one loaded with
        :py:func:`load_atomistic_model`)
    :param max_o3_lambda_target: maximum O(3) angular momentum expected among the
        model outputs to back-rotate
    :param max_o3_lambda_character: maximum O(3) angular momentum used for the
        character projections. If ``None`` (default), character projections are
        unavailable (calling :py:meth:`forward` with
        ``compute_character_projections=True`` raises an error) and the default
        quadrature grid follows ``max_o3_lambda_target`` instead.
    :param batch_size: number of rotations to evaluate in a single batch
    :param max_o3_lambda_grid: maximum O(3) angular momentum the quadrature grid
        integrates exactly. If ``None`` (default), set to
        ``2 * max_o3_lambda_character + 1`` when ``max_o3_lambda_character`` is
        given, and to ``2 * max_o3_lambda_target + 1`` otherwise.
    :param storage_device: device on which intermediate and final results are
        kept. If ``None`` (default), everything stays on the model device. If set
        (e.g. ``"cpu"`` for a model on GPU), base-model outputs are moved there
        right after each forward pass, trading transfer bandwidth for GPU
        memory, and back-rotation, accumulation, and the returned TensorMaps
        live there.
    """

    def __init__(
        self,
        base_model,
        max_o3_lambda_target: int,
        max_o3_lambda_character: Optional[int] = None,
        batch_size: int = 32,
        max_o3_lambda_grid: Optional[int] = None,
        storage_device: Optional[str] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self._base_is_atomistic = is_atomistic_model(base_model)

        try:
            ref_param = next(base_model.parameters())
            device = ref_param.device
        except StopIteration:
            device = torch.device("cpu")

        self.max_o3_lambda_target = max_o3_lambda_target
        self.batch_size = batch_size
        self.storage_device = (
            None if storage_device is None else torch.device(storage_device)
        )
        if max_o3_lambda_grid is None:
            if max_o3_lambda_character is not None:
                max_o3_lambda_grid = int(2 * max_o3_lambda_character + 1)
            else:
                max_o3_lambda_grid = int(2 * max_o3_lambda_target + 1)
        self.max_o3_lambda_grid = max_o3_lambda_grid
        self.max_o3_lambda_character = max_o3_lambda_character

        lebedev_order, n_inplane_rotations = _choose_quadrature(self.max_o3_lambda_grid)
        # kept for the sufficiency check in forward() when character projections
        # are requested
        self._lebedev_order = lebedev_order
        alpha, beta, gamma, w_so3 = get_euler_angles_quadrature(
            lebedev_order, n_inplane_rotations
        )
        # the grid buffers are kept in float64: all statistics are accumulated
        # in double precision (the variance is a difference of two second
        # moments and cancels catastrophically in float32 for outputs of large
        # magnitude); rotations are downcast to the model dtype only when
        # transforming the input systems
        so3_weights = torch.from_numpy(w_so3).to(device=device, dtype=torch.float64)
        self.register_buffer("so3_weights", so3_weights)
        # private full-precision copy used by the accumulation: a plain
        # attribute, so a user cast like `.to(torch.float32)` cannot touch it
        # (weights quantized to float32 no longer sum to 1 within ~1e-8, which
        # alone reintroduces the variance cancellation)
        self._so3_weights_float64 = so3_weights.to(device="cpu", copy=True)

        so3_rotations = torch.from_numpy(
            _rotations_from_angles(alpha, beta, gamma).as_matrix()
        ).to(device=device, dtype=torch.float64)
        self.register_buffer("so3_rotations", so3_rotations)

        angles_inverse_rotations = (np.pi - gamma, beta, np.pi - alpha)
        so3_inverse_rotations = torch.from_numpy(
            _rotations_from_angles(*angles_inverse_rotations).as_matrix()
        ).to(device=device, dtype=torch.float64)
        self.register_buffer("so3_inverse_rotations", so3_inverse_rotations)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
        compute_character_projections: bool = False,
        compute_gradients: bool = False,
        energy_name: str = "energy",
    ) -> Dict[str, TensorMap]:
        """
        Symmetrize the model outputs over :math:`O(3)` and compute equivariance
        metrics.

        :param systems: list of systems to evaluate
        :param outputs: dictionary of model outputs to symmetrize
        :param selected_atoms: optional Labels specifying which atoms to consider
        :param compute_character_projections: if True, also compute character
            projections. Requires ``max_o3_lambda_character`` to be set and a
            quadrature grid able to integrate the projections exactly.
        :param compute_gradients: if True, compute conservative forces and stress
            via autograd. When False (default), the grid evaluation runs under
            ``torch.no_grad()`` to save memory.
        :param energy_name: name of the output used to derive forces and stress
            when ``compute_gradients=True``; it must be present in ``outputs``.
            The derived quantities are always returned under the ``forces`` and
            ``stress`` names.
        :return: dictionary with symmetrized outputs and equivariance metrics.
            Statistics are accumulated and returned in float64, independently
            of the model dtype; the model itself runs in its own dtype, so
            errors below the round-off of its outputs remain unmeasurable.
        """
        if compute_character_projections:
            if self.max_o3_lambda_character is None:
                raise ValueError(
                    "max_o3_lambda_character must be set to compute character "
                    "projections"
                )
            if self._lebedev_order < 2 * self.max_o3_lambda_character:
                raise ValueError(
                    "the quadrature grid is too coarse for character projections "
                    f"up to lambda={self.max_o3_lambda_character} (Lebedev order "
                    f"{self._lebedev_order} < {2 * self.max_o3_lambda_character}); "
                    "set max_o3_lambda_grid >= 2 * max_o3_lambda_character, or "
                    "leave it None to use a sufficient default"
                )

        device = self.so3_weights.device
        # outputs are detached from any autograd graph before back-rotation,
        # so offloading is safe in compute_gradients mode too
        offload = self.storage_device is not None
        result_device = (
            self.storage_device if self.storage_device is not None else device
        )

        with torch.enable_grad() if compute_gradients else torch.no_grad():
            results = self._eval_over_grid(
                systems,
                outputs,
                selected_atoms,
                return_transformed=compute_character_projections,
                compute_gradients=compute_gradients,
                offload=offload,
                energy_name=energy_name,
            )

        return {
            name: tensor.to(device=result_device) for name, tensor in results.items()
        }

    def equivariance_error(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
        compute_gradients: bool = False,
        energy_name: str = "energy",
    ) -> Dict[str, TensorMap]:
        """
        Compute the per-system equivariance error of the model outputs.

        This is the recommended entry point for evaluation pipelines: it runs
        :py:meth:`forward` (with character projections off) and reduces each
        ``<name>_var`` entry with :py:func:`per_system_equivariance_rmse`.

        The values are in the raw units of the corresponding output;
        extensivity normalization (e.g. dividing a total-energy RMSE by the
        number of atoms per system) is up to the caller.

        :param systems: list of systems to evaluate
        :param outputs: dictionary of model outputs to symmetrize
        :param selected_atoms: optional Labels specifying which atoms to consider
        :param compute_gradients: if True, also compute the error of the
            conservative forces (and stress, for periodic systems) obtained
            from the ``energy_name`` output via autograd
        :param energy_name: name of the output used to derive forces and stress
            when ``compute_gradients=True``
        :return: dictionary keyed by decomposed output name (e.g.
            ``"energy_l0"``, ``"forces_l1"``). Each entry is a
            :py:class:`TensorMap` with the block and property structure of the
            corresponding output (see :py:func:`per_system_equivariance_rmse`)
            and one RMSE value per ``system`` sample; to pool per-system values
            into a dataset-level RMSE, combine the squares weighted by the
            number of samples of each system,
            ``sqrt(sum_A N_A rmse_A^2 / sum_A N_A)``.
        """
        results = self.forward(
            systems,
            outputs,
            selected_atoms,
            compute_character_projections=False,
            compute_gradients=compute_gradients,
            energy_name=energy_name,
        )

        n_systems = len(systems)
        errors: Dict[str, TensorMap] = {}
        for key in results:
            if not key.endswith("_var"):
                continue
            name = key[: -len("_var")]
            errors[name] = per_system_equivariance_rmse(
                results, name, n_systems=n_systems
            )
        return errors

    def _run_base_model(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        """Evaluate the base model, adapting the call to its kind.

        Modules following the :py:class:`ModelInterface` convention are called
        directly; exported :py:class:`AtomisticModel` instances are called with
        a :py:class:`ModelEvaluationOptions` (empty ``length_unit``, i.e. no
        conversion of the input systems) and ``check_consistency=False``.

        The requested outputs are forwarded unchanged: each base-model kind
        keeps its usual metatomic semantics, so exported models apply their
        normal output-unit conversion when a requested ``unit`` differs from
        the declared one, while raw modules ignore units entirely.
        """
        if self._base_is_atomistic:
            options = ModelEvaluationOptions(
                length_unit="",
                outputs=outputs,
                selected_atoms=selected_atoms,
            )
            return self.base_model(systems, options, check_consistency=False)
        return self.base_model(systems, outputs, selected_atoms)

    def _eval_over_grid(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
        return_transformed: bool,
        compute_gradients: bool = False,
        offload: bool = False,
        energy_name: str = "energy",
    ) -> Dict[str, TensorMap]:
        """
        Stream the model over the O(3) quadrature, accumulating mean, variance, and
        character projections without ever materializing the full-grid TensorMap.

        :param systems: list of systems to evaluate
        :param outputs: dictionary of model outputs to symmetrize
        :param selected_atoms: optional Labels specifying which atoms to consider
        :param return_transformed: if True, also compute character projections
        :param compute_gradients: if True, compute forces/stress via autograd
        :param offload: if True, move base-model outputs to ``storage_device``
            right after each forward pass
        :param energy_name: name of the output forces/stress are derived from
        :return: dictionary with all symmetrized outputs and metrics
        """
        n_rotations = self.so3_rotations.size(0)
        requested_output_names = list(outputs.keys())
        if compute_gradients:
            if energy_name not in outputs:
                raise ValueError(
                    f"compute_gradients=True requires '{energy_name}' in outputs"
                )
            for reserved in ("forces", "stress"):
                if reserved in outputs:
                    raise ValueError(
                        f"'{reserved}' is reserved for the autograd-derived "
                        "output when compute_gradients=True; rename the model "
                        "output or evaluate it in a separate call"
                    )

            requested_output_names = list(
                dict.fromkeys(requested_output_names + ["forces"])
            )
            if any(bool(torch.any(s.pbc).item()) for s in systems):
                requested_output_names = list(
                    dict.fromkeys(requested_output_names + ["stress"])
                )

        mean_accumulators: Dict[str, List[TensorMap]] = {}
        second_moment_accumulators: Dict[str, List[TensorMap]] = {}
        character_projection_accumulators: Dict[str, List[TensorMap]] = {}

        # Wigner-D caches beyond the target order are only needed for
        # character projections
        ell_max = self.max_o3_lambda_target
        if return_transformed:
            assert self.max_o3_lambda_character is not None
            ell_max = max(ell_max, self.max_o3_lambda_character)

        for i_sys, system in enumerate(systems):
            system_mean_accumulators: Dict[str, TensorMap] = {}
            system_second_moment_accumulators: Dict[str, TensorMap] = {}
            system_proj_pos: Dict[str, Dict] = {}
            system_proj_neg: Dict[str, Dict] = {}

            work_device = system.positions.device
            work_dtype = system.positions.dtype
            backrotation_device = self.storage_device if offload else work_device
            # back-rotation and accumulation always happen in float64
            augmentation_system = system.to(
                device=backrotation_device, dtype=torch.float64
            )

            # fail loud on data beyond the declared band limit, instead of
            # dying later inside the Wigner-D cache with an opaque message
            for data_name in system.known_data():
                data_lambda = _max_spherical_lambda(system.get_data(data_name))
                if data_lambda > self.max_o3_lambda_target:
                    raise ValueError(
                        f"system data '{data_name}' contains "
                        f"o3_lambda={data_lambda} components, larger than "
                        f"max_o3_lambda_target={self.max_o3_lambda_target}; "
                        "increase max_o3_lambda_target"
                    )

            # the forward transformation only needs Wigner-D caches to rotate
            # custom data with spherical components; positions, cell, and
            # neighbor lists use the 3x3 matrix only
            forward_ell_max = ell_max if len(system.known_data()) > 0 else 0

            for batch_start in range(0, n_rotations, self.batch_size):
                batch_stop = min(batch_start + self.batch_size, n_rotations)
                n_local_systems = batch_stop - batch_start
                weights = self._so3_weights_float64[batch_start:batch_stop]
                batch_rotations = self.so3_rotations[batch_start:batch_stop].to(
                    device=work_device, dtype=work_dtype
                )
                local_selected_atoms = _selected_atoms_for_local_systems(
                    selected_atoms,
                    i_sys,
                    n_local_systems,
                )

                for inversion in (1, -1):
                    inversion_batch = inversion * batch_rotations

                    # in both branches, outputs are moved off the model device
                    # (when offloading) and upcast right after the forward:
                    # everything downstream (back-rotation, mean/second-moment,
                    # projections) runs in float64 to avoid catastrophic
                    # cancellation in the variance for outputs of large
                    # magnitude
                    if compute_gradients:
                        raw = _evaluate_with_gradients(
                            self._run_base_model,
                            system,
                            inversion_batch,
                            outputs,
                            local_selected_atoms,
                            work_device,
                            work_dtype,
                            energy_name=energy_name,
                            ell_max=forward_ell_max,
                        )
                        # the statistics need no gradients, and the forces and
                        # stress are already materialized: detach so each
                        # batch's autograd graph is freed instead of being
                        # kept alive by the accumulators for the whole grid
                        out = {
                            k: mts.detach(v).to(
                                device=backrotation_device, dtype=torch.float64
                            )
                            for k, v in raw.items()
                        }
                    else:
                        transformed_systems = [
                            transform_system(
                                system,
                                O3Transformation(
                                    R.to(device=work_device), forward_ell_max
                                ),
                            )
                            for R in inversion_batch
                        ]
                        raw = self._run_base_model(
                            transformed_systems,
                            outputs,
                            local_selected_atoms,
                        )
                        out = {
                            k: v.to(device=backrotation_device, dtype=torch.float64)
                            for k, v in raw.items()
                        }

                    present_output_names = [
                        name for name in requested_output_names if name in out
                    ]
                    if len(present_output_names) == 0:
                        continue

                    inverse_mats = (
                        inversion * self.so3_inverse_rotations[batch_start:batch_stop]
                    ).to(device=backrotation_device, dtype=torch.float64)
                    transformations = [
                        O3Transformation(R, ell_max) for R in inverse_mats.unbind(0)
                    ]
                    wigner_stacks: Dict[int, torch.Tensor] = (
                        _wigner_stacks(transformations, ell_max)
                        if return_transformed
                        else {}
                    )

                    for name in present_output_names:
                        tensor = out[name]

                        output_lambda = _max_spherical_lambda(tensor)
                        if output_lambda > self.max_o3_lambda_target:
                            raise ValueError(
                                f"output '{name}' contains "
                                f"o3_lambda={output_lambda} components, larger "
                                f"than max_o3_lambda_target="
                                f"{self.max_o3_lambda_target}; increase "
                                "max_o3_lambda_target"
                            )

                        backtransformed = transform_tensor(
                            tensor,
                            [augmentation_system] * n_local_systems,
                            transformations,
                        )
                        for final_name, decomposed_tensor in _decompose_output(
                            name, backtransformed
                        ).items():
                            _accumulate_tensormap(
                                system_mean_accumulators,
                                final_name,
                                _reduce_weighted_batch_tensor(
                                    decomposed_tensor,
                                    weights,
                                    i_sys,
                                    component_norm=False,
                                ),
                            )
                            _accumulate_tensormap(
                                system_second_moment_accumulators,
                                final_name,
                                _reduce_weighted_batch_tensor(
                                    decomposed_tensor,
                                    weights,
                                    i_sys,
                                    component_norm=True,
                                ),
                            )

                        if return_transformed:
                            _accumulate_batch(
                                system_proj_pos,
                                system_proj_neg,
                                _decompose_output(name, tensor),
                                weights,
                                wigner_stacks,
                                self.max_o3_lambda_character,
                                inversion,
                            )

            if return_transformed:
                projections = _finalize_system(
                    system_proj_pos,
                    system_proj_neg,
                    i_sys,
                    self.max_o3_lambda_character,
                )
                for name, tensor in projections.items():
                    _append_tensormap(character_projection_accumulators, name, tensor)

            for name, tensor in system_mean_accumulators.items():
                _append_tensormap(mean_accumulators, name, tensor)

            for name, tensor in system_second_moment_accumulators.items():
                _append_tensormap(second_moment_accumulators, name, tensor)

        results: Dict[str, TensorMap] = {}
        for name, mean_tensors in mean_accumulators.items():
            mean_tensor = _join_tensormap_list(mean_tensors)
            results[name + "_mean"] = mean_tensor

            norm_squared = _join_tensormap_list(second_moment_accumulators[name])
            results[name + "_var"] = _finalize_variance(norm_squared, mean_tensor)
            results[name + "_norm_squared"] = norm_squared

        for name, tensors in character_projection_accumulators.items():
            if tensors:
                results[name + "_character_projection"] = _join_tensormap_list(tensors)

        return results
