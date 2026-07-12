from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    is_atomistic_model,
    register_autograd_neighbors,
)

from ..o3 import O3Transformation, transform_system, transform_tensor
from ._decompose import _decompose_output, _decomposed_output_names
from ._gradients import _evaluate_with_gradients
from ._projections import (
    _accumulate_batch,
    _finalize_system,
    _wigner_stacks_from_matrices,
)
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
    _validated_integer,
)


_DEFAULT_WIGNER_CACHE_MAX_BYTES = 64 * 1024**2
_MPS_UNSUPPORTED_MESSAGE = (
    "SymmetrizedModel does not support MPS because stable O(3) quadrature, "
    "back-rotation, and statistical reduction require torch.float64, which "
    "the MPS backend does not support. Use CPU or CUDA for the base model, "
    "input Systems, and storage_device."
)


def _transform_system_batch(
    system: System,
    matrices: torch.Tensor,
    max_angular_momentum: int,
    *,
    is_inverted: bool,
) -> List[System]:
    """Transform geometry in one batch; route heterogeneous custom data normally."""
    if matrices.dim() != 3 or tuple(matrices.shape[1:]) != (3, 3):
        raise ValueError(
            f"matrices must have shape (N, 3, 3), got {tuple(matrices.shape)}"
        )
    # Batched-matmul setup is slower for one operation.
    if len(matrices) == 1:
        return [
            transform_system(
                system,
                O3Transformation._from_validated_matrix(
                    matrices[0],
                    max_angular_momentum,
                    is_inverted=is_inverted,
                ),
            )
        ]
    if matrices.device != system.positions.device or (
        matrices.dtype != system.positions.dtype
    ):
        raise ValueError("system and quadrature matrices must share dtype/device")

    transposed = matrices.transpose(1, 2)
    positions = system.positions.unsqueeze(0) @ transposed
    cells = system.cell.unsqueeze(0) @ transposed

    transformations: List[O3Transformation] = []
    data_names = system.known_data()
    if len(data_names) != 0:
        transformations = [
            O3Transformation._from_validated_matrix(
                matrix,
                max_angular_momentum,
                is_inverted=is_inverted,
            )
            for matrix in matrices.unbind(0)
        ]

    neighbor_batches = []
    for options in system.known_neighbor_lists():
        neighbors = system.get_neighbor_list(options)
        # The input list can already be registered against ``system.positions``.
        # Start a fresh graph for each transformed System, as the generic path does.
        values = neighbors.values.detach().squeeze(-1).unsqueeze(0) @ transposed
        neighbor_batches.append((options, neighbors, values))

    transformed_systems: List[System] = []
    for index in range(len(matrices)):
        transformed = System(
            types=system.types,
            positions=positions[index],
            cell=cells[index],
            pbc=system.pbc,
        )
        for options, neighbors, values in neighbor_batches:
            rotated_neighbors = TensorBlock(
                values=values[index].unsqueeze(-1),
                samples=neighbors.samples,
                components=neighbors.components,
                properties=neighbors.properties,
            )
            register_autograd_neighbors(transformed, rotated_neighbors)
            transformed.add_neighbor_list(options, rotated_neighbors)
        for data_name in data_names:
            transformed.add_data(
                data_name,
                transform_tensor(
                    system.get_data(data_name),
                    [system],
                    [transformations[index]],
                ),
            )
        transformed_systems.append(transformed)

    return transformed_systems


def _reduce_weighted_batch_moments(
    tensor: TensorMap,
    weights: torch.Tensor,
    system_index: int,
    reference: Optional[TensorMap] = None,
) -> Tuple[TensorMap, TensorMap, TensorMap, TensorMap]:
    """Return signed/absolute centered moments and the fixed reference.

    Here ``y = tensor - reference``; the first copy becomes the reference on
    the first call. The absolute second moment supplies a scale for distinguishing
    signed-quadrature aliasing from floating-point round-off.
    """
    n_local_systems = weights.numel()
    mean_blocks: List[TensorBlock] = []
    second_moment_blocks: List[TensorBlock] = []
    absolute_second_moment_blocks: List[TensorBlock] = []
    reference_blocks: List[TensorBlock] = []
    for key, block in tensor.items():
        values, sample_names, sample_values = _reshape_block_by_local_system(
            block, n_local_systems
        )
        if reference is None:
            reference_values = values[0].clone()
        else:
            reference_values = reference.block(key).values
            if tuple(reference_values.shape) != tuple(values.shape[1:]):
                raise ValueError("reference and batch block shapes do not match")
        values = values - reference_values.unsqueeze(0)

        weight = weights.to(dtype=values.dtype, device=values.device)
        mean_view = [values.shape[0]] + [1] * (values.ndim - 1)
        mean_values = torch.sum(0.5 * weight.view(mean_view) * values, dim=0)

        component_dims = tuple(range(2, 2 + len(block.components)))
        if len(component_dims) == 0:
            squared_norms = values**2
        else:
            squared_norms = torch.sum(values**2, dim=component_dims)
        moment_view = [squared_norms.shape[0]] + [1] * (squared_norms.ndim - 1)
        second_moment_values = torch.sum(
            0.5 * weight.view(moment_view) * squared_norms, dim=0
        )
        absolute_second_moment_values = torch.sum(
            0.5 * torch.abs(weight).view(moment_view) * squared_norms, dim=0
        )

        samples = _prepend_system_to_samples(
            sample_names,
            sample_values,
            system_index,
            device=block.samples.values.device,
        )
        mean_blocks.append(
            TensorBlock(
                values=mean_values,
                samples=samples,
                components=block.components,
                properties=block.properties,
            )
        )
        second_moment_blocks.append(
            TensorBlock(
                values=second_moment_values,
                samples=samples,
                components=[],
                properties=block.properties,
            )
        )
        absolute_second_moment_blocks.append(
            TensorBlock(
                values=absolute_second_moment_values,
                samples=samples,
                components=[],
                properties=block.properties,
            )
        )
        if reference is None:
            reference_blocks.append(
                TensorBlock(
                    values=reference_values,
                    samples=samples,
                    components=block.components,
                    properties=block.properties,
                )
            )

    if reference is None:
        reference = TensorMap(tensor.keys, reference_blocks)
    return (
        TensorMap(tensor.keys, mean_blocks),
        TensorMap(tensor.keys, second_moment_blocks),
        TensorMap(tensor.keys, absolute_second_moment_blocks),
        reference,
    )


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


def _wigner_stack_nbytes(n_rotations: int, ell_max: int, dtype: torch.dtype) -> int:
    """Exact storage of stacked Wigner-D matrices through ``ell_max``."""
    # sum_{ell=0}^L (2 ell + 1)^2
    elements_per_rotation = (ell_max + 1) * (2 * ell_max + 1) * (2 * ell_max + 3) // 3
    element_size = torch.empty((), dtype=dtype).element_size()
    return n_rotations * elements_per_rotation * element_size


def _attach_wigner_stacks(
    transformations: List[O3Transformation],
    stacks: Dict[int, torch.Tensor],
    batch_start: int,
) -> None:
    """Seed transformations with views into immutable full-grid stacks."""
    for local_index, transformation in enumerate(transformations):
        grid_index = batch_start + local_index
        transformation._wigner_D_cache = {
            ell: stack[grid_index] for ell, stack in stacks.items()
        }


def _validate_base_outputs(
    raw: Dict[str, TensorMap], requested_outputs: Dict[str, ModelOutput]
) -> None:
    """Check the base-model output contract before quadrature processing."""
    missing = [name for name in requested_outputs if name not in raw]
    if len(missing) != 0:
        formatted = ", ".join(f"'{name}'" for name in missing)
        raise ValueError(f"base model did not return requested output(s): {formatted}")

    for name in requested_outputs:
        for block in raw[name].blocks():
            gradient_names = block.gradients_list()
            if len(gradient_names) != 0:
                raise ValueError(
                    f"base model output '{name}' contains explicit gradient "
                    f"'{gradient_names[0]}', which SymmetrizedModel does not support"
                )


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


def _validate_nonnegative_diagnostic(
    tensor: TensorMap,
    scale: TensorMap,
    *,
    n_grid_points: int,
    quantity: str,
    max_o3_lambda_grid: int,
) -> TensorMap:
    """Clamp round-off-sized negatives and reject signed-grid aliasing."""
    blocks: List[TensorBlock] = []
    for key, block in tensor.items():
        scale_values = scale.block(key).values
        invalid = (
            (~torch.isfinite(block.values))
            | (~torch.isfinite(scale_values))
            | (scale_values < 0)
        )
        if bool(torch.any(invalid).item()):
            raise ValueError(f"O(3) {quantity} or its error scale is invalid")

        epsilon = torch.finfo(block.values.dtype).eps
        n_epsilon = n_grid_points * epsilon
        gamma = n_epsilon / (1.0 - n_epsilon)
        tolerance = (
            64.0
            * gamma
            * torch.clamp(scale_values, min=torch.finfo(block.values.dtype).tiny)
        )
        if bool(torch.any(block.values < -tolerance).item()):
            raise ValueError(
                f"finite O(3) {quantity} is materially negative; the quadrature "
                "does not resolve this response. Increase max_o3_lambda_grid "
                f"above {max_o3_lambda_grid} and check convergence"
            )

        blocks.append(
            TensorBlock(
                values=torch.clamp(block.values, min=0.0),
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )
    return TensorMap(tensor.keys, blocks)


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
    component axes) is divided by the total component multiplicity read from
    the matching ``<name>_mean`` block. This is :math:`2\\lambda+1` for one
    irrep axis, and the product of the axis sizes for a higher-rank target. The
    result is then averaged over each system's samples before taking the square
    root, matching the normalization of an element-wise accuracy RMSE.

    Aggregating afterwards is up to the caller: to pool per-system values into
    a dataset-level RMSE, combine the squares weighted by the number of
    samples of each system, ``sqrt(sum_A N_A rmse_A^2 / sum_A N_A)``; a plain
    mean of per-system RMSEs is not the pooled RMSE when system sizes differ.
    Similarly, pooling over properties or blocks is a mean of squares. Systems
    that contribute no samples to a block get an RMSE of zero. Extensivity is
    not corrected either: for a total-energy output, dividing by the number of
    atoms per system is up to the caller.

    Variances returned by :class:`SymmetrizedModel` are non-negative: round-off
    residuals are set to zero, while materially negative signed-grid estimates
    raise with grid-convergence guidance. This helper likewise rejects negative
    or non-finite manually supplied variance entries.

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
        if bool(torch.any((~torch.isfinite(values)) | (values < 0)).item()):
            raise ValueError("variance values must be finite and non-negative")
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
                values=torch.sqrt(total),
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
    Wrapper around an atomistic model that approximates symmetrization over
    :math:`O(3)` with a deterministic finite quadrature and computes
    equivariance metrics. The result equals the Haar projection when the
    relevant orbit response and products are resolved by the chosen grid;
    otherwise convergence must be checked.

    For each grid operation :math:`g`, the direct response :math:`f(gX)` is
    back-rotated to a common frame before its mean, norm, and orientation
    variance are accumulated. Optional character projections instead act on
    the direct orbit and report its norm in the included :math:`O(3)` sectors.
    Generated standard spherical maps use canonical
    ``o3_lambda``/``o3_sigma`` keys.
    See the module reference for the projector, basis conventions, finite-grid
    theorem, output schema, and convergence requirements.

    :param base_model: atomistic model to symmetrize, either a module following
        the :py:class:`ModelInterface` call convention or an exported
        :py:class:`AtomisticModel` (including one loaded with
        :py:func:`load_atomistic_model`)
    :param max_o3_lambda_target: validation ceiling for spherical ``o3_lambda``
        components returned directly by the base model. Actual back-rotation
        caches are sized from the ranks present in each batch. Canonical
        Cartesian force/stress outputs are decomposed after this validation and
        are not bounded by it. This value also controls the default grid when no
        character cutoff is given, but is not an orbit-response bandwidth.
    :param max_o3_lambda_character: maximum O(3) angular momentum used for the
        character projections. If ``None`` (default), character projections are
        unavailable (calling :py:meth:`forward` with
        ``compute_character_projections=True`` raises an error) and the default
        quadrature grid follows ``max_o3_lambda_target`` instead. If given, it
        must be a non-negative integer.
    :param batch_size: positive integer number of rotations evaluated in one
        base-model call. It changes peak batch memory, not the total number of
        rotated copies.
    :param max_o3_lambda_grid: declared product-quadrature integration degree
        ``D``. It selects the smallest available Lebedev order at least ``D``
        and ``D + 1`` in-plane angles. If ``None`` (default), set to
        ``2 * max_o3_lambda_character`` when ``max_o3_lambda_character`` is
        given, and to ``2 * max_o3_lambda_target + 1`` otherwise. The character
        default is the exact product-quadrature boundary: a grid degree
        ``2 * L`` selects ``2 * L + 1`` in-plane angles. The target-only
        default keeps one additional in-plane angle as a conservative heuristic
        because a target-rank ceiling is not an orbit-response bandwidth. It
        must be a non-negative integer no larger than 131. For an unrestricted
        response, increase it until the requested statistics converge.
    :param storage_device: device on which intermediate and final results are
        kept. If ``None`` (default), back-rotation follows the System device and
        final results are moved to the wrapper's quadrature-buffer device. If
        set (e.g. ``"cpu"`` for a model on GPU), base-model outputs are moved
        there after each forward pass, trading transfer bandwidth for GPU
        memory; back-rotation, accumulation, and returned TensorMaps live there.
    :param wigner_cache_max_bytes: tensor-memory budget for the lazy persistent
        cache of full-grid Wigner-D matrices (default: 64 MiB). Transformation
        wrappers are recreated one batch at a time and are not retained.
        Reusing the matrices avoids rebuilding the same Wigner-D values for
        every system, forward call, and molecular-dynamics step. The cache is
        sized from the largest rank actually requested; if the complete tensor
        allocation would exceed this byte limit, the wrapper falls back to its
        bounded per-batch cache. Set this to zero to disable persistent caching.

    ``SymmetrizedModel`` does not support MPS execution. Stable quadrature,
    back-rotation, and reduction require float64, which the MPS backend does
    not provide. Registered MPS base state is rejected at construction;
    MPS storage, input Systems, and wrapper-controlled moves are rejected before
    evaluation. Manually moving an individual base submodule after construction
    is outside this preflight contract. This limitation is specific to this
    wrapper and does not change general metatomic O(3) or metatrain MPS support.

    Keep the wrapper in the dtype it was constructed with: casting it to a
    lower precision (e.g. ``.to(torch.float32)``) degrades the quadrature grid
    stored in its buffers.
    """

    def __init__(
        self,
        base_model,
        max_o3_lambda_target: int,
        max_o3_lambda_character: Optional[int] = None,
        batch_size: int = 32,
        max_o3_lambda_grid: Optional[int] = None,
        storage_device: Optional[str] = None,
        wigner_cache_max_bytes: int = _DEFAULT_WIGNER_CACHE_MAX_BYTES,
    ):
        super().__init__()
        self.base_model = base_model
        self._base_is_atomistic = is_atomistic_model(base_model)

        max_o3_lambda_target = _validated_integer(
            "max_o3_lambda_target", max_o3_lambda_target, 0
        )
        if max_o3_lambda_character is not None:
            max_o3_lambda_character = _validated_integer(
                "max_o3_lambda_character", max_o3_lambda_character, 0
            )
        if max_o3_lambda_grid is not None:
            max_o3_lambda_grid = _validated_integer(
                "max_o3_lambda_grid", max_o3_lambda_grid, 0
            )
        batch_size = _validated_integer("batch_size", batch_size, 1)
        wigner_cache_max_bytes = _validated_integer(
            "wigner_cache_max_bytes", wigner_cache_max_bytes, 0
        )

        storage_device = (
            None if storage_device is None else torch.device(storage_device)
        )
        if storage_device is not None and storage_device.type == "mps":
            raise ValueError(_MPS_UNSUPPORTED_MESSAGE)

        device = torch.device("cpu")
        found_reference_parameter = False
        for parameter in base_model.parameters():
            if not found_reference_parameter:
                device = parameter.device
                found_reference_parameter = True
            if parameter.device.type == "mps":
                raise ValueError(_MPS_UNSUPPORTED_MESSAGE)
        for buffer in base_model.buffers():
            if buffer.device.type == "mps":
                raise ValueError(_MPS_UNSUPPORTED_MESSAGE)

        self.max_o3_lambda_target = max_o3_lambda_target
        self.batch_size = batch_size
        self.wigner_cache_max_bytes = wigner_cache_max_bytes
        self.storage_device = storage_device
        if max_o3_lambda_grid is None:
            if max_o3_lambda_character is not None:
                max_o3_lambda_grid = 2 * max_o3_lambda_character
            else:
                max_o3_lambda_grid = 2 * max_o3_lambda_target + 1
        self.max_o3_lambda_grid = max_o3_lambda_grid
        self.max_o3_lambda_character = max_o3_lambda_character

        lebedev_order, n_inplane_rotations = _choose_quadrature(self.max_o3_lambda_grid)
        alpha, beta, gamma, w_so3 = get_euler_angles_quadrature(
            lebedev_order, n_inplane_rotations
        )
        w_so3 = w_so3 / np.sum(w_so3, dtype=np.float64)
        # the grid buffers are kept in float64: all statistics are accumulated
        # in double precision around a fixed reference value; rotations are
        # downcast to the model dtype only when transforming the input systems
        # the weights are a plain CPU float64 attribute, not a buffer, so a
        # user cast like `.to(torch.float32)` cannot touch them (weights
        # quantized to float32 no longer sum to 1 within ~1e-8, which alone
        # reintroduces the variance cancellation)
        self._so3_weights_float64 = torch.from_numpy(w_so3).to(dtype=torch.float64)

        so3_rotations = torch.from_numpy(
            _rotations_from_angles(alpha, beta, gamma).as_matrix()
        ).to(device=device, dtype=torch.float64)
        self.register_buffer("so3_rotations", so3_rotations)

        # Derived runtime data only: this is intentionally neither a parameter
        # nor a buffer, and is removed from whole-module pickle state below.
        # One device/dtype/grid entry is retained at a time, bounding persistent
        # memory and avoiding stale multi-device copies.
        self._wigner_cache: Dict[int, torch.Tensor] = {}
        self._wigner_cache_key: Optional[Tuple] = None

    def clear_wigner_cache(self) -> None:
        """Release cached quadrature Wigner-D matrices.

        The cache is also cleared automatically by :meth:`torch.nn.Module.to`
        and related dtype/device transformations. Normal in-place changes to
        the canonical quadrature buffer are detected through its tensor version;
        call this method after unsupported ``.data`` mutation.
        """
        self._wigner_cache = {}
        self._wigner_cache_key = None

    def _validate_no_mps_execution(self, systems: List[System]) -> None:
        """Reject MPS storage or inputs before model evaluation."""
        if self.storage_device is not None and self.storage_device.type == "mps":
            raise ValueError(_MPS_UNSUPPORTED_MESSAGE)
        if any(system.device.type == "mps" for system in systems):
            raise ValueError(_MPS_UNSUPPORTED_MESSAGE)

    def _apply(self, fn, recurse=True):
        # Preflight standard Module.to(...) targets before clearing derived
        # caches or moving any parameter/buffer in this wrapper subtree. An
        # integral probe follows the target device without requesting an
        # unsupported float64 allocation on MPS.
        try:
            target = fn(torch.empty(0, dtype=torch.int64, device="cpu"))
        except (RuntimeError, NotImplementedError) as error:
            if "mps" in str(error).casefold():
                raise ValueError(_MPS_UNSUPPORTED_MESSAGE) from error
            raise
        if target.device.type == "mps":
            raise ValueError(_MPS_UNSUPPORTED_MESSAGE)

        # A cached tensor dictionary is not registered as module state, so it
        # would not follow `.to(...)`. Drop it before parameters/buffers move;
        # the next spherical operation rebuilds it on the requested device.
        self.clear_wigner_cache()
        return super()._apply(fn, recurse=recurse)

    def __getstate__(self):
        # Wigner stacks are deterministically derived from serialized
        # quadrature buffers. Excluding them keeps whole-module torch.save and
        # deepcopy payloads independent of whether a wrapper has been warmed.
        state = super().__getstate__()
        state["_wigner_cache"] = {}
        state["_wigner_cache_key"] = None
        return state

    def _persistent_wigner_stacks(
        self,
        ell_max: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[int, torch.Tensor]]:
        """Return full-grid Wigner stacks, or ``None`` when they exceed the cap."""
        # ``torch.device("cuda")`` follows the current device and therefore
        # can denote different physical devices across calls. Resolve it before
        # keying or allocating so such a change replaces, rather than reuses,
        # an entry on the old device.
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        required_bytes = _wigner_stack_nbytes(
            self.so3_rotations.shape[0], ell_max, dtype
        )
        if required_bytes > self.wigner_cache_max_bytes:
            # Honor a budget lowered after the cache was built. Avoid replacing
            # empty dictionaries on every batch of an always-uncached run.
            if self._wigner_cache_key is not None:
                self.clear_wigner_cache()
            return None

        source = self.so3_rotations
        key = (
            device.type,
            -1 if device.index is None else device.index,
            str(dtype),
            source.device.type,
            -1 if source.device.index is None else source.device.index,
            str(source.dtype),
            tuple(source.shape),
            id(source),
            source.data_ptr(),
            source._version,
        )
        if key == self._wigner_cache_key and all(
            ell in self._wigner_cache for ell in range(ell_max + 1)
        ):
            retained_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in self._wigner_cache.values()
            )
            if retained_bytes <= self.wigner_cache_max_bytes:
                return self._wigner_cache

        # Release old stacks before allocating replacements to bound peak memory.
        self.clear_wigner_cache()

        # Use the fallback path so cached and uncached results remain identical.
        n_rotations = source.shape[0]
        # Publish only after every batch succeeds.
        new_cache = {
            ell: torch.empty(
                (n_rotations, 2 * ell + 1, 2 * ell + 1),
                device=device,
                dtype=dtype,
            )
            for ell in range(ell_max + 1)
        }
        for batch_start in range(0, n_rotations, self.batch_size):
            batch_stop = min(batch_start + self.batch_size, n_rotations)
            batch_matrices = source[batch_start:batch_stop].transpose(-1, -2)
            batch_stacks = _wigner_stacks_from_matrices(
                batch_matrices,
                ell_max,
                is_inverted=False,
                output_device=device,
                output_dtype=dtype,
            )
            for ell, stack in batch_stacks.items():
                new_cache[ell][batch_start:batch_stop] = stack
        self._wigner_cache = new_cache
        # Publish the validity key last, after all potentially failing work.
        self._wigner_cache_key = key
        return self._wigner_cache

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
            projections. Requires ``max_o3_lambda_character`` to be set and
            ``max_o3_lambda_grid >= 2 * max_o3_lambda_character``. Exact Haar
            projections additionally require the response bandwidth to be
            covered by the cutoff and grid.
        :param compute_gradients: if True, compute conservative forces and, for
            fully periodic systems, 3D stress via autograd. When False (default),
            the grid evaluation runs under ``torch.no_grad()`` to save memory.
        :param energy_name: name of the output used to derive forces and stress
            when ``compute_gradients=True``; it must be present in ``outputs``.
            The derived quantities are always returned under the ``forces`` and
            ``stress`` names, which are reserved in this mode.
        :return: dictionary with symmetrized outputs and equivariance metrics.
            Statistics are accumulated and returned in float64, independently
            of the model dtype; the model itself runs in its own dtype, so
            errors below the round-off of its outputs remain unmeasurable.
        """
        self._validate_no_mps_execution(systems)

        for name, output in outputs.items():
            if len(output.explicit_gradients) != 0:
                raise ValueError(
                    "SymmetrizedModel does not support explicit gradients for "
                    f"output '{name}'; request it without explicit_gradients. "
                    "Use compute_gradients=True to derive conservative forces "
                    "and stress from an energy output."
                )

        if compute_character_projections:
            if self.max_o3_lambda_character is None:
                raise ValueError(
                    "max_o3_lambda_character must be set to compute character "
                    "projections"
                )
            required_grid_degree = 2 * self.max_o3_lambda_character
            if self.max_o3_lambda_grid < required_grid_degree:
                raise ValueError(
                    "the quadrature grid is too coarse for character projections "
                    f"up to lambda={self.max_o3_lambda_character} "
                    f"(max_o3_lambda_grid={self.max_o3_lambda_grid} < "
                    f"{required_grid_degree}); set max_o3_lambda_grid >= 2 * "
                    "max_o3_lambda_character, or leave it None to use a "
                    "sufficient default"
                )

        device = self.so3_rotations.device
        # outputs are detached from any autograd graph before back-rotation,
        # so offloading is safe in compute_gradients mode too
        result_device = (
            self.storage_device if self.storage_device is not None else device
        )

        with torch.enable_grad() if compute_gradients else torch.no_grad():
            results = self._eval_over_grid(
                systems,
                outputs,
                selected_atoms,
                compute_character_projections=compute_character_projections,
                compute_gradients=compute_gradients,
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
            conservative forces (and stress, for fully periodic systems) obtained
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
        """Call a raw module or adapt options for an exported AtomisticModel."""
        if self._base_is_atomistic:
            options = ModelEvaluationOptions(
                length_unit="",
                outputs=outputs,
                selected_atoms=selected_atoms,
            )
            return self.base_model(systems, options, check_consistency=False)
        return self.base_model(systems, outputs, selected_atoms)

    def _evaluate_batch(
        self,
        system: System,
        inversion_batch: torch.Tensor,
        outputs: Dict[str, ModelOutput],
        local_selected_atoms: Optional[Labels],
        compute_gradients: bool,
        energy_name: str,
        forward_ell_max: int,
        backrotation_device: torch.device,
        cell_volume: Optional[torch.Tensor],
        is_inverted: bool,
    ) -> Dict[str, TensorMap]:
        """Evaluate one rotation batch and move detached outputs to float64 storage."""
        work_device = system.positions.device
        work_dtype = system.positions.dtype

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
                cell_volume=cell_volume,
                quadrature_is_inverted=is_inverted,
            )
            _validate_base_outputs(raw, outputs)
            # the statistics need no gradients, and the forces and stress are
            # already materialized: detach so each batch's autograd graph is
            # freed instead of being kept alive by the accumulators for the
            # whole grid
            return {
                k: mts.detach(v).to(device=backrotation_device, dtype=torch.float64)
                for k, v in raw.items()
            }

        transformed_systems = _transform_system_batch(
            system,
            inversion_batch.to(device=work_device, dtype=work_dtype),
            forward_ell_max,
            is_inverted=is_inverted,
        )
        raw = self._run_base_model(
            transformed_systems,
            outputs,
            local_selected_atoms,
        )
        _validate_base_outputs(raw, outputs)
        return {
            k: v.to(device=backrotation_device, dtype=torch.float64)
            for k, v in raw.items()
        }

    def _accumulate_system(
        self,
        system: System,
        i_sys: int,
        outputs: Dict[str, ModelOutput],
        requested_output_names: List[str],
        selected_atoms: Optional[Labels],
        compute_character_projections: bool,
        compute_gradients: bool,
        energy_name: str,
    ) -> Tuple[
        Dict[str, TensorMap],
        Dict[str, TensorMap],
        Dict[str, TensorMap],
        Dict[str, TensorMap],
    ]:
        """Stream and reduce all rotation batches for one input system."""
        references: Dict[str, TensorMap] = {}
        centered_mean_accumulators: Dict[str, TensorMap] = {}
        centered_second_moment_accumulators: Dict[str, TensorMap] = {}
        absolute_second_moment_accumulators: Dict[str, TensorMap] = {}
        proj_pos: Dict[str, Dict] = {}
        proj_neg: Dict[str, Dict] = {}

        n_rotations = self.so3_rotations.size(0)
        work_device = system.positions.device
        work_dtype = system.positions.dtype
        backrotation_device = (
            self.storage_device if self.storage_device is not None else work_device
        )
        selected_atoms_by_batch_size: Dict[int, Optional[Labels]] = {}

        # Validate the 3D volume once per original system. The gradient helper is
        # called once per rotation batch and inversion, so doing this there would
        # introduce repeated device synchronizations for the same cell.
        cell_volume: Optional[torch.Tensor] = None
        if compute_gradients and bool(torch.all(system.pbc).item()):
            candidate_volume = torch.abs(torch.linalg.det(system.cell.detach()))
            invalid_volume = (~torch.isfinite(candidate_volume)) | (
                candidate_volume == 0
            )
            if bool(invalid_volume.item()):
                raise ValueError(
                    "can not compute 3D stress for a singular or non-finite "
                    "periodic cell"
                )
            cell_volume = candidate_volume

        # Input and output angular momenta are independent: inspect custom data
        # directly instead of coupling its transformation cache to the declared
        # output limit.
        forward_ell_max = 0
        for data_name in system.known_data():
            data_lambda = _max_spherical_lambda(system.get_data(data_name))
            forward_ell_max = max(forward_ell_max, data_lambda)

        for batch_start in range(0, n_rotations, self.batch_size):
            batch_stop = min(batch_start + self.batch_size, n_rotations)
            n_local_systems = batch_stop - batch_start
            weights = self._so3_weights_float64[batch_start:batch_stop]
            batch_rotations = self.so3_rotations[batch_start:batch_stop].to(
                device=work_device, dtype=work_dtype
            )
            if n_local_systems not in selected_atoms_by_batch_size:
                selected_atoms_by_batch_size[n_local_systems] = (
                    _selected_atoms_for_local_systems(
                        selected_atoms,
                        i_sys,
                        n_local_systems,
                    )
                )
            local_selected_atoms = selected_atoms_by_batch_size[n_local_systems]
            proper_uncached_wigner_stacks: Dict[int, torch.Tensor] = {}
            proper_transformation_ell_max = -1

            for inversion in (1, -1):
                out = self._evaluate_batch(
                    system,
                    inversion * batch_rotations,
                    outputs,
                    local_selected_atoms,
                    compute_gradients,
                    energy_name,
                    forward_ell_max,
                    backrotation_device,
                    cell_volume,
                    is_inverted=inversion == -1,
                )

                present_output_names = [
                    name for name in requested_output_names if name in out
                ]
                if len(present_output_names) == 0:
                    continue

                # Size inverse-frame Wigner caches from the spherical ranks that
                # are actually present in this batch. ``max_o3_lambda_target`` is
                # a validation ceiling, and can be much larger than the output in
                # a particular call; using it as an allocation target needlessly
                # makes every character operation scale with that ceiling.
                output_ell_max = 0
                has_spherical_output = False
                for name in present_output_names:
                    output_lambda = _max_spherical_lambda(out[name])
                    if output_lambda > self.max_o3_lambda_target:
                        raise ValueError(
                            f"output '{name}' contains "
                            f"o3_lambda={output_lambda} components, larger "
                            f"than max_o3_lambda_target="
                            f"{self.max_o3_lambda_target}; increase "
                            "max_o3_lambda_target"
                        )
                    if output_lambda >= 0:
                        has_spherical_output = True
                        output_ell_max = max(output_ell_max, output_lambda)

                transformation_ell_max = output_ell_max
                if compute_character_projections:
                    assert self.max_o3_lambda_character is not None
                    transformation_ell_max = max(
                        transformation_ell_max,
                        self.max_o3_lambda_character,
                    )

                needs_wigner = has_spherical_output or compute_character_projections
                persistent_wigner_stacks: Optional[Dict[int, torch.Tensor]] = None
                if needs_wigner:
                    persistent_wigner_stacks = self._persistent_wigner_stacks(
                        transformation_ell_max,
                        device=backrotation_device,
                        dtype=torch.float64,
                    )

                inverse_mats = (
                    inversion
                    * self.so3_rotations[batch_start:batch_stop].transpose(-1, -2)
                ).to(device=backrotation_device, dtype=torch.float64)
                transformations = [
                    O3Transformation._from_validated_matrix(
                        R,
                        transformation_ell_max,
                        is_inverted=inversion == -1,
                    )
                    for R in inverse_mats.unbind(0)
                ]
                if persistent_wigner_stacks is not None:
                    _attach_wigner_stacks(
                        transformations,
                        persistent_wigner_stacks,
                        batch_start,
                    )

                uncached_wigner_stacks: Dict[int, torch.Tensor] = {}
                if persistent_wigner_stacks is None:
                    if (
                        inversion == -1
                        and needs_wigner
                        and len(proper_uncached_wigner_stacks) != 0
                        and transformation_ell_max <= proper_transformation_ell_max
                    ):
                        uncached_wigner_stacks = {
                            ell: proper_uncached_wigner_stacks[ell]
                            for ell in range(transformation_ell_max + 1)
                        }
                        _attach_wigner_stacks(
                            transformations,
                            uncached_wigner_stacks,
                            0,
                        )
                    elif needs_wigner:
                        uncached_wigner_stacks = _wigner_stacks_from_matrices(
                            inverse_mats,
                            transformation_ell_max,
                            is_inverted=inversion == -1,
                            output_device=backrotation_device,
                            output_dtype=torch.float64,
                        )
                        _attach_wigner_stacks(
                            transformations,
                            uncached_wigner_stacks,
                            0,
                        )

                    if inversion == 1:
                        proper_uncached_wigner_stacks = uncached_wigner_stacks
                        proper_transformation_ell_max = transformation_ell_max
                wigner_stacks: Dict[int, torch.Tensor] = {}
                if compute_character_projections:
                    assert self.max_o3_lambda_character is not None
                    if persistent_wigner_stacks is None:
                        wigner_stacks = {
                            ell: uncached_wigner_stacks[ell]
                            for ell in range(self.max_o3_lambda_character + 1)
                        }
                    else:
                        wigner_stacks = {
                            ell: persistent_wigner_stacks[ell][batch_start:batch_stop]
                            for ell in range(self.max_o3_lambda_character + 1)
                        }

                for name in present_output_names:
                    tensor = out[name]

                    backtransformed = transform_tensor(
                        tensor,
                        # transform_tensor uses these Systems only to establish
                        # batch cardinality; reusing the original avoids copying
                        # all custom data and neighbor lists to the reduction device.
                        [system] * n_local_systems,
                        transformations,
                    )
                    for final_name, decomposed_tensor in _decompose_output(
                        name, backtransformed
                    ).items():
                        (
                            mean_contribution,
                            second_moment_contribution,
                            absolute_second_moment_contribution,
                            reference,
                        ) = _reduce_weighted_batch_moments(
                            decomposed_tensor,
                            weights,
                            i_sys,
                            reference=references.get(final_name),
                        )
                        references[final_name] = reference
                        _accumulate_tensormap(
                            centered_mean_accumulators,
                            final_name,
                            mean_contribution,
                        )
                        _accumulate_tensormap(
                            centered_second_moment_accumulators,
                            final_name,
                            second_moment_contribution,
                        )
                        _accumulate_tensormap(
                            absolute_second_moment_accumulators,
                            final_name,
                            absolute_second_moment_contribution,
                        )

                    if compute_character_projections:
                        _accumulate_batch(
                            proj_pos,
                            proj_neg,
                            _decompose_output(name, tensor),
                            weights,
                            wigner_stacks,
                            self.max_o3_lambda_character,
                            inversion,
                        )

        projections: Dict[str, TensorMap] = {}
        if compute_character_projections:
            projections = _finalize_system(
                proj_pos,
                proj_neg,
                i_sys,
                self.max_o3_lambda_character,
            )

        means: Dict[str, TensorMap] = {}
        variances: Dict[str, TensorMap] = {}
        norm_squared: Dict[str, TensorMap] = {}
        n_grid_points = 2 * n_rotations
        absolute_weight_sum = float(
            torch.sum(torch.abs(self._so3_weights_float64)).item()
        )
        for name, reference in references.items():
            centered_mean = centered_mean_accumulators[name]
            mean = mts.add(reference, centered_mean)
            raw_variance = _finalize_variance(
                centered_second_moment_accumulators[name],
                centered_mean,
            )
            absolute_second_moment = absolute_second_moment_accumulators[name]
            variance_scale = mts.add(
                absolute_second_moment,
                _mean_norm_squared_tensor(centered_mean),
            )
            variance = _validate_nonnegative_diagnostic(
                raw_variance,
                variance_scale,
                n_grid_points=n_grid_points,
                quantity="variance",
                max_o3_lambda_grid=self.max_o3_lambda_grid,
            )
            raw_norm_squared = mts.add(
                raw_variance,
                _mean_norm_squared_tensor(mean),
            )
            norm_scale = mts.multiply(
                mts.add(
                    mts.multiply(
                        _mean_norm_squared_tensor(reference), absolute_weight_sum
                    ),
                    absolute_second_moment,
                ),
                2.0,
            )
            means[name] = mean
            variances[name] = variance
            norm_squared[name] = _validate_nonnegative_diagnostic(
                raw_norm_squared,
                norm_scale,
                n_grid_points=n_grid_points,
                quantity="squared norm",
                max_o3_lambda_grid=self.max_o3_lambda_grid,
            )

        return means, variances, norm_squared, projections

    def _eval_over_grid(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
        compute_character_projections: bool,
        compute_gradients: bool = False,
        energy_name: str = "energy",
    ) -> Dict[str, TensorMap]:
        """
        Stream the model over the O(3) quadrature, accumulating mean, variance, and
        character projections without ever materializing the full-grid TensorMap.

        :param systems: list of systems to evaluate
        :param outputs: dictionary of model outputs to symmetrize
        :param selected_atoms: optional Labels specifying which atoms to consider
        :param compute_character_projections: if True, also compute character
            projections
        :param compute_gradients: if True, compute forces/stress via autograd
        :param energy_name: name of the output forces/stress are derived from
        :return: dictionary with all symmetrized outputs and metrics
        """
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

            requested_output_names.append("forces")
            if any(bool(torch.all(s.pbc).item()) for s in systems):
                requested_output_names.append("stress")

        decomposed_origins: Dict[str, str] = {}
        for source_name in requested_output_names:
            for final_name in _decomposed_output_names(source_name):
                if final_name in decomposed_origins:
                    raise ValueError(
                        f"output names '{decomposed_origins[final_name]}' and "
                        f"'{source_name}' both produce '{final_name}' after "
                        "spherical decomposition; rename one requested output"
                    )
                decomposed_origins[final_name] = source_name

        mean_accumulators: Dict[str, List[TensorMap]] = {}
        variance_accumulators: Dict[str, List[TensorMap]] = {}
        norm_squared_accumulators: Dict[str, List[TensorMap]] = {}
        character_projection_accumulators: Dict[str, List[TensorMap]] = {}

        for i_sys, system in enumerate(systems):
            means, variances, norm_squared, projections = self._accumulate_system(
                system,
                i_sys,
                outputs,
                requested_output_names,
                selected_atoms,
                compute_character_projections,
                compute_gradients,
                energy_name,
            )
            for name, tensor in means.items():
                _append_tensormap(mean_accumulators, name, tensor)
            for name, tensor in variances.items():
                _append_tensormap(variance_accumulators, name, tensor)
            for name, tensor in norm_squared.items():
                _append_tensormap(norm_squared_accumulators, name, tensor)
            for name, tensor in projections.items():
                _append_tensormap(character_projection_accumulators, name, tensor)

        results: Dict[str, TensorMap] = {}
        for name, mean_tensors in mean_accumulators.items():
            results[name + "_mean"] = _join_tensormap_list(mean_tensors)
            results[name + "_var"] = _join_tensormap_list(variance_accumulators[name])
            results[name + "_norm_squared"] = _join_tensormap_list(
                norm_squared_accumulators[name]
            )

        for name, tensors in character_projection_accumulators.items():
            results[name + "_character_projection"] = _join_tensormap_list(tensors)

        return results
