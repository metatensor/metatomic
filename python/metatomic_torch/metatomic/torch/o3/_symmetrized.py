"""
:py:class:`SymmetrizedModel`, which averages another model's outputs over a finite
O(3) quadrature. The wrapped model is evaluated on rotated and inverted copies of
each system, and the results are transformed back to the input frame to build the
O(3) average together with the equivariance diagnostics of the requested outputs.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap, dtype_name

from .. import (
    ModelCapabilities,
    ModelOutput,
    NeighborListOptions,
    System,
    register_autograd_neighbors,
)
from .._quantities import (
    MAX_ANGULAR_MOMENTUM_PER_CATEGORY,
    QUANTITIES_WITH_GRADIENTS,
    STANDARD_QUANTITY_CATEGORIES,
    current_quantity_name,
)
from ..model import (
    AtomisticModel,
    ModelInterface,
)
from ._decompose import decompose_quantity
from ._projections import (
    character_projection_coefficients_from_batch,
    character_projection_tensormap_from_cosets,
)
from ._quadrature import choose_quadrature, get_rotation_quadrature
from ._transformations import (
    O3Transformations,
    max_o3_lambda_in_tensor,
)
from ._utils import (
    copy_tensormap_info,
    group_samples_by_rotated_copy,
    map_selected_atoms_to_rotated_copies,
    restore_input_system_to_samples,
    validate_integer,
)


def _check_o3_lambda_limit(
    tensor: TensorMap,
    tensor_description: str,
    max_angular_momentum: int,
    limit_name: str,
) -> None:
    """Check a TensorMap's spherical component ranks against one limit."""
    tensor_max_o3_lambda = max_o3_lambda_in_tensor(tensor)
    if tensor_max_o3_lambda > max_angular_momentum:
        raise ValueError(
            f"{tensor_description} contains o3_lambda={tensor_max_o3_lambda}, "
            f"exceeding {limit_name}={max_angular_momentum}"
        )


def _parse_output_request(requested_name: str) -> Tuple[str, str]:
    """Return the underlying output name and requested calculation."""
    variance_prefix = "o3::variance::"
    character_projection_prefix = "o3::character_projection::"

    if requested_name.startswith(variance_prefix):
        source_name = requested_name[len(variance_prefix) :]
        calculation = "variance"
    elif requested_name.startswith(character_projection_prefix):
        source_name = requested_name[len(character_projection_prefix) :]
        calculation = "character_projection"
    else:
        if requested_name.startswith("o3::"):
            raise ValueError(
                f"requested output '{requested_name}' uses the 'o3::' prefix "
                "reserved by SymmetrizedModel, but is neither a variance nor a "
                "character-projection request"
            )
        source_name = requested_name
        calculation = "average"

    if len(source_name) == 0:
        raise ValueError(
            f"requested output '{requested_name}' does not identify an "
            "underlying model output"
        )

    return source_name, calculation


def _group_output_requests(
    outputs: Dict[str, ModelOutput],
) -> Tuple[
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
]:
    """Group public requests by underlying output and calculation.

    The returned dictionaries map each source name to the exact spelling the
    caller requested it under.
    """
    source_sample_kinds: Dict[str, str] = {}
    average_names: Dict[str, str] = {}
    variance_names: Dict[str, str] = {}
    character_projection_names: Dict[str, str] = {}

    for requested_name, output in outputs.items():
        source_name, calculation = _parse_output_request(requested_name)
        sample_kind = output.sample_kind
        if source_name in source_sample_kinds:
            previous_sample_kind = source_sample_kinds[source_name]
            if sample_kind != previous_sample_kind:
                raise ValueError(
                    f"all requests derived from '{source_name}' must use the same "
                    f"sample_kind; got '{previous_sample_kind}' and '{sample_kind}'"
                )
        else:
            source_sample_kinds[source_name] = sample_kind

        if calculation == "average":
            average_names[source_name] = requested_name
        elif calculation == "variance":
            variance_names[source_name] = requested_name
        else:
            character_projection_names[source_name] = requested_name

    return (
        source_sample_kinds,
        average_names,
        variance_names,
        character_projection_names,
    )


def _infer_max_angular_momentum(
    names: Dict[str, ModelOutput],
    kind: str,
    argument: str,
) -> int:
    """Guess an angular-momentum limit from standard quantity names."""
    max_angular_momentum = 0
    found_standard = False
    custom_names: List[str] = []
    for name in names.keys():
        quantity = current_quantity_name(name).split("/", 1)[0]
        if quantity == "feature":
            # features are not an irreducible representation of O(3): they are
            # passed through unchanged and never rotated back
            found_standard = True
            continue
        if quantity not in STANDARD_QUANTITY_CATEGORIES:
            # a custom name says nothing about its angular momenta, so it is
            # skipped: if it turns out to carry a larger one and is requested,
            # _check_o3_lambda_limit rejects it at forward time, naming the limit
            custom_names.append(name)
            continue
        found_standard = True
        category = STANDARD_QUANTITY_CATEGORIES[quantity]
        max_angular_momentum = max(
            max_angular_momentum, MAX_ANGULAR_MOMENTUM_PER_CATEGORY[category]
        )

    if not found_standard and len(custom_names) != 0:
        raise ValueError(
            f"no standard quantities were found among the {kind}s "
            f"{custom_names}, please set {argument} explicitly"
        )

    return max_angular_momentum


def _reduce_weighted_centered_batch(
    tensor: TensorMap,
    weights: torch.Tensor,
    input_system_index: int,
    reference: Optional[TensorMap],
    compute_second_moments: bool,
) -> Tuple[
    TensorMap,
    Optional[TensorMap],
    Optional[TensorMap],
    TensorMap,
]:
    """Accumulate one rotation batch's weighted moments, centered on a reference.

    Centering on the first rotated copy keeps both terms of ``E[X^2] - E[X]^2``
    of the order of the variation itself, so their subtraction does not lose
    significant digits to cancellation when the mean response is large.
    """
    n_rotated_copies = weights.numel()
    centered_first_moment_blocks: List[TensorBlock] = []
    second_moment_blocks: List[TensorBlock] = []
    absolute_second_moment_blocks: List[TensorBlock] = []
    reference_blocks: List[TensorBlock] = []

    for key, block in tensor.items():
        values, sample_names, sample_values = group_samples_by_rotated_copy(
            block, n_rotated_copies
        )
        if reference is None:
            # clone so the reference does not keep the full batch tensor alive
            reference_values = values[0].clone()
        else:
            reference_values = reference.block(key).values
            matching_shape = reference_values.dim() + 1 == values.dim()
            if matching_shape:
                for axis in range(reference_values.dim()):
                    if reference_values.size(axis) != values.size(axis + 1):
                        matching_shape = False
            if not matching_shape:
                raise ValueError(
                    "reference and batch block shapes do not match: reference is "
                    f"{list(reference_values.shape)}, batch is {list(values.shape)}"
                )
        centered_values = values - reference_values.unsqueeze(0)

        batch_weights = weights.to(
            dtype=centered_values.dtype,
            device=centered_values.device,
        )
        weight_shape = [centered_values.shape[0]] + [1] * (centered_values.ndim - 1)
        centered_first_moment_values = torch.sum(
            batch_weights.view(weight_shape) * centered_values,
            dim=0,
        )

        samples = restore_input_system_to_samples(
            sample_names,
            sample_values,
            input_system_index,
            device=block.samples.values.device,
        )
        centered_first_moment_blocks.append(
            TensorBlock(
                values=centered_first_moment_values,
                samples=samples,
                components=block.components,
                properties=block.properties,
            )
        )

        if compute_second_moments:
            squared_norms = centered_values**2
            if len(block.components) != 0:
                n_components = 1
                for component in block.components:
                    n_components *= len(component)
                squared_norms = squared_norms.reshape(
                    centered_values.shape[0],
                    centered_values.shape[1],
                    n_components,
                    centered_values.shape[-1],
                ).sum(dim=2)
            moment_weight_shape = [squared_norms.shape[0]] + [1] * (
                squared_norms.ndim - 1
            )
            second_moment_values = torch.sum(
                batch_weights.view(moment_weight_shape) * squared_norms,
                dim=0,
            )
            absolute_second_moment_values = torch.sum(
                torch.abs(batch_weights).view(moment_weight_shape) * squared_norms,
                dim=0,
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

    second_moment: Optional[TensorMap] = None
    absolute_second_moment: Optional[TensorMap] = None
    if compute_second_moments:
        second_moment = TensorMap(tensor.keys, second_moment_blocks)
        absolute_second_moment = TensorMap(
            tensor.keys,
            absolute_second_moment_blocks,
        )

    return (
        TensorMap(tensor.keys, centered_first_moment_blocks),
        second_moment,
        absolute_second_moment,
        reference,
    )


def _add_tensormap_contribution(
    accumulator: Dict[str, TensorMap],
    output_name: str,
    contribution: TensorMap,
) -> None:
    """Add a TensorMap contribution to the running sum for one output."""
    if output_name in accumulator:
        accumulator[output_name] = mts.add(accumulator[output_name], contribution)
    else:
        accumulator[output_name] = contribution


def _component_norm_squared(tensor: TensorMap) -> TensorMap:
    """Return squared values summed over all component axes."""
    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        values = block.values.square()
        if len(block.components) != 0:
            values = values.flatten(start_dim=1, end_dim=-2).sum(dim=1)
        blocks.append(
            TensorBlock(
                values=values,
                samples=block.samples,
                components=[],
                properties=block.properties,
            )
        )
    return TensorMap(tensor.keys, blocks)


def _clamp_roundoff_negative_diagnostic(
    tensor: TensorMap,
    scale: TensorMap,
    *,
    n_grid_points: int,
    quantity: str,
    max_angular_momentum_grid: int,
) -> TensorMap:
    """Clamp round-off negatives and reject invalid or materially negative values.

    The variance and character projections are non-negative by construction, but
    the finite quadrature evaluates them as differences of large accumulated
    sums, so exact zeros come out as tiny values of either sign. Values within
    the accumulated round-off bound (estimated from ``scale``) are clamped to
    zero; more negative values mean the quadrature did not resolve the response,
    which is reported instead of silently returned.
    """
    blocks: List[TensorBlock] = []
    for key, block in tensor.items():
        scale_values = scale.block(key).values
        if bool(torch.any(~torch.isfinite(block.values)).item()):
            raise ValueError(f"O(3) {quantity} is not finite for block {key.print()}")
        if bool(torch.any(~torch.isfinite(scale_values)).item()):
            raise ValueError(
                f"round-off scale of the O(3) {quantity} is not finite for "
                f"block {key.print()}"
            )

        # TorchScript does not support torch.finfo; use the IEEE-754 values for
        # the floating-point dtypes supported by metatomic models.
        if block.values.dtype == torch.float64:
            epsilon = 2.220446049250313e-16
            tiny = 2.2250738585072014e-308
        elif block.values.dtype == torch.float32:
            epsilon = 1.1920928955078125e-07
            tiny = 1.1754943508222875e-38
        else:
            raise TypeError(
                "O(3) diagnostics require float32 or float64 values, got "
                f"{dtype_name(block.values.dtype)}"
            )

        n_epsilon = n_grid_points * epsilon
        gamma = n_epsilon / (1.0 - n_epsilon)
        tolerance = (
            64.0
            * gamma
            * torch.clamp(
                scale_values,
                min=tiny,
            )
        )
        if bool(torch.any(block.values < -tolerance).item()):
            raise ValueError(
                f"finite O(3) {quantity} is materially negative; the quadrature "
                "does not resolve this response. Increase max_angular_momentum_grid "
                f"above {max_angular_momentum_grid} and check convergence"
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


def _variance_from_centered_moments(
    centered_first_moment: TensorMap,
    centered_second_moment: TensorMap,
    absolute_centered_second_moment: TensorMap,
    *,
    n_grid_points: int,
    max_angular_momentum_grid: int,
) -> TensorMap:
    """Compute a validated component-summed variance from centered moments."""
    centered_first_moment_norm_squared = _component_norm_squared(centered_first_moment)
    variance = mts.subtract(
        centered_second_moment,
        centered_first_moment_norm_squared,
    )
    roundoff_scale = mts.add(
        absolute_centered_second_moment,
        centered_first_moment_norm_squared,
    )
    return _clamp_roundoff_negative_diagnostic(
        variance,
        roundoff_scale,
        n_grid_points=n_grid_points,
        quantity="variance",
        max_angular_momentum_grid=max_angular_momentum_grid,
    )


def _mean_variance_over_components(
    variance: TensorMap,
    component_layout: TensorMap,
) -> TensorMap:
    """Average component-summed variance over each block's components."""
    # both maps were built from the same moments earlier in this forward
    assert variance.keys == component_layout.keys

    blocks: List[TensorBlock] = []
    for key, block in variance.items():
        assert len(block.components) == 0

        layout_block = component_layout.block(key)
        assert (
            layout_block.samples == block.samples
            and layout_block.properties == block.properties
        )

        n_components = 1
        for component in layout_block.components:
            n_components *= len(component)

        blocks.append(
            TensorBlock(
                values=block.values / n_components,
                samples=block.samples,
                components=[],
                properties=block.properties,
            )
        )

    return TensorMap(variance.keys, blocks)


def _differentiable_system(
    system: System,
    positions: torch.Tensor,
    strain: torch.Tensor,
) -> System:
    """Copy ``system`` with ``positions``, and positions and cell strained by
    ``strain``."""
    differentiable = System(
        types=system.types,
        positions=positions @ strain,
        cell=system.cell.detach() @ strain,
        pbc=system.pbc,
    )
    for options in system.known_neighbor_lists():
        neighbors = system.get_neighbor_list(options)
        neighbors = TensorBlock(
            values=neighbors.values.detach(),
            samples=neighbors.samples,
            components=neighbors.components,
            properties=neighbors.properties,
        )
        register_autograd_neighbors(differentiable, neighbors)
        differentiable.add_neighbor_list(options, neighbors)
    for name in system.known_data():
        differentiable.add_data(
            name, system.get_data(name), _private_warn_on_deprecated=False
        )
    return differentiable


def _add_explicit_gradients(
    tensor: TensorMap,
    positions_gradients: List[torch.Tensor],
    strain_gradients: List[torch.Tensor],
    parameters: List[str],
    input_system_index: int,
) -> TensorMap:
    """Attach the requested explicit gradients to the blocks of ``tensor``."""
    device = tensor.device
    xyz = torch.arange(3, dtype=torch.int32, device=device).reshape(-1, 1)
    blocks: List[TensorBlock] = []
    for block_index, block in enumerate(tensor.blocks()):
        result = TensorBlock(
            values=block.values,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        # gradient values are flattened over the block components and properties
        value_shape = list(block.values.shape[1:])
        for parameter in parameters:
            if parameter == "positions":
                n_atoms = positions_gradients[block_index].shape[0]
                values = positions_gradients[block_index].reshape(
                    [n_atoms, 3] + value_shape
                )
                samples_values = torch.zeros(
                    (n_atoms, 3), dtype=torch.int32, device=device
                )
                samples_values[:, 1] = input_system_index
                samples_values[:, 2] = torch.arange(n_atoms, device=device)
                samples = Labels(
                    ["sample", "system", "atom"], samples_values, assume_unique=True
                )
                components = [Labels(["xyz"], xyz)]
            else:
                values = strain_gradients[block_index].reshape([1, 3, 3] + value_shape)
                samples = Labels(
                    ["sample"], torch.zeros((1, 1), dtype=torch.int32, device=device)
                )
                components = [Labels(["xyz_1"], xyz), Labels(["xyz_2"], xyz)]
            result.add_gradient(
                parameter,
                TensorBlock(
                    values=values,
                    samples=samples,
                    components=components + block.components,
                    properties=block.properties,
                ),
            )
        blocks.append(result)
    return TensorMap(tensor.keys, blocks)


def _accumulate_gradients(
    backrotated_outputs: Dict[str, TensorMap],
    weights: torch.Tensor,
    positions: torch.Tensor,
    strain: torch.Tensor,
    positions_gradients: Dict[str, List[torch.Tensor]],
    strain_gradients: Dict[str, List[torch.Tensor]],
):
    """Add the positions and strain gradients of the weighted sums of
    ``backrotated_outputs`` over the copies of one batch, freeing its graph."""
    targets: Dict[str, List[torch.Tensor]] = {}
    n_targets = 0
    for name, tensor in backrotated_outputs.items():
        targets[name] = []
        for block in tensor.blocks():
            copy_weights = weights[block.samples.column("system")]
            copy_weights = copy_weights.reshape([-1] + [1] * (block.values.dim() - 1))
            target = (copy_weights * block.values).sum(dim=0).reshape(-1)
            targets[name].append(target)
            if target.requires_grad:
                n_targets += target.shape[0]

        if name not in positions_gradients:
            positions_gradients[name] = [
                torch.zeros(
                    list(positions.shape) + [target.shape[0]],
                    dtype=positions.dtype,
                    device=positions.device,
                )
                for target in targets[name]
            ]
            strain_gradients[name] = [
                torch.zeros(
                    (3, 3, target.shape[0]), dtype=strain.dtype, device=strain.device
                )
                for target in targets[name]
            ]

    # the graph is only kept alive until the last gradient of the batch
    target_index = 0
    for name, blocks in targets.items():
        for block_index, target in enumerate(blocks):
            # outputs that do not depend on the inputs have zero gradients
            if not target.requires_grad:
                continue
            for entry in range(target.shape[0]):
                target_index += 1
                gradients = torch.autograd.grad(
                    [target[entry]],
                    [positions, strain],
                    retain_graph=target_index < n_targets,
                    allow_unused=True,
                )
                positions_gradient = gradients[0]
                strain_gradient = gradients[1]
                if positions_gradient is not None:
                    positions_gradients[name][block_index][..., entry] += (
                        positions_gradient
                    )
                if strain_gradient is not None:
                    strain_gradients[name][block_index][..., entry] += strain_gradient


class SymmetrizedModel(torch.nn.Module):
    """
    Wrap a model with finite-quadrature O(3) averaging and equivariance
    diagnostics.

    Requesting an output declared by the wrapped model returns its O(3)
    average, evaluated over rotated and inverted copies of the input and
    transformed back to the input frame. Requests named
    ``o3::variance::<name>`` return the component-averaged equivariance
    variance of the ``<name>`` output and, when ``max_angular_momentum_character`` is
    set, ``o3::character_projection::<name>`` requests return its unnormalized
    squared character-projection contributions. Outputs whose blocks carry no
    recognized component labels are not rotated back, so their variance
    measures the deviation from invariance only. The definition of these
    quantities, their TensorMap representation, and convergence guidance for
    the quadrature are documented in :ref:`symmetrized-model`.

    When an input requires gradients, differentiating an averaged result
    through PyTorch autograd retains the source-model activations from all
    quadrature batches; ``batch_size`` does not bound their total size. Use
    :py:func:`torch.inference_mode` or :py:func:`torch.no_grad` when derivatives
    are not required. Alternatively, averages of per-system outputs (such as
    ``energy``) can be requested with explicit ``"positions"`` and ``"strain"``
    gradients: these are computed with autograd and accumulated one batch at a
    time, so that ``batch_size`` bounds the memory usage. The values of such
    requests are not differentiable with respect to the inputs.

    :param model: underlying :py:class:`ModelInterface`. The :py:meth:`wrap` method
        obtains this module from :py:attr:`AtomisticModel.module`.
    :param max_angular_momentum_target: maximum angular momentum that can be transformed
        back to the input frame when an average or variance of an
        already-spherical output is requested. Cartesian outputs and
        character-only requests are not limited by this value.
    :param max_angular_momentum_input: maximum angular momentum that can be rotated in
        already-spherical custom System data. The default of zero still allows
        Cartesian custom inputs. The ``ModelOutput`` declarations returned by a
        model's ``requested_inputs()`` do not specify which angular momenta
        may occur in the corresponding TensorMaps, so this limit must be
        supplied before export for all required Wigner-D matrices to be
        serialized.
    :param max_angular_momentum_character: maximum angular momentum included in
        character projections. ``None`` disables character-projection outputs; zero
        enables the scalar (``o3_lambda = 0``) contribution only.
    :param max_angular_momentum_grid: quadrature integration degree. If ``None``, use
        the larger of ``2 * max_angular_momentum_target + 1`` and
        ``2 * max_angular_momentum_character`` when character projections are enabled.
        An explicit value must be non-negative and no larger than the highest available
        Lebedev order, 131; a value below ``2 * max_angular_momentum_character`` is
        rejected. Zero replaces the grid by the identity, so that the outputs are only
        averaged over the inversion.
    :param include_inversion: if ``False``, average over the proper rotations only,
        i.e. over SO(3) instead of O(3), without character projections.
    :param batch_size: positive number of transformed systems evaluated in one call to
        ``model``. The default is 32.
    """

    max_angular_momentum_character: Optional[int]
    _requested_inputs: Dict[str, ModelOutput]
    _requested_neighbor_lists: List[NeighborListOptions]

    def __init__(
        self,
        model: ModelInterface,
        *,
        max_angular_momentum_target: int,
        max_angular_momentum_input: int = 0,
        max_angular_momentum_character: Optional[int] = None,
        max_angular_momentum_grid: Optional[int] = None,
        include_inversion: bool = True,
        batch_size: int = 32,
    ):
        super().__init__()

        self._model = model
        self._requested_inputs = {}
        self._requested_neighbor_lists = []
        self.max_angular_momentum_target = validate_integer(
            "max_angular_momentum_target", max_angular_momentum_target, 0
        )
        self.max_angular_momentum_input = validate_integer(
            "max_angular_momentum_input", max_angular_momentum_input, 0
        )
        if max_angular_momentum_character is not None:
            max_angular_momentum_character = validate_integer(
                "max_angular_momentum_character", max_angular_momentum_character, 0
            )
        self.max_angular_momentum_character = max_angular_momentum_character
        self.batch_size = validate_integer("batch_size", batch_size, 1)
        self.include_inversion = include_inversion

        if self.max_angular_momentum_character is not None and not include_inversion:
            raise ValueError("character projections require include_inversion=True")

        if max_angular_momentum_grid is None:
            max_angular_momentum_grid = 2 * self.max_angular_momentum_target + 1
            if self.max_angular_momentum_character is not None:
                max_angular_momentum_grid = max(
                    max_angular_momentum_grid,
                    2 * self.max_angular_momentum_character,
                )
        else:
            max_angular_momentum_grid = validate_integer(
                "max_angular_momentum_grid", max_angular_momentum_grid, 0
            )
        if (
            self.max_angular_momentum_character is not None
            and max_angular_momentum_grid < 2 * self.max_angular_momentum_character
        ):
            raise ValueError(
                "max_angular_momentum_grid must be at least twice "
                "max_angular_momentum_character"
            )
        self.max_angular_momentum_grid = max_angular_momentum_grid

        device = torch.device("cpu")
        dtype = torch.float64
        for parameter in model.parameters():
            device = parameter.device
            dtype = parameter.dtype
            break
        else:
            for buffer in model.buffers():
                device = buffer.device
                dtype = buffer.dtype
                break

        if self.max_angular_momentum_grid == 0:
            # a single point integrates constants exactly
            rotation_matrices = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
            rotation_weights = torch.ones(1, dtype=dtype, device=device)
        else:
            lebedev_order, n_rotations = choose_quadrature(
                self.max_angular_momentum_grid
            )
            rotations, weights = get_rotation_quadrature(
                lebedev_order,
                n_rotations,
            )
            rotation_matrices = torch.from_numpy(rotations).to(
                dtype=dtype,
                device=device,
            )
            rotation_weights = torch.from_numpy(weights).to(
                dtype=dtype,
                device=device,
            )

        max_angular_momentum_wigner = max(
            self.max_angular_momentum_input,
            self.max_angular_momentum_target,
            0
            if self.max_angular_momentum_character is None
            else self.max_angular_momentum_character,
        )
        self._max_angular_momentum_wigner = max_angular_momentum_wigner

        # a single module holding all the rotations, sliced into batches in forward:
        # TorchScript unrolls loops over a ModuleList, which would inline a copy of
        # the wrapped model for every batch
        self._transformations = O3Transformations(
            rotation_matrices,
            max_angular_momentum_wigner,
        )
        self.register_buffer("_rotation_weights", rotation_weights)

    @staticmethod
    def wrap(
        model: AtomisticModel,
        *,
        max_angular_momentum_target: Optional[int] = None,
        max_angular_momentum_input: Optional[int] = None,
        max_angular_momentum_character: Optional[int] = None,
        max_angular_momentum_grid: Optional[int] = None,
        include_inversion: bool = True,
        batch_size: int = 32,
    ) -> AtomisticModel:
        """
        Wrap an exported model with O(3) averaging and diagnostics.

        The returned model retains every output declared by ``model`` under its
        original name. Requesting such an output evaluates its O(3) average (SO(3)
        average if ``include_inversion`` is ``False``). Additional outputs named
        ``o3::variance::<name>`` provide the component-averaged equivariance
        variance, unless the identity is the only operation. If
        ``max_angular_momentum_character`` is set, ``o3::character_projection::<name>``
        outputs provide squared character projections through that angular momentum.

        The original metadata, requested inputs, neighbor lists, dtype, and compatible
        capabilities are preserved. Per-system averages of energy and custom outputs
        also offer explicit ``"positions"`` and ``"strain"`` gradients, which
        simulation engines such as :py:class:`~metatomic_ase.MetatomicCalculator`
        use when available.

        Constructing a wrapper requires SciPy 1.15 or newer for its Lebedev
        quadrature. SciPy is not required to evaluate a wrapper that has
        already been saved.

        :param model: the :py:class:`AtomisticModel` to wrap
        :param max_angular_momentum_target: maximum angular momentum accepted in
            already-spherical model outputs requested for averaging or variance.
            When ``None``, it is guessed as the largest angular momentum of the
            standard quantities declared by ``model``; non-standard outputs are
            skipped, and an explicit value is required if ``model`` declares
            outputs but none of them is a standard quantity.
        :param max_angular_momentum_input: maximum angular momentum accepted in custom
            System data. When ``None``, it is guessed the same way from the
            quantities in ``model.requested_inputs()``.
        :param max_angular_momentum_character: maximum angular momentum in character
            projections, or ``None`` to disable them
        :param max_angular_momentum_grid: quadrature integration degree, selected
            automatically when ``None``. Zero uses the identity only.
        :param include_inversion: if ``False``, average over SO(3) instead of O(3)
        :param batch_size: number of transformed Systems evaluated in one model call
        """
        if not isinstance(model, AtomisticModel):
            raise TypeError("model must be an AtomisticModel")

        capabilities = model.capabilities()

        if max_angular_momentum_target is None:
            max_angular_momentum_target = _infer_max_angular_momentum(
                capabilities.outputs,
                "output",
                "max_angular_momentum_target",
            )
        if max_angular_momentum_input is None:
            max_angular_momentum_input = _infer_max_angular_momentum(
                model.requested_inputs(use_new_names=True),
                "input",
                "max_angular_momentum_input",
            )

        wrapper = SymmetrizedModel(
            model.module,
            max_angular_momentum_target=max_angular_momentum_target,
            max_angular_momentum_input=max_angular_momentum_input,
            max_angular_momentum_character=max_angular_momentum_character,
            max_angular_momentum_grid=max_angular_momentum_grid,
            include_inversion=include_inversion,
            batch_size=batch_size,
        )
        group = "O(3)" if include_inversion else "SO(3)"

        outputs: Dict[str, ModelOutput] = {}
        # private field: the as-declared output names, deliberately without the
        # deprecation aliases added by the public accessors
        for name in model._model_capabilities_outputs_names:
            if name.startswith("o3::"):
                raise ValueError(
                    "the wrapped model output '"
                    + name
                    + "' uses a prefix reserved by SymmetrizedModel"
                )

            source_output = capabilities.outputs[name]
            average_description = group + " average of the '" + name + "' output."
            if source_output.description != "":
                average_description += " " + source_output.description
            explicit_gradients: List[str] = []
            # custom outputs, containing "::", are free to define gradients, and
            # only atom pair outputs can not be requested per system
            quantity = current_quantity_name(name).split("/", 1)[0]
            if source_output.sample_kind != "atom_pair" and (
                quantity in QUANTITIES_WITH_GRADIENTS or "::" in quantity
            ):
                explicit_gradients = ["positions", "strain"]
            outputs[name] = ModelOutput(
                unit=source_output.unit,
                sample_kind=source_output.sample_kind,
                explicit_gradients=explicit_gradients,
                description=average_description,
            )

            squared_unit = ""
            if source_output.unit != "":
                squared_unit = "(" + source_output.unit + ")^2"
            if wrapper._has_variance():
                outputs["o3::variance::" + name] = ModelOutput(
                    unit=squared_unit,
                    sample_kind=source_output.sample_kind,
                    explicit_gradients=[],
                    description=(
                        group
                        + " equivariance variance of the '"
                        + name
                        + "' output for each sample, averaged over components."
                    ),
                )
            if max_angular_momentum_character is not None:
                outputs["o3::character_projection::" + name] = ModelOutput(
                    unit=squared_unit,
                    sample_kind=source_output.sample_kind,
                    explicit_gradients=[],
                    description=(
                        "Unnormalized squared O(3) character-projection "
                        "contributions of the '"
                        + name
                        + "' output, resolved by chi_lambda and chi_sigma."
                    ),
                )

        # private field: the as-declared inputs, deliberately without deprecation
        # aliases
        wrapper._requested_inputs = {
            name: requested_input
            for name, requested_input in model._requested_inputs.items()
        }
        # copy the options: constructing the AtomisticModel below mutates them by
        # adding requestors and setting the length unit
        for options in model.requested_neighbor_lists():
            copied_options = NeighborListOptions(
                options.cutoff,
                options.full_list,
                options.strict,
            )
            for requestor in options.requestors():
                copied_options.add_requestor(requestor)
            wrapper._requested_neighbor_lists.append(copied_options)
        new_capabilities = ModelCapabilities(
            outputs=outputs,
            atomic_types=capabilities.atomic_types,
            interaction_range=capabilities.interaction_range,
            length_unit=capabilities.length_unit,
            supported_devices=capabilities.supported_devices,
            dtype=capabilities.dtype,
        )
        # the dtype of the quadrature buffers follows the parameters of the model, use
        # the declared one instead for models without parameters
        dtype = torch.float32 if capabilities.dtype == "float32" else torch.float64
        return AtomisticModel(
            wrapper.to(dtype=dtype).eval(),
            model.metadata(),
            capabilities=new_capabilities,
        )

    def _has_variance(self) -> bool:
        """Whether there is more than one operation to compute variances over."""
        return self.include_inversion or self.max_angular_momentum_grid != 0

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        """Return the neighbor lists requested by the wrapped model."""
        return self._requested_neighbor_lists

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        """Return the custom System data requested by the wrapped model."""
        return self._requested_inputs

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        """Evaluate the requested O(3) averages and diagnostics."""
        if len(outputs) == 0:
            empty: Dict[str, TensorMap] = {}
            return empty
        if len(systems) == 0:
            # the metadata of the outputs (keys, sample and property labels) only
            # becomes known by evaluating the wrapped model on at least one
            # system, so there is no way to build correctly-labelled empty results
            raise ValueError("SymmetrizedModel requires at least one System")

        (
            source_sample_kinds,
            average_names,
            variance_names,
            character_projection_names,
        ) = _group_output_requests(outputs)
        if (
            len(character_projection_names) != 0
            and self.max_angular_momentum_character is None
        ):
            raise ValueError(
                "max_angular_momentum_character must be set to request "
                "character projections"
            )
        if len(variance_names) != 0 and not self._has_variance():
            raise ValueError(
                "variances require more than one operation, they can not be "
                "requested with max_angular_momentum_grid=0 and "
                "include_inversion=False"
            )

        gradient_parameters: Dict[str, List[str]] = {}
        for requested_name, output in outputs.items():
            if len(output.explicit_gradients) == 0:
                continue
            source_name, calculation = _parse_output_request(requested_name)
            if calculation != "average" or output.sample_kind != "system":
                raise ValueError(
                    "explicit gradients are only supported for per-system averages, "
                    f"not for '{requested_name}'"
                )
            gradient_parameters[source_name] = output.explicit_gradients

        source_outputs: Dict[str, ModelOutput] = {}
        for source_name in source_sample_kinds:
            source_outputs[source_name] = ModelOutput(
                sample_kind=source_sample_kinds[source_name],
            )

        per_output_results: Dict[str, List[TensorMap]] = {}
        for requested_name in outputs:
            empty_results: List[TensorMap] = []
            per_output_results[requested_name] = empty_results

        integration_dtype = self._transformations.dtype
        if integration_dtype != torch.float64:
            if integration_dtype != torch.float32:
                raise TypeError(
                    "SymmetrizedModel integration buffers must use float32 or "
                    f"float64, got {dtype_name(integration_dtype)}"
                )
            warnings.warn(
                "SymmetrizedModel is running in float32; averages and "
                "diagnostics will be less accurate",
                stacklevel=2,
            )

        if len(gradient_parameters) != 0 and not torch.is_grad_enabled():
            raise ValueError(
                "explicit gradients are computed with autograd and can not be "
                "requested when it is disabled, e.g. by torch.no_grad() or "
                "torch.inference_mode()"
            )

        for input_system_index, system in enumerate(systems):
            system_results = self._evaluate_system(
                system,
                input_system_index,
                source_outputs,
                average_names,
                variance_names,
                character_projection_names,
                gradient_parameters,
                selected_atoms,
            )
            for requested_name in outputs:
                per_output_results[requested_name].append(
                    system_results[requested_name]
                )

        results: Dict[str, TensorMap] = {}
        for requested_name in outputs:
            results[requested_name] = mts.join(
                per_output_results[requested_name],
                "samples",
                different_keys="union",
            )
        return results

    def _evaluate_system(
        self,
        system: System,
        input_system_index: int,
        source_outputs: Dict[str, ModelOutput],
        average_names: Dict[str, str],
        variance_names: Dict[str, str],
        character_projection_names: Dict[str, str],
        gradient_parameters: Dict[str, List[str]],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        """Stream all quadrature batches for one input System."""
        work_dtype = system.positions.dtype
        work_device = system.positions.device
        integration_dtype = self._transformations.dtype
        if work_dtype != torch.float32 and work_dtype != torch.float64:
            raise TypeError(
                "SymmetrizedModel requires float32 or float64 Systems, got "
                f"{dtype_name(work_dtype)}"
            )
        if work_dtype != integration_dtype:
            raise TypeError(
                "SymmetrizedModel and input Systems must use the same dtype; got "
                f"{dtype_name(work_dtype)} systems and "
                f"{dtype_name(integration_dtype)} quadrature"
            )
        if (
            self._transformations.device != work_device
            or self._rotation_weights.device != work_device
        ):
            raise ValueError(
                "SymmetrizedModel and input Systems must use the same device"
            )

        for data_name in system.known_data():
            _check_o3_lambda_limit(
                system.get_data(data_name),
                f"custom input '{data_name}'",
                self.max_angular_momentum_input,
                "max_angular_momentum_input",
            )

        character_max = 0
        configured_character_max = self.max_angular_momentum_character
        if configured_character_max is not None:
            character_max = configured_character_max

        positions = system.positions
        strain = torch.eye(3, dtype=work_dtype, device=work_device)
        if len(gradient_parameters) != 0:
            positions = positions.detach().requires_grad_(True)
            strain.requires_grad_(True)
        # weighted sums of the positions and strain gradients of each block of
        # the averages, flattened over the block components and properties
        positions_gradients: Dict[str, List[torch.Tensor]] = {}
        strain_gradients: Dict[str, List[torch.Tensor]] = {}

        average_references: Dict[str, TensorMap] = {}
        average_first_moments: Dict[str, TensorMap] = {}
        variance_references: Dict[str, TensorMap] = {}
        variance_first_moments: Dict[str, TensorMap] = {}
        variance_second_moments: Dict[str, TensorMap] = {}
        variance_absolute_second_moments: Dict[str, TensorMap] = {}
        proper_character_coefficients: Dict[str, TensorMap] = {}
        improper_character_coefficients: Dict[str, TensorMap] = {}

        n_cosets = 2 if self.include_inversion else 1
        n_rotations = self._transformations.matrices.size(0)
        for start in range(0, n_rotations, self.batch_size):
            stop = min(start + self.batch_size, n_rotations)
            n_rotated_copies = stop - start
            is_first_batch = start == 0

            so3_weights = self._rotation_weights[start:stop]
            o3_weights = so3_weights / n_cosets
            local_selected_atoms = map_selected_atoms_to_rotated_copies(
                selected_atoms,
                input_system_index,
                n_rotated_copies,
            )

            inverse_character_wigner_matrices: List[torch.Tensor] = []
            if len(character_projection_names) != 0:
                for chi_lambda in range(character_max + 1):
                    inverse_character_wigner_matrices.append(
                        self._transformations.inverse_wigner_D_matrices(chi_lambda)[
                            start:stop
                        ]
                    )

            for coset_index in range(n_cosets):
                is_improper = coset_index == 1
                batch_system = system
                if len(gradient_parameters) != 0:
                    # rebuilt for every model call, since computing the gradients
                    # frees the graph of the previous one
                    batch_system = _differentiable_system(system, positions, strain)
                transformed_systems = self._transformations.transform_systems(
                    [batch_system for _ in range(n_rotated_copies)],
                    is_improper,
                    start,
                    stop,
                )
                raw_outputs = self._model(
                    transformed_systems,
                    source_outputs,
                    local_selected_atoms,
                )
                for source_name in source_outputs:
                    if source_name not in raw_outputs:
                        raise ValueError(
                            "underlying model did not return requested output "
                            f"'{source_name}'"
                        )
                    raw_tensor = raw_outputs[source_name]
                    for block in raw_tensor.blocks():
                        gradient_names = block.gradients_list()
                        if len(gradient_names) != 0:
                            raise ValueError(
                                f"underlying output '{source_name}' contains "
                                f"unsupported explicit gradient '{gradient_names[0]}'"
                            )
                    # the component metadata does not change across batches:
                    # check it once per output
                    if (
                        is_first_batch
                        and not is_improper
                        and (
                            source_name in average_names
                            or source_name in variance_names
                        )
                    ):
                        _check_o3_lambda_limit(
                            raw_tensor,
                            f"output '{source_name}'",
                            self.max_angular_momentum_target,
                            "max_angular_momentum_target",
                        )

                if len(gradient_parameters) != 0:
                    backrotated_outputs: Dict[str, TensorMap] = {}
                    for source_name in gradient_parameters:
                        backrotated_outputs[source_name] = (
                            self._transformations.inverse_transform_tensormap(
                                raw_outputs[source_name],
                                None,
                                is_improper,
                                start,
                                stop,
                            )
                        )
                    _accumulate_gradients(
                        backrotated_outputs,
                        o3_weights,
                        positions,
                        strain,
                        positions_gradients,
                        strain_gradients,
                    )
                    detached: Dict[str, TensorMap] = {}
                    for name, tensor in raw_outputs.items():
                        detached[name] = mts.detach(tensor)
                    raw_outputs = detached

                for source_name in source_outputs:
                    tensor = raw_outputs[source_name].to(
                        dtype=integration_dtype,
                        device=work_device,
                    )
                    if source_name in average_names or source_name in variance_names:
                        backrotated = self._transformations.inverse_transform_tensormap(
                            tensor,
                            None,
                            is_improper,
                            start,
                            stop,
                        )

                        if source_name in average_names:
                            has_average_reference = source_name in average_references
                            average_reference: Optional[TensorMap] = None
                            if has_average_reference:
                                average_reference = average_references[source_name]
                            (
                                first_moment,
                                _,
                                _,
                                updated_average_reference,
                            ) = _reduce_weighted_centered_batch(
                                backrotated,
                                o3_weights,
                                input_system_index,
                                average_reference,
                                compute_second_moments=False,
                            )
                            if not has_average_reference:
                                updated_average_reference = copy_tensormap_info(
                                    backrotated,
                                    updated_average_reference,
                                )
                            average_references[source_name] = updated_average_reference
                            _add_tensormap_contribution(
                                average_first_moments,
                                source_name,
                                first_moment,
                            )

                        if source_name in variance_names:
                            diagnostic_tensor = decompose_quantity(
                                source_name,
                                backrotated,
                            )
                            variance_reference: Optional[TensorMap] = None
                            if source_name in variance_references:
                                variance_reference = variance_references[source_name]
                            (
                                first_moment,
                                second_moment,
                                absolute_second_moment,
                                variance_reference,
                            ) = _reduce_weighted_centered_batch(
                                diagnostic_tensor,
                                o3_weights,
                                input_system_index,
                                variance_reference,
                                compute_second_moments=True,
                            )
                            # always computed with compute_second_moments=True
                            assert second_moment is not None
                            assert absolute_second_moment is not None
                            variance_references[source_name] = variance_reference
                            _add_tensormap_contribution(
                                variance_first_moments,
                                source_name,
                                first_moment,
                            )
                            _add_tensormap_contribution(
                                variance_second_moments,
                                source_name,
                                second_moment,
                            )
                            _add_tensormap_contribution(
                                variance_absolute_second_moments,
                                source_name,
                                absolute_second_moment,
                            )

                    if source_name in character_projection_names:
                        direct_tensor = decompose_quantity(source_name, tensor)
                        contribution = character_projection_coefficients_from_batch(
                            direct_tensor,
                            so3_weights,
                            inverse_character_wigner_matrices,
                            input_system_index,
                        )
                        if is_improper:
                            _add_tensormap_contribution(
                                improper_character_coefficients,
                                source_name,
                                contribution,
                            )
                        else:
                            _add_tensormap_contribution(
                                proper_character_coefficients,
                                source_name,
                                contribution,
                            )

        results: Dict[str, TensorMap] = {}
        for source_name, requested_name in average_names.items():
            mean = mts.add(
                average_references[source_name],
                average_first_moments[source_name],
            )
            mean = copy_tensormap_info(average_references[source_name], mean)
            if source_name in positions_gradients:
                mean = _add_explicit_gradients(
                    mean,
                    positions_gradients[source_name],
                    strain_gradients[source_name],
                    gradient_parameters[source_name],
                    input_system_index,
                )
            results[requested_name] = mean.to(
                dtype=work_dtype,
                device=work_device,
            )

        for source_name, requested_name in variance_names.items():
            variance = _variance_from_centered_moments(
                variance_first_moments[source_name],
                variance_second_moments[source_name],
                variance_absolute_second_moments[source_name],
                n_grid_points=n_cosets * n_rotations,
                max_angular_momentum_grid=self.max_angular_momentum_grid,
            )
            variance = _mean_variance_over_components(
                variance,
                variance_references[source_name],
            )
            results[requested_name] = variance.to(
                dtype=work_dtype,
                device=work_device,
            )

        for source_name, requested_name in character_projection_names.items():
            projection = character_projection_tensormap_from_cosets(
                proper_character_coefficients[source_name],
                improper_character_coefficients[source_name],
            )
            results[requested_name] = projection.to(
                dtype=work_dtype,
                device=work_device,
            )

        return results
