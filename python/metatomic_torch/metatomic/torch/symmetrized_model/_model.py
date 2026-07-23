from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelInterface,
    ModelOutput,
    NeighborListOptions,
    System,
    register_autograd_neighbors,
)

from ..o3._tranformations import (
    _max_o3_lambda_in_tensor,
    _transform_tensor_with_precomputed_matrices,
)
from ._decompose import _decompose_output
from ._projections import (
    _character_projection_coefficients_from_batch,
    _character_projection_tensormap_from_cosets,
)
from ._quadrature import _choose_quadrature, get_rotation_quadrature
from ._utils import (
    _group_samples_by_rotated_copy,
    _map_selected_atoms_to_rotated_copies,
    _restore_input_system_to_samples,
    _validate_integer,
)
from ._wigner_storage import (
    _build_packed_wigner_matrices,
    _wigner_matrices_for_lambda,
)


_DEFAULT_MAX_WIGNER_STORAGE_BYTES = 64 * 1024 * 1024  # 64 MiB


def _transform_system_geometry_batch(
    system: System,
    matrices: torch.Tensor,
) -> List[System]:
    """Transform System geometry and neighbor lists with internal O(3) matrices."""
    if (
        matrices.dim() != 3
        or matrices.size(0) == 0
        or matrices.size(1) != 3
        or matrices.size(2) != 3
    ):
        raise ValueError("matrices must have shape (N, 3, 3) with N > 0")
    if (
        matrices.dtype != system.positions.dtype
        or matrices.device != system.positions.device
    ):
        raise ValueError("system and matrices must have the same dtype and device")

    if matrices.size(0) == 1:
        positions = (system.positions @ matrices[0].transpose(0, 1)).unsqueeze(0)
        cells = (system.cell @ matrices[0].transpose(0, 1)).unsqueeze(0)
    else:
        positions = system.positions.unsqueeze(0) @ matrices.transpose(1, 2)
        cells = system.cell.unsqueeze(0) @ matrices.transpose(1, 2)

    transformed_systems: List[System] = []
    for index in range(matrices.size(0)):
        transformed_systems.append(
            System(
                types=system.types,
                positions=positions[index],
                cell=cells[index],
                pbc=system.pbc,
            )
        )

    for options in system.known_neighbor_lists():
        neighbors = system.get_neighbor_list(options)
        source_values = neighbors.values.detach().squeeze(-1)
        if matrices.size(0) == 1:
            neighbor_values = (source_values @ matrices[0].transpose(0, 1)).unsqueeze(0)
        else:
            neighbor_values = source_values.unsqueeze(0) @ matrices.transpose(1, 2)
        for index in range(matrices.size(0)):
            rotated_neighbors = TensorBlock(
                values=neighbor_values[index].unsqueeze(-1),
                samples=neighbors.samples,
                components=neighbors.components,
                properties=neighbors.properties,
            )
            register_autograd_neighbors(
                transformed_systems[index],
                rotated_neighbors,
            )
            transformed_systems[index].add_neighbor_list(
                options,
                rotated_neighbors,
            )

    return transformed_systems


def _check_o3_lambda_limit(
    tensor: TensorMap,
    tensor_description: str,
    max_o3_lambda: int,
    limit_name: str,
) -> None:
    """Check a TensorMap's spherical component ranks against one limit."""
    tensor_max_o3_lambda = _max_o3_lambda_in_tensor(tensor)
    if tensor_max_o3_lambda > max_o3_lambda:
        raise ValueError(
            tensor_description
            + " contains o3_lambda="
            + str(tensor_max_o3_lambda)
            + ", exceeding "
            + limit_name
            + "="
            + str(max_o3_lambda)
        )


def _transform_system_batch(
    system: System,
    matrices: torch.Tensor,
    wigner_matrices: List[torch.Tensor],
    max_o3_lambda_input: int,
    is_improper: bool,
) -> List[System]:
    """Transform a System batch, including its custom TensorMap data."""
    data_names = system.known_data()
    for data_name in data_names:
        _check_o3_lambda_limit(
            system.get_data(data_name),
            "custom input '" + data_name + "'",
            max_o3_lambda_input,
            "max_o3_lambda_input",
        )

    transformed_systems = _transform_system_geometry_batch(system, matrices)
    if len(data_names) == 0:
        return transformed_systems

    for index in range(len(transformed_systems)):
        wigner_matrices_for_copy: List[torch.Tensor] = []
        for rank_matrices in wigner_matrices:
            wigner_matrices_for_copy.append(rank_matrices[index : index + 1])

        for data_name in data_names:
            transformed_systems[index].add_data(
                data_name,
                _transform_tensor_with_precomputed_matrices(
                    system.get_data(data_name),
                    matrices[index : index + 1],
                    wigner_matrices_for_copy,
                    is_improper,
                ),
            )

    return transformed_systems


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
        source_name = requested_name
        calculation = "average"

    if len(source_name) == 0:
        raise ValueError(
            "requested output '"
            + requested_name
            + "' does not identify an underlying model output"
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
    """Group public requests by underlying output and calculation."""
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
                    "all requests derived from '"
                    + source_name
                    + "' must use the same sample_kind; got '"
                    + previous_sample_kind
                    + "' and '"
                    + sample_kind
                    + "'"
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
    """Accumulate one rotation batch's reference-centered weighted moments."""
    n_rotated_copies = weights.numel()
    centered_first_moment_blocks: List[TensorBlock] = []
    second_moment_blocks: List[TensorBlock] = []
    absolute_second_moment_blocks: List[TensorBlock] = []
    reference_blocks: List[TensorBlock] = []

    for key, block in tensor.items():
        values, sample_names, sample_values = _group_samples_by_rotated_copy(
            block, n_rotated_copies
        )
        if reference is None:
            reference_values = values[0].clone()
        else:
            reference_values = reference.block(key).values
            matching_shape = reference_values.dim() + 1 == values.dim()
            if matching_shape:
                for axis in range(reference_values.dim()):
                    if reference_values.size(axis) != values.size(axis + 1):
                        matching_shape = False
            if not matching_shape:
                raise ValueError("reference and batch block shapes do not match")
        centered_values = values - reference_values.unsqueeze(0)

        # Any proper/improper weight split is applied by the caller.
        batch_weights = weights.to(
            dtype=centered_values.dtype,
            device=centered_values.device,
        )
        weight_shape = [centered_values.shape[0]] + [1] * (centered_values.ndim - 1)
        centered_first_moment_values = torch.sum(
            batch_weights.view(weight_shape) * centered_values,
            dim=0,
        )

        samples = _restore_input_system_to_samples(
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


def _copy_tensormap_info(source: TensorMap, result: TensorMap) -> TensorMap:
    """Copy global information from ``source`` to ``result``."""
    for info_name, info_value in source.info().items():
        result.set_info(info_name, info_value)
    return result


def _join_per_system_tensormaps(tensors: List[TensorMap]) -> TensorMap:
    """Join one TensorMap per input system along their sample axes."""
    if len(tensors) == 0:
        raise ValueError("expected at least one per-system TensorMap")

    keys = tensors[0].keys
    different_keys = "error"
    for index in range(1, len(tensors)):
        if tensors[index].keys != keys:
            different_keys = "union"
            break

    return mts.join(tensors, "samples", different_keys=different_keys)


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
    max_o3_lambda_grid: int,
) -> TensorMap:
    """Clamp round-off negatives and reject invalid or materially negative values."""
    blocks: List[TensorBlock] = []
    for key, block in tensor.items():
        scale_values = scale.block(key).values
        invalid = (
            (~torch.isfinite(block.values))
            | (~torch.isfinite(scale_values))
            | (scale_values < 0)
        )
        if bool(torch.any(invalid).item()):
            raise ValueError(f"O(3) {quantity} or its round-off scale is invalid")

        # TorchScript does not support torch.finfo; use the IEEE-754 values for
        # the floating-point dtypes supported by metatomic models.
        if block.values.dtype == torch.float64:
            epsilon = 2.220446049250313e-16
            tiny = 2.2250738585072014e-308
        elif block.values.dtype == torch.float32:
            epsilon = 1.1920928955078125e-07
            tiny = 1.1754943508222875e-38
        else:
            raise TypeError("O(3) diagnostics require float32 or float64 values")

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


def _variance_from_centered_moments(
    centered_first_moment: TensorMap,
    centered_second_moment: TensorMap,
    absolute_centered_second_moment: TensorMap,
    *,
    n_grid_points: int,
    max_o3_lambda_grid: int,
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
        max_o3_lambda_grid=max_o3_lambda_grid,
    )


def _mean_variance_over_components(
    variance: TensorMap,
    component_layout: TensorMap,
) -> TensorMap:
    """Average component-summed variance over each block's components."""
    if variance.keys != component_layout.keys:
        raise ValueError("variance and component-layout keys do not match")

    blocks: List[TensorBlock] = []
    for key, block in variance.items():
        if len(block.components) != 0:
            raise ValueError("component-summed variance must not have components")

        layout_block = component_layout.block(key)
        if (
            layout_block.samples != block.samples
            or layout_block.properties != block.properties
        ):
            raise ValueError("variance and component-layout metadata do not match")

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


class SymmetrizedModel(torch.nn.Module):
    r"""
    Wrap a model with finite-quadrature O(3) averaging and equivariance
    diagnostics.

    For a target representation :math:`\rho_\alpha`, define the model response
    transformed back to the input frame as

    .. math::

        z_\alpha(g;x) = \rho_\alpha(g^{-1}) f(gx).

    An ordinary requested output is the normalized Haar average

    .. math::

        \Pi_\alpha(f,x)
        = \int_{\mathrm{O}(3)} z_\alpha(g;x)\,\mathrm{d}\mu(g).

    The integrals are approximated by evaluating the underlying model on batches of
    proper and improper transformations. For a TensorMap block with :math:`d`
    component entries, ``o3::variance::<output>`` returns

    .. math::

        v_\alpha(f,x)
        = \frac{1}{d}\left[
          \int_{\mathrm{O}(3)} \lVert z_\alpha(g;x) \rVert_2^2\,
          \mathrm{d}\mu(g)
          - \lVert \Pi_\alpha(f,x) \rVert_2^2
          \right]
        = \frac{A_\alpha(f,x)^2}{d}.

    Thus, :math:`A_\alpha^2=d\,v_\alpha` is the squared component-summed
    equivariance error. The returned value is the component-averaged variance for
    every retained sample and property: this class neither takes its square root nor
    aggregates it over samples.

    Character projections act on the direct response :math:`u(g;x) = f(gx)`. For a
    character sector :math:`\beta=(\lambda,\sigma)` with
    :math:`d_\beta=2\lambda+1`, the corresponding squared projection norm is

    .. math::

        B_\beta(u,x)
        = d_\beta \iint_{\mathrm{O}(3)}
          u(g_1;x)^\dagger\,
          \chi_\beta(g_1g_2^{-1})\,u(g_2;x)\,
          \mathrm{d}\mu(g_1)\,\mathrm{d}\mu(g_2).

    Writing an O(3) operation as :math:`\Phi(R,s)`, with :math:`s=+1` for a proper
    rotation and :math:`s=-1` for an improper operation, the character convention is

    .. math::

        \chi_{\lambda,\sigma}(\Phi(R,s))
        = \left[\sigma(-1)^\lambda\right]^{(1-s)/2}
          \operatorname{tr} D^\lambda(R).

    Requests named ``o3::character_projection::<output>`` return the unnormalized
    contributions to :math:`B_\beta`, labeled by ``chi_lambda`` and ``chi_sigma``.
    Target component axes are retained; summing over them recovers the full
    component norm in the equation above.

    The deterministic quadrature is exact only when it resolves the angular dependence
    of the transformed model response. For unrestricted responses, convergence must be
    checked by increasing ``max_o3_lambda_grid``. ``batch_size`` changes how many
    transformed systems are evaluated in one model call, but does not change the grid
    or the result.

    Rotation matrices, quadrature weights, and Wigner-D matrices are stored as float64
    buffers so they follow ordinary module device movement and serialization. The
    packed Wigner-D allocation is checked against ``max_wigner_storage_bytes`` before
    it is created.

    :param model: underlying :py:class:`ModelInterface`. The :py:meth:`wrap` method
        obtains this module from :py:attr:`AtomisticModel.module`.
    :param max_o3_lambda_target: largest ``o3_lambda`` accepted on an
        already-spherical output component axis when an average or variance is
        requested. Cartesian outputs and character-only requests are not limited by
        this value.
    :param max_o3_lambda_input: largest ``o3_lambda`` accepted on an
        already-spherical component axis in custom System data. The default of zero
        still allows Cartesian custom inputs.
    :param max_o3_lambda_character: largest character sector included in character
        projections. ``None`` disables character-projection outputs; zero enables the
        scalar character sector only.
    :param batch_size: positive number of transformed systems evaluated in one call to
        ``model``. The default is 32.
    :param max_o3_lambda_grid: quadrature integration degree. If ``None``, use the
        larger of ``2 * max_o3_lambda_target + 1`` and
        ``2 * max_o3_lambda_character`` when character projections are enabled. An
        explicit value must be non-negative and no larger than the highest available
        Lebedev order, 131.
    :param max_wigner_storage_bytes: maximum number of bytes used by the serialized
        packed Wigner-D matrices. Construction fails before allocation when this limit
        would be exceeded. The default is 64 MiB.
    """

    max_o3_lambda_character: Optional[int]
    _requested_inputs: Dict[str, ModelOutput]
    _requested_neighbor_lists: List[NeighborListOptions]

    def __init__(
        self,
        model: ModelInterface,
        max_o3_lambda_target: int,
        max_o3_lambda_input: int = 0,
        max_o3_lambda_character: Optional[int] = None,
        batch_size: int = 32,
        max_o3_lambda_grid: Optional[int] = None,
        max_wigner_storage_bytes: int = _DEFAULT_MAX_WIGNER_STORAGE_BYTES,
    ):
        super().__init__()

        self._model = model
        self._requested_inputs = {}
        self._requested_neighbor_lists = []
        self.max_o3_lambda_target = _validate_integer(
            "max_o3_lambda_target", max_o3_lambda_target, 0
        )
        self.max_o3_lambda_input = _validate_integer(
            "max_o3_lambda_input", max_o3_lambda_input, 0
        )
        if max_o3_lambda_character is not None:
            max_o3_lambda_character = _validate_integer(
                "max_o3_lambda_character", max_o3_lambda_character, 0
            )
        self.max_o3_lambda_character = max_o3_lambda_character
        self.batch_size = _validate_integer("batch_size", batch_size, 1)
        self.max_wigner_storage_bytes = _validate_integer(
            "max_wigner_storage_bytes", max_wigner_storage_bytes, 1
        )

        if max_o3_lambda_grid is None:
            max_o3_lambda_grid = 2 * self.max_o3_lambda_target + 1
            if self.max_o3_lambda_character is not None:
                max_o3_lambda_grid = max(
                    max_o3_lambda_grid,
                    2 * self.max_o3_lambda_character,
                )
        else:
            max_o3_lambda_grid = _validate_integer(
                "max_o3_lambda_grid", max_o3_lambda_grid, 0
            )
        if (
            self.max_o3_lambda_character is not None
            and max_o3_lambda_grid < 2 * self.max_o3_lambda_character
        ):
            raise ValueError(
                "max_o3_lambda_grid must be at least twice max_o3_lambda_character"
            )
        self.max_o3_lambda_grid = max_o3_lambda_grid

        device = torch.device("cpu")
        for parameter in model.parameters():
            device = parameter.device
            break
        else:
            for buffer in model.buffers():
                device = buffer.device
                break
        if device.type != "cpu" and device.type != "cuda":
            raise ValueError("SymmetrizedModel supports CPU and CUDA execution")

        lebedev_order, n_rotations = _choose_quadrature(self.max_o3_lambda_grid)
        rotations, weights = get_rotation_quadrature(
            lebedev_order,
            n_rotations,
        )
        rotation_matrices = torch.from_numpy(rotations).to(
            dtype=torch.float64,
            device=device,
        )
        rotation_weights = torch.from_numpy(weights).to(
            dtype=torch.float64,
            device=device,
        )

        max_o3_lambda_wigner = max(
            self.max_o3_lambda_input,
            self.max_o3_lambda_target,
            0 if self.max_o3_lambda_character is None else self.max_o3_lambda_character,
        )
        n_wigner_elements_per_matrix = (
            (max_o3_lambda_wigner + 1)
            * (2 * max_o3_lambda_wigner + 1)
            * (2 * max_o3_lambda_wigner + 3)
            // 3
        )
        required_wigner_storage_bytes = (
            len(rotation_matrices)
            * n_wigner_elements_per_matrix
            * rotation_matrices.element_size()
        )
        if required_wigner_storage_bytes > self.max_wigner_storage_bytes:
            raise ValueError(
                "packed Wigner-D matrices require "
                + str(required_wigner_storage_bytes)
                + " bytes, exceeding max_wigner_storage_bytes="
                + str(self.max_wigner_storage_bytes)
            )
        packed_wigner_matrices = _build_packed_wigner_matrices(
            rotation_matrices,
            max_o3_lambda_wigner,
        )

        self.register_buffer("_rotation_matrices", rotation_matrices)
        self.register_buffer("_rotation_weights", rotation_weights)
        self.register_buffer("_packed_wigner_matrices", packed_wigner_matrices)

    @staticmethod
    def wrap(
        model: AtomisticModel,
        *,
        max_o3_lambda_target: int,
        max_o3_lambda_input: int = 0,
        max_o3_lambda_character: Optional[int] = None,
        batch_size: int = 32,
        max_o3_lambda_grid: Optional[int] = None,
        max_wigner_storage_bytes: int = _DEFAULT_MAX_WIGNER_STORAGE_BYTES,
    ) -> AtomisticModel:
        """
        Wrap an exported model with O(3) averaging and diagnostics.

        The returned model retains every output declared by ``model`` under its
        original name. Requesting such an output evaluates its O(3) average.
        Additional outputs named ``o3::variance::<name>`` provide the
        component-averaged equivariance variance. If ``max_o3_lambda_character``
        is set, ``o3::character_projection::<name>`` outputs provide squared
        character projections through that angular momentum.

        The original metadata, requested inputs, neighbor lists, and compatible
        capabilities are preserved.

        :param model: the :py:class:`AtomisticModel` to wrap
        :param max_o3_lambda_target: largest spherical rank accepted in
            already-spherical model outputs requested for averaging or variance
        :param max_o3_lambda_input: largest spherical rank accepted in custom System
            data
        :param max_o3_lambda_character: largest character sector to report, or ``None``
            to disable character projections
        :param batch_size: number of transformed Systems evaluated in one model call
        :param max_o3_lambda_grid: quadrature integration degree, selected
            automatically when ``None``
        :param max_wigner_storage_bytes: maximum size of the packed Wigner-D storage
        """
        if not isinstance(model, AtomisticModel):
            raise TypeError("model must be an AtomisticModel")

        capabilities = model.capabilities()
        supported_devices = [
            device
            for device in capabilities.supported_devices
            if device == "cpu" or device == "cuda"
        ]
        if len(supported_devices) == 0:
            raise ValueError(
                "SymmetrizedModel supports CPU and CUDA execution, but the "
                "wrapped model declares " + str(capabilities.supported_devices)
            )

        outputs: Dict[str, ModelOutput] = {}
        for name in model._model_capabilities_outputs_names:
            if name.startswith("o3::variance::") or name.startswith(
                "o3::character_projection::"
            ):
                raise ValueError(
                    "the wrapped model output '"
                    + name
                    + "' uses a prefix reserved by SymmetrizedModel"
                )

            source_output = capabilities.outputs[name]
            average_description = "O(3) average of the '" + name + "' output."
            if source_output.description != "":
                average_description += " " + source_output.description
            outputs[name] = ModelOutput(
                unit=source_output.unit,
                sample_kind=source_output.sample_kind,
                explicit_gradients=[],
                description=average_description,
            )

            squared_unit = ""
            if source_output.unit != "":
                squared_unit = "(" + source_output.unit + ")^2"
            outputs["o3::variance::" + name] = ModelOutput(
                unit=squared_unit,
                sample_kind=source_output.sample_kind,
                explicit_gradients=[],
                description=(
                    "O(3) equivariance variance of the '"
                    + name
                    + "' output for each sample, averaged over components."
                ),
            )
            if max_o3_lambda_character is not None:
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

        wrapper = SymmetrizedModel(
            model.module,
            max_o3_lambda_target=max_o3_lambda_target,
            max_o3_lambda_input=max_o3_lambda_input,
            max_o3_lambda_character=max_o3_lambda_character,
            batch_size=batch_size,
            max_o3_lambda_grid=max_o3_lambda_grid,
            max_wigner_storage_bytes=max_wigner_storage_bytes,
        )
        wrapper._requested_inputs = {
            name: requested_input
            for name, requested_input in model._requested_inputs.items()
        }
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
            supported_devices=supported_devices,
            dtype=capabilities.dtype,
        )
        return AtomisticModel(
            wrapper.eval(),
            model.metadata(),
            capabilities=new_capabilities,
        )

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
            return torch.jit.annotate(Dict[str, TensorMap], {})
        if len(systems) == 0:
            raise ValueError("SymmetrizedModel requires at least one System")

        for requested_name, output in outputs.items():
            if len(output.explicit_gradients) != 0:
                raise ValueError(
                    "SymmetrizedModel does not support explicit gradients for output '"
                    + requested_name
                    + "'"
                )

        (
            source_sample_kinds,
            average_names,
            variance_names,
            character_projection_names,
        ) = _group_output_requests(outputs)
        if (
            len(character_projection_names) != 0
            and self.max_o3_lambda_character is None
        ):
            raise ValueError(
                "max_o3_lambda_character must be set to request character projections"
            )

        source_outputs = torch.jit.annotate(Dict[str, ModelOutput], {})
        for source_name in source_sample_kinds:
            source_outputs[source_name] = ModelOutput(
                sample_kind=source_sample_kinds[source_name],
            )

        per_output_results = torch.jit.annotate(
            Dict[str, List[TensorMap]],
            {},
        )
        for requested_name in outputs:
            per_output_results[requested_name] = torch.jit.annotate(List[TensorMap], [])

        for input_system_index, system in enumerate(systems):
            system_results = self._evaluate_system(
                system,
                input_system_index,
                source_outputs,
                average_names,
                variance_names,
                character_projection_names,
                selected_atoms,
            )
            for requested_name in outputs:
                if requested_name not in system_results:
                    raise ValueError(
                        "SymmetrizedModel did not produce requested output '"
                        + requested_name
                        + "'"
                    )
                per_output_results[requested_name].append(
                    system_results[requested_name]
                )

        results = torch.jit.annotate(Dict[str, TensorMap], {})
        for requested_name in outputs:
            results[requested_name] = _join_per_system_tensormaps(
                per_output_results[requested_name]
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
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        """Stream all quadrature batches for one input System."""
        work_dtype = system.positions.dtype
        work_device = system.positions.device
        if work_dtype != torch.float32 and work_dtype != torch.float64:
            raise TypeError("SymmetrizedModel requires float32 or float64 Systems")
        if work_device.type != "cpu" and work_device.type != "cuda":
            raise ValueError("SymmetrizedModel supports CPU and CUDA execution")
        if (
            self._rotation_matrices.dtype != torch.float64
            or self._rotation_weights.dtype != torch.float64
            or self._packed_wigner_matrices.dtype != torch.float64
        ):
            raise ValueError("SymmetrizedModel integration buffers must remain float64")
        if (
            self._rotation_matrices.device != work_device
            or self._rotation_weights.device != work_device
            or self._packed_wigner_matrices.device != work_device
        ):
            raise ValueError(
                "SymmetrizedModel and input Systems must use the same device"
            )

        character_max = 0
        configured_character_max = self.max_o3_lambda_character
        if configured_character_max is not None:
            character_max = configured_character_max

        average_references = torch.jit.annotate(Dict[str, TensorMap], {})
        average_first_moments = torch.jit.annotate(Dict[str, TensorMap], {})
        variance_references = torch.jit.annotate(Dict[str, TensorMap], {})
        variance_first_moments = torch.jit.annotate(Dict[str, TensorMap], {})
        variance_second_moments = torch.jit.annotate(Dict[str, TensorMap], {})
        variance_absolute_second_moments = torch.jit.annotate(
            Dict[str, TensorMap],
            {},
        )
        proper_character_coefficients = torch.jit.annotate(
            Dict[str, TensorMap],
            {},
        )
        improper_character_coefficients = torch.jit.annotate(
            Dict[str, TensorMap],
            {},
        )

        n_rotations = self._rotation_matrices.size(0)
        needs_backrotation = len(average_names) != 0 or len(variance_names) != 0
        for batch_start in range(0, n_rotations, self.batch_size):
            batch_stop = min(batch_start + self.batch_size, n_rotations)
            n_rotated_copies = batch_stop - batch_start
            proper_matrices = self._rotation_matrices[batch_start:batch_stop]
            so3_weights = self._rotation_weights[batch_start:batch_stop]
            o3_weights = 0.5 * so3_weights
            local_selected_atoms = _map_selected_atoms_to_rotated_copies(
                selected_atoms,
                input_system_index,
                n_rotated_copies,
            )

            input_wigner_matrices: List[torch.Tensor] = []
            for o3_lambda in range(self.max_o3_lambda_input + 1):
                input_wigner_matrices.append(
                    _wigner_matrices_for_lambda(
                        self._packed_wigner_matrices,
                        n_rotations,
                        o3_lambda,
                    )[batch_start:batch_stop].to(
                        dtype=work_dtype,
                        device=work_device,
                    )
                )

            inverse_target_wigner_matrices: List[torch.Tensor] = []
            if needs_backrotation:
                for o3_lambda in range(self.max_o3_lambda_target + 1):
                    inverse_target_wigner_matrices.append(
                        _wigner_matrices_for_lambda(
                            self._packed_wigner_matrices,
                            n_rotations,
                            o3_lambda,
                        )[batch_start:batch_stop].transpose(1, 2)
                    )

            inverse_character_wigner_matrices: List[torch.Tensor] = []
            if len(character_projection_names) != 0:
                for chi_lambda in range(character_max + 1):
                    inverse_character_wigner_matrices.append(
                        _wigner_matrices_for_lambda(
                            self._packed_wigner_matrices,
                            n_rotations,
                            chi_lambda,
                        )[batch_start:batch_stop].transpose(1, 2)
                    )

            for coset_index in range(2):
                is_improper = coset_index == 1
                sign = -1.0 if is_improper else 1.0
                matrices = (sign * proper_matrices).to(
                    dtype=work_dtype,
                    device=work_device,
                )
                transformed_systems = _transform_system_batch(
                    system,
                    matrices,
                    input_wigner_matrices,
                    self.max_o3_lambda_input,
                    is_improper,
                )
                raw_outputs = self._model(
                    transformed_systems,
                    source_outputs,
                    local_selected_atoms,
                )

                for source_name in source_outputs:
                    if source_name not in raw_outputs:
                        raise ValueError(
                            "underlying model did not return requested output '"
                            + source_name
                            + "'"
                        )
                for returned_name in raw_outputs:
                    if returned_name not in source_outputs:
                        raise ValueError(
                            "underlying model returned unrequested output '"
                            + returned_name
                            + "'"
                        )

                inverse_matrices = (sign * proper_matrices).transpose(1, 2)
                for source_name in source_outputs:
                    raw_tensor = raw_outputs[source_name]
                    for block in raw_tensor.blocks():
                        gradient_names = block.gradients_list()
                        if len(gradient_names) != 0:
                            raise ValueError(
                                "underlying output '"
                                + source_name
                                + "' contains unsupported explicit gradient '"
                                + gradient_names[0]
                                + "'"
                            )

                    tensor = raw_tensor.to(
                        dtype=torch.float64,
                        device=work_device,
                    )
                    if source_name in average_names or source_name in variance_names:
                        _check_o3_lambda_limit(
                            tensor,
                            "output '" + source_name + "'",
                            self.max_o3_lambda_target,
                            "max_o3_lambda_target",
                        )
                        backrotated = _transform_tensor_with_precomputed_matrices(
                            tensor,
                            inverse_matrices,
                            inverse_target_wigner_matrices,
                            is_improper,
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
                                updated_average_reference = _copy_tensormap_info(
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
                            diagnostic_tensor = _decompose_output(
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
                            if second_moment is None or absolute_second_moment is None:
                                raise RuntimeError("variance moments were not computed")
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
                        direct_tensor = _decompose_output(source_name, tensor)
                        contribution = _character_projection_coefficients_from_batch(
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

        results = torch.jit.annotate(Dict[str, TensorMap], {})
        for source_name, requested_name in average_names.items():
            if (
                source_name not in average_references
                or source_name not in average_first_moments
            ):
                raise RuntimeError("average accumulation is incomplete")
            mean = mts.add(
                average_references[source_name],
                average_first_moments[source_name],
            )
            mean = _copy_tensormap_info(average_references[source_name], mean)
            results[requested_name] = mean.to(
                dtype=work_dtype,
                device=work_device,
            )

        for source_name, requested_name in variance_names.items():
            if (
                source_name not in variance_references
                or source_name not in variance_first_moments
                or source_name not in variance_second_moments
                or source_name not in variance_absolute_second_moments
            ):
                raise RuntimeError("variance accumulation is incomplete")
            variance = _variance_from_centered_moments(
                variance_first_moments[source_name],
                variance_second_moments[source_name],
                variance_absolute_second_moments[source_name],
                n_grid_points=2 * n_rotations,
                max_o3_lambda_grid=self.max_o3_lambda_grid,
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
            if (
                source_name not in proper_character_coefficients
                or source_name not in improper_character_coefficients
            ):
                raise RuntimeError("character-projection accumulation is incomplete")
            projection = _character_projection_tensormap_from_cosets(
                proper_character_coefficients[source_name],
                improper_character_coefficients[source_name],
            )
            results[requested_name] = projection.to(
                dtype=work_dtype,
                device=work_device,
            )

        return results
