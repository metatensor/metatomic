"""
Rotate systems and tensor maps under O(3) transformations, routing rows of
multi-system tensors by their ``"system"`` sample label.
"""

from numbers import Integral

import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap

from .. import System, register_autograd_neighbors
from ._wigner import build_wigner_D_cache


# Component-axis names recognised by the augmentation machinery.
_SUFFIXES = [""] + [f"_{i}" for i in range(1, 10)]
_CARTESIAN_AXES = frozenset({f"xyz{s}" for s in _SUFFIXES})
_SPHERICAL_AXIS_TO_LAMBDA = {f"o3_mu{s}": f"o3_lambda{s}" for s in _SUFFIXES}
_SPHERICAL_AXIS_TO_SIGMA = {f"o3_mu{s}": f"o3_sigma{s}" for s in _SUFFIXES}
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _validate_nonnegative_integer(name: str, value: int) -> int:
    """Normalize a public integer argument without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {normalized}.")
    return normalized


def _spherical_parity(ell: int, sigma: int, is_inverted: bool) -> int:
    """Return the discrete O(3) factor outside the proper-rotation Wigner-D."""
    if isinstance(sigma, bool) or not isinstance(sigma, Integral):
        raise TypeError("sigma must be either -1 or +1")
    sigma = int(sigma)
    if sigma not in (-1, 1):
        raise ValueError(f"sigma must be either -1 or +1, got {sigma}")
    if is_inverted:
        return ((-1) ** ell) * sigma
    return 1


def _validate_system_routing(
    systems: list[System],
    transformations: list["O3Transformation"],
    system_ids: list[int] | torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Validate and normalize the mapping from sample labels to systems."""
    n_systems = len(systems)
    n_transformations = len(transformations)
    if n_systems != n_transformations:
        raise ValueError(
            "Expected one transformation per system, but got "
            f"len(systems)={n_systems} and "
            f"len(transformations)={n_transformations}."
        )

    if system_ids is None:
        # This is the hot path used by SymmetrizedModel. The generated mapping
        # has the required length and is unique by construction.
        return torch.arange(n_systems, dtype=torch.long, device=device)

    if isinstance(system_ids, torch.Tensor):
        if system_ids.ndim != 1:
            raise ValueError(
                "system_ids must be one-dimensional, but got a tensor with shape "
                f"{tuple(system_ids.shape)}."
            )
        if system_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError(
                "system_ids must contain integers, but got a tensor with dtype "
                f"{system_ids.dtype}."
            )
        if system_ids.device != device:
            raise ValueError(
                f"system_ids are on device {system_ids.device}, but the values to "
                f"transform are on device {device}."
            )
        normalized_ids = system_ids.to(dtype=torch.long)
    else:
        python_ids: list[int] = []
        for system_id in system_ids:
            if isinstance(system_id, bool) or not isinstance(system_id, Integral):
                raise ValueError("system_ids must contain integers.")
            python_ids.append(int(system_id))
        normalized_ids = torch.tensor(python_ids, dtype=torch.long, device=device)

    if len(normalized_ids) != n_systems:
        raise ValueError(
            "system_ids must contain exactly one entry per system, but got "
            f"len(system_ids)={len(normalized_ids)} and len(systems)={n_systems}."
        )
    if torch.unique(normalized_ids).numel() != n_systems:
        raise ValueError(
            "system_ids must contain one distinct entry per system, but got "
            f"{normalized_ids.tolist()}."
        )
    return normalized_ids


def _validate_transformation_dtype_device(
    transformations: list["O3Transformation"],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Require every transformation to match the values it will transform."""
    for index, transformation in enumerate(transformations):
        if transformation.dtype != dtype or transformation.device != device:
            raise ValueError(
                f"Transformation at index {index} has dtype/device "
                f"({transformation.dtype}, {transformation.device}), differing from "
                f"the values to transform ({dtype}, {device})."
            )


class O3Transformation:
    """
    A single O(3) transformation, represented by a (3, 3) rotation or improper-rotation
    matrix.
    """

    def __init__(self, matrix: torch.Tensor, max_angular_momentum: int):
        """
        :param matrix: (3, 3) rotation or improper-rotation matrix
        :param max_angular_momentum: maximum angular momentum of any spherical
            representation to be transformed by this transformation; Wigner-D matrices
            are computed on first use for all ``ell <= max_angular_momentum``
        """
        max_angular_momentum = _validate_nonnegative_integer(
            "max_angular_momentum", max_angular_momentum
        )

        if matrix.shape != (3, 3):
            raise ValueError(
                f"Transformation has shape {tuple(matrix.shape)}; expected (3, 3)."
            )

        identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
        if not torch.allclose(matrix @ matrix.T, identity, atol=1e-5):
            raise ValueError(
                "Transformation is not orthogonal (R @ R.T deviates from I)."
            )

        # Keep the validated transformation immutable. The determinant parity and
        # Wigner-D matrices are cached, so retaining caller-owned storage would let a
        # later in-place mutation make the Cartesian and spherical paths disagree.
        self._matrix = matrix.clone()
        self._max_angular_momentum = max_angular_momentum
        self._is_inverted = bool(torch.det(matrix) < 0)

        # Cartesian transformations only need ``matrix``. Building Wigner-D
        # matrices eagerly is particularly expensive for quadrature workloads,
        # where most transformations never touch a spherical component.
        self._wigner_D_cache: dict[int, torch.Tensor] | None = None

    @classmethod
    def _from_validated_matrix(
        cls,
        matrix: torch.Tensor,
        max_angular_momentum: int,
        *,
        is_inverted: bool,
    ) -> "O3Transformation":
        """Build a transformation whose orthogonality/parity is already known.

        This private constructor is for internally generated quadrature
        matrices only. It avoids the device synchronizations required by the
        public constructor's orthogonality and determinant checks; callers are
        responsible for providing a valid matrix and the correct component of
        O(3).
        """
        transformation = cls.__new__(cls)
        transformation._matrix = matrix
        transformation._max_angular_momentum = max_angular_momentum
        transformation._is_inverted = is_inverted
        transformation._wigner_D_cache = None
        return transformation

    def _ensure_wigner_D_cache(self) -> dict[int, torch.Tensor]:
        if self._wigner_D_cache is None:
            self._wigner_D_cache = build_wigner_D_cache(
                self._max_angular_momentum,
                self._matrix,
                device=self._matrix.device,
                dtype=self._matrix.dtype,
            )
        return self._wigner_D_cache

    @property
    def matrix(self) -> torch.Tensor:
        """A copy of the (3, 3) rotation or improper-rotation matrix."""
        return self._matrix.clone()

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the transformation matrix."""
        return self._matrix.dtype

    @property
    def device(self) -> torch.device:
        """The device of the transformation matrix."""
        return self._matrix.device

    @property
    def is_inverted(self) -> bool:
        """Whether this transformation is an improper rotation (det < 0)."""
        return self._is_inverted

    def transform_cartesian(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to Cartesian vectors.

        :param vectors: (..., 3) tensor of Cartesian vectors
        :return: (..., 3) tensor of transformed vectors
        """
        return vectors @ self._matrix.T

    def transform_spherical(
        self, values: torch.Tensor, ell: int, sigma: int
    ) -> torch.Tensor:
        """Apply the transformation to spherical values.

        :param values: (..., 2*ell+1) tensor of spherical values
        :param ell: angular momentum of the spherical representation
        :param sigma: metatensor O(3) parity label (+1 for a proper tensor and
            -1 for a pseudotensor). Under inversion the representation acquires
            the factor ``sigma * (-1)**ell``.
        :return: (..., 2*ell+1) tensor of transformed spherical values
        """
        ell = self._validate_angular_momentum(ell)
        parity = _spherical_parity(ell, sigma, self.is_inverted)
        transformed = values @ self._wigner_D_matrix(ell).T
        if parity != 1:
            transformed = transformed * parity
        return transformed

    def _wigner_D_matrix(self, ell: int) -> torch.Tensor:
        """Return the cached Wigner-D matrix without copying it."""
        ell = self._validate_angular_momentum(ell)
        D = self._ensure_wigner_D_cache().get(ell)
        assert D is not None
        return D

    def _validate_angular_momentum(self, ell: int) -> int:
        """Normalize an angular momentum and enforce this instance's bound."""
        ell = _validate_nonnegative_integer("ell", ell)
        if ell > self._max_angular_momentum:
            raise ValueError(
                f"ell={ell} exceeds max_angular_momentum={self._max_angular_momentum}."
            )
        return ell

    def wigner_D_matrix(self, ell: int) -> torch.Tensor:
        """Return a copy of the proper-part Wigner-D matrix for angular momentum
        ``ell``.

        For an improper transformation, :meth:`transform_spherical` applies
        the discrete inversion-parity factor separately.

        :param ell: angular momentum of the spherical representation
        :return: (2*ell+1, 2*ell+1) Wigner-D matrix
        """
        return self._wigner_D_matrix(ell).clone()


def random_transformations(
    n: int,
    max_angular_momentum: int = 0,
    *,
    device: torch.device,
    dtype: torch.dtype,
    include_inversions: bool = False,
    generator: torch.Generator | None = None,
) -> list[O3Transformation]:
    """Sample ``n`` uniformly distributed O(3) transformations.

    Rotations are sampled from the Haar measure on SO(3) via random unit quaternions.
    When ``include_inversions`` is ``True``, each matrix is independently negated with
    probability 0.5, giving a uniform distribution over the full O(3) group.

    :param n: number of transformations to generate
    :param max_angular_momentum: maximum angular momentum for Wigner-D matrices
    :param device: target device for the output tensors
    :param dtype: target dtype for the output tensors
    :param include_inversions: if ``True``, sample from O(3) instead of SO(3)
    :param generator: optional :class:`torch.Generator` for reproducible sampling; when
        ``None`` the global RNG is used
    :return: list of ``n`` :class:`O3Transformation` objects
    """
    n = _validate_nonnegative_integer("n", n)
    max_angular_momentum = _validate_nonnegative_integer(
        "max_angular_momentum", max_angular_momentum
    )

    if not dtype.is_floating_point:
        raise TypeError(
            f"random O(3) transformations require a real dtype, got {dtype}"
        )

    q = torch.randn(n, 4, device=device, dtype=dtype, generator=generator)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    # Quaternion to rotation matrix (standard formula)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=1,
    ).reshape(n, 3, 3)

    inverted = [False] * n
    if include_inversions:
        signs = torch.randint(0, 2, (n,), device=device, generator=generator) * 2 - 1
        R = R * signs.to(dtype=dtype).reshape(n, 1, 1)
        # One bulk transfer avoids a determinant synchronization per matrix.
        inverted = (signs < 0).tolist()

    # Preserve the public constructor's orthogonality check, but validate the
    # internally generated batch at once instead of synchronizing once per matrix.
    identity = torch.eye(3, device=R.device, dtype=R.dtype).expand(n, 3, 3)
    if not torch.allclose(R @ R.transpose(-1, -2), identity, atol=1e-5):
        raise ValueError("Generated transformations are not orthogonal.")

    return [
        O3Transformation._from_validated_matrix(
            matrix,
            max_angular_momentum,
            is_inverted=is_inverted,
        )
        for matrix, is_inverted in zip(R.unbind(0), inverted, strict=True)
    ]


def _block_row_indices_by_system(
    block: TensorBlock,
    system_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Route value rows using the ``system`` sample label.

    For one system all rows belong to it; otherwise ``system_ids[i]`` identifies
    the rows transformed by operation ``i``.
    """
    n_systems = len(system_ids)
    if n_systems == 1:
        return [torch.arange(block.values.shape[0], device=block.values.device)]

    if "system" not in block.samples.names:
        raise ValueError(
            "Rotational augmentation expects output samples to include a 'system' "
            "dimension when transforming multiple systems."
        )
    system_column = block.samples.column("system").to(dtype=torch.long)
    distinct = torch.unique(system_column)
    covered = torch.isin(distinct, system_ids)
    if not covered.all():
        uncovered = distinct[~covered]
        raise ValueError(
            f"Block samples contain system labels {uncovered.tolist()} that are "
            f"not in system_ids={system_ids.tolist()}. Every sample must be "
            f"assigned to a system in the transformation."
        )
    return [
        torch.nonzero(system_column == label, as_tuple=False).reshape(-1)
        for label in system_ids
    ]


def _gradient_row_indices_by_system(
    grad_block: TensorBlock,
    parent_block: TensorBlock,
    system_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Route gradient rows through their parent ``sample`` indices."""
    n_systems = len(system_ids)
    if n_systems == 1:
        return [
            torch.arange(grad_block.values.shape[0], device=grad_block.values.device)
        ]
    if "system" not in parent_block.samples.names:
        raise ValueError(
            "Rotational augmentation expects the values samples to include a 'system' "
            "dimension when transforming gradients of multiple systems."
        )
    parent_system = parent_block.samples.column("system").to(dtype=torch.long)
    sample_index = grad_block.samples.column("sample").to(dtype=torch.long)
    grad_system = parent_system[sample_index]
    return [
        torch.nonzero(grad_system == label, as_tuple=False).reshape(-1)
        for label in system_ids
    ]


def transform_system(system: System, transformation: O3Transformation) -> System:
    """Apply an O(3) transformation to a single System.

    This function will transform positions, cell vectors, neighbor-list displacement
    vectors, and any custom data.

    :param system: input system
    :param transformation: O(3) transformation to apply
    :return: new System with transformed geometry
    """
    if (
        system.positions.dtype != transformation.dtype
        or system.positions.device != transformation.device
    ):
        raise ValueError(
            f"System has positions with dtype/device "
            f"({system.positions.dtype}, {system.positions.device}) differing "
            f"from the transformations ({transformation.dtype}, "
            f"{transformation.device})."
        )

    new_system = System(
        positions=transformation.transform_cartesian(system.positions),
        types=system.types,
        cell=transformation.transform_cartesian(system.cell),
        pbc=system.pbc,
    )

    for data_name in system.known_data():
        data = system.get_data(data_name)
        new_system.add_data(
            data_name, transform_tensor(data, [system], [transformation])
        )

    for options in system.known_neighbor_lists():
        neighbors = system.get_neighbor_list(options)
        # neighbor vectors are stored as (N, 3, 1); squeeze/unsqueeze around the matmul
        # A neighbor list can already be registered with the input geometry. Start
        # from detached values and register a fresh graph against ``new_system``.
        neighbors_values = neighbors.values.detach().squeeze(-1)
        new_values = transformation.transform_cartesian(neighbors_values)
        rotated_neighbors = TensorBlock(
            values=new_values.unsqueeze(-1),
            samples=neighbors.samples,
            components=neighbors.components,
            properties=neighbors.properties,
        )
        register_autograd_neighbors(new_system, rotated_neighbors)
        new_system.add_neighbor_list(options, rotated_neighbors)

    return new_system


def _contract_component_axes(
    values: torch.Tensor,
    matrices: list[torch.Tensor],
) -> torch.Tensor:
    """Rotate each component axis of ``values`` by its matrix.

    ``values`` has shape ``(n_rows, d_1, ..., d_k, n_properties)`` and ``matrices[j]``
    (shape ``(d_j, d_j)``) is contracted with component axis ``j`` as
    ``out[..., A, ...] = sum_a matrices[j][A, a] * values[..., a, ...]``.

    :param values: values tensor of a value or gradient block
    :param matrices: one rotation matrix per component axis (empty for scalars)
    :return: rotated values, same shape as the input
    """
    # einsum index letters for the per-axis component contraction (input lower,
    # output upper). Keep these in sync with the ten recognized component suffixes.
    _EINSUM_IN = "abcdefghjk"
    _EINSUM_OUT = "ABCDEFGHIJ"

    if len(matrices) == 0:
        return values
    n_axes = len(matrices)
    if n_axes > len(_EINSUM_IN):
        raise ValueError(f"can not transform a tensor with {n_axes} component axes")
    in_subscript = "i" + _EINSUM_IN[:n_axes] + "p"
    out_subscript = "i" + _EINSUM_OUT[:n_axes] + "p"
    matrix_subscripts = [_EINSUM_OUT[j] + _EINSUM_IN[j] for j in range(n_axes)]
    equation = ",".join(matrix_subscripts + [in_subscript]) + "->" + out_subscript
    return torch.einsum(equation, *matrices, values)


def _component_axis_metadata(
    components: list[Labels],
    key: LabelsEntry,
) -> list[tuple[str, int, int]]:
    """Validate component axes independently of the number of value rows."""
    if len(components) > len(_SUFFIXES):
        raise ValueError(
            f"can not transform a tensor with {len(components)} component axes; "
            f"at most {len(_SUFFIXES)} are supported"
        )
    metadata: list[tuple[str, int, int]] = []
    for component in components:
        axis_name = component.names[0]
        if axis_name in _CARTESIAN_AXES:
            if len(component) != 3:
                raise ValueError(
                    f"Cartesian component axis '{axis_name}' must contain 3 "
                    f"entries, got {len(component)}."
                )
            expected = torch.arange(
                3,
                device=component.values.device,
                dtype=component.values.dtype,
            )
            if not torch.equal(component.values[:, 0], expected):
                raise ValueError(
                    f"Cartesian component axis '{axis_name}' must use labels "
                    "[0, 1, 2] in x, y, z order."
                )
            metadata.append((axis_name, 0, 1))
        elif axis_name in _SPHERICAL_AXIS_TO_LAMBDA:
            ell = int(key[_SPHERICAL_AXIS_TO_LAMBDA[axis_name]])
            sigma = int(key[_SPHERICAL_AXIS_TO_SIGMA[axis_name]])
            if ell < 0:
                raise ValueError(f"ell must be non-negative, got {ell}")
            expected_size = 2 * ell + 1
            if len(component) != expected_size:
                raise ValueError(
                    f"Spherical component axis '{axis_name}' for ell={ell} must "
                    f"contain {expected_size} entries, got {len(component)}."
                )
            expected = torch.arange(
                -ell,
                ell + 1,
                device=component.values.device,
                dtype=component.values.dtype,
            )
            if not torch.equal(component.values[:, 0], expected):
                raise ValueError(
                    f"Spherical component axis '{axis_name}' for ell={ell} must "
                    f"use labels from {-ell} through {ell} in ascending order."
                )
            # Validate sigma even when a selected-atom block has no rows.
            _spherical_parity(ell, sigma, False)
            metadata.append((axis_name, ell, sigma))
        else:
            raise ValueError(
                f"Found a component axis '{axis_name}', which is neither a Cartesian "
                "('xyz'/'xyz_1'/'xyz_2'/...) nor spherical ('o3_mu'/'o3_mu_1'/...) "
                "axis; it can not be transformed."
            )
    return metadata


def _axis_matrices_and_parity(
    metadata: list[tuple[str, int, int]],
    transformation: O3Transformation,
) -> tuple[list[torch.Tensor], int]:
    """Return each axis matrix and the combined spherical inversion parity."""
    matrices: list[torch.Tensor] = []
    parity = 1
    for axis_name, ell, sigma in metadata:
        if axis_name in _CARTESIAN_AXES:
            matrices.append(transformation._matrix)
        else:
            parity *= _spherical_parity(ell, sigma, transformation.is_inverted)
            matrices.append(transformation._wigner_D_matrix(ell))
    return matrices, parity


def _transform_component_values(
    values: torch.Tensor,
    components: list[Labels],
    key: LabelsEntry,
    row_indices: list[torch.Tensor],
    transformations: list[O3Transformation],
) -> torch.Tensor:
    """Rotate value or gradient rows with their assigned transformation."""
    metadata = _component_axis_metadata(components, key)
    new_values = values.clone()
    for system_index, rows in enumerate(row_indices):
        if len(rows) == 0:
            continue
        matrices, parity = _axis_matrices_and_parity(
            metadata, transformations[system_index]
        )
        rotated = _contract_component_axes(values[rows], matrices)
        if parity != 1:
            rotated = rotated * parity
        new_values[rows] = rotated
    return new_values


def transform_block(
    key: LabelsEntry,
    block: TensorBlock,
    systems: list[System],
    transformations: list[O3Transformation],
    system_ids: list[int] | torch.Tensor | None = None,
) -> TensorBlock:
    """Rotate one block and all of its gradients.

    :param key: the block's key, containing ``o3_lambda``/``o3_sigma`` values for
        spherical blocks
    :param block: the block to rotate
    :param systems: list of systems, one per transformation
    :param transformations: per-system O(3) transformation matrices, all with the same
        dtype and device as ``block.values``
    :param system_ids: Python list or one-dimensional integer tensor containing
        one distinct ``"system"`` sample label per entry in ``systems``. Entry
        ``i`` identifies the samples transformed with ``transformations[i]``.
        Labels need not be sorted or consecutive. A tensor-valued ``system_ids``
        must be on the same device as ``block.values``. Defaults to
        ``list(range(len(systems)))``
    :return: new block with rotated values and gradients
    """
    system_ids = _validate_system_routing(
        systems,
        transformations,
        system_ids,
        device=block.values.device,
    )
    _validate_transformation_dtype_device(
        transformations,
        dtype=block.values.dtype,
        device=block.values.device,
    )
    if len(systems) == 0:
        return block

    return _transform_block_impl(key, block, transformations, system_ids)


def _transform_block_impl(
    key: LabelsEntry,
    block: TensorBlock,
    transformations: list[O3Transformation],
    system_ids: torch.Tensor,
) -> TensorBlock:
    """Implementation of ``transform_block`` without the initial checks"""
    row_indices = _block_row_indices_by_system(block, system_ids)
    new_block = TensorBlock(
        values=_transform_component_values(
            block.values,
            block.components,
            key,
            row_indices,
            transformations,
        ),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )
    for gradient_name in block.gradients_list():
        grad_block = block.gradient(gradient_name)
        grad_rows = _gradient_row_indices_by_system(grad_block, block, system_ids)
        new_block.add_gradient(
            gradient_name,
            TensorBlock(
                values=_transform_component_values(
                    grad_block.values,
                    grad_block.components,
                    key,
                    grad_rows,
                    transformations,
                ),
                samples=grad_block.samples,
                components=grad_block.components,
                properties=grad_block.properties,
            ),
        )
    return new_block


def transform_tensor(
    tensor: TensorMap,
    systems: list[System],
    transformations: list[O3Transformation],
    system_ids: list[int] | torch.Tensor | None = None,
) -> TensorMap:
    """Rotate every block (and its gradients) of a TensorMap.

    The tensor character of each component axis is inferred from its name, so a single
    :py:class:`TensorMap` may freely mix scalar, Cartesian and spherical blocks, and
    blocks may carry gradients.

    With multiple systems, one sample dimension must be ``"system"`` and is
    used to route each row to the correct system and transformation. With one
    system this column is optional and, when present, ignored.

    :param tensor: TensorMap to rotate
    :param systems: input systems
    :param transformations: per-system O(3) transformation matrices, all with the same
        dtype and device as the tensor values
    :param system_ids: Python list or one-dimensional integer tensor containing
        one distinct ``"system"`` sample label per entry in ``systems``. Entry
        ``i`` identifies the samples transformed with ``transformations[i]``.
        Labels need not be sorted or consecutive. A tensor-valued ``system_ids``
        must be on the same device as the tensor values. Defaults to
        ``list(range(len(systems)))``
    :return: new TensorMap with rotated values and gradients
    """
    reference_values = (
        tensor.block(0).values
        if len(tensor) != 0
        else transformations[0]._matrix
        if len(transformations) != 0
        else torch.empty(0)
    )
    system_ids = _validate_system_routing(
        systems,
        transformations,
        system_ids,
        device=reference_values.device,
    )
    if len(systems) == 0:
        return tensor

    _validate_transformation_dtype_device(
        transformations,
        dtype=reference_values.dtype,
        device=reference_values.device,
    )

    new_blocks = [
        _transform_block_impl(key, block, transformations, system_ids)
        for key, block in tensor.items()
    ]
    transformed = TensorMap(keys=tensor.keys, blocks=new_blocks)
    for info_key, info_value in tensor.info().items():
        transformed.set_info(info_key, info_value)
    return transformed
