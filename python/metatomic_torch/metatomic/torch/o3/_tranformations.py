"""
Rotate systems and tensor maps under O(3) transformations, routing rows of
multi-system tensors by their ``"system"`` sample label.
"""

from numbers import Integral

import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap

from .. import System, register_autograd_neighbors
from ._wigner import build_wigner_D_cache


_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _validate_nonnegative_integer(name: str, value: int) -> int:
    """Validate a non-negative integer and return it as a Python int."""
    if torch.jit.is_scripting():
        integer_value = value
    else:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a non-negative integer.")
        integer_value = int(value)
    if integer_value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {integer_value}.")

    return integer_value


def _spherical_parity_factor(
    ell: int,
    sigma: int,
    is_improper: bool,
) -> int:
    """Return ``sigma * (-1) ** ell`` for an improper transformation, else ``1``."""
    if torch.jit.is_scripting():
        integer_sigma = sigma
    else:
        if isinstance(sigma, bool) or not isinstance(sigma, Integral):
            raise TypeError("sigma must be either -1 or +1.")
        integer_sigma = int(sigma)
    if integer_sigma not in (-1, 1):
        raise ValueError(f"sigma must be either -1 or +1, got {integer_sigma}.")

    if is_improper:
        return integer_sigma * int((-1) ** ell)

    return 1


def _validate_system_ids(
    systems: list[System],
    transformations: list["O3Transformation"],
    system_ids: list[int] | torch.Tensor | None,
    *,
    expected_device: torch.device | None,
) -> torch.Tensor:
    """Return one distinct ``torch.long`` sample label per
    system-transformation pair.
    """
    n_systems = len(systems)
    n_transformations = len(transformations)
    if n_systems != n_transformations:
        raise ValueError(
            "Expected one transformation per system, but got "
            f"len(systems)={n_systems} and "
            f"len(transformations)={n_transformations}."
        )

    if system_ids is None:
        return torch.arange(n_systems, dtype=torch.long, device=expected_device)

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
        if expected_device is not None and system_ids.device != expected_device:
            raise ValueError(
                f"system_ids are on device {system_ids.device}, but the values to "
                f"transform are on device {expected_device}."
            )
        validated_ids = system_ids.to(dtype=torch.long)
    else:
        python_ids: list[int] = []
        for system_id in system_ids:
            if isinstance(system_id, bool) or not isinstance(system_id, Integral):
                raise ValueError("system_ids must contain integers.")
            python_ids.append(int(system_id))
        validated_ids = torch.tensor(
            python_ids,
            dtype=torch.long,
            device=expected_device,
        )

    if len(validated_ids) != n_systems:
        raise ValueError(
            "system_ids must contain exactly one entry per system, but got "
            f"len(system_ids)={len(validated_ids)} and len(systems)={n_systems}."
        )
    if torch.unique(validated_ids).numel() != n_systems:
        raise ValueError(
            "system_ids must contain one distinct entry per system, but got "
            f"{validated_ids.tolist()}."
        )

    return validated_ids


def _validate_transformations_dtype_device(
    transformations: list["O3Transformation"],
    *,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> None:
    """Check that every transformation has the expected dtype and device."""
    for index, transformation in enumerate(transformations):
        if (
            transformation.dtype != expected_dtype
            or transformation.device != expected_device
        ):
            raise ValueError(
                f"Transformation at index {index} has dtype/device "
                f"({transformation.dtype}, {transformation.device}), differing from "
                f"the values to transform ({expected_dtype}, {expected_device})."
            )


class O3Transformation:
    """
    A single O(3) transformation, represented by a (3, 3) rotation or improper-rotation
    matrix.

    The constructor stores a copy of ``matrix``.
    """

    def __init__(self, matrix: torch.Tensor, max_angular_momentum: int):
        """
        :param matrix: (3, 3) rotation or improper-rotation matrix
        :param max_angular_momentum: non-negative maximum angular momentum for
            Wigner-D matrices, which are computed and cached on first use
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

        # Keep an independent copy so modifying the input tensor later cannot make
        # the matrix disagree with the cached parity and Wigner-D matrices.
        self._matrix = matrix.clone()
        self._max_angular_momentum = max_angular_momentum
        self._is_improper = bool(torch.det(self._matrix) < 0)

        self._wigner_D_cache: dict[int, torch.Tensor] | None = None

    @classmethod
    def _create_from_internal_matrix(
        cls,
        matrix: torch.Tensor,
        max_angular_momentum: int,
        *,
        is_improper: bool,
    ) -> "O3Transformation":
        """Create a transformation after validation in ``random_transformations``.

        The random factory validates its arguments and matrices before calling this
        method. This avoids repeating the public constructor's checks, matrix copy,
        and determinant calculation for every matrix. ``is_improper`` must match
        ``matrix``.
        """
        transformation = cls.__new__(cls)
        transformation._matrix = matrix
        transformation._max_angular_momentum = max_angular_momentum
        transformation._is_improper = is_improper
        transformation._wigner_D_cache = None
        return transformation

    def _ensure_wigner_D_cache(self) -> dict[int, torch.Tensor]:
        """Ensure that the Wigner-D cache has been built and return it."""
        if self._wigner_D_cache is None:
            self._wigner_D_cache = build_wigner_D_cache(
                self._max_angular_momentum,
                self._matrix,
                device=self._matrix.device,
                dtype=self._matrix.dtype,
            )

        return self._wigner_D_cache

    def _wigner_D_cache_entry(self, ell: int) -> torch.Tensor:
        """Return the internal cache entry for ``ell`` without copying it."""
        ell = self._validate_ell_range(ell)

        D = self._ensure_wigner_D_cache().get(ell)
        if D is None:
            raise ValueError(f"Wigner-D matrix for ell={ell} not found in cache.")

        return D

    @property
    def matrix(self) -> torch.Tensor:
        """The (3, 3) rotation or improper-rotation matrix."""
        return self._matrix

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the transformation matrix."""
        return self._matrix.dtype

    @property
    def device(self) -> torch.device:
        """The device of the transformation matrix."""
        return self._matrix.device

    @property
    def is_improper(self) -> bool:
        """Whether this transformation is improper, with negative determinant."""
        return self._is_improper

    def transform_cartesian(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to Cartesian vectors.

        :param vectors: (..., 3) tensor of Cartesian vectors
        :return: (..., 3) tensor of transformed vectors
        """
        return vectors @ self._matrix.T

    def _validate_ell_range(self, ell: int) -> int:
        """Check that ``ell`` is an integer in ``[0, max_angular_momentum]``."""
        ell = _validate_nonnegative_integer("ell", ell)

        if ell > self._max_angular_momentum:
            raise ValueError(
                f"ell={ell} exceeds max_angular_momentum={self._max_angular_momentum}."
            )

        return ell

    def transform_spherical(
        self, values: torch.Tensor, ell: int, sigma: int
    ) -> torch.Tensor:
        """Apply the transformation to spherical values.

        :param values: (..., 2*ell+1) tensor of spherical values
        :param ell: angular momentum in ``[0, max_angular_momentum]``
        :param sigma: ``+1`` for a proper spherical representation or ``-1`` for
            a pseudo one. Under an improper transformation, the representation
            acquires the factor ``sigma * (-1) ** ell``.
        :return: (..., 2*ell+1) tensor of transformed spherical values
        """
        ell = self._validate_ell_range(ell)
        parity_factor = _spherical_parity_factor(
            ell,
            sigma,
            is_improper=self.is_improper,
        )

        D = self._wigner_D_cache_entry(ell)
        transformed = values @ D.T
        if parity_factor != 1:
            transformed = transformed * parity_factor

        return transformed

    def wigner_D_matrix(self, ell: int):
        """Return the proper-part Wigner-D matrix for ``ell``.

        For an improper transformation, :meth:`transform_spherical` applies the
        inversion-parity factor separately.

        :param ell: angular momentum in ``[0, max_angular_momentum]``
        :return: (2*ell+1, 2*ell+1) Wigner-D matrix
        """
        return self._wigner_D_cache_entry(ell)


def random_transformations(
    n: int,
    max_angular_momentum: int = 0,
    *,
    device: torch.device,
    dtype: torch.dtype,
    include_inversions: bool = False,
    generator: torch.Generator | None = None,
) -> list[O3Transformation]:
    """Sample ``n`` transformations uniformly from SO(3), or from O(3) when
    inversions are included.

    Rotations are sampled from the Haar measure on SO(3) via random unit quaternions.
    When ``include_inversions`` is ``True``, each matrix is independently negated with
    probability 0.5, giving a uniform distribution over the full O(3) group.

    :param n: non-negative number of transformations to generate
    :param max_angular_momentum: non-negative maximum angular momentum for
        Wigner-D matrices
    :param device: target device for the output tensors
    :param dtype: target dtype for the output tensors; must be
        :attr:`torch.float32` or :attr:`torch.float64`
    :param include_inversions: if ``True``, sample from O(3) instead of SO(3)
    :param generator: optional :class:`torch.Generator` for reproducible sampling; when
        ``None`` the global RNG is used
    :return: list of ``n`` :class:`O3Transformation` objects
    """
    n = _validate_nonnegative_integer("n", n)
    max_angular_momentum = _validate_nonnegative_integer(
        "max_angular_momentum", max_angular_momentum
    )

    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}.")

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

    matrices_are_improper = [False] * n

    if include_inversions:
        signs = torch.randint(0, 2, (n,), device=device, generator=generator) * 2 - 1
        R = R * signs.to(dtype=dtype).reshape(n, 1, 1)
        matrices_are_improper = (signs < 0).tolist()

    identity = torch.eye(
        3,
        device=R.device,
        dtype=R.dtype,
    ).expand(n, 3, 3)
    if not torch.allclose(
        R @ R.transpose(-1, -2),
        identity,
        atol=1e-5,
    ):
        raise ValueError("Generated transformations are not orthogonal.")

    return [
        O3Transformation._create_from_internal_matrix(
            matrix,
            max_angular_momentum,
            is_improper=is_improper,
        )
        for matrix, is_improper in zip(
            R.unbind(0),
            matrices_are_improper,
            strict=True,
        )
    ]


def _value_row_indices_by_system(
    block: TensorBlock,
    system_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Return value-row indices in ``system_ids`` order, or all rows for one system."""
    if len(system_ids) == 1:
        return [torch.arange(block.values.shape[0], device=block.values.device)]

    if "system" not in block.samples.names:
        raise ValueError(
            "Rotational augmentation expects output samples to include a 'system' "
            "dimension when transforming multiple systems."
        )
    system_labels = block.samples.column("system").to(dtype=torch.long)
    unique_labels = torch.unique(system_labels)
    labels_are_known = torch.isin(unique_labels, system_ids)
    if not labels_are_known.all():
        unknown_labels = unique_labels[~labels_are_known]
        raise ValueError(
            f"Block samples contain system labels {unknown_labels.tolist()} that are "
            f"not in system_ids={system_ids.tolist()}. Every sample must be "
            f"assigned to a system in the transformation."
        )
    return [
        torch.nonzero(system_labels == system_id, as_tuple=False).reshape(-1)
        for system_id in system_ids
    ]


def _gradient_row_indices_by_system(
    grad_block: TensorBlock,
    parent_block: TensorBlock,
    system_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Group gradient rows by the system of their referenced value row."""
    if len(system_ids) == 1:
        return [
            torch.arange(grad_block.values.shape[0], device=grad_block.values.device)
        ]

    if "system" not in parent_block.samples.names:
        raise ValueError(
            "Rotational augmentation expects the values samples to include a 'system' "
            "dimension when transforming gradients of multiple systems."
        )

    parent_system_labels = parent_block.samples.column("system").to(dtype=torch.long)
    parent_value_rows = grad_block.samples.column("sample").to(dtype=torch.long)
    gradient_system_labels = parent_system_labels[parent_value_rows]

    return [
        torch.nonzero(
            gradient_system_labels == system_id,
            as_tuple=False,
        ).reshape(-1)
        for system_id in system_ids
    ]


def transform_system(system: System, transformation: O3Transformation) -> System:
    """Apply an O(3) transformation to a single System.

    Positions, cell vectors, neighbor-list displacements, and custom data following
    :ref:`o3-conventions` are transformed. Atomic types and periodic-boundary flags
    are preserved.

    :param system: input system
    :param transformation: O(3) transformation to apply, matching
        ``system.positions`` in dtype and device
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
        # Detach the input graph before registering the rotated values below.
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
    # Reserve einsum indices for all ten component axes supported by Metatomic.
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


def _component_axis_suffix(axis_name: str, prefix: str) -> tuple[bool, str]:
    """Match a component-axis name and return its supported suffix."""
    suffixes = ["", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9"]
    for suffix in suffixes:
        if axis_name == prefix + suffix:
            return True, suffix
    return False, ""


def _validate_component_axis_metadata(
    components: list[Labels],
    key: LabelsEntry,
) -> list[tuple[bool, int, int]]:
    """Validate component axes and return ``(is_spherical, ell, sigma)`` metadata."""
    if len(components) > 10:
        raise ValueError(
            f"can not transform a tensor with {len(components)} component axes; "
            "at most 10 are supported"
        )

    metadata: list[tuple[bool, int, int]] = []
    for component in components:
        axis_name = component.names[0]
        is_cartesian, _ = _component_axis_suffix(axis_name, "xyz")
        is_spherical, suffix = _component_axis_suffix(axis_name, "o3_mu")
        if is_cartesian:
            expected_labels = torch.arange(
                3,
                device=component.values.device,
                dtype=component.values.dtype,
            )
            if not torch.equal(component.values[:, 0], expected_labels):
                raise ValueError(
                    f"Cartesian component axis '{axis_name}' must use labels "
                    "[0, 1, 2] in x, y, z order."
                )
            metadata.append((False, 0, 1))
        elif is_spherical:
            ell = _validate_nonnegative_integer(
                "ell",
                int(key["o3_lambda" + suffix]),
            )
            sigma = int(key["o3_sigma" + suffix])
            _spherical_parity_factor(ell, sigma, is_improper=False)

            expected_labels = torch.arange(
                -ell,
                ell + 1,
                device=component.values.device,
                dtype=component.values.dtype,
            )
            if not torch.equal(component.values[:, 0], expected_labels):
                raise ValueError(
                    f"Spherical component axis '{axis_name}' for ell={ell} must use "
                    f"labels from {-ell} through {ell} in ascending order."
                )
            metadata.append((True, ell, sigma))
        else:
            raise ValueError(
                f"Found a component axis '{axis_name}', which is neither a Cartesian "
                "('xyz'/'xyz_1'/'xyz_2'/...) nor spherical ('o3_mu'/'o3_mu_1'/...) "
                "axis; it can not be transformed."
            )

    return metadata


def _max_o3_lambda_in_tensor(tensor: TensorMap) -> int:
    """Return the largest spherical rank in block values or attached gradients.

    A TensorMap containing only scalar or Cartesian component axes returns ``-1``.
    """
    max_o3_lambda = -1
    for key, block in tensor.items():
        metadata = _validate_component_axis_metadata(block.components, key)
        for is_spherical, ell, _sigma in metadata:
            if is_spherical and ell > max_o3_lambda:
                max_o3_lambda = ell

        for _gradient_name, gradient in block.gradients():
            gradient_metadata = _validate_component_axis_metadata(
                gradient.components,
                key,
            )
            for is_spherical, ell, _sigma in gradient_metadata:
                if is_spherical and ell > max_o3_lambda:
                    max_o3_lambda = ell

    return max_o3_lambda


def _axis_matrices_and_parity(
    metadata: list[tuple[bool, int, int]],
    transformation: O3Transformation,
) -> tuple[list[torch.Tensor], int]:
    """Return the axis matrices and their combined spherical parity factor."""
    matrices: list[torch.Tensor] = []
    parity = 1
    for is_spherical, ell, sigma in metadata:
        if is_spherical:
            matrices.append(transformation._wigner_D_cache_entry(ell))
            parity *= _spherical_parity_factor(
                ell,
                sigma,
                transformation.is_improper,
            )
        else:
            matrices.append(transformation._matrix)

    return matrices, parity


def _transform_component_values(
    values: torch.Tensor,
    components: list[Labels],
    key: LabelsEntry,
    row_indices: list[torch.Tensor],
    transformations: list[O3Transformation],
) -> torch.Tensor:
    """Rotate value or gradient rows with their assigned transformation."""
    metadata = _validate_component_axis_metadata(components, key)
    new_values = values.clone()
    for system_index, rows in enumerate(row_indices):
        if len(rows) == 0:
            continue
        matrices, parity = _axis_matrices_and_parity(
            metadata,
            transformations[system_index],
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
    """Apply per-system O(3) transformations to a block and its gradients.

    :param key: parent block key, supplying the O(3) labels required by spherical
        component axes
    :param block: block to transform
    :param systems: systems corresponding positionally to ``transformations``
    :param transformations: one O(3) transformation per system, matching
        ``block.values`` in dtype and device
    :param system_ids: one distinct integer ``"system"`` sample label per system;
        entry ``i`` is paired with ``transformations[i]``. A tensor argument must
        be one-dimensional and use the same device as ``block.values``. Defaults
        to ``range(len(systems))``
    :return: block with transformed values and gradients and unchanged labels; when
        ``systems`` is empty, the block is unchanged
    """
    system_ids = _validate_system_ids(
        systems,
        transformations,
        system_ids,
        expected_device=block.values.device,
    )
    if len(systems) == 0:
        return block

    _validate_transformations_dtype_device(
        transformations,
        expected_dtype=block.values.dtype,
        expected_device=block.values.device,
    )

    return _transform_block_impl(key, block, transformations, system_ids)


def _transform_block_impl(
    key: LabelsEntry,
    block: TensorBlock,
    transformations: list[O3Transformation],
    system_ids: torch.Tensor,
) -> TensorBlock:
    """Transform block values and gradients using validated system assignments."""
    value_sample_indices = _value_row_indices_by_system(block, system_ids)
    new_block = TensorBlock(
        values=_transform_component_values(
            block.values,
            block.components,
            key,
            value_sample_indices,
            transformations,
        ),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )
    for gradient_name, gradient in block.gradients():
        gradient_sample_indices = _gradient_row_indices_by_system(
            gradient,
            block,
            system_ids,
        )
        new_block.add_gradient(
            gradient_name,
            TensorBlock(
                values=_transform_component_values(
                    gradient.values,
                    gradient.components,
                    key,
                    gradient_sample_indices,
                    transformations,
                ),
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            ),
        )
    return new_block


def transform_tensor(
    tensor: TensorMap,
    systems: list[System],
    transformations: list[O3Transformation],
    system_ids: list[int] | torch.Tensor | None = None,
) -> TensorMap:
    """Apply per-system O(3) transformations to a TensorMap and its gradients.

    Blocks without component axes are scalar. Cartesian and spherical component
    axes are identified by name, so one :py:class:`TensorMap` may contain all three
    kinds of data.

    With multiple systems, the ``"system"`` sample label assigns each value sample
    to a transformation. With one system, this label is optional and ignored.

    :param tensor: TensorMap to transform
    :param systems: systems corresponding positionally to ``transformations``
    :param transformations: one O(3) transformation per system, matching the tensor
        values in dtype and device when present
    :param system_ids: one distinct integer ``"system"`` sample label per system;
        entry ``i`` is paired with ``transformations[i]``. A tensor argument must
        be one-dimensional and use the transformations' device. Defaults to
        ``range(len(systems))``
    :return: transformed TensorMap with the same keys and global information; when
        ``systems`` is empty, the tensor is unchanged
    """
    if len(tensor) != 0:
        system_ids_device = tensor.block(0).values.device
    elif len(transformations) != 0:
        system_ids_device = transformations[0].device
    else:
        system_ids_device = None

    system_ids = _validate_system_ids(
        systems,
        transformations,
        system_ids,
        expected_device=system_ids_device,
    )
    if len(systems) == 0:
        return tensor

    if len(tensor) != 0:
        values = tensor.block(0).values
        _validate_transformations_dtype_device(
            transformations,
            expected_dtype=values.dtype,
            expected_device=values.device,
        )

    new_blocks = [
        _transform_block_impl(key, block, transformations, system_ids)
        for key, block in tensor.items()
    ]
    transformed = TensorMap(keys=tensor.keys, blocks=new_blocks)
    for info_key, info_value in tensor.info().items():
        transformed.set_info(info_key, info_value)

    return transformed


def _transformation_indices(
    samples: Labels,
    n_transformations: int,
) -> torch.Tensor:
    """Map sample rows to local transformation indices."""
    if n_transformations <= 0:
        raise ValueError("n_transformations must be positive")
    if n_transformations == 1:
        return torch.zeros(
            len(samples),
            dtype=torch.long,
            device=samples.device,
        )
    if "system" not in samples.names:
        raise ValueError("multiple transformations require a 'system' sample dimension")

    indices = samples.column("system").to(dtype=torch.long)
    if bool(torch.any((indices < 0) | (indices >= n_transformations)).item()):
        raise ValueError("sample system indices exceed the transformation batch")
    return indices


def _transform_component_values_with_precomputed_matrices(
    values: torch.Tensor,
    components: list[Labels],
    key: LabelsEntry,
    transformation_indices: torch.Tensor,
    matrices: torch.Tensor,
    wigner_matrices: list[torch.Tensor],
    is_improper: bool,
) -> torch.Tensor:
    """Transform component axes with precomputed O(3) matrices."""
    metadata = _validate_component_axis_metadata(components, key)
    if len(metadata) == 0:
        return values.clone()

    transformed = values
    parity = 1
    for component_index, (is_spherical, ell, sigma) in enumerate(metadata):
        if is_spherical:
            if ell >= len(wigner_matrices):
                raise ValueError("spherical rank exceeds the Wigner-D storage")
            axis_matrices = wigner_matrices[ell]
            parity *= _spherical_parity_factor(ell, sigma, is_improper)
        else:
            axis_matrices = matrices

        component_axis = component_index + 1
        moved = torch.movedim(transformed, component_axis, -1)
        moved_shape = moved.shape
        flattened = moved.flatten(start_dim=1, end_dim=-2)
        matrices_for_rows = axis_matrices.index_select(
            0,
            transformation_indices,
        )
        transformed = torch.bmm(
            flattened,
            matrices_for_rows.transpose(1, 2),
        )
        transformed = transformed.reshape(moved_shape)
        transformed = torch.movedim(transformed, -1, component_axis)

    if parity != 1:
        transformed = transformed * parity
    return transformed


def _transform_tensor_with_precomputed_matrices(
    tensor: TensorMap,
    matrices: torch.Tensor,
    wigner_matrices: list[torch.Tensor],
    is_improper: bool,
) -> TensorMap:
    """Transform a TensorMap using precomputed matrices from one O(3) coset.

    ``matrices[i]`` is the actual Cartesian operation for local system ``i``,
    while ``wigner_matrices[ell][i]`` is the Wigner-D matrix for its proper
    rotational part. Every operation in the batch must be either proper or
    improper, as selected by ``is_improper``.

    With multiple operations, ``"system"`` sample labels are local indices into
    the matrix batch. A singleton batch does not require this sample dimension.
    The caller chooses the transformation direction by supplying either the
    forward matrices or their inverses.
    """
    if (
        matrices.dim() != 3
        or matrices.size(0) == 0
        or matrices.size(1) != 3
        or matrices.size(2) != 3
    ):
        raise ValueError("matrices must have shape (N, 3, 3) with N > 0")
    if matrices.dtype != torch.float32 and matrices.dtype != torch.float64:
        raise TypeError("matrices must use float32 or float64")
    if len(tensor) != 0:
        reference_values = tensor.block(0).values
        if (
            matrices.dtype != reference_values.dtype
            or matrices.device != reference_values.device
        ):
            raise ValueError("tensor and matrices must have the same dtype and device")

    blocks: list[TensorBlock] = []
    for key, block in tensor.items():
        value_indices = _transformation_indices(
            block.samples,
            matrices.size(0),
        )
        new_block = TensorBlock(
            values=_transform_component_values_with_precomputed_matrices(
                block.values,
                block.components,
                key,
                value_indices,
                matrices,
                wigner_matrices,
                is_improper,
            ),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for gradient_name, gradient in block.gradients():
            parent_rows = gradient.samples.column("sample").to(dtype=torch.long)
            gradient_indices = value_indices.index_select(0, parent_rows)
            new_block.add_gradient(
                gradient_name,
                TensorBlock(
                    values=_transform_component_values_with_precomputed_matrices(
                        gradient.values,
                        gradient.components,
                        key,
                        gradient_indices,
                        matrices,
                        wigner_matrices,
                        is_improper,
                    ),
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=gradient.properties,
                ),
            )
        blocks.append(new_block)

    transformed = TensorMap(tensor.keys, blocks)
    for info_name, info_value in tensor.info().items():
        transformed.set_info(info_name, info_value)
    return transformed
