"""
Rotate systems and tensor maps under O(3) transformations, routing rows of
multi-system tensors by their ``"system"`` sample label.
"""

import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap

from .. import System, register_autograd_neighbors
from ._wigner import build_wigner_D_cache


# Component-axis names recognised by the augmentation machinery.
_SUFFIXES = [""] + [f"_{i}" for i in range(1, 10)]
_CARTESIAN_AXES = frozenset({f"xyz{s}" for s in _SUFFIXES})
_SPHERICAL_AXIS_TO_LAMBDA = {f"o3_mu{s}": f"o3_lambda{s}" for s in _SUFFIXES}
_SPHERICAL_AXIS_TO_SIGMA = {f"o3_mu{s}": f"o3_sigma{s}" for s in _SUFFIXES}


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
            will be precomputed for all ``ell <= max_angular_momentum``
        """
        if matrix.shape != (3, 3):
            raise ValueError(
                f"Transformation has shape {tuple(matrix.shape)}; expected (3, 3)."
            )

        identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
        if not torch.allclose(matrix @ matrix.T, identity, atol=1e-5):
            raise ValueError(
                "Transformation is not orthogonal (R @ R.T deviates from I)."
            )

        self._matrix = matrix
        self._max_angular_momentum = max_angular_momentum
        self._is_inverted = bool(torch.det(matrix) < 0)

        self._wigner_D_cache = build_wigner_D_cache(
            max_angular_momentum,
            matrix,
            device=matrix.device,
            dtype=matrix.dtype,
        )

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
        :param sigma: inversion parity of the spherical representation
        :return: (..., 2*ell+1) tensor of transformed spherical values
        """
        D = self._wigner_D_cache.get(ell)
        if D is None:
            raise ValueError(f"Wigner-D matrix for ell={ell} not found in cache.")
        transformed = values @ D.T
        if sigma == -1:
            transformed = transformed * ((-1) ** ell)
        return transformed

    def wigner_D_matrix(self, ell: int):
        """Return the Wigner-D matrix for this transformation and angular momentum ell.

        :param ell: angular momentum of the spherical representation
        :return: (2*ell+1, 2*ell+1) Wigner-D matrix
        """
        D = self._wigner_D_cache.get(ell)
        if D is None:
            raise ValueError(f"Wigner-D matrix for ell={ell} not found in cache.")
        return D


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
    :return: list of ``n`` orthogonal (3, 3) tensors
    """
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

    if include_inversions:
        signs = torch.randint(0, 2, (n,), device=device, generator=generator) * 2 - 1
        R = R * signs.to(dtype=dtype).reshape(n, 1, 1)

    return [O3Transformation(r, max_angular_momentum) for r in R.unbind(0)]


def _block_row_indices_by_system(
    block: TensorBlock,
    system_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Return row-index tensors into ``block.values``, one per system.

    With a single system every row belongs to it (any ``"system"`` label is ignored).

    With several systems the ``"system"`` column is required and each row is routed by
    matching its ``"system"`` label against ``system_ids``.

    :param block: block whose ``samples`` may contain a ``"system"`` column
    :param system_ids: one value per system; ``system_ids[i]`` is the value of the
        ``"system"`` column identifying the rows belonging to the i-th entry in the
        ``systems`` list passed to ``transform_tensor`` or ``transform_block``
    :return: list of length ``len(system_ids)``; entry ``i`` selects the rows of system
        ``i``
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
    """Return row-index tensors into a gradient block, one per system.

    Gradient samples carry a ``"sample"`` column indexing into the parent block rather
    than their own ``"system"`` column, so the system of each gradient row is read from
    the parent block's ``"system"`` column.

    :param grad_block: gradient block to route
    :param parent_block: the value block this gradient is attached to
    :param system_ids: one value per system; ``system_ids[i]`` is the value of the
        ``"system"`` column identifying the rows belonging to the i-th entry in the
        ``systems`` list passed to ``transform_tensor`` or ``transform_block``
    :return: list of length ``len(system_ids)``; entry ``i`` selects the gradient
        rows of system ``i``
    """
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
        neighbors_values = neighbors.values.squeeze(-1)
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
    # einsum index letters for the per-axis component contraction (input lower, output
    # upper). Six axes is far more than any realistic block (value + gradient) needs.
    _EINSUM_IN = "abcdef"
    _EINSUM_OUT = "ABCDEF"

    if len(matrices) == 0:
        return values
    n_axes = len(matrices)
    in_subscript = "i" + _EINSUM_IN[:n_axes] + "p"
    out_subscript = "i" + _EINSUM_OUT[:n_axes] + "p"
    matrix_subscripts = [_EINSUM_OUT[j] + _EINSUM_IN[j] for j in range(n_axes)]
    equation = ",".join(matrix_subscripts + [in_subscript]) + "->" + out_subscript
    return torch.einsum(equation, *matrices, values)


def _axis_matrices_and_parity(
    components: list[Labels],
    key: LabelsEntry,
    transformation: O3Transformation,
) -> tuple[list[torch.Tensor], int]:
    """Pick the rotation matrix for each component axis and the O(3) inversion parity.

    Cartesian axes (``xyz``/``xyz_1``/``xyz_2``/...) use the rotation directly, so
    improper rotations flip vectors automatically. Spherical axes
    (``o3_mu``/``o3_mu_1``/ ``o3_mu_2``) use the proper-rotation Wigner-D matrix of the
    matching ``o3_lambda`` plus a ``(-1)^ell * sigma`` parity factor accumulated
    whenever ``transformation`` is improper.

    :param name: TensorMap name, used only in error messages
    :param components: component :class:`Labels` of the block (or gradient block)
    :param key: the parent block's key, supplying ``o3_lambda``/``o3_sigma`` values
    :param transformation: this system's O3 transformation
    :return: ``(matrices, parity)`` with one matrix per component axis
    """
    matrices: list[torch.Tensor] = []
    parity = 1
    for component in components:
        axis_name = component.names[0]
        if axis_name in _CARTESIAN_AXES:
            matrices.append(transformation.matrix)
        elif axis_name in _SPHERICAL_AXIS_TO_LAMBDA:
            ell = int(key[_SPHERICAL_AXIS_TO_LAMBDA[axis_name]])
            matrices.append(transformation.wigner_D_matrix(ell))
            if transformation.is_inverted:
                sigma = int(key[_SPHERICAL_AXIS_TO_SIGMA[axis_name]])
                parity *= ((-1) ** ell) * sigma
        else:
            raise ValueError(
                f"Found a component axis '{axis_name}', which is neither a Cartesian "
                "('xyz'/'xyz_1'/'xyz_2'/...) nor spherical ('o3_mu'/'o3_mu_1'/...) "
                "axis; it can not be transformed."
            )
    return matrices, parity


def _transform_component_values(
    values: torch.Tensor,
    components: list[Labels],
    key: LabelsEntry,
    row_indices: list[torch.Tensor],
    transformations: list[O3Transformation],
) -> torch.Tensor:
    """Rotate the values of a single value or gradient block, per system.

    :param name: TensorMap name, used only in error messages
    :param values: the block's values tensor
    :param components: the block's component :class:`Labels`
    :param key: the parent block's key (for spherical ``o3_lambda``/``o3_sigma``)
    :param row_indices: per-system row indices into ``values``
    :param transformations: per-system (3, 3) transformation matrices
    :param wigner_D_matrices: ``{ell: [D_0, ..., D_{N-1}]}`` real Wigner-D matrices
    :return: new values tensor with each system's rows rotated
    """
    new_values = values.clone()
    for system_index, rows in enumerate(row_indices):
        if len(rows) == 0:
            continue
        matrices, parity = _axis_matrices_and_parity(
            components, key, transformations[system_index]
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
    :param transformations: per-system O(3) transformation matrices
    :param system_ids: index of the ``systems`` used in the samples of ``block``;
        defaults to ``list(range(len(systems)))``
    :return: new block with rotated values and gradients
    """
    assert len(systems) == len(transformations)
    if len(systems) == 0:
        return block

    if system_ids is None:
        system_ids = list(range(len(systems)))

    if not isinstance(system_ids, torch.Tensor):
        system_ids = torch.tensor(
            system_ids, dtype=torch.int32, device=block.values.device
        )

    if (
        block.values.dtype != transformations[0].dtype
        or block.values.device != transformations[0].device
    ):
        raise ValueError(
            f"TensorMap has values with dtype/device "
            f"({block.values.dtype}, {block.values.device}) differing "
            f"from the transformations ({transformations[0].dtype}, "
            f"{transformations[0].device})."
        )

    return _transform_block_impl(key, block, systems, transformations, system_ids)


def _transform_block_impl(
    key: LabelsEntry,
    block: TensorBlock,
    systems: list[System],
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

    One of the samples dimensions must be ``"system"``, which is used to route each row
    to the correct system and transformation.

    :param tensor: TensorMap to rotate
    :param systems: input systems
    :param transformations: per-system O(3) transformation matrices
    :param system_ids: index of the ``systems`` used in the samples of ``tensor``;
        defaults to ``list(range(len(systems)))``
    :return: new TensorMap with rotated values and gradients
    """
    assert len(systems) == len(transformations)
    if len(systems) == 0:
        return tensor

    if system_ids is None:
        system_ids = list(range(len(systems)))

    if not isinstance(system_ids, torch.Tensor):
        system_ids = torch.tensor(
            system_ids, dtype=torch.int32, device=transformations[0].device
        )

    if len(tensor) != 0:
        block = tensor.block(0)
        if (
            block.values.dtype != transformations[0].dtype
            or block.values.device != transformations[0].device
        ):
            raise ValueError(
                f"TensorMap has values with dtype/device "
                f"({block.values.dtype}, {block.values.device}) differing "
                f"from the transformations ({transformations[0].dtype}, "
                f"{transformations[0].device})."
            )

    new_blocks = [
        _transform_block_impl(key, block, systems, transformations, system_ids)
        for key, block in tensor.items()
    ]
    return TensorMap(keys=tensor.keys, blocks=new_blocks)
