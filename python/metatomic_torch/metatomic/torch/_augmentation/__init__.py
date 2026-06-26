"""
O(3) augmentation: apply batched rotation/inversion transformations to Systems and
TensorMaps.

metatomic stores positions and cell as row vectors (shape ``(N, 3)``), so a rotation
matrix ``R`` (3x3) acts as ``x @ R.T`` throughout this module.
"""

import metatensor.torch as mts
import numpy as np
import torch
from metatensor.torch import TensorBlock, TensorMap

from .. import System, register_autograd_neighbors
from ._wigner import compute_wigner_batch


# Component-axis names recognised by the augmentation machinery. Cartesian axes are
# rotated by ``R`` directly (so improper rotations flip vectors); spherical axes are
# rotated by the Wigner-D matrix of the matching ``o3_lambda`` plus an explicit
# ``(-1)^ell * sigma`` inversion parity factor.
_CARTESIAN_AXES = frozenset({"xyz", "xyz_1", "xyz_2"})
_SPHERICAL_AXIS_TO_LAMBDA = {
    "o3_mu": "o3_lambda",
    "o3_mu_1": "o3_lambda_1",
    "o3_mu_2": "o3_lambda_2",
}
_SPHERICAL_AXIS_TO_SIGMA = {
    "o3_mu": "o3_sigma",
    "o3_mu_1": "o3_sigma_1",
    "o3_mu_2": "o3_sigma_2",
}

# einsum index letters for the per-axis component contraction (input lower, output
# upper). Six axes is far more than any realistic block (value + gradient) needs.
_EINSUM_IN = "abcdef"
_EINSUM_OUT = "ABCDEF"


def _row_indices_from_system_ids(
    system_ids: torch.Tensor,
    n_systems: int,
) -> list[torch.Tensor]:
    """Group row indices by system, mapping each distinct ``system`` id to one system.

    The ``"system"`` column is not necessarily a 0-based batch index: metatrain keeps
    each structure's original dataset index there, so the labels in a batch are an
    arbitrary set (e.g. ``[67, 92, 38, ...]``). The i-th distinct id, **in sorted
    order**, is taken to be system ``i``; when the ids already span ``[0, n_systems)``
    this reduces to the identity grouping.

    .. note::

        This pairs the i-th sorted label with ``systems[i]``/``transformations[i]``,
        which is only consistent with the (positional) system rotation when the caller
        passes ``systems`` ordered by their label. metatrain satisfies this.

    :param system_ids: 1-D tensor with one system id per row
    :param n_systems: number of systems being augmented
    :return: list of length ``n_systems``; entry ``i`` selects the rows of system ``i``
    :raises ValueError: if an id is negative or there are more distinct ids than systems
    """
    device = system_ids.device
    if len(system_ids) == 0:
        return [
            torch.zeros(0, dtype=torch.long, device=device) for _ in range(n_systems)
        ]

    min_id = int(system_ids.min().item())
    max_id = int(system_ids.max().item())
    if min_id < 0:
        raise ValueError("Encountered output samples with negative system indices.")

    if max_id < n_systems:
        # ids already index into [0, n_systems): group directly (sparse rows are fine).
        return [
            torch.nonzero(system_ids == i, as_tuple=False).reshape(-1)
            for i in range(n_systems)
        ]

    # ids are arbitrary labels: map the sorted distinct ids onto systems 0..n_systems-1.
    unique_ids: list[int] = sorted(set(system_ids.tolist()))
    if len(unique_ids) > n_systems:
        raise ValueError(
            f"TensorMap block has {len(unique_ids)} distinct system indices "
            f"but only {n_systems} systems were provided."
        )
    id_to_pos = {uid: pos for pos, uid in enumerate(unique_ids)}
    remapped = torch.tensor(
        [id_to_pos[int(sid)] for sid in system_ids.tolist()],
        dtype=torch.long,
        device=device,
    )
    return [
        torch.nonzero(remapped == i, as_tuple=False).reshape(-1)
        for i in range(n_systems)
    ]


def _block_row_indices_by_system(
    block: TensorBlock,
    n_systems: int,
) -> list[torch.Tensor]:
    """Return row-index tensors into ``block.values``, one per system.

    With a single system every row belongs to it (any ``"system"`` label is ignored).
    With several systems the ``"system"`` column is required and is interpreted by
    :func:`_row_indices_from_system_ids`.

    :param block: block whose ``samples`` may contain a ``"system"`` column
    :param n_systems: number of systems being augmented
    :return: list of length ``n_systems``; entry ``i`` selects all rows of system ``i``
    """
    if "system" not in block.samples.names:
        if n_systems == 1:
            return [torch.arange(block.values.shape[0], device=block.values.device)]
        raise ValueError(
            "Rotational augmentation expects output samples to include a 'system' "
            "dimension when transforming multiple systems."
        )
    system_ids = block.samples.column("system").to(dtype=torch.long)
    return _row_indices_from_system_ids(system_ids, n_systems)


def _gradient_row_indices_by_system(
    grad_block: TensorBlock,
    parent_block: TensorBlock,
    n_systems: int,
) -> list[torch.Tensor]:
    """Return row-index tensors into a gradient block, one per system.

    Gradient samples carry a ``"sample"`` column indexing into the parent block (the
    metatensor convention) rather than their own ``"system"`` column, so the system of
    each gradient row is read from the parent block's ``"system"`` column.

    :param grad_block: gradient block to route
    :param parent_block: the value block this gradient is attached to
    :param n_systems: number of systems being augmented
    :return: list of length ``n_systems``; entry ``i`` selects the gradient rows of
        system ``i``
    """
    if n_systems == 1:
        return [
            torch.arange(grad_block.values.shape[0], device=grad_block.values.device)
        ]
    if "sample" not in grad_block.samples.names:
        raise ValueError(
            "Gradient samples are expected to include a 'sample' dimension indexing "
            "into the parent block."
        )
    if "system" not in parent_block.samples.names:
        raise ValueError(
            "Rotational augmentation expects parent samples to include a 'system' "
            "dimension when transforming gradients of multiple systems."
        )
    parent_system = parent_block.samples.column("system").to(dtype=torch.long)
    sample_index = grad_block.samples.column("sample").to(dtype=torch.long)
    return _row_indices_from_system_ids(parent_system[sample_index], n_systems)


def _has_spherical_axis(tmap: TensorMap) -> bool:
    """Whether any block of ``tmap`` carries a spherical component axis.

    :param tmap: TensorMap to inspect
    :return: ``True`` if any block has an ``o3_mu``/``o3_mu_1``/``o3_mu_2`` component
    """
    for block in tmap.blocks():
        for component in block.components:
            if component.names[0] in _SPHERICAL_AXIS_TO_LAMBDA:
                return True
    return False


def _transform_single_system(
    system: System,
    transformation: torch.Tensor,
) -> System:
    """Apply an O(3) transformation to a single System.

    Rotates positions, cell vectors, any registered per-atom data, and all neighbor-list
    displacement vectors. Types and pbc flags are unchanged.

    Registered data is rotated by the same machinery as targets (see
    :func:`_transform_tmap`), as a single-system batch: scalar blocks pass through and
    Cartesian (``xyz``/``xyz_1``/``xyz_2``) blocks are rotated by ``transformation``.
    No Wigner-D matrices are computed for System data, so data carrying a spherical
    (``o3_mu``) axis cannot be rotated and is passed through unchanged.

    :param system: input system
    :param transformation: (3, 3) rotation or improper-rotation matrix
    :return: new System with transformed geometry
    """
    new_system = System(
        positions=system.positions @ transformation.T,
        types=system.types,
        cell=system.cell @ transformation.T,
        pbc=system.pbc,
    )
    for data_name in system.known_data():
        data = system.get_data(data_name)
        if _has_spherical_axis(data):
            # Rotating spherical data needs Wigner-D matrices, which are not computed
            # for System data; rather than rotate it incorrectly we pass it through
            # unchanged (attaching spherical data to a System is allowed but exotic).
            new_system.add_data(data_name, data)
        else:
            new_system.add_data(
                data_name,
                _transform_tmap(data_name, data, [system], [transformation], {}),
            )
    for options in system.known_neighbor_lists():
        neighbors = mts.detach_block(system.get_neighbor_list(options))
        # neighbor vectors are stored as (N, 3, 1); squeeze/unsqueeze around the matmul
        neighbors.values[:] = (
            neighbors.values.squeeze(-1) @ transformation.T
        ).unsqueeze(-1)
        register_autograd_neighbors(new_system, neighbors)
        new_system.add_neighbor_list(options, neighbors)
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
    if len(matrices) == 0:
        return values
    n_axes = len(matrices)
    in_subscript = "i" + _EINSUM_IN[:n_axes] + "p"
    out_subscript = "i" + _EINSUM_OUT[:n_axes] + "p"
    matrix_subscripts = [_EINSUM_OUT[j] + _EINSUM_IN[j] for j in range(n_axes)]
    equation = ",".join(matrix_subscripts + [in_subscript]) + "->" + out_subscript
    return torch.einsum(equation, *matrices, values)


def _axis_matrices_and_parity(
    name: str,
    components: list,
    key,
    R: torch.Tensor,
    wigner_D_matrices: dict[int, list[torch.Tensor]],
    system_index: int,
    is_inverted: bool,
) -> tuple[list[torch.Tensor], int]:
    """Pick the rotation matrix for each component axis and the O(3) inversion parity.

    Cartesian axes (``xyz``/``xyz_1``/``xyz_2``) use ``R`` directly, so improper
    rotations flip vectors automatically. Spherical axes (``o3_mu``/``o3_mu_1``/
    ``o3_mu_2``) use the proper-rotation Wigner-D matrix of the matching ``o3_lambda``
    plus a ``(-1)^ell * sigma`` parity factor accumulated whenever ``R`` is improper.

    :param name: TensorMap name, used only in error messages
    :param components: component :class:`Labels` of the block (or gradient block)
    :param key: the parent block's key, supplying ``o3_lambda``/``o3_sigma`` values
    :param R: this system's (3, 3) transformation matrix
    :param wigner_D_matrices: ``{ell: [D_0, ..., D_{N-1}]}`` real Wigner-D matrices
    :param system_index: index selecting this system's Wigner-D matrix
    :param is_inverted: whether ``R`` is an improper rotation (``det(R) < 0``)
    :return: ``(matrices, parity)`` with one matrix per component axis
    :raises ValueError: if a component axis is neither Cartesian nor spherical
    """
    matrices: list[torch.Tensor] = []
    parity = 1
    for component in components:
        axis_name = component.names[0]
        if axis_name in _CARTESIAN_AXES:
            matrices.append(R)
        elif axis_name in _SPHERICAL_AXIS_TO_LAMBDA:
            ell = int(key[_SPHERICAL_AXIS_TO_LAMBDA[axis_name]])
            matrices.append(wigner_D_matrices[ell][system_index])
            if is_inverted:
                sigma = int(key[_SPHERICAL_AXIS_TO_SIGMA[axis_name]])
                parity *= ((-1) ** ell) * sigma
        else:
            raise ValueError(
                f"TensorMap '{name}' has component axis '{axis_name}', which is "
                "neither a Cartesian ('xyz'/'xyz_1'/'xyz_2') nor spherical "
                "('o3_mu'/'o3_mu_1'/'o3_mu_2') axis; rotational augmentation cannot "
                "transform it."
            )
    return matrices, parity


def _transform_component_values(
    name: str,
    values: torch.Tensor,
    components: list,
    key,
    row_indices: list[torch.Tensor],
    transformations: list[torch.Tensor],
    wigner_D_matrices: dict[int, list[torch.Tensor]],
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
        R = transformations[system_index]
        is_inverted = bool(torch.det(R) < 0)
        matrices, parity = _axis_matrices_and_parity(
            name, components, key, R, wigner_D_matrices, system_index, is_inverted
        )
        rotated = _contract_component_axes(values[rows], matrices)
        if parity != 1:
            rotated = rotated * parity
        new_values[rows] = rotated
    return new_values


def _transform_block(
    name: str,
    key,
    block: TensorBlock,
    n_systems: int,
    transformations: list[torch.Tensor],
    wigner_D_matrices: dict[int, list[torch.Tensor]],
) -> TensorBlock:
    """Rotate one block and all of its gradients.

    Value rows are routed to systems by their ``"system"`` column; every gradient is
    routed by the parent block's ``"system"`` label via
    :func:`_gradient_row_indices_by_system` (the gradient's ``"sample"`` column indexes
    the parent). Each gradient reuses the parent ``key``, so a spherical component axis
    inherited from the value keeps the value's ``o3_lambda``/``o3_sigma``, while the
    extra Cartesian gradient-direction axis is rotated by ``R``.

    :param name: TensorMap name, used only in error messages
    :param key: the block's key
    :param block: the value block to rotate
    :param n_systems: number of systems being augmented
    :param transformations: per-system (3, 3) transformation matrices
    :param wigner_D_matrices: ``{ell: [D_0, ..., D_{N-1}]}`` real Wigner-D matrices
    :return: new block with rotated values and gradients
    """
    row_indices = _block_row_indices_by_system(block, n_systems)
    new_block = TensorBlock(
        values=_transform_component_values(
            name,
            block.values,
            block.components,
            key,
            row_indices,
            transformations,
            wigner_D_matrices,
        ),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )
    for gradient_name in block.gradients_list():
        grad_block = block.gradient(gradient_name)
        grad_rows = _gradient_row_indices_by_system(grad_block, block, n_systems)
        new_block.add_gradient(
            gradient_name,
            TensorBlock(
                values=_transform_component_values(
                    name,
                    grad_block.values,
                    grad_block.components,
                    key,
                    grad_rows,
                    transformations,
                    wigner_D_matrices,
                ),
                samples=grad_block.samples,
                components=grad_block.components,
                properties=grad_block.properties,
            ),
        )
    return new_block


def _transform_tmap(
    name: str,
    tmap: TensorMap,
    systems: list[System],
    transformations: list[torch.Tensor],
    wigner_D_matrices: dict[int, list[torch.Tensor]],
) -> TensorMap:
    """Rotate every block (and its gradients) of a TensorMap.

    The tensor character of each component axis is inferred from its name, so a single
    TensorMap may freely mix scalar, Cartesian and spherical blocks, and blocks may
    carry gradients.

    :param name: used only in error messages
    :param tmap: TensorMap to rotate
    :param systems: input systems; length gives the batch size
    :param transformations: per-system (3, 3) transformation matrices
    :param wigner_D_matrices: ``{ell: [D_0, ..., D_{N-1}]}`` real Wigner-D matrices
    :return: new TensorMap with rotated values and gradients
    :raises ValueError: if a component axis name is not recognised
    """
    n_systems = len(systems)
    new_blocks = [
        _transform_block(
            name, key, block, n_systems, transformations, wigner_D_matrices
        )
        for key, block in tmap.items()
    ]
    return TensorMap(keys=tmap.keys, blocks=new_blocks)


def _apply_transformations(
    systems: list[System],
    targets: dict[str, TensorMap],
    transformations: list[torch.Tensor],
    wigner_D_matrices: dict[int, list[torch.Tensor]],
    extra_data: dict[str, TensorMap] | None = None,
) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
    """Apply a batch of O(3) transformations to systems and TensorMaps simultaneously.

    Each element of ``transformations`` is a (3, 3) matrix (rotation or improper
    rotation) applied to the corresponding system. TensorMaps in ``targets`` and
    ``extra_data`` are transformed per system (Cartesian axes by ``R``, spherical axes
    by the matching Wigner-D matrix), except that keys ending in ``"_mask"`` pass
    through unchanged.

    :param systems: input systems, one per transformation
    :param targets: TensorMaps to transform (e.g. model predictions to back-rotate)
    :param transformations: per-system (3, 3) transformation matrices
    :param wigner_D_matrices: ``{ell: [D_0, ..., D_{N-1}]}`` real Wigner-D matrices
    :param extra_data: additional TensorMaps to transform alongside targets
    :return: ``(new_systems, new_targets, new_extra_data)``
    """
    new_systems = [
        _transform_single_system(system, R)
        for system, R in zip(systems, transformations, strict=True)
    ]

    new_targets: dict[str, TensorMap] = {
        name: _transform_tmap(name, tmap, systems, transformations, wigner_D_matrices)
        for name, tmap in targets.items()
    }

    new_extra_data: dict[str, TensorMap] = {}
    if extra_data is not None:
        for key, value in extra_data.items():
            if key.endswith("_mask"):
                new_extra_data[key] = value
            else:
                new_extra_data[key] = _transform_tmap(
                    key, value, systems, transformations, wigner_D_matrices
                )

    return new_systems, new_targets, new_extra_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SPHERICAL_KEY_NAMES = frozenset({"o3_lambda", "o3_lambda_1", "o3_lambda_2"})


def _rotations_to_zyz(
    rotations: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a list of O(3) matrices into ZYZ Euler angles :math:`(\\alpha, \\beta,
    \\gamma)`.

    For improper rotations (det < 0) the proper part ``-R`` is decomposed; the inversion
    parity factor is handled separately when applying Wigner-D matrices.

    :param rotations: list of (3, 3) orthogonal tensors
    :return: ``(alphas, betas, gammas)`` as 1-D numpy arrays of length
        ``len(rotations)``
    """
    alphas = np.empty(len(rotations))
    betas = np.empty(len(rotations))
    gammas = np.empty(len(rotations))
    for i, R in enumerate(rotations):
        proper_R = R if torch.det(R) > 0 else -R
        # R = Rz(alpha) Ry(beta) Rz(gamma): element [2,2] = cos(beta)
        cos_beta = float(proper_R[2, 2].clamp(-1.0, 1.0))
        beta = np.arccos(cos_beta)
        sin_beta = np.sin(beta)
        if abs(sin_beta) < 1e-10:
            # Gimbal lock: only alpha +/- gamma is defined; fix gamma=0
            if cos_beta > 0:
                alpha = float(torch.atan2(proper_R[1, 0], proper_R[0, 0]))
            else:
                alpha = float(torch.atan2(-proper_R[1, 0], -proper_R[0, 0]))
            gamma = 0.0
        else:
            # R[0,2]=cos(alpha)*sin(beta), R[1,2]=sin(alpha)*sin(beta): alpha via atan2
            # R[2,1]=sin(beta)*sin(gamma), R[2,0]=-sin(beta)*cos(gamma): gamma via atan2
            alpha = float(torch.atan2(proper_R[1, 2], proper_R[0, 2]))
            gamma = float(torch.atan2(proper_R[2, 1], -proper_R[2, 0]))
        alphas[i] = alpha
        betas[i] = beta
        gammas[i] = gamma
    return alphas, betas, gammas


def random_rotations(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    include_inversions: bool = False,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Sample ``n`` uniformly distributed O(3) transformations.

    Rotations are sampled from the Haar measure on SO(3) via random unit quaternions.
    When ``include_inversions`` is ``True``, each matrix is independently negated with
    probability 0.5, giving a uniform distribution over the full O(3) group.

    :param n: number of transformations to generate
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
    return list(R.unbind(0))


def apply_transformations(
    systems: list[System],
    targets: dict[str, TensorMap],
    transformations: list[torch.Tensor],
    extra_data: dict[str, TensorMap] | None = None,
) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
    """Apply a batch of O(3) transformations to systems and TensorMaps simultaneously.

    Wigner-D matrices are derived automatically from ``transformations``; the tensor
    type (scalar, Cartesian, spherical) is inferred from each TensorMap's component axis
    names. Keys in ``extra_data`` that end in ``"_mask"`` pass through unchanged.

    :param systems: input systems, one per transformation
    :param targets: model output TensorMaps to back-rotate (e.g. predicted energies,
        forces, or spherical features)
    :param transformations: per-system (3, 3) orthogonal matrices; use
        :func:`random_rotations` to generate these
    :param extra_data: additional TensorMaps to transform alongside ``targets``
    :return: ``(new_systems, new_targets, new_extra_data)``
    :raises ValueError: if ``len(systems) != len(transformations)``, any matrix is not a
        (3, 3) orthogonal matrix, or the transformations, systems and TensorMaps do not
        share a common dtype and device
    """
    if len(systems) != len(transformations):
        raise ValueError(
            f"Expected one transformation per system, got {len(transformations)} "
            f"transformations for {len(systems)} systems."
        )
    for i, R in enumerate(transformations):
        if R.shape != (3, 3):
            raise ValueError(
                f"Transformation {i} has shape {tuple(R.shape)}; expected (3, 3)."
            )
        identity = torch.eye(3, device=R.device, dtype=R.dtype)
        if not torch.allclose(R @ R.T, identity, atol=1e-5):
            raise ValueError(
                f"Transformation {i} is not orthogonal (R @ R.T deviates from I)."
            )

    if len(transformations) > 0:
        # Everything is contracted with the transformations (or the Wigner-D matrices
        # derived from them), so dtype and device must match throughout, otherwise the
        # matmuls below fail with a much less helpful message.
        reference = transformations[0]
        for i, R in enumerate(transformations):
            if R.dtype != reference.dtype or R.device != reference.device:
                raise ValueError(
                    f"Transformation {i} has dtype/device ({R.dtype}, {R.device}) "
                    f"differing from transformation 0 ({reference.dtype}, "
                    f"{reference.device}); all transformations must agree."
                )
        for i, system in enumerate(systems):
            if (
                system.positions.dtype != reference.dtype
                or system.positions.device != reference.device
            ):
                raise ValueError(
                    f"System {i} has positions with dtype/device "
                    f"({system.positions.dtype}, {system.positions.device}) differing "
                    f"from the transformations ({reference.dtype}, "
                    f"{reference.device})."
                )
        for label, tmap in list(targets.items()) + (
            [(k, v) for k, v in extra_data.items() if not k.endswith("_mask")]
            if extra_data is not None
            else []
        ):
            for block in tmap.blocks():
                if (
                    block.values.dtype != reference.dtype
                    or block.values.device != reference.device
                ):
                    raise ValueError(
                        f"TensorMap '{label}' has values with dtype/device "
                        f"({block.values.dtype}, {block.values.device}) differing "
                        f"from the transformations ({reference.dtype}, "
                        f"{reference.device})."
                    )

    # Determine the highest angular momentum present across all TensorMaps
    ell_max = 0
    all_tmaps = list(targets.values())
    if extra_data is not None:
        all_tmaps += [v for k, v in extra_data.items() if not k.endswith("_mask")]
    for tmap in all_tmaps:
        for name in tmap.keys.names:
            if name in _SPHERICAL_KEY_NAMES:
                col = tmap.keys.column(name)
                if len(col) > 0:
                    ell_max = max(ell_max, int(col.max()))

    if len(transformations) > 0:
        device = transformations[0].device
        dtype = transformations[0].dtype
        angles = _rotations_to_zyz(transformations)
        wigner_batch = compute_wigner_batch(ell_max, angles, device=device, dtype=dtype)
        # Unbind the batch dim: {ell: (N, 2l+1, 2l+1)} to {ell: [D_0, ..., D_{N-1}]}
        wigner_D_matrices: dict[int, list[torch.Tensor]] = {
            ell: list(D.unbind(0)) for ell, D in wigner_batch.items()
        }
    else:
        wigner_D_matrices = {}

    return _apply_transformations(
        systems, targets, transformations, wigner_D_matrices, extra_data
    )
