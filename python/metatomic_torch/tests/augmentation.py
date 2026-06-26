import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    NeighborListOptions,
    System,
    apply_transformations,
    random_rotations,
    register_autograd_neighbors,
)
from metatomic.torch._augmentation import _apply_transformations, _rotations_to_zyz
from metatomic.torch._augmentation._wigner import compute_wigner_batch


def _axis_angle(axis, theta):
    """A general (non-degenerate, beta != 0) rotation matrix from axis and angle."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ]
    )


# Change of basis from Cartesian (x, y, z) to real ell=1 spherical harmonics, ordered
# (m=-1, 0, +1) = (y, z, x). The real ell=1 Wigner-D matrix satisfies D1 = C @ R @ C.T,
# which lets us cross-check the spherical path (Wigner-D) against the trivially-correct
# Cartesian path under arbitrary rotations.
_CART_TO_SPHERICAL_L1 = torch.tensor(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64
)


def _make_system(types, positions=None, cell=None, pbc=None):
    n_atoms = len(types)
    if positions is None:
        positions = torch.zeros((n_atoms, 3), dtype=torch.float64)
    if cell is None:
        cell = torch.zeros((3, 3), dtype=torch.float64)
    if pbc is None:
        pbc = torch.tensor([False, False, False])
    return System(
        types=torch.tensor(types, dtype=torch.int32),
        positions=positions,
        cell=cell,
        pbc=pbc,
    )


def _rotation_batch(alphas):
    transformations = []
    for alpha in alphas:
        transformations.append(
            torch.tensor(
                [
                    [np.cos(alpha), -np.sin(alpha), 0.0],
                    [np.sin(alpha), np.cos(alpha), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float64,
            )
        )

    zeros = np.zeros(len(alphas))
    wigner_D_matrices = {
        ell: list(matrix.unbind(0))
        for ell, matrix in compute_wigner_batch(
            1,
            (np.asarray(alphas), zeros, zeros),
            device=torch.device("cpu"),
            dtype=torch.float64,
        ).items()
    }
    return transformations, wigner_D_matrices


def _row_indices(samples, n_systems):
    system_ids = samples.column("system").to(dtype=torch.long)
    return [
        torch.nonzero(system_ids == system_index, as_tuple=False).reshape(-1)
        for system_index in range(n_systems)
    ]


def test_sparse_atomic_basis_rank2_augmentation_with_missing_system_rows():
    """Rank-2 spherical features rotate on each mu axis, and empty system row-groups
    are a no-op.

    The single block carries two component axes (``o3_mu_1``, ``o3_mu_2``) and all of
    its rows belong to system 0; system 1 contributes no rows (the "missing system
    rows"). The test confirms only system 0's rows are transformed -- by
    ``D1 @ v @ D1.T`` on the two mu axes -- that the empty row-group for system 1 leaves
    the block untouched, and that the values actually change (guarding against an
    accidental identity transform).
    """
    systems = [_make_system([1, 1]), _make_system([8, 8])]
    transformations, wigner_D_matrices = _rotation_batch([np.pi / 2, np.pi])

    components = [
        Labels(
            ["o3_mu_1"],
            torch.arange(-1, 2, dtype=torch.int32).reshape(-1, 1),
        ),
        Labels(
            ["o3_mu_2"],
            torch.arange(-1, 2, dtype=torch.int32).reshape(-1, 1),
        ),
    ]
    property_labels = Labels(
        ["n_1", "n_2"],
        torch.tensor([[0, 0]], dtype=torch.int32),
    )
    values = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3, 1)
    tensor = TensorMap(
        Labels(
            ["o3_lambda_1", "o3_lambda_2", "o3_sigma_1", "o3_sigma_2", "atom_type"],
            torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.int32),
        ),
        [
            TensorBlock(
                values=values,
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
                ),
                components=components,
                properties=property_labels,
            )
        ],
    )

    _, augmented_targets, _ = _apply_transformations(
        systems,
        {"target": tensor},
        transformations,
        wigner_D_matrices,
    )
    augmented = augmented_targets["target"]

    expected_values = values.clone()
    rows = _row_indices(tensor.block().samples, len(systems))[0]
    expected_values[rows] = torch.einsum(
        "Aa,iabp,bB->iABp",
        wigner_D_matrices[1][0],
        values[rows],
        wigner_D_matrices[1][0].T,
    )
    expected = TensorMap(
        tensor.keys,
        [
            TensorBlock(
                values=expected_values,
                samples=tensor.block().samples,
                components=tensor.block().components,
                properties=tensor.block().properties,
            )
        ],
    )

    assert augmented.block().samples == expected.block().samples
    assert torch.allclose(augmented.block().values, expected.block().values, atol=1e-12)
    assert not torch.allclose(augmented.block().values, values)


def test_system_positions_and_cell_are_rotated():
    # Non-trivial positions and cell so the rotation is observable; verifies that
    # `_apply_transformations` does not silently leave the System unchanged.
    positions_a = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    positions_b = torch.tensor(
        [[0.5, 0.5, 0.0], [1.0, -1.0, 0.5]],
        dtype=torch.float64,
    )
    cell_a = torch.eye(3, dtype=torch.float64) * 3.0
    cell_b = torch.eye(3, dtype=torch.float64) * 4.0
    pbc = torch.tensor([True, True, True])
    systems = [
        _make_system([1, 1, 1], positions=positions_a, cell=cell_a, pbc=pbc),
        _make_system([8, 8], positions=positions_b, cell=cell_b, pbc=pbc),
    ]
    transformations, wigner_D_matrices = _rotation_batch([np.pi / 3, np.pi / 4])

    new_systems, _, _ = _apply_transformations(
        systems,
        {},
        transformations,
        wigner_D_matrices,
    )

    assert len(new_systems) == 2
    for original, new, R in zip(systems, new_systems, transformations, strict=True):
        assert torch.allclose(new.positions, original.positions @ R.T, atol=1e-12)
        assert torch.allclose(new.cell, original.cell @ R.T, atol=1e-12)
        # types and pbc must pass through unchanged
        assert torch.equal(new.types, original.types)
        assert torch.equal(new.pbc, original.pbc)


def test_random_rotations_are_orthogonal():
    rotations = random_rotations(20, device=torch.device("cpu"), dtype=torch.float64)
    assert len(rotations) == 20
    identity = torch.eye(3, dtype=torch.float64)
    for R in rotations:
        assert R.shape == (3, 3)
        assert torch.allclose(R @ R.T, identity, atol=1e-10)
        assert abs(float(torch.det(R)) - 1.0) < 1e-10


def test_random_rotations_include_inversions():
    # With n=100 the probability that all determinants have the same sign is 2^{-99}.
    rotations = random_rotations(
        100, device=torch.device("cpu"), dtype=torch.float64, include_inversions=True
    )
    dets = torch.tensor([float(torch.det(R)) for R in rotations], dtype=torch.float64)
    assert torch.allclose(dets.abs(), torch.ones(100, dtype=torch.float64), atol=1e-10)
    assert (dets > 0).any() and (dets < 0).any()


def test_apply_transformations_raises_on_length_mismatch():
    systems = [_make_system([1])]
    two_rotations = [torch.eye(3, dtype=torch.float64)] * 2
    with pytest.raises(ValueError, match="one transformation per system"):
        apply_transformations(systems, {}, two_rotations)


def test_apply_transformations_raises_on_non_orthogonal():
    systems = [_make_system([1])]
    not_orthogonal = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    with pytest.raises(ValueError, match="not orthogonal"):
        apply_transformations(systems, {}, [not_orthogonal])


# Rotations exercising the full ZYZ decomposition + real Wigner-D path, including both
# gimbal-lock branches of `_rotations_to_zyz`: a generic beta != 0 rotation, a z-axis
# rotation (beta == 0), a 180-degree rotation about x (beta == pi), and an improper one.
_GENERAL_ROTATIONS = [
    torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64),
    torch.tensor(_axis_angle([-2.0, 1.0, 0.5], 2.4), dtype=torch.float64),
    torch.tensor(_axis_angle([0.0, 0.0, 1.0], 0.9), dtype=torch.float64),  # beta=0
    torch.tensor(_axis_angle([1.0, 0.0, 0.0], np.pi), dtype=torch.float64),  # beta=pi
    # improper rotation (det = -1): a proper rotation composed with inversion
    -torch.tensor(_axis_angle([0.3, -1.0, 2.0], 1.1), dtype=torch.float64),
]


@pytest.mark.parametrize("R", _GENERAL_ROTATIONS)
def test_general_rotation_ell1_wigner_matches_cartesian(R):
    # For ell=1 the real Wigner-D matrix must equal C @ R_proper @ C.T, where R_proper
    # is the proper part of R. This validates the Euler decomposition and complex->real
    # conversion for general (non-degenerate) rotations, independently of the rest of
    # the augmentation machinery.
    proper_R = R if torch.det(R) > 0 else -R
    angles = _rotations_to_zyz([R])
    D = compute_wigner_batch(1, angles, device=torch.device("cpu"), dtype=torch.float64)
    expected = _CART_TO_SPHERICAL_L1 @ proper_R @ _CART_TO_SPHERICAL_L1.T
    assert torch.allclose(D[1][0], expected, atol=1e-12)


@pytest.mark.parametrize("R", _GENERAL_ROTATIONS)
def test_general_rotation_spherical_matches_cartesian_vector(R):
    # End-to-end through the public API: a Cartesian vector target and a spherical ell=1
    # (sigma=1, i.e. a true vector) target encoding the same vectors via C must stay
    # related by C after augmentation. Covers general rotations *and* the inversion
    # parity factor for the improper case.
    systems = [_make_system([1, 8])]
    cartesian_vectors = torch.randn(2, 3, 1, dtype=torch.float64)

    cartesian = TensorMap(
        Labels(["_"], torch.tensor([[0]], dtype=torch.int32)),
        [
            TensorBlock(
                values=cartesian_vectors,
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
                ),
                components=[
                    Labels(["xyz"], torch.arange(3, dtype=torch.int32).reshape(-1, 1))
                ],
                properties=Labels(["p"], torch.tensor([[0]], dtype=torch.int32)),
            )
        ],
    )
    # spherical encoding: w = C @ v along the component axis
    spherical_values = torch.einsum(
        "Aa,iap->iAp", _CART_TO_SPHERICAL_L1, cartesian_vectors
    )
    spherical = TensorMap(
        Labels(["o3_lambda", "o3_sigma"], torch.tensor([[1, 1]], dtype=torch.int32)),
        [
            TensorBlock(
                values=spherical_values,
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
                ),
                components=[
                    Labels(
                        ["o3_mu"], torch.arange(-1, 2, dtype=torch.int32).reshape(-1, 1)
                    )
                ],
                properties=Labels(["p"], torch.tensor([[0]], dtype=torch.int32)),
            )
        ],
    )

    _, out, _ = apply_transformations(
        systems, {"cart": cartesian, "spher": spherical}, [R]
    )

    rotated_cart = out["cart"].block().values
    rotated_spher = out["spher"].block().values
    expected_spher = torch.einsum("Aa,iap->iAp", _CART_TO_SPHERICAL_L1, rotated_cart)
    assert torch.allclose(rotated_spher, expected_spher, atol=1e-12)


def test_scalar_energy_gradients_are_rotated():
    # A scalar (energy-like) block carrying positions and strain gradients across two
    # systems. Positions gradients transform as vectors (R @ g); strain gradients as
    # rank-2 Cartesian tensors (R @ S @ R.T). Verifies gradient support and per-system
    # routing through the parent block's "system" column.
    systems = [
        _make_system([1, 1]),
        _make_system([8, 8, 8]),
    ]
    R0 = torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64)
    R1 = torch.tensor(_axis_angle([0.0, 1.0, 1.0], 1.9), dtype=torch.float64)

    values = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    pos_grad = torch.randn(5, 3, 1, dtype=torch.float64)  # 2 + 3 atoms
    strain_grad = torch.randn(2, 3, 3, 1, dtype=torch.float64)

    block = TensorBlock(
        values=values,
        samples=Labels(["system"], torch.tensor([[0], [1]], dtype=torch.int32)),
        components=[],
        properties=Labels(["energy"], torch.tensor([[0]], dtype=torch.int32)),
    )
    block.add_gradient(
        "positions",
        TensorBlock(
            values=pos_grad,
            samples=Labels(
                ["sample", "atom"],
                torch.tensor(
                    [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], dtype=torch.int32
                ),
            ),
            components=[
                Labels(["xyz"], torch.arange(3, dtype=torch.int32).reshape(-1, 1))
            ],
            properties=Labels(["energy"], torch.tensor([[0]], dtype=torch.int32)),
        ),
    )
    block.add_gradient(
        "strain",
        TensorBlock(
            values=strain_grad,
            samples=Labels(["sample"], torch.tensor([[0], [1]], dtype=torch.int32)),
            components=[
                Labels(["xyz_1"], torch.arange(3, dtype=torch.int32).reshape(-1, 1)),
                Labels(["xyz_2"], torch.arange(3, dtype=torch.int32).reshape(-1, 1)),
            ],
            properties=Labels(["energy"], torch.tensor([[0]], dtype=torch.int32)),
        ),
    )
    tensor = TensorMap(Labels(["_"], torch.tensor([[0]], dtype=torch.int32)), [block])

    _, out, _ = apply_transformations(systems, {"energy": tensor}, [R0, R1])
    out_block = out["energy"].block()

    # scalar values unchanged
    assert torch.allclose(out_block.values, values)

    # positions gradients: rows 0,1 -> R0, rows 2,3,4 -> R1
    expected_pos = pos_grad.clone()
    expected_pos[:2] = torch.einsum("Aa,iap->iAp", R0, pos_grad[:2])
    expected_pos[2:] = torch.einsum("Aa,iap->iAp", R1, pos_grad[2:])
    assert torch.allclose(out_block.gradient("positions").values, expected_pos)

    # strain gradients: row 0 -> R0 S R0.T, row 1 -> R1 S R1.T
    expected_strain = strain_grad.clone()
    expected_strain[0] = torch.einsum("Aa,abp,Bb->ABp", R0, strain_grad[0], R0)
    expected_strain[1] = torch.einsum("Aa,abp,Bb->ABp", R1, strain_grad[1], R1)
    assert torch.allclose(out_block.gradient("strain").values, expected_strain)


def test_positions_gradient_routes_by_parent_system_label():
    """Positions gradients follow the parent block's "system" label, not row position.

    The parent energy rows are given in non-sorted label order ([1, 0]); each gradient
    row must be rotated by the transformation of the system named in *its parent row's*
    "system" label, exactly as the values are routed. A positional routing (pairing
    gradient ``sample == i`` with ``transformations[i]``) would mis-pair the gradient
    rows with the systems whenever the parent rows are not sorted by label.
    """
    # systems are passed sorted by label: systems[0] <-> label 0, systems[1] <-> label 1
    systems = [_make_system([1, 1]), _make_system([8, 8, 8])]
    R0 = torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64)
    R1 = torch.tensor(_axis_angle([0.0, 1.0, 1.0], 1.9), dtype=torch.float64)

    # parent rows in non-sorted order: row 0 -> system label 1, row 1 -> system label 0
    values = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    block = TensorBlock(
        values=values,
        samples=Labels(["system"], torch.tensor([[1], [0]], dtype=torch.int32)),
        components=[],
        properties=Labels(["energy"], torch.tensor([[0]], dtype=torch.int32)),
    )
    # sample 0 -> parent row 0 (label 1, 3 atoms); sample 1 -> parent row 1 (label 0, 2)
    pos_grad = torch.randn(5, 3, 1, dtype=torch.float64)
    block.add_gradient(
        "positions",
        TensorBlock(
            values=pos_grad,
            samples=Labels(
                ["sample", "atom"],
                torch.tensor(
                    [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]], dtype=torch.int32
                ),
            ),
            components=[
                Labels(["xyz"], torch.arange(3, dtype=torch.int32).reshape(-1, 1))
            ],
            properties=Labels(["energy"], torch.tensor([[0]], dtype=torch.int32)),
        ),
    )
    tensor = TensorMap(Labels(["_"], torch.tensor([[0]], dtype=torch.int32)), [block])

    _, out, _ = apply_transformations(systems, {"energy": tensor}, [R0, R1])
    grad = out["energy"].block().gradient("positions").values

    expected = pos_grad.clone()
    # parent row 0 has label 1 -> R1 (gradient rows with sample == 0: indices 0,1,2)
    expected[:3] = torch.einsum("Aa,iap->iAp", R1, pos_grad[:3])
    # parent row 1 has label 0 -> R0 (gradient rows with sample == 1: indices 3,4)
    expected[3:] = torch.einsum("Aa,iap->iAp", R0, pos_grad[3:])
    assert torch.allclose(grad, expected, atol=1e-12)


def test_unsupported_component_axis_raises():
    # A component axis that is neither Cartesian nor spherical must raise rather than be
    # silently passed through unchanged.
    systems = [_make_system([1])]
    tensor = TensorMap(
        Labels(["_"], torch.tensor([[0]], dtype=torch.int32)),
        [
            TensorBlock(
                values=torch.zeros(1, 3, 1, dtype=torch.float64),
                samples=Labels(["system"], torch.tensor([[0]], dtype=torch.int32)),
                components=[
                    Labels(
                        ["direction"], torch.arange(3, dtype=torch.int32).reshape(-1, 1)
                    )
                ],
                properties=Labels(["p"], torch.tensor([[0]], dtype=torch.int32)),
            )
        ],
    )
    with pytest.raises(ValueError, match="neither a Cartesian"):
        apply_transformations(
            systems, {"bad": tensor}, [torch.eye(3, dtype=torch.float64)]
        )


def test_neighbor_list_vectors_are_rotated():
    R = torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64)
    system = _make_system(
        [1, 1],
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    options = NeighborListOptions(cutoff=2.0, full_list=True, strict=False)
    vectors = torch.tensor(
        [[[1.0], [0.0], [0.0]]], dtype=torch.float64
    )  # (1 pair, 3, 1)
    neighbors = TensorBlock(
        values=vectors.clone(),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.int32),
        ),
        components=[Labels(["xyz"], torch.arange(3, dtype=torch.int32).reshape(-1, 1))],
        properties=Labels(["distance"], torch.tensor([[0]], dtype=torch.int32)),
    )
    register_autograd_neighbors(system, neighbors)
    system.add_neighbor_list(options, neighbors)

    new_systems, _, _ = apply_transformations([system], {}, [R])

    new_vectors = new_systems[0].get_neighbor_list(options).values
    expected = (vectors.squeeze(-1) @ R.T).unsqueeze(-1)
    assert torch.allclose(new_vectors, expected, atol=1e-12)


def test_system_custom_data_is_rotated_by_tensor_type():
    """Registered System data is rotated by the same machinery as targets.

    Scalar blocks must pass through unchanged, ``xyz`` vector blocks must rotate by
    ``R``, and a multi-block TensorMap must be handled block-by-block (the previous
    implementation rejected anything other than a single block). This is the path
    exercised when a model carries per-atom geometric quantities (e.g. local frames) as
    System data that has to follow the rotation of the structure.
    """
    R = torch.tensor(_axis_angle([0.3, -0.7, 0.5], 0.9), dtype=torch.float64)
    system = _make_system(
        [1, 8],
        positions=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64),
    )

    # scalar per-atom data spread over two blocks: both must pass through untouched
    scalar = TensorMap(
        keys=Labels(["block"], torch.tensor([[0], [1]], dtype=torch.int32)),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0], [2.0]], dtype=torch.float64),
                samples=Labels.range("atom", 2),
                components=[],
                properties=Labels.range("p", 1),
            ),
            TensorBlock(
                values=torch.tensor([[3.0], [4.0]], dtype=torch.float64),
                samples=Labels.range("atom", 2),
                components=[],
                properties=Labels.range("p", 1),
            ),
        ],
    )
    # xyz vector per-atom data: must rotate by R on the component axis
    vector_values = torch.tensor(
        [[[1.0], [0.0], [0.0]], [[0.0], [2.0], [0.0]]], dtype=torch.float64
    )  # (2 atoms, 3, 1)
    vector = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=vector_values.clone(),
                samples=Labels.range("atom", 2),
                components=[
                    Labels(["xyz"], torch.arange(3, dtype=torch.int32).reshape(-1, 1))
                ],
                properties=Labels.range("p", 1),
            )
        ],
    )
    system.add_data("custom::scalar", scalar)
    system.add_data("custom::vector", vector)

    new_systems, _, _ = apply_transformations([system], {}, [R])
    new_system = new_systems[0]

    new_scalar = new_system.get_data("custom::scalar")
    for block_id in range(len(scalar.keys)):
        assert torch.allclose(
            new_scalar.block_by_id(block_id).values,
            scalar.block_by_id(block_id).values,
        )

    new_vector = new_system.get_data("custom::vector").block().values
    expected_vector = (vector_values.squeeze(-1) @ R.T).unsqueeze(-1)
    assert torch.allclose(new_vector, expected_vector, atol=1e-12)
    # sanity: the rotation actually changed the vector data
    assert not torch.allclose(new_vector, vector_values)


def test_spherical_system_data_is_passed_through_unrotated():
    """Spherical System data is allowed but not rotated (no Wigner-D computed for it).

    Attaching an ``o3_mu`` block to a System via ``add_data`` is permitted, but the
    augmentation does not build Wigner-D matrices for System data. Rather than crash or
    rotate it incorrectly, such data must be passed through unchanged while the geometry
    is still rotated -- so a model relying on it is responsible for its equivariance.
    """
    R = torch.tensor(_axis_angle([0.2, 0.5, -0.8], 1.1), dtype=torch.float64)
    system = _make_system(
        [1, 8],
        positions=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64),
    )
    spherical_values = torch.tensor(
        [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], dtype=torch.float64
    )  # (2 atoms, 2*ell+1 = 3, 1)
    mu = Labels(["o3_mu"], torch.arange(-1, 2, dtype=torch.int32).reshape(-1, 1))
    spherical = TensorMap(
        keys=Labels(
            ["o3_lambda", "o3_sigma"], torch.tensor([[1, 1]], dtype=torch.int32)
        ),
        blocks=[
            TensorBlock(
                values=spherical_values.clone(),
                samples=Labels.range("atom", 2),
                components=[mu],
                properties=Labels.range("p", 1),
            )
        ],
    )
    system.add_data("custom::spherical", spherical)

    # no targets -> no Wigner-D matrices are computed; this would KeyError if the
    # spherical block were (incorrectly) sent through the rotation path
    new_systems, _, _ = apply_transformations([system], {}, [R])

    new_spherical = new_systems[0].get_data("custom::spherical").block().values
    assert torch.allclose(new_spherical, spherical_values)
    # the geometry itself is still rotated
    assert torch.allclose(new_systems[0].positions, system.positions @ R.T, atol=1e-12)


def test_random_rotations_generator_is_reproducible():
    g1 = torch.Generator().manual_seed(12345)
    g2 = torch.Generator().manual_seed(12345)
    a = random_rotations(
        8,
        device=torch.device("cpu"),
        dtype=torch.float64,
        include_inversions=True,
        generator=g1,
    )
    b = random_rotations(
        8,
        device=torch.device("cpu"),
        dtype=torch.float64,
        include_inversions=True,
        generator=g2,
    )
    for Ra, Rb in zip(a, b, strict=True):
        assert torch.equal(Ra, Rb)


def test_apply_transformations_raises_on_dtype_mismatch():
    systems = [_make_system([1, 8])]
    R = torch.eye(
        3, dtype=torch.float32
    )  # mismatched vs float64 target/positions below
    tensor = TensorMap(
        Labels(["_"], torch.tensor([[0]], dtype=torch.int32)),
        [
            TensorBlock(
                values=torch.zeros(2, 3, 1, dtype=torch.float64),
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
                ),
                components=[
                    Labels(["xyz"], torch.arange(3, dtype=torch.int32).reshape(-1, 1))
                ],
                properties=Labels(["p"], torch.tensor([[0]], dtype=torch.int32)),
            )
        ],
    )
    with pytest.raises(ValueError, match="dtype/device"):
        apply_transformations(systems, {"t": tensor}, [R])


def test_arbitrary_system_labels_are_remapped():
    # metatrain keeps each structure's original dataset index in the "system" column,
    # so a 2-system batch can be labelled e.g. [92, 38] rather than [0, 1]. The i-th
    # sorted label maps to system i: 38 -> system 0 (R0), 92 -> system 1 (R1).
    systems = [_make_system([1, 8]), _make_system([1, 8])]
    R0 = torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64)
    R1 = torch.tensor(_axis_angle([0.0, 1.0, 1.0], 1.9), dtype=torch.float64)
    vectors = torch.randn(2, 3, 1, dtype=torch.float64)
    tensor = TensorMap(
        Labels(["_"], torch.tensor([[0]], dtype=torch.int32)),
        [
            TensorBlock(
                values=vectors,
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[92, 0], [38, 0]], dtype=torch.int32),
                ),
                components=[
                    Labels(["xyz"], torch.arange(3, dtype=torch.int32).reshape(-1, 1))
                ],
                properties=Labels(["p"], torch.tensor([[0]], dtype=torch.int32)),
            )
        ],
    )
    _, out, _ = apply_transformations(systems, {"t": tensor}, [R0, R1])
    result = out["t"].block().values
    # row 0 has label 92 (sorted position 1) -> R1; row 1 has label 38 -> R0
    expected = vectors.clone()
    expected[0] = torch.einsum("Aa,ap->Ap", R1, vectors[0])
    expected[1] = torch.einsum("Aa,ap->Ap", R0, vectors[1])
    assert torch.allclose(result, expected, atol=1e-12)


def test_too_many_distinct_system_ids_raises():
    # More distinct "system" labels than systems is genuinely ambiguous and must raise.
    systems = [_make_system([1]), _make_system([8])]
    R = torch.eye(3, dtype=torch.float64)
    tensor = TensorMap(
        Labels(["o3_lambda", "o3_sigma"], torch.tensor([[1, 1]], dtype=torch.int32)),
        [
            TensorBlock(
                values=torch.zeros(3, 3, 1, dtype=torch.float64),
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[10, 0], [20, 0], [30, 0]], dtype=torch.int32),
                ),
                components=[
                    Labels(
                        ["o3_mu"], torch.arange(-1, 2, dtype=torch.int32).reshape(-1, 1)
                    )
                ],
                properties=Labels(["p"], torch.tensor([[0]], dtype=torch.int32)),
            )
        ],
    )
    with pytest.raises(ValueError, match="distinct system indices"):
        apply_transformations(systems, {"t": tensor}, [R, R])
