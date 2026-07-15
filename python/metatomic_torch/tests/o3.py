import re

import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    NeighborListOptions,
    System,
    register_autograd_neighbors,
)
from metatomic.torch.o3 import (
    O3Transformation,
    random_transformations,
    transform_block,
    transform_system,
    transform_tensor,
)
from metatomic.torch.o3._wigner import (
    _complex_to_real_spherical_harmonics_transform,
    _compute_real_wigner_d_matrices,
    build_wigner_D_cache,
)

from ._tests_utils import can_use_mps_backend


ALL_DEVICE_DTYPE = [("cpu", "float64"), ("cpu", "float32")]

if torch.cuda.is_available():
    ALL_DEVICE_DTYPE.append(("cuda", "float64"))
    ALL_DEVICE_DTYPE.append(("cuda", "float32"))

if can_use_mps_backend():
    ALL_DEVICE_DTYPE.append(("mps", "float32"))


def _make_system(
    types,
    positions=None,
    cell=None,
    pbc=None,
    *,
    device="cpu",
    dtype=torch.float64,
):
    n_atoms = len(types)
    if positions is None:
        positions = torch.zeros((n_atoms, 3), dtype=dtype, device=device)
    if cell is None:
        cell = torch.zeros((3, 3), dtype=dtype, device=device)
    if pbc is None:
        pbc = torch.tensor([False, False, False], device=device)
    elif torch.is_tensor(pbc):
        pbc = pbc.to(device=device)
    else:
        pbc = torch.tensor(pbc, device=device)
    return System(
        types=torch.tensor(types, device=device),
        positions=positions,
        cell=cell,
        pbc=pbc,
    )


def _tensor_map(values, samples, components, *, keys=None, properties=None):
    """Single-block TensorMap factory for transformation tests."""
    device = values.device
    if keys is None:
        keys = Labels(["_"], torch.tensor([[0]], device=device))
    if properties is None:
        properties = Labels(["p"], torch.tensor([[0]], device=device))
    return TensorMap(
        keys,
        [TensorBlock(values, samples, components, properties)],
    )


@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_transform_system(device, dtype):
    dtype = getattr(torch, dtype)
    atol = 1e-6 if dtype == torch.float32 else 1e-12

    # create a system with neighbor list and custom data
    system = _make_system(
        [1, 1, 1],
        positions=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
            device=device,
        ),
        cell=torch.eye(3, dtype=dtype, device=device) * 3.0,
        pbc=torch.tensor([True, True, True]),
        device=device,
        dtype=dtype,
    )

    options = NeighborListOptions(cutoff=2.0, full_list=True, strict=False)
    neighbors = TensorBlock(
        values=torch.tensor([[[1.0], [0.0], [0.0]]], dtype=dtype, device=device),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            torch.tensor([[0, 1, 0, 0, 0]], device=device),
        ),
        components=[Labels(["xyz"], torch.arange(3, device=device).reshape(-1, 1))],
        properties=Labels(["distance"], torch.tensor([[0]], device=device)),
    )
    system.add_neighbor_list(options, neighbors)

    # scalar per-atom data spread over two blocks: both must pass through untouched
    samples = Labels(
        ["system", "atom"],
        torch.tensor([[0, 0], [0, 2], [0, 1]], device=device),
    )
    scalar = TensorMap(
        keys=Labels(["k"], torch.tensor([[0], [1]], device=device)),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0], [2.0], [3.0]], dtype=dtype, device=device),
                samples=samples,
                components=[],
                properties=Labels(["p"], torch.tensor([[0]], device=device)),
            ),
            TensorBlock(
                values=torch.tensor([[3.0], [4.0], [5.0]], dtype=dtype, device=device),
                samples=samples,
                components=[],
                properties=Labels(["p"], torch.tensor([[0]], device=device)),
            ),
        ],
    )
    scalar.set_info("unit", "eV")
    scalar.set_info("custom", "preserved by rotation")

    # xyz vector per-atom data: must rotate by R on the component axis
    vector_values = torch.tensor(
        [[[1.0], [0.0], [0.0]], [[0.0], [2.0], [0.0]], [[0.0], [0.0], [3.0]]],
        dtype=dtype,
        device=device,
    )
    vector = TensorMap(
        keys=Labels(["_"], torch.tensor([[0]], device=device)),
        blocks=[
            TensorBlock(
                values=vector_values.clone(),
                samples=samples,
                components=[
                    Labels(["xyz"], torch.arange(3, device=device).reshape(-1, 1))
                ],
                properties=Labels(["p"], torch.tensor([[0]], device=device)),
            )
        ],
    )
    system.add_data("custom::scalar", scalar)
    system.add_data("custom::vector", vector)

    matrix = _zyz_matrix(np.pi / 3, 0.0, 0.0, dtype).to(device=device)
    transformation = O3Transformation(matrix, max_angular_momentum=0)

    rotated = transform_system(system, transformation)

    assert torch.allclose(rotated.positions, system.positions @ matrix.T, atol=atol)
    assert torch.allclose(rotated.cell, system.cell @ matrix.T, atol=atol)
    assert torch.equal(rotated.types, system.types)
    assert torch.equal(rotated.pbc, system.pbc)

    new_neighbors = rotated.get_neighbor_list(options).values
    expected = (
        system.get_neighbor_list(options).values.squeeze(-1) @ matrix.T
    ).unsqueeze(-1)
    assert torch.allclose(new_neighbors, expected, atol=atol)

    new_scalar = rotated.get_data("custom::scalar")
    assert new_scalar.info() == scalar.info()
    for block_id in range(len(scalar.keys)):
        assert torch.allclose(
            new_scalar.block_by_id(block_id).values,
            scalar.block_by_id(block_id).values,
        )

    new_vector = rotated.get_data("custom::vector").block().values
    expected_vector = (vector_values.squeeze(-1) @ matrix.T).unsqueeze(-1)
    assert torch.allclose(new_vector, expected_vector, atol=atol)


def test_transform_system_reregisters_autograd_neighbors():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    system = _make_system([1, 1], positions=positions)
    options = NeighborListOptions(cutoff=2.0, full_list=True, strict=False)
    neighbors = TensorBlock(
        values=torch.tensor([[[1.0], [0.0], [0.0]]], dtype=torch.float64),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            torch.tensor([[0, 1, 0, 0, 0]]),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    register_autograd_neighbors(system, neighbors)
    system.add_neighbor_list(options, neighbors)

    rotation = _zyz_matrix(np.pi / 3, np.pi / 4, 0.0, torch.float64)
    rotated = transform_system(system, O3Transformation(rotation, 0))
    loss = torch.sum(rotated.get_neighbor_list(options).values ** 2)

    gradient = torch.autograd.grad(loss, positions)[0]
    expected = torch.tensor([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(gradient, expected, atol=1e-12)


def test_random_rotations_are_orthogonal():
    transformations = random_transformations(
        20, device=torch.device("cpu"), dtype=torch.float64
    )
    assert len(transformations) == 20
    identity = torch.eye(3, dtype=torch.float64)
    for transformation in transformations:
        assert transformation.matrix.shape == (3, 3)
        assert torch.allclose(
            transformation.matrix @ transformation.matrix.T, identity, atol=1e-10
        )
        assert abs(float(torch.det(transformation.matrix)) - 1.0) < 1e-10


def test_random_rotations_include_inversions():
    # With n=100 the probability that all determinants have the same sign is 2^{-99}.
    transformations = random_transformations(
        100,
        device="cpu",
        dtype=torch.float64,
        include_inversions=True,
    )
    dets = torch.stack([torch.det(T.matrix) for T in transformations])
    assert torch.allclose(dets.abs(), torch.ones(100, dtype=torch.float64), atol=1e-10)
    assert (dets > 0).any() and (dets < 0).any()
    assert all(
        T.is_inverted == bool(det < 0)
        for T, det in zip(transformations, dets, strict=True)
    )


def test_random_transformations_use_trusted_batch_validation(monkeypatch):
    def unexpected_checked_construction(*args, **kwargs):
        raise AssertionError("random_transformations repeated per-matrix validation")

    monkeypatch.setattr(O3Transformation, "__init__", unexpected_checked_construction)
    transformations = random_transformations(
        8,
        max_angular_momentum=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
        include_inversions=True,
        generator=torch.Generator().manual_seed(20260712),
    )
    assert len(transformations) == 8


def test_public_transformation_constructor_still_validates_matrix():
    with pytest.raises(ValueError, match="shape"):
        O3Transformation(torch.eye(2, dtype=torch.float64), 0)

    non_orthogonal = torch.eye(3, dtype=torch.float64)
    non_orthogonal[0, 0] = 2.0
    with pytest.raises(ValueError, match="not orthogonal"):
        O3Transformation(non_orthogonal, 0)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_angular_momentum_limit_validation(value, error):
    with pytest.raises(error, match="max_angular_momentum"):
        O3Transformation(torch.eye(3, dtype=torch.float64), value)

    with pytest.raises(error, match="max_angular_momentum"):
        random_transformations(
            0,
            max_angular_momentum=value,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_random_transformation_count_validation(value, error):
    with pytest.raises(error, match="n must"):
        random_transformations(
            value,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )


@pytest.mark.parametrize("inversion", [1.0, -1.0])
def test_trusted_transformation_matches_checked_constructor(inversion):
    matrix = inversion * _zyz_matrix(np.pi / 3, 0.0, 0.0, torch.float64)
    checked = O3Transformation(matrix, max_angular_momentum=3)
    trusted = O3Transformation._from_validated_matrix(
        matrix,
        max_angular_momentum=3,
        is_inverted=inversion < 0,
    )

    assert trusted.is_inverted == checked.is_inverted
    assert torch.equal(trusted.matrix, checked.matrix)
    for ell in range(4):
        assert torch.allclose(
            trusted.wigner_D_matrix(ell),
            checked.wigner_D_matrix(ell),
            atol=1e-14,
            rtol=1e-14,
        )


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


def _zyz_matrix(alpha, beta, gamma, dtype):
    """Construct ``Rz(alpha) Ry(beta) Rz(gamma)`` without angle recovery."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    return torch.tensor(
        [
            [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
            [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
            [-sb * cg, sb * sg, cb],
        ],
        dtype=dtype,
    )


def _direct_wigner(ell_max, alpha, beta, gamma, dtype):
    transforms = {
        ell: _complex_to_real_spherical_harmonics_transform(ell)
        for ell in range(ell_max + 1)
    }
    return {
        ell: matrix.to(dtype=dtype)
        for ell, matrix in _compute_real_wigner_d_matrices(
            ell_max,
            (alpha, beta, gamma),
            transforms,
        ).items()
    }


# Change of basis from Cartesian (x, y, z) to real ell=1 spherical harmonics, ordered
# (m=-1, 0, +1) = (y, z, x). The real ell=1 Wigner-D matrix satisfies D1 = C @ R @ C.T,
# which lets us cross-check the spherical path (Wigner-D) against the trivially-correct
# Cartesian path under arbitrary rotations.
CARTESIAN_TO_SPHERICAL_L1 = torch.tensor(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64
)

# Rotations exercising the full ZYZ decomposition path: generic, beta=0, beta=pi,
# and improper cases.
TRANSFORMATIONS = [
    pytest.param(
        torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64),
        id="rotation-1",
    ),
    pytest.param(
        torch.tensor(_axis_angle([-2.0, 1.0, 0.5], 2.4), dtype=torch.float64),
        id="rotation-2",
    ),
    pytest.param(
        torch.tensor(_axis_angle([0.0, 0.0, 1.0], 0.9), dtype=torch.float64),
        id="gimbal-lock-beta-0",
    ),
    pytest.param(
        torch.tensor(_axis_angle([1.0, 0.0, 0.0], np.pi), dtype=torch.float64),
        id="gimbal-lock-beta-pi",
    ),
    pytest.param(
        -torch.tensor(_axis_angle([0.3, -1.0, 2.0], 1.1), dtype=torch.float64),
        id="improper-rotation",
    ),
]


@pytest.mark.parametrize("matrix", TRANSFORMATIONS)
def test_general_rotation_L1_wigner_matches_cartesian(matrix):
    """For ell=1, Wigner-D matches the Cartesian basis-change formula."""
    proper = matrix if torch.det(matrix) > 0 else -matrix
    transformation = O3Transformation(matrix, max_angular_momentum=1)

    expected = CARTESIAN_TO_SPHERICAL_L1 @ proper @ CARTESIAN_TO_SPHERICAL_L1.T
    assert torch.allclose(transformation.wigner_D_matrix(1), expected, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("inverted", [False, True])
def test_wigner_angle_recovery_random_and_near_poles(dtype, inverted):
    """Recover generating ZYZ matrices through ell=8 across difficult angles."""
    ell_max = 8
    atol = 2.0e-11 if dtype == torch.float64 else 4.0e-5
    rng = np.random.default_rng(20260711)

    angles = []
    # Haar-distributed generic rotations exercise the non-degenerate branch.
    for _ in range(32):
        alpha = rng.uniform(-np.pi, np.pi)
        beta = np.arccos(rng.uniform(-1.0, 1.0))
        gamma = rng.uniform(-np.pi, np.pi)
        angles.append((alpha, beta, gamma))

    # Exact and near-pole rotations cover both gimbal branches and the stable
    # atan2 recovery immediately outside them. The scales straddle float32 and
    # float64 machine precision.
    distances = [0.0, 1.0e-14, 1.0e-10, 1.0e-8, 1.0e-5, 1.0e-3]
    for distance in distances:
        angles.append((0.37, distance, -1.19))
        angles.append((-0.83, np.pi - distance, 2.11))

    for alpha, beta, gamma in angles:
        proper = _zyz_matrix(alpha, beta, gamma, dtype)
        matrix = -proper if inverted else proper
        recovered = build_wigner_D_cache(
            ell_max, matrix, device=matrix.device, dtype=dtype
        )
        expected = _direct_wigner(ell_max, alpha, beta, gamma, dtype)
        for ell in range(ell_max + 1):
            assert torch.allclose(recovered[ell], expected[ell], rtol=0.0, atol=atol), (
                f"Wigner mismatch for ell={ell}, dtype={dtype}, "
                f"inverted={inverted}, beta={beta}"
            )


@pytest.mark.parametrize("matrix", [torch.eye(3), -torch.eye(3)])
@pytest.mark.parametrize("ell", [0, 1])
@pytest.mark.parametrize("sigma", [1, -1])
def test_transform_spherical_o3_parity(matrix, ell, sigma):
    """The public helper follows the same O(3) irrep convention as TensorMaps."""
    matrix = matrix.to(dtype=torch.float64)
    transformation = O3Transformation(matrix, max_angular_momentum=ell)
    values = torch.arange(
        1,
        2 * (2 * ell + 1) + 1,
        dtype=torch.float64,
    ).reshape(2, 2 * ell + 1)

    parity = 1 if torch.det(matrix) > 0 else sigma * (-1) ** ell
    expected = values * parity

    assert torch.allclose(
        transformation.transform_spherical(values, ell, sigma),
        expected,
        atol=1e-12,
    )


@pytest.mark.parametrize("inversion", [1.0, -1.0])
def test_invalid_spherical_parity_is_rejected(inversion):
    transformation = O3Transformation(
        inversion * torch.eye(3, dtype=torch.float64), max_angular_momentum=0
    )
    with pytest.raises(ValueError, match="sigma must be either -1 or \\+1"):
        transformation.transform_spherical(
            torch.ones((1, 1), dtype=torch.float64), ell=0, sigma=2
        )

    tensor = _tensor_map(
        torch.ones((1, 1, 1), dtype=torch.float64),
        Labels(["system"], torch.tensor([[0]], dtype=torch.int64)),
        [Labels(["o3_mu"], torch.tensor([[0]], dtype=torch.int64))],
        keys=Labels(
            ["o3_lambda", "o3_sigma"],
            torch.tensor([[0, 2]], dtype=torch.int64),
        ),
    )
    with pytest.raises(ValueError, match="sigma must be either -1 or \\+1"):
        transform_tensor(tensor, [_make_system([1])], [transformation])


@pytest.mark.parametrize("ell", [True, 1.0])
def test_public_wigner_methods_reject_non_integer_ell(ell):
    transformation = O3Transformation(torch.eye(3, dtype=torch.float64), 1)

    with pytest.raises(TypeError, match="ell must be a non-negative integer"):
        transformation.wigner_D_matrix(ell)
    with pytest.raises(TypeError, match="ell must be a non-negative integer"):
        transformation.transform_spherical(
            torch.ones((1, 3), dtype=torch.float64), ell=ell, sigma=1
        )


def test_public_wigner_methods_enforce_configured_ell_bound():
    transformation = O3Transformation(torch.eye(3, dtype=torch.float64), 0)

    with pytest.raises(ValueError, match="ell=1 exceeds max_angular_momentum=0"):
        transformation.wigner_D_matrix(1)
    with pytest.raises(ValueError, match="ell=1 exceeds max_angular_momentum=0"):
        transformation.transform_spherical(
            torch.ones((1, 3), dtype=torch.float64), ell=1, sigma=1
        )


def test_wigner_cache_is_built_lazily():
    transformation = O3Transformation(torch.eye(3, dtype=torch.float64), 2)
    assert transformation._wigner_D_cache is None
    transformation.transform_cartesian(torch.ones(4, 3, dtype=torch.float64))
    assert transformation._wigner_D_cache is None

    transformation.wigner_D_matrix(1)
    cache = transformation._wigner_D_cache
    assert cache is not None
    transformation.wigner_D_matrix(2)
    assert transformation._wigner_D_cache is cache


def test_transformation_matrix_is_immutable():
    source = torch.eye(3, dtype=torch.float64)
    transformation = O3Transformation(source, max_angular_momentum=1)
    expected_wigner = transformation.wigner_D_matrix(1).clone()

    source[:] = _zyz_matrix(np.pi / 2, 0.0, 0.0, torch.float64)
    exposed = transformation.matrix
    exposed[:] = -torch.eye(3, dtype=torch.float64)
    transformation.wigner_D_matrix(1).zero_()

    assert torch.equal(transformation.matrix, torch.eye(3, dtype=torch.float64))
    assert not transformation.is_inverted
    assert torch.equal(transformation.wigner_D_matrix(1), expected_wigner)


@pytest.mark.parametrize("matrix", TRANSFORMATIONS)
@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_spherical_rotation_matches_cartesian(matrix, device, dtype):
    dtype = getattr(torch, dtype)
    atol = 1e-6 if dtype == torch.float32 else 1e-12

    matrix = matrix.to(device=device, dtype=dtype)

    system = _make_system([1, 1], device=device, dtype=dtype)

    cartesian_vectors = torch.randn(2, 3, 1, dtype=dtype, device=device)

    samples = Labels(["system", "atom"], torch.tensor([[0, 0], [0, 1]], device=device))
    cartesian = _tensor_map(
        cartesian_vectors,
        samples,
        [Labels(["xyz"], torch.arange(3, device=device).reshape(-1, 1))],
    )
    # spherical encoding: w = C @ v along the component axis
    cart_to_sph = CARTESIAN_TO_SPHERICAL_L1.to(device=device, dtype=dtype)
    spherical_values = torch.einsum(
        "Aa,iap->iAp", cart_to_sph, cartesian_vectors
    ).requires_grad_()
    spherical = _tensor_map(
        spherical_values,
        samples,
        [Labels(["o3_mu"], torch.arange(-1, 2, device=device).reshape(-1, 1))],
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor([[1, 1]], device=device)),
    )

    transformation = O3Transformation(matrix, max_angular_momentum=1)

    cartesian_transformed = transform_tensor(cartesian, [system], [transformation])
    spherical_transformed = transform_tensor(spherical, [system], [transformation])

    expected_spherical = torch.einsum(
        "Aa,iap->iAp", cart_to_sph, cartesian_transformed.block().values
    )

    assert torch.allclose(
        spherical_transformed.block().values, expected_spherical, atol=atol
    )
    gradient = torch.autograd.grad(
        spherical_transformed.block().values.square().sum(), spherical_values
    )[0]
    assert torch.allclose(gradient, 2.0 * spherical_values, atol=atol)


def test_gradients_are_rotated():
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
        samples=Labels(["system"], torch.tensor([[0], [1]])),
        components=[],
        properties=Labels(["energy"], torch.tensor([[0]])),
    )
    block.add_gradient(
        "positions",
        TensorBlock(
            values=pos_grad,
            samples=Labels(
                ["sample", "atom"],
                # non-sorted samples values
                torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1], [1, 2]]),
            ),
            components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
            properties=Labels(["energy"], torch.tensor([[0]])),
        ),
    )
    block.add_gradient(
        "strain",
        TensorBlock(
            values=strain_grad,
            samples=Labels(["sample"], torch.tensor([[0], [1]])),
            components=[
                Labels(["xyz_1"], torch.arange(3).reshape(-1, 1)),
                Labels(["xyz_2"], torch.arange(3).reshape(-1, 1)),
            ],
            properties=Labels(["energy"], torch.tensor([[0]])),
        ),
    )
    tensor = TensorMap(Labels(["_"], torch.tensor([[0]])), [block])

    transformed = transform_tensor(
        tensor,
        systems,
        [
            O3Transformation(R0, max_angular_momentum=1),
            O3Transformation(R1, max_angular_momentum=1),
        ],
    )

    # values should not change
    assert torch.allclose(transformed.block().values, values)

    # positions gradients: rows 0,2 -> R0, rows 1,3,4 -> R1
    expected_pos = pos_grad.clone()
    expected_pos[0] = R0 @ pos_grad[0]
    expected_pos[1] = R1 @ pos_grad[1]
    expected_pos[2] = R0 @ pos_grad[2]
    expected_pos[3] = R1 @ pos_grad[3]
    expected_pos[4] = R1 @ pos_grad[4]

    assert torch.allclose(
        transformed.block().gradient("positions").values, expected_pos
    )

    # strain gradients: row 0 -> R0 S R0.T, row 1 -> R1 S R1.T
    expected_strain = strain_grad.clone()
    expected_strain[0] = torch.einsum("Aa,abp,Bb->ABp", R0, strain_grad[0], R0)
    expected_strain[1] = torch.einsum("Aa,abp,Bb->ABp", R1, strain_grad[1], R1)
    assert torch.allclose(
        transformed.block().gradient("strain").values, expected_strain
    )


def test_unsupported_component_axis_raises():
    """Unknown component-axis names fail loudly."""
    systems = [_make_system([1])]
    tensor = _tensor_map(
        torch.zeros(1, 3, 1, dtype=torch.float64),
        Labels(["system"], torch.tensor([[0]])),
        [Labels(["direction"], torch.arange(3).reshape(-1, 1))],
    )

    message = (
        "Found a component axis 'direction', which is neither a Cartesian "
        "('xyz'/'xyz_1'/'xyz_2'/...) nor spherical ('o3_mu'/'o3_mu_1'/...) axis; "
        "it can not be transformed."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        transform_tensor(
            tensor, systems, [O3Transformation(torch.eye(3, dtype=torch.float64), 1)]
        )


_EMPTY_SCHEMA_CASES = [
    pytest.param("o3_mu", [0], (0, 2), "sigma must", id="parity"),
    pytest.param("direction", range(3), None, "component axis 'direction'", id="axis"),
    pytest.param(
        "xyz", range(2), None, "axis 'xyz' must contain 3", id="cartesian-size"
    ),
    pytest.param("o3_mu", [-1, 0], (1, 1), "ell=1 must contain 3", id="spherical-size"),
    pytest.param(
        "xyz", [2, 0, 1], None, "axis 'xyz' must use labels", id="cartesian-order"
    ),
    pytest.param(
        "o3_mu", [1, -1, 0], (1, 1), "ell=1 must use labels", id="spherical-order"
    ),
]


@pytest.mark.parametrize("location", ["values", "gradient"])
@pytest.mark.parametrize(
    ("component_name", "component_values", "irrep", "message"),
    _EMPTY_SCHEMA_CASES,
)
def test_empty_blocks_still_validate_component_schema(
    location, component_name, component_values, irrep, message
):
    """Selected-atom outputs must not bypass representation validation."""
    components = [
        Labels(
            [component_name],
            torch.as_tensor(component_values, dtype=torch.int64).reshape(-1, 1),
        )
    ]
    if irrep is None:
        keys = Labels(["_"], torch.tensor([[0]], dtype=torch.int64))
    else:
        keys = Labels(
            ["o3_lambda", "o3_sigma"], torch.tensor([irrep], dtype=torch.int64)
        )

    component_size = len(components[0])
    if location == "values":
        tensor = _tensor_map(
            torch.empty((0, component_size, 1), dtype=torch.float64),
            Labels(["system"], torch.empty((0, 1), dtype=torch.int64)),
            components,
            keys=keys,
        )
    else:
        block = TensorBlock(
            values=torch.ones((1, 1), dtype=torch.float64),
            samples=Labels(["system"], torch.tensor([[0]], dtype=torch.int64)),
            components=[],
            properties=Labels(["p"], torch.tensor([[0]], dtype=torch.int64)),
        )
        block.add_gradient(
            "positions",
            TensorBlock(
                values=torch.empty((0, component_size, 1), dtype=torch.float64),
                samples=Labels(
                    ["sample", "atom"], torch.empty((0, 2), dtype=torch.int64)
                ),
                components=components,
                properties=block.properties,
            ),
        )
        tensor = TensorMap(keys, [block])

    with pytest.raises(ValueError, match=re.escape(message)):
        transform_tensor(
            tensor,
            [_make_system([1])],
            [O3Transformation(torch.eye(3, dtype=torch.float64), 0)],
        )


def test_empty_blocks_enforce_component_axis_limit():
    components = [
        Labels([f"xyz{suffix}"], torch.arange(3).reshape(-1, 1))
        for suffix in [""] + [f"_{index}" for index in range(1, 10)]
    ]
    components.append(Labels(["o3_mu"], torch.tensor([[0]])))
    tensor = _tensor_map(
        torch.empty([0] + [3] * 10 + [1, 1], dtype=torch.float64),
        Labels(["system"], torch.empty((0, 1), dtype=torch.int64)),
        components,
        keys=Labels(
            ["o3_lambda", "o3_sigma"], torch.tensor([[0, 1]], dtype=torch.int64)
        ),
    )

    with pytest.raises(
        ValueError,
        match="can not transform a tensor with 11 component axes; at most 10",
    ):
        transform_tensor(
            tensor,
            [_make_system([1])],
            [O3Transformation(torch.eye(3, dtype=torch.float64), 0)],
        )


def _system_ids_test_case(sample_system_ids=(92, -7)):
    systems = [_make_system([1, 8]), _make_system([1, 8])]

    R0 = O3Transformation(torch.eye(3, dtype=torch.float64), 1)
    R1 = O3Transformation(_zyz_matrix(np.pi / 2, 0, 0, torch.float64).round(), 1)

    vector_values = torch.tensor(
        [[[1.0], [0.0], [0.0]], [[0.0], [2.0], [0.0]]],
        dtype=torch.float64,
    )[: len(sample_system_ids)]
    tensor = _tensor_map(
        vector_values.clone(),
        Labels(
            ["system", "atom"],
            torch.tensor(
                [[system_id, atom] for atom, system_id in enumerate(sample_system_ids)]
            ),
        ),
        [Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
    )
    return tensor, systems, [R0, R1], vector_values


def _transform_system_ids_case(api, tensor, systems, transformations, system_ids):
    if api == "tensor":
        return transform_tensor(
            tensor,
            systems,
            transformations,
            system_ids=system_ids,
        ).block()
    return transform_block(
        tensor.keys[0],
        tensor.block(),
        systems,
        transformations,
        system_ids=system_ids,
    )


_CARDINALITY_ERROR = "exactly one entry per system"
_DIMENSION_ERROR = "system_ids must be one-dimensional"
_INTEGER_ERROR = "system_ids must contain integers"


@pytest.mark.parametrize("api", ["tensor", "block"])
@pytest.mark.parametrize(
    "system_ids",
    [
        pytest.param([92, -7], id="python-list"),
        pytest.param(torch.tensor([92, -7]), id="tensor"),
    ],
)
def test_system_ids_explicit_match(api, system_ids):
    """Arbitrary, unsorted IDs preserve their positional system mapping."""
    tensor, systems, transformations, vector_values = _system_ids_test_case()

    result = _transform_system_ids_case(
        api, tensor, systems, transformations, system_ids
    ).values
    # row 0 (system 92) rotated by R0 (identity)
    assert torch.equal(result[0], vector_values[0])
    # row 1 (system -7) rotated by R1 (z-90)
    expected = torch.tensor([[-2.0], [0.0], [0.0]], dtype=torch.float64)
    assert torch.equal(result[1], expected)


@pytest.mark.parametrize("api", ["tensor", "block"])
def test_system_ids_can_be_absent_from_a_block(api):
    """A system may legitimately contribute no samples to an individual block."""
    tensor, systems, transformations, vector_values = _system_ids_test_case((92,))
    result = _transform_system_ids_case(
        api, tensor, systems, transformations, [92, 2**40]
    ).values
    assert torch.equal(result, vector_values)


@pytest.mark.parametrize("api", ["tensor", "block"])
@pytest.mark.parametrize(
    "system_ids,message",
    [
        pytest.param([92], _CARDINALITY_ERROR, id="too-few"),
        pytest.param([92, 92], "one distinct entry per system", id="duplicate"),
        pytest.param(torch.tensor([[92, -7]]), _DIMENSION_ERROR, id="row-tensor"),
        pytest.param(torch.tensor([92.0, -7.0]), _INTEGER_ERROR, id="float-tensor"),
        pytest.param([92.0, -7.0], _INTEGER_ERROR, id="float-list"),
        pytest.param([True, False], _INTEGER_ERROR, id="boolean-list"),
    ],
)
def test_system_ids_invalid_raises(api, system_ids, message):
    tensor, systems, transformations, _ = _system_ids_test_case()
    with pytest.raises(ValueError, match=message):
        _transform_system_ids_case(api, tensor, systems, transformations, system_ids)


@pytest.mark.parametrize("api", ["tensor", "block"])
def test_system_and_transformation_count_must_match(api):
    tensor, systems, transformations, _ = _system_ids_test_case()
    with pytest.raises(ValueError, match="Expected one transformation per system"):
        _transform_system_ids_case(
            api,
            tensor,
            systems,
            transformations[:1],
            [92, -7],
        )


@pytest.mark.parametrize("api", ["tensor", "block"])
def test_every_transformation_must_match_values(api):
    """A mismatch after the first transformation is rejected before routing."""
    tensor, systems, transformations, _ = _system_ids_test_case()
    transformations[1] = O3Transformation(torch.eye(3, dtype=torch.float32), 1)

    with pytest.raises(ValueError, match="Transformation at index 1 has dtype/device"):
        _transform_system_ids_case(api, tensor, systems, transformations, [92, -7])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("api", ["tensor", "block"])
def test_cpu_system_ids_reject_for_cuda_values(api):
    tensor, systems, transformations, _ = _system_ids_test_case()
    device = torch.device("cuda")
    tensor = tensor.to(device=device)
    systems = [system.to(device=device) for system in systems]
    transformations = [
        O3Transformation(transformation.matrix.to(device=device), 1)
        for transformation in transformations
    ]

    with pytest.raises(
        ValueError,
        match="system_ids are on device cpu, but the values to transform are on device",
    ):
        _transform_system_ids_case(
            api,
            tensor,
            systems,
            transformations,
            torch.tensor([92, -7]),
        )


@pytest.mark.parametrize(
    "kwargs,missing_ids",
    [
        pytest.param({"system_ids": [99, 100]}, [99, 100], id="explicit"),
        pytest.param({}, [0, 1], id="default"),
    ],
)
def test_system_ids_uncovered_samples_raises(kwargs, missing_ids):
    """All distinct system indices in samples must appear in ``system_ids``."""
    systems = [_make_system([1, 8]), _make_system([1, 8])]
    R = O3Transformation(torch.eye(3, dtype=torch.float64), 1)

    tensor = _tensor_map(
        torch.zeros(2, 3, 1, dtype=torch.float64),
        Labels(["system", "atom"], torch.tensor([[92, 0], [38, 0]])),
        [Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
    )

    message = (
        "Block samples contain system labels [38, 92] that are not in "
        f"system_ids={missing_ids}. Every sample must be assigned to a system "
        "in the transformation."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        transform_tensor(tensor, systems, [R, R], **kwargs)


def test_system_ids_single_system_ignores_label():
    """Single-system blocks return all rows regardless of the ``"system"`` label."""
    systems = [_make_system([1])]
    values = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float64)

    tensor = _tensor_map(
        values.clone(),
        Labels(["system", "atom"], torch.tensor([[4, 0]])),
        [Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
    )

    transformation = O3Transformation(torch.eye(3, dtype=torch.float64), 1)
    transformed = transform_tensor(tensor, systems, [transformation])
    assert torch.allclose(transformed.block().values, values)


@pytest.mark.parametrize("inverted", [False, True])
@pytest.mark.parametrize("sigma_2", [1, -1])
def test_rank2_transform_with_missing_system_rows(inverted, sigma_2):
    """Rank-2 blocks combine per-axis parity even with missing system rows."""
    systems = [_make_system([1, 1]), _make_system([8, 8])]
    matrix = _zyz_matrix(np.pi / 2, 0.0, 0.0, torch.float64)
    if inverted:
        matrix = -matrix
    T0 = O3Transformation(
        matrix,
        max_angular_momentum=1,
    )
    T1 = O3Transformation(
        _zyz_matrix(np.pi, 0.0, 0.0, torch.float64), max_angular_momentum=1
    )

    components = [
        Labels(["o3_mu_1"], torch.arange(-1, 2, dtype=torch.int32).reshape(-1, 1)),
        Labels(["o3_mu_2"], torch.arange(-1, 2, dtype=torch.int32).reshape(-1, 1)),
    ]
    property_labels = Labels(["n_1", "n_2"], torch.tensor([[0, 0]], dtype=torch.int32))
    values = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3, 1)
    tensor = TensorMap(
        Labels(
            ["o3_lambda_1", "o3_lambda_2", "o3_sigma_1", "o3_sigma_2", "atom_type"],
            torch.tensor([[1, 1, 1, sigma_2, 1]], dtype=torch.int32),
        ),
        [
            TensorBlock(
                values=values,
                # both rows belong to system 0; system 1 has no rows in this block
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
                ),
                components=components,
                properties=property_labels,
            )
        ],
    )

    transformed = transform_tensor(tensor, systems, [T0, T1])

    D0 = T0.wigner_D_matrix(1)
    expected_values = torch.einsum("Aa,iabp,bB->iABp", D0, values, D0.T)
    if inverted:
        # The first ell=1, sigma=+1 axis contributes -1; the second contributes
        # -sigma_2. Their product is therefore sigma_2.
        expected_values = sigma_2 * expected_values

    assert transformed.block().samples == tensor.block().samples
    assert torch.allclose(transformed.block().values, expected_values, atol=1e-12)
    assert not torch.allclose(transformed.block().values, values)


def test_transform_supports_all_recognized_component_suffixes():
    """The public ``o3_mu`` naming convention recognizes ten component axes."""
    suffixes = [""] + [f"_{index}" for index in range(1, 10)]
    components = [
        Labels([f"o3_mu{suffix}"], torch.tensor([[0]], dtype=torch.int32))
        for suffix in suffixes
    ]
    key_names = [f"o3_lambda{suffix}" for suffix in suffixes] + [
        f"o3_sigma{suffix}" for suffix in suffixes
    ]
    tensor = TensorMap(
        Labels(key_names, torch.tensor([[0] * 10 + [1] * 10], dtype=torch.int32)),
        [
            TensorBlock(
                values=torch.ones([1] + [1] * 10 + [1], dtype=torch.float64),
                samples=Labels(["system"], torch.tensor([[0]], dtype=torch.int32)),
                components=components,
                properties=Labels(["p"], torch.tensor([[0]], dtype=torch.int32)),
            )
        ],
    )

    transformed = transform_tensor(
        tensor,
        [_make_system([1])],
        [O3Transformation(torch.eye(3, dtype=torch.float64), 0)],
    )

    assert torch.equal(transformed.block().values, tensor.block().values)
