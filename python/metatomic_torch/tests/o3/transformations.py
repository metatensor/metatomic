import io
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
from metatomic.torch.o3 import O3Transformations, random_transformations

# The complex-to-real spherical harmonics conversion is defined only here for now.
from metatomic.torch.o3._wigner import _complex_to_real_spherical_harmonics_transform

from .._tests_utils import can_use_mps_backend


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


def _rotation_90_degrees_around_z():
    """Return the exact matrix for a 90-degree rotation around z."""
    return torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def _single_block_tensor_map(
    *,
    values,
    samples,
    components,
    keys=None,
    properties=None,
):
    """Return a TensorMap containing one TensorBlock."""
    if keys is None:
        keys = Labels(["_"], torch.tensor([[0]], device=values.device))
    if properties is None:
        properties = Labels(["p"], torch.tensor([[0]], device=values.device))

    return TensorMap(
        keys,
        [
            TensorBlock(
                values=values,
                samples=samples,
                components=components,
                properties=properties,
            )
        ],
    )


def test_gradient_components_require_wigner_ranks():
    """Angular momenta appearing only in gradient components still require the
    matching Wigner-D matrices, while purely Cartesian data needs none."""
    properties = Labels("property", torch.tensor([[0]]))
    block = TensorBlock(
        values=torch.ones((1, 3, 1), dtype=torch.float64),
        samples=Labels("system", torch.tensor([[0]])),
        components=[Labels("o3_mu", torch.arange(-1, 2).reshape(-1, 1))],
        properties=properties,
    )
    block.add_gradient(
        "parameter",
        TensorBlock(
            values=torch.ones((1, 7, 3, 1), dtype=torch.float64),
            samples=Labels(
                ["sample", "parameter"],
                torch.tensor([[0, 0]]),
            ),
            components=[
                Labels("o3_mu_1", torch.arange(-3, 4).reshape(-1, 1)),
                Labels("o3_mu", torch.arange(-1, 2).reshape(-1, 1)),
            ],
            properties=properties,
        ),
    )
    spherical = TensorMap(
        Labels(
            ["o3_lambda", "o3_sigma", "o3_lambda_1", "o3_sigma_1"],
            torch.tensor([[1, 1, 3, -1]]),
        ),
        [block],
    )

    # the values only reach ell=1, but the gradient carries an ell=3 axis
    rotation = O3Transformations(
        _rotation_90_degrees_around_z().unsqueeze(0),
        max_angular_momentum=1,
    )
    message = re.escape("ell=3 exceeds max_angular_momentum=1.")
    with pytest.raises(ValueError, match=f"^{message}$"):
        rotation.transform_tensormap(spherical)

    # Cartesian data transforms without any Wigner-D matrices
    cartesian = _single_block_tensor_map(
        values=torch.ones((1, 3, 1), dtype=torch.float64),
        samples=Labels("system", torch.tensor([[0]])),
        components=[Labels("xyz", torch.arange(3).reshape(-1, 1))],
    )
    no_wigner = O3Transformations(
        _rotation_90_degrees_around_z().unsqueeze(0),
        max_angular_momentum=0,
    )
    transformed = no_wigner.transform_tensormap(cartesian)
    assert torch.allclose(
        transformed.block().values,
        torch.tensor([[[-1.0], [1.0], [1.0]]], dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_transform_system(device, dtype):
    """Geometry, neighbor vectors, and custom TensorMaps receive one transformation."""
    dtype = getattr(torch, dtype)
    atol = 1e-6 if dtype == torch.float32 else 1e-12

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

    samples = Labels(
        ["system", "atom"],
        torch.tensor([[0, 0], [0, 2], [0, 1]], device=device),
    )
    scalar = TensorMap(
        keys=Labels(["k"], torch.tensor([[0]], device=device)),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0], [2.0], [3.0]], dtype=dtype, device=device),
                samples=samples,
                components=[],
                properties=Labels(["p"], torch.tensor([[0]], device=device)),
            ),
        ],
    )
    scalar.set_info("unit", "eV")

    vector_values = torch.tensor(
        [[[1.0], [0.0], [0.0]], [[0.0], [2.0], [0.0]], [[0.0], [0.0], [3.0]]],
        dtype=dtype,
        device=device,
    )
    vector = _single_block_tensor_map(
        values=vector_values.clone(),
        samples=samples,
        components=[Labels(["xyz"], torch.arange(3, device=device).reshape(-1, 1))],
    )
    system.add_data("custom::scalar", scalar)
    system.add_data("custom::vector", vector)

    matrix = torch.tensor(
        [
            [np.cos(np.pi / 3), -np.sin(np.pi / 3), 0.0],
            [np.sin(np.pi / 3), np.cos(np.pi / 3), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    transformations = O3Transformations(matrix.unsqueeze(0), max_angular_momentum=0)

    rotated = transformations.transform_systems([system])[0]

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
    assert torch.allclose(new_scalar.block().values, scalar.block().values)

    new_vector = rotated.get_data("custom::vector").block().values
    expected_vector = (vector_values.squeeze(-1) @ matrix.T).unsqueeze(-1)
    assert torch.allclose(new_vector, expected_vector, atol=atol)


def test_transform_system_preserves_neighbor_gradients():
    """Transformed registered neighbor lists remain differentiable from positions."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    system = _make_system([1, 1], positions=positions)

    options = NeighborListOptions(
        cutoff=2.0,
        full_list=True,
        strict=False,
    )
    neighbors = TensorBlock(
        values=torch.tensor(
            [[[1.0], [0.0], [0.0]]],
            dtype=torch.float64,
        ),
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

    rotation = _rotation_90_degrees_around_z()
    transformations = O3Transformations(rotation.unsqueeze(0), max_angular_momentum=0)
    transformed = transformations.transform_systems([system])[0]

    loss = torch.sum(transformed.get_neighbor_list(options).values ** 2)
    gradient = torch.autograd.grad(loss, positions)[0]

    expected = torch.tensor(
        [
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(gradient, expected, atol=1e-12)


def test_transformation_validation():
    """Realistic construction mistakes fail with a clear error."""
    # negative counts and angular momentum limits
    message = "max_angular_momentum must be non-negative, got -1"
    with pytest.raises(ValueError, match=f"^{message}$"):
        O3Transformations(torch.eye(3, dtype=torch.float64).unsqueeze(0), -1)
    message = "n must be positive, got 0"
    with pytest.raises(ValueError, match=f"^{message}$"):
        random_transformations(
            0, max_angular_momentum=-1, device=torch.device("cpu"), dtype=torch.float64
        )
    message = "n must be positive, got -1"
    with pytest.raises(ValueError, match=f"^{message}$"):
        random_transformations(-1, device=torch.device("cpu"), dtype=torch.float64)
    message = "n must be positive, got 0"
    with pytest.raises(ValueError, match=f"^{message}$"):
        random_transformations(
            0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

    # models only declare float32/float64 capabilities
    message = "dtype must be torch.float32 or torch.float64, got torch.float16."
    with pytest.raises(ValueError, match=f"^{message}$"):
        random_transformations(1, device=torch.device("cpu"), dtype=torch.float16)

    # matrices must be (N, 3, 3) and orthogonal
    message = re.escape(
        "O3Transformations `matrices` has shape (1, 2, 2); expected (N, 3, 3)"
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        O3Transformations(
            torch.eye(2, dtype=torch.float64).unsqueeze(0), max_angular_momentum=0
        )
    matrix = torch.eye(3, dtype=torch.float64)
    matrix[0, 0] = 2.0
    message = "O3Transformations `matrices` must be orthogonal"
    with pytest.raises(ValueError, match=f"^{message}$"):
        O3Transformations(matrix.unsqueeze(0), max_angular_momentum=0)

    # Wigner-D requests need a valid ell
    transformations = O3Transformations(
        torch.eye(3, dtype=torch.float64).unsqueeze(0), 1
    )
    message = "ell must be non-negative, got -1"
    with pytest.raises(ValueError, match=f"^{message}$"):
        transformations.wigner_D_matrices(-1)

    # systems must match the transformations dtype/device
    system = _make_system([1], dtype=torch.float64)
    transformations = O3Transformations(
        torch.eye(3, dtype=torch.float32).unsqueeze(0), 0
    )
    message = "system and transformation matrices must have the same dtype and device"
    with pytest.raises(ValueError, match=f"^{message}$"):
        transformations.transform_systems([system])


def test_random_rotations_are_orthogonal():
    """Sampling without inversions yields proper rotations."""
    atol = 1e-10

    transformations = random_transformations(
        20,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert transformations.matrices.shape == (20, 3, 3)
    identity = torch.eye(3, dtype=torch.float64)

    for matrix in transformations.matrices:
        assert matrix.shape == (3, 3)
        assert torch.allclose(
            matrix @ matrix.T,
            identity,
            atol=atol,
        )
        assert abs(float(torch.det(matrix)) - 1.0) < atol


def test_random_transformations_add_inversions():
    """Sampling with inversions covers both O(3) cosets and sets improper."""
    transformations = random_transformations(
        20,
        device="cpu",
        dtype=torch.float64,
        add_inversions=True,
        generator=torch.Generator().manual_seed(20260718),
    )

    determinants = torch.stack([torch.det(m) for m in transformations.matrices])
    assert (determinants > 0).any() and (determinants < 0).any()
    assert torch.equal(transformations.improper_mask, determinants < 0)


def test_o3_parity_factor():
    """Improper transformations scale spherical values by ``sigma * (-1)**ell``."""
    transformations = O3Transformations(
        (-torch.eye(3, dtype=torch.float64)).unsqueeze(0),
        max_angular_momentum=1,
    )

    for ell in (0, 1):
        values = torch.arange(
            1,
            2 * (2 * ell + 1) + 1,
            dtype=torch.float64,
        ).reshape(2, 2 * ell + 1)
        for sigma in (1, -1):
            expected = values * (sigma * (-1) ** ell)
            assert torch.allclose(
                transformations.transform_spherical(values, ell, sigma),
                expected,
                rtol=0.0,
                atol=1e-12,
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


def test_complex_to_real_spherical_harmonics_transform_is_unitary():
    """The complex-to-real basis transformations are unitary."""
    for ell in range(9):
        transform = _complex_to_real_spherical_harmonics_transform(ell)
        size = 2 * ell + 1

        np.testing.assert_allclose(
            transform @ transform.conj().T,
            np.eye(size),
            rtol=0.0,
            atol=1e-15,
            err_msg=f"ell={ell}",
        )


def test_wigner_D_depends_only_on_proper_part():
    """The inversion enters as a separate parity factor, not in the Wigner-D."""
    matrix = torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64)
    proper = O3Transformations(matrix.unsqueeze(0), max_angular_momentum=3)
    improper = O3Transformations((-matrix).unsqueeze(0), max_angular_momentum=3)

    for ell in range(4):
        assert torch.equal(
            proper.wigner_D_matrices(ell), improper.wigner_D_matrices(ell)
        )


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
def test_L1_wigner_matches_cartesian(matrix):
    """For ell=1, Wigner-D is the Cartesian rotation in the (y, z, x) basis."""
    proper = matrix if torch.det(matrix) > 0 else -matrix
    transformations = O3Transformations(
        torch.stack([matrix, matrix]),
        max_angular_momentum=1,
    )

    C = CARTESIAN_TO_SPHERICAL_L1
    expected = C @ proper @ C.T
    assert torch.allclose(
        transformations.wigner_D_matrices(1),
        torch.stack([expected, expected]),
        atol=1e-12,
    )

    # the spherical action of an ell=1, sigma=+1 vector equals the Cartesian
    # action, inversion included
    vectors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.2, -1.3, 0.7], [-0.5, 0.1, 2.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        transformations.transform_spherical(vectors @ C.T, ell=1, sigma=1),
        transformations.transform_cartesian(vectors) @ C.T,
        rtol=0.0,
        atol=1e-12,
    )


def _real_solid_harmonics_l2(v):
    """Real ell=2 solid harmonics of a Cartesian vector, ordered m = -2..2, in
    the same real-spherical-harmonics convention as the ell=1 (y, z, x) map."""
    x, y, z = v
    r2 = x * x + y * y + z * z
    s3 = np.sqrt(3.0)
    return torch.stack(
        [
            s3 * x * y,
            s3 * y * z,
            0.5 * (3.0 * z * z - r2),
            s3 * x * z,
            0.5 * s3 * (x * x - y * y),
        ]
    )


def test_l2_wigner_D_equivariance():
    """D2 @ Y2(v) == Y2(R @ v): independent reference above ell=1."""
    matrices = torch.stack(
        [
            torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64),
            torch.tensor(_axis_angle([-2.0, 1.0, 0.5], 2.4), dtype=torch.float64),
        ]
    )
    transformations = O3Transformations(matrices, max_angular_momentum=2)
    D2 = transformations.wigner_D_matrices(2)

    vectors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.2, -1.3, 0.7], [-0.5, 0.1, 2.0]],
        dtype=torch.float64,
    )
    for batch, matrix in enumerate(matrices):
        for v in vectors:
            torch.testing.assert_close(
                D2[batch] @ _real_solid_harmonics_l2(v),
                _real_solid_harmonics_l2(matrix @ v),
                rtol=0.0,
                atol=1e-12,
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_wigner_D_roundoff_at_euler_poles(dtype):
    """One-ULP matrix roundoff must preserve rotations at beta=0 and beta=pi."""
    alpha = 0.73
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    basis = CARTESIAN_TO_SPHERICAL_L1.to(dtype=dtype)
    atol = 1e-6 if dtype == torch.float32 else 1e-14

    for cos_beta in (1.0, -1.0):
        proper = torch.tensor(
            [
                [cos_beta * cos_alpha, -sin_alpha, 0.0],
                [cos_beta * sin_alpha, cos_alpha, 0.0],
                [0.0, 0.0, cos_beta],
            ],
            dtype=dtype,
        )

        matrix = proper.clone()
        matrix[2, 2] = torch.nextafter(matrix[2, 2], torch.zeros_like(matrix[2, 2]))

        transformations = O3Transformations(
            torch.stack([matrix, matrix]),
            max_angular_momentum=1,
        )
        expected = basis @ proper @ basis.T
        actual = transformations.wigner_D_matrices(1)

        for batch in range(2):
            torch.testing.assert_close(actual[batch], expected, rtol=0.0, atol=atol)


def test_gradient_rows_follow_parent_system():
    """Gradient rows use their parent sample's transformations; inputs unchanged."""
    R_92 = torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64)
    R_38 = torch.tensor(_axis_angle([0.0, 1.0, 1.0], 1.9), dtype=torch.float64)

    values = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    pos_grad = torch.randn(
        5,
        3,
        1,
        dtype=torch.float64,
        requires_grad=True,
    )  # 2 + 3 atoms
    strain_grad = torch.randn(2, 3, 3, 1, dtype=torch.float64)

    block = TensorBlock(
        values=values,
        samples=Labels(["system"], torch.tensor([[92], [38]])),
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
    values_before = values.detach().clone()
    pos_grad_before = pos_grad.detach().clone()
    strain_grad_before = strain_grad.detach().clone()

    batch = O3Transformations(
        torch.stack([R_92, R_38]),
        max_angular_momentum=1,
    )
    transformed = batch.transform_tensormap(tensor, torch.tensor([92, 38]))
    transformed_block = transformed.block()

    assert torch.equal(transformed_block.values, values_before)

    expected_pos = pos_grad_before.clone()
    expected_pos[0] = R_92 @ pos_grad_before[0]
    expected_pos[1] = R_38 @ pos_grad_before[1]
    expected_pos[2] = R_92 @ pos_grad_before[2]
    expected_pos[3] = R_38 @ pos_grad_before[3]
    expected_pos[4] = R_38 @ pos_grad_before[4]

    transformed_pos = transformed_block.gradient("positions").values
    assert torch.allclose(transformed_pos, expected_pos)

    expected_strain = strain_grad_before.clone()
    expected_strain[0] = torch.einsum(
        "Aa,abp,Bb->ABp", R_92, strain_grad_before[0], R_92
    )
    expected_strain[1] = torch.einsum(
        "Aa,abp,Bb->ABp", R_38, strain_grad_before[1], R_38
    )
    assert torch.allclose(
        transformed_block.gradient("strain").values,
        expected_strain,
    )

    input_block = tensor.block()
    assert torch.equal(input_block.values, values_before)
    assert torch.equal(input_block.gradient("positions").values, pos_grad_before)
    assert torch.equal(input_block.gradient("strain").values, strain_grad_before)

    autograd_gradient = torch.autograd.grad(
        transformed_pos.square().sum(),
        pos_grad,
    )[0]
    assert torch.allclose(
        autograd_gradient,
        2.0 * pos_grad,
        rtol=0.0,
        atol=1e-12,
    )


def test_inverse_transform_block_roundtrips():
    """``inverse_transform_block`` undoes ``transform_block``."""
    batch = random_transformations(
        3,
        max_angular_momentum=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(99),
    )

    values = torch.randn(6, 5, 2, dtype=torch.float64)
    keys = Labels(["o3_lambda", "o3_sigma"], torch.tensor([[2, 1]]))
    samples = Labels(
        ["system", "atom"],
        torch.tensor([[0, 0], [1, 1], [2, 0], [0, 2], [1, 0], [2, 1]]),
    )
    components = [Labels("o3_mu", torch.arange(-2, 3).reshape(-1, 1))]
    properties = Labels("property", torch.tensor([[0], [1]]))
    block = TensorBlock(
        values=values,
        samples=samples,
        components=components,
        properties=properties,
    )

    forward = batch.transform_block(keys, block)
    roundtrip = batch.inverse_transform_block(keys, forward)

    assert torch.allclose(roundtrip.values, values, atol=1e-12)
    assert roundtrip.samples == block.samples
    assert roundtrip.components == block.components
    assert roundtrip.properties == block.properties


def test_transform_block_add_inversion():
    """``transform_block`` with ``add_inversion`` applies the inverted operations."""
    matrices = torch.stack(
        [
            torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64),
            torch.tensor(_axis_angle([-2.0, 1.0, 0.5], 2.4), dtype=torch.float64),
        ]
    )
    batch = O3Transformations(matrices, max_angular_momentum=1)
    inverted = O3Transformations(-matrices, max_angular_momentum=1)

    keys = Labels(["o3_lambda", "o3_sigma"], torch.tensor([[1, 1]]))
    values = torch.randn(2, 3, 1, dtype=torch.float64)
    samples = Labels(["system"], torch.tensor([[0], [1]]))
    components = [Labels("o3_mu", torch.arange(-1, 2).reshape(-1, 1))]
    properties = Labels("p", torch.tensor([[0]]))
    block = TensorBlock(
        values=values, samples=samples, components=components, properties=properties
    )

    with_inversion = batch.transform_block(keys, block, add_inversion=True)
    reference = inverted.transform_block(keys, block)

    assert torch.allclose(with_inversion.values, reference.values, atol=1e-12)


def test_component_metadata_validation():
    """Hand-built blocks with wrong component metadata fail with clear errors."""
    transformations = O3Transformations(
        torch.eye(3, dtype=torch.float64).unsqueeze(0), 1
    )

    # unknown component axis name
    tensor = _single_block_tensor_map(
        values=torch.zeros(1, 3, 1, dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[0]])),
        components=[Labels(["direction"], torch.arange(3).reshape(-1, 1))],
    )
    message = re.escape(
        "Found a component axis 'direction', which is neither a Cartesian "
        "('xyz'/'xyz_1'/'xyz_2'/...) nor spherical ('o3_mu'/'o3_mu_1'/...) axis; "
        "it can not be transformed."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        transformations.transform_tensormap(tensor)

    # o3_sigma outside {-1, +1}
    tensor = _single_block_tensor_map(
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor([[0, 2]])),
        values=torch.ones((1, 1, 1), dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[0]])),
        components=[Labels(["o3_mu"], torch.tensor([[0]]))],
    )
    message = re.escape("sigma must be either -1 or +1, got 2")
    with pytest.raises(ValueError, match=f"^{message}$"):
        transformations.transform_tensormap(tensor)

    # misordered Cartesian labels
    tensor = _single_block_tensor_map(
        values=torch.ones((1, 3, 1), dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[0]])),
        components=[Labels(["xyz"], torch.tensor([[2], [0], [1]]))],
    )
    message = re.escape(
        "Cartesian component axis 'xyz' must use labels [0, 1, 2] in x, y, z order."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        transformations.transform_tensormap(tensor)

    # misordered spherical labels on an empty block: the validation is
    # metadata-driven, not row-driven
    block = TensorBlock(
        values=torch.empty((0, 3, 1), dtype=torch.float64),
        samples=Labels(["system"], torch.empty((0, 1), dtype=torch.int64)),
        components=[Labels(["o3_mu"], torch.tensor([[1], [-1], [0]]))],
        properties=Labels(["p"], torch.tensor([[0]])),
    )
    tensor = TensorMap(
        Labels(["o3_lambda", "o3_sigma"], torch.tensor([[1, 1]])),
        [block],
    )
    message = re.escape(
        "Spherical component axis 'o3_mu' for ell=1 must use labels from -1 "
        "through 1 in ascending order."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        transformations.transform_tensormap(tensor)


def test_insufficient_max_angular_momentum():
    """The default ``max_angular_momentum=0`` fails clearly on spherical data."""
    tensor = _single_block_tensor_map(
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor([[1, 1]])),
        values=torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[0]])),
        components=[Labels(["o3_mu"], torch.arange(-1, 2).reshape(-1, 1))],
    )

    transformations = random_transformations(
        1,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    message = re.escape("ell=1 exceeds max_angular_momentum=0.")
    with pytest.raises(ValueError, match=f"^{message}$"):
        transformations.transform_tensormap(tensor)

    transformations = random_transformations(
        1,
        max_angular_momentum=1,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    transformed = transformations.transform_tensormap(tensor)
    # a rotation (with sigma parity +-1) preserves the norm of ell=1 values
    torch.testing.assert_close(
        transformed.block().values.norm(),
        tensor.block().values.norm(),
        rtol=0.0,
        atol=1e-12,
    )


def test_transform_tensor_combines_spherical_axis_parities():
    """Improper parity factors multiply across spherical component axes."""
    values = torch.arange(1, 10, dtype=torch.float64).reshape(1, 3, 3, 1)
    for sigma_9 in (1, -1):
        tensor = _single_block_tensor_map(
            keys=Labels(
                ["o3_lambda_1", "o3_lambda_9", "o3_sigma_1", "o3_sigma_9"],
                torch.tensor([[1, 1, 1, sigma_9]]),
            ),
            values=values,
            samples=Labels(["system"], torch.tensor([[0]])),
            components=[
                Labels(["o3_mu_1"], torch.arange(-1, 2).reshape(-1, 1)),
                Labels(["o3_mu_9"], torch.arange(-1, 2).reshape(-1, 1)),
            ],
        )

        transformed = O3Transformations(
            (-torch.eye(3, dtype=torch.float64)).unsqueeze(0), 1
        ).transform_tensormap(tensor)

        assert torch.allclose(
            transformed.block().values,
            sigma_9 * values,
            rtol=0.0,
            atol=1e-12,
        )


def test_rows_route_by_system_id():
    """Default and explicit IDs route each row to its own transformations."""
    batch = O3Transformations(
        torch.stack(
            [
                torch.tensor(
                    [
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float64,
                ),
                _rotation_90_degrees_around_z(),
            ]
        ),
        max_angular_momentum=0,
    )
    keys = Labels(["_"], torch.tensor([[0]]))
    expected = torch.tensor(
        [
            [[-1.0], [0.0], [0.0]],
            [[-2.0], [0.0], [0.0]],
        ],
        dtype=torch.float64,
    )

    def make_block(sample_system_ids):
        # TensorMap takes ownership of its blocks, so each call needs a new one
        return TensorBlock(
            values=torch.tensor(
                [
                    [[1.0], [0.0], [0.0]],
                    [[0.0], [2.0], [0.0]],
                ],
                dtype=torch.float64,
            ),
            samples=Labels(
                ["system", "atom"],
                torch.tensor(
                    [
                        [sample_system_ids[0], 0],
                        [sample_system_ids[1], 0],
                    ]
                ),
            ),
            components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
            properties=Labels(["p"], torch.tensor([[0]])),
        )

    for system_ids, sample_system_ids in [
        (None, [0, 1]),
        (torch.tensor([92, 38]), [92, 38]),
        (torch.tensor([92, -7], dtype=torch.int32), [92, -7]),
    ]:
        transformed = batch.transform_tensormap(
            TensorMap(keys, [make_block(sample_system_ids)]),
            system_ids,
        ).block()
        assert torch.equal(transformed.values, expected)


def _scalar_two_system_tensor(device="cpu"):
    """A minimal two-row scalar TensorMap with ``"system"`` labels 92 and 38."""
    return _single_block_tensor_map(
        values=torch.zeros((2, 1), dtype=torch.float64, device=device),
        samples=Labels(["system"], torch.tensor([[92], [38]], device=device)),
        components=[],
    )


def test_rejects_unassigned_system_labels():
    """Every ``"system"`` label present in a block must appear in ``system_ids``."""
    batch = O3Transformations(
        torch.eye(3, dtype=torch.float64).expand(2, 3, 3),
        max_angular_momentum=0,
    )

    message = re.escape(
        "Block samples contain system labels [38] that are not in "
        "system_ids=[92, 99]. Every sample must be assigned to a system in the "
        "transformation."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        batch.transform_tensormap(_scalar_two_system_tensor(), torch.tensor([92, 99]))


def test_transform_empty_inputs_are_no_ops():
    """Blocks with no samples and TensorMaps with no blocks pass through."""
    transformations = O3Transformations(
        _rotation_90_degrees_around_z().unsqueeze(0),
        max_angular_momentum=0,
    )

    no_samples = _single_block_tensor_map(
        values=torch.empty((0, 1), dtype=torch.float64),
        samples=Labels(["system"], torch.empty((0, 1), dtype=torch.int64)),
        components=[],
    )
    transformed = transformations.transform_tensormap(no_samples)
    assert torch.equal(transformed.block().values, no_samples.block().values)
    assert transformed.block().samples == no_samples.block().samples

    empty = TensorMap(
        Labels(
            ["_"],
            torch.empty((0, 1), dtype=torch.int64),
        ),
        [],
    )
    transformed_map = transformations.transform_tensormap(empty)

    assert len(transformed_map) == 0
    assert transformed_map.keys == empty.keys


def test_single_system_routes_all_rows():
    """One operation transforms all rows regardless of ``"system"`` labels."""
    transformations = O3Transformations(_rotation_90_degrees_around_z().unsqueeze(0), 1)
    values = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[-1.0], [4.0], [0.0]],
        ],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [
            [[-2.0], [1.0], [3.0]],
            [[-4.0], [-1.0], [0.0]],
        ],
        dtype=torch.float64,
    )

    for samples in [
        Labels(["atom"], torch.tensor([[0], [1]])),
        Labels(["system", "atom"], torch.tensor([[4, 0], [4, 1]])),
    ]:
        tensor = _single_block_tensor_map(
            values=values,
            samples=samples,
            components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
        )
        transformed = transformations.transform_tensormap(tensor)
        assert torch.equal(transformed.block().values, expected)


def test_block_with_subset_of_systems():
    """Blocks may cover only some systems; ids must survive beyond int32."""
    transformations = O3Transformations(
        torch.stack(
            [_rotation_90_degrees_around_z(), torch.eye(3, dtype=torch.float64)]
        ),
        max_angular_momentum=1,
    )

    tensor = _single_block_tensor_map(
        values=torch.tensor(
            [[[1.0], [2.0], [3.0]]],
            dtype=torch.float64,
        ),
        samples=Labels(
            ["system", "atom"],
            torch.tensor([[92, 0]]),
        ),
        components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
    )

    transformed = transformations.transform_tensormap(
        tensor,
        torch.tensor([92, 2**40]),
    )

    expected = torch.tensor(
        [[[-2.0], [1.0], [3.0]]],
        dtype=torch.float64,
    )
    assert torch.equal(transformed.block().values, expected)


def test_pair_samples_routing():
    """Row routing by the "system" column also works for pair-sampled blocks (e.g.
    atom-pair targets), which carry extra sample columns beyond "system"/"atom"."""
    identity = torch.eye(3, dtype=torch.float64)
    # 90-degree rotation around z: (x,y) -> (-y, x)
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    rot90 = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)
    transformations = O3Transformations(
        torch.stack([identity, rot90]),
        max_angular_momentum=1,
    )

    # row 0 (system 0) -> identity: unchanged
    # row 1 (system 1) -> rot90 (z-90): (0,2,0) -> (-2,0,0)
    vector_values = torch.tensor(
        [[[1.0], [0.0], [0.0]], [[0.0], [2.0], [0.0]]],
        dtype=torch.float64,
    )
    samples = Labels(
        [
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        torch.tensor([[0, 0, 1, 0, 0, 0], [1, 1, 0, -1, 0, 0]]),
    )
    tensor = TensorMap(
        Labels(["_"], torch.tensor([[0]])),
        [
            TensorBlock(
                values=vector_values.clone(),
                samples=samples,
                components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
                properties=Labels(["p"], torch.tensor([[0]])),
            )
        ],
    )

    transformed = transformations.transform_tensormap(tensor)
    result_block = transformed.block()

    expected = torch.tensor(
        [[[1.0], [0.0], [0.0]], [[-2.0], [0.0], [0.0]]], dtype=torch.float64
    )
    assert torch.allclose(result_block.values, expected)
    # only "system" is used for routing and only values are rotated: the extra pair
    # columns (first_atom/second_atom/cell_shift_*) must pass through untouched
    assert result_block.samples == samples


@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
@pytest.mark.parametrize("parities", [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)])
def test_batched_tensor_transform_matches_singleton_reference(
    device,
    dtype,
    parities,
):
    """Rows routed to each operation of a batch match the single-operation
    result, including for batches mixing proper and improper operations."""
    dtype = getattr(torch, dtype)
    atol = 1.0e-5 if dtype == torch.float32 else 1.0e-12
    proper_matrices = [
        _rotation_90_degrees_around_z().to(device=device, dtype=dtype),
        torch.tensor(
            _axis_angle([1.0, 2.0, 3.0], 0.7),
            device=device,
            dtype=dtype,
        ),
    ]
    single_matrices = torch.stack(
        [sign * matrix for sign, matrix in zip(parities, proper_matrices, strict=True)]
    )
    batch = O3Transformations(single_matrices, max_angular_momentum=2)

    keys = Labels(
        ["o3_lambda_1", "o3_sigma_1", "o3_lambda_2", "o3_sigma_2"],
        torch.tensor([[1, 1, 2, 1]], device=device),
    )
    properties = Labels("property", torch.tensor([[0], [1]], device=device))
    components = [
        Labels("xyz", torch.arange(3, device=device).reshape(-1, 1)),
        Labels("o3_mu_1", torch.arange(-1, 2, device=device).reshape(-1, 1)),
        Labels("o3_mu_2", torch.arange(-2, 3, device=device).reshape(-1, 1)),
    ]
    values = torch.linspace(
        -2.0,
        3.0,
        3 * 3 * 3 * 5 * 2,
        device=device,
        dtype=dtype,
    ).reshape(3, 3, 3, 5, 2)
    values.requires_grad_()
    gradient_values = torch.linspace(
        1.0,
        5.0,
        4 * 3 * 3 * 5 * 2,
        device=device,
        dtype=dtype,
    ).reshape(4, 3, 3, 5, 2)
    gradient_values.requires_grad_()

    block = TensorBlock(
        values=values,
        samples=Labels(
            ["system", "atom"],
            torch.tensor([[1, 2], [0, 0], [1, 1]], device=device),
        ),
        components=components,
        properties=properties,
    )
    block.add_gradient(
        "parameter",
        TensorBlock(
            values=gradient_values,
            samples=Labels(
                ["sample", "parameter"],
                torch.tensor(
                    [[2, 0], [0, 1], [1, 2], [2, 3]],
                    device=device,
                ),
            ),
            components=components,
            properties=properties,
        ),
    )
    tensor = TensorMap(keys, [block])
    tensor.set_info("unit", "arbitrary")
    values_before = values.detach().clone()
    gradient_values_before = gradient_values.detach().clone()

    result = batch.transform_tensormap(tensor)

    # reference: a single operation transforms every row through the singleton
    # path, so the batch rows routed to it must match row by row
    system_column = tensor.block(0).samples.column("system").to(dtype=torch.long)
    gradient_parents = (
        tensor.block(0).gradient("parameter").samples.column("sample")
    ).to(dtype=torch.long)
    for batch, single_matrix in enumerate(single_matrices):
        single = O3Transformations(single_matrix.unsqueeze(0), max_angular_momentum=2)
        reference = single.transform_tensormap(tensor)
        rows = system_column == batch
        assert torch.allclose(
            result.block(0).values[rows],
            reference.block(0).values[rows],
            rtol=0.0,
            atol=atol,
        )
        gradient_rows = rows[gradient_parents]
        assert torch.allclose(
            result.block(0).gradient("parameter").values[gradient_rows],
            reference.block(0).gradient("parameter").values[gradient_rows],
            rtol=0.0,
            atol=atol,
        )

    assert result.info() == tensor.info()
    assert torch.equal(values, values_before)
    assert torch.equal(gradient_values, gradient_values_before)

    result_loss = result.block_by_id(0).values.square().sum()
    result_loss = (
        result_loss + result.block_by_id(0).gradient("parameter").values.square().sum()
    )
    value_gradient, explicit_gradient_gradient = torch.autograd.grad(
        result_loss,
        (values, gradient_values),
    )
    assert torch.allclose(value_gradient, 2.0 * values, rtol=0.0, atol=atol)
    assert torch.allclose(
        explicit_gradient_gradient,
        2.0 * gradient_values,
        rtol=0.0,
        atol=atol,
    )


def test_single_transformation_needs_no_system_label():
    """A single transformation applies to samples without a ``system`` label."""
    transformations = O3Transformations(
        _rotation_90_degrees_around_z().unsqueeze(0),
        max_angular_momentum=1,
    )
    tensor = _single_block_tensor_map(
        keys=Labels(
            ["o3_lambda", "o3_sigma"],
            torch.tensor([[1, 1]]),
        ),
        values=torch.tensor(
            [[[1.0], [2.0], [3.0]], [[-1.0], [4.0], [0.0]]],
            dtype=torch.float64,
        ),
        samples=Labels("atom", torch.tensor([[0], [1]])),
        components=[
            Labels("o3_mu", torch.arange(-1, 2).reshape(-1, 1)),
        ],
    )

    result = transformations.transform_tensormap(tensor)
    expected = transformations.transform_spherical(
        tensor.block().values.squeeze(-1),
        ell=1,
        sigma=1,
    ).unsqueeze(-1)

    assert torch.allclose(
        result.block().values,
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_batched_tensor_transform_rejects_invalid_routing_and_wigner_rank():
    """Ambiguous routing or missing Wigner-D ranks fail instead of misrotating."""
    batch = O3Transformations(
        torch.stack(
            [torch.eye(3, dtype=torch.float64), _rotation_90_degrees_around_z()]
        ),
        max_angular_momentum=1,
    )

    missing_system = _single_block_tensor_map(
        values=torch.ones((1, 1), dtype=torch.float64),
        samples=Labels("sample", torch.tensor([[0]])),
        components=[],
    )
    message = "multiple transformations require a 'system' sample dimension"
    with pytest.raises(ValueError, match=f"^{message}$"):
        batch.transform_tensormap(missing_system)

    for system_index in (-1, 2):
        out_of_range = _single_block_tensor_map(
            values=torch.ones((1, 1), dtype=torch.float64),
            samples=Labels("system", torch.tensor([[system_index]])),
            components=[],
        )
        message = "sample system indices exceed the transformation batch"
        with pytest.raises(ValueError, match=f"^{message}$"):
            batch.transform_tensormap(out_of_range)

    unavailable_rank = _single_block_tensor_map(
        keys=Labels(
            ["o3_lambda", "o3_sigma"],
            torch.tensor([[1, 1]]),
        ),
        values=torch.ones((1, 3, 1), dtype=torch.float64),
        samples=Labels("system", torch.tensor([[0]])),
        components=[
            Labels("o3_mu", torch.arange(-1, 2).reshape(-1, 1)),
        ],
    )
    too_low = O3Transformations(
        torch.eye(3, dtype=torch.float64).unsqueeze(0),
        max_angular_momentum=0,
    )
    message = re.escape("ell=1 exceeds max_angular_momentum=0.")
    with pytest.raises(ValueError, match=f"^{message}$"):
        too_low.transform_tensormap(unavailable_rank)


def test_inverse_and_add_inversion():
    """``inverse_transform_*`` and ``add_inversion`` compose correctly."""
    matrices = torch.stack(
        [
            torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64),
            torch.tensor(_axis_angle([-2.0, 1.0, 0.5], 2.4), dtype=torch.float64),
        ]
    )
    transformations = O3Transformations(matrices, max_angular_momentum=2)

    # the inverse matrices are the transposed matrices
    for matrix in matrices:
        assert torch.allclose(
            matrix.T @ matrix,
            torch.eye(3, dtype=torch.float64),
            atol=1e-12,
        )
    for ell in range(3):
        assert torch.allclose(
            transformations.inverse_wigner_D_matrices(ell),
            transformations.wigner_D_matrices(ell).transpose(1, 2),
            atol=1e-12,
        )

    # add_inversion negates the matrices and flips the parity, while the
    # proper rotational part -- and with it the Wigner-D matrices -- is unchanged
    assert torch.equal(
        transformations._effective_matrices(add_inversion=True, transpose=False),
        -transformations.matrices,
    )
    assert torch.all(
        transformations._effective_improper(add_inversion=True)
        != transformations._improper
    )
    for ell in range(3):
        assert torch.equal(
            transformations.wigner_D_matrices(ell),
            transformations.wigner_D_matrices(ell),
        )

    # values round-trip through a transformation and its inverse
    values = torch.randn(4, 5, dtype=torch.float64)
    forward = transformations.transform_spherical(values, ell=2, sigma=-1)
    assert forward.shape[0] == 2
    for batch in range(2):
        roundtrip = transformations.inverse_transform_spherical(
            forward[batch],
            ell=2,
            sigma=-1,
        )
        assert torch.allclose(roundtrip[batch], values, atol=1e-12)

    # the same round-trip, composed with an inversion on both sides
    forward_inverted = transformations.transform_spherical(
        values, ell=2, sigma=-1, add_inversion=True
    )
    for batch in range(2):
        roundtrip_inverted = transformations.inverse_transform_spherical(
            forward_inverted[batch], ell=2, sigma=-1, add_inversion=True
        )
        assert torch.allclose(roundtrip_inverted[batch], values, atol=1e-12)


def test_script():
    """A scripted model can use an O3Transformations submodule inside forward
    and still be saved and loaded."""

    class BackRotate(torch.nn.Module):
        def __init__(self, transformations: O3Transformations):
            super().__init__()
            self.transformations = transformations

        def forward(self, tensor: TensorMap) -> TensorMap:
            return self.transformations.inverse_transform_tensormap(
                tensor, add_inversion=True
            )

    transformations = random_transformations(
        3,
        max_angular_momentum=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(3),
    )
    module = BackRotate(transformations)
    scripted = torch.jit.script(module)

    values = torch.randn(6, 5, 2, dtype=torch.float64)
    tensor = _single_block_tensor_map(
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor([[2, 1]])),
        values=values,
        samples=Labels(
            ["system", "sample"],
            torch.stack([torch.arange(6) % 3, torch.arange(6)], dim=1),
        ),
        components=[Labels(["o3_mu"], torch.arange(-2, 3).reshape(-1, 1))],
        properties=Labels(["p"], torch.tensor([[0], [1]])),
    )

    eager_result = module(tensor)
    scripted_result = scripted(tensor)
    assert torch.allclose(
        eager_result.block().values,
        scripted_result.block().values,
        atol=1e-12,
    )

    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    loaded = torch.jit.load(buffer)
    loaded_result = loaded(tensor)
    assert torch.allclose(
        eager_result.block().values,
        loaded_result.block().values,
        atol=1e-12,
    )
