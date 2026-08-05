import re

import metatensor.torch as mts
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

# These private helpers back exported symmetrized-model wrappers and have no public
# entry point yet; their tests below compare them against the public transform_tensor.
from metatomic.torch.o3._transformations import (
    _max_o3_lambda_in_tensor,
    _transform_tensormap_batched,
)

# The complex-to-real spherical harmonics conversion is defined only here for now.
from metatomic.torch.o3._wigner import _complex_to_real_spherical_harmonics_transform

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


def _stack_o3_matrices(transformations, max_angular_momentum):
    """Stack Cartesian, Wigner, and parity tensors from O3 transformations."""
    matrices = torch.cat(
        [transformation.matrices for transformation in transformations]
    )
    wigner_matrices = [
        torch.cat(
            [
                transformation.wigner_D_matrices(ell)
                for transformation in transformations
            ]
        )
        for ell in range(max_angular_momentum + 1)
    ]
    improper = torch.cat(
        [transformation.improper for transformation in transformations]
    )
    return matrices, wigner_matrices, improper


def test_max_o3_lambda_in_tensor():
    """Check `_max_o3_lambda_in_tensor`, including in torchscript mode"""
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
    cartesian = _single_block_tensor_map(
        values=torch.ones((1, 3, 1), dtype=torch.float64),
        samples=Labels("system", torch.tensor([[0]])),
        components=[Labels("xyz", torch.arange(3).reshape(-1, 1))],
    )

    scripted_maximum = torch.jit.script(_max_o3_lambda_in_tensor)
    assert scripted_maximum(spherical) == 3
    assert scripted_maximum(cartesian) == -1


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
    transformed = transform_system(
        system,
        O3Transformation(rotation, max_angular_momentum=0),
    )

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
    message = re.escape("max_angular_momentum must be a non-negative integer, got -1.")
    with pytest.raises(ValueError, match=f"^{message}$"):
        O3Transformation(torch.eye(3, dtype=torch.float64), -1)
    with pytest.raises(ValueError, match=f"^{message}$"):
        random_transformations(
            0, max_angular_momentum=-1, device=torch.device("cpu"), dtype=torch.float64
        )
    message = re.escape("n must be a non-negative integer, got -1.")
    with pytest.raises(ValueError, match=f"^{message}$"):
        random_transformations(-1, device=torch.device("cpu"), dtype=torch.float64)

    # models only declare float32/float64 capabilities
    message = re.escape(
        "dtype must be torch.float32 or torch.float64, got torch.float16."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        random_transformations(0, device=torch.device("cpu"), dtype=torch.float16)

    # matrices must be (3, 3) or (N, 3, 3) and orthogonal
    message = re.escape(
        "Transformation has shape (2, 2); expected (3, 3) or (N, 3, 3) with N > 0."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        O3Transformation(torch.eye(2, dtype=torch.float64), max_angular_momentum=0)
    matrix = torch.eye(3, dtype=torch.float64)
    matrix[0, 0] = 2.0
    message = re.escape("Transformation is not orthogonal (R @ R.T deviates from I).")
    with pytest.raises(ValueError, match=f"^{message}$"):
        O3Transformation(matrix, max_angular_momentum=0)

    # Wigner-D requests need a valid ell
    transformation = O3Transformation(torch.eye(3, dtype=torch.float64), 1)
    message = re.escape("ell must be a non-negative integer, got -1.")
    with pytest.raises(ValueError, match=f"^{message}$"):
        transformation.wigner_D_matrix(-1)

    # systems must match the transformation dtype/device
    system = _make_system([1], dtype=torch.float64)
    transformation = O3Transformation(torch.eye(3, dtype=torch.float32), 0)
    message = re.escape(
        "System has positions with dtype/device (torch.float64, cpu) differing "
        "from the transformations (torch.float32, cpu)."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        transform_system(system, transformation)


def test_random_rotations_are_orthogonal():
    """Sampling without inversions yields proper rotations, including n=0."""
    atol = 1e-10

    assert (
        random_transformations(
            0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        == []
    )

    transformations = random_transformations(
        20,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert len(transformations) == 20
    identity = torch.eye(3, dtype=torch.float64)

    for transformation in transformations:
        matrix = transformation.matrix

        assert matrix.shape == (3, 3)
        assert torch.allclose(
            matrix @ matrix.T,
            identity,
            atol=atol,
        )
        assert abs(float(torch.det(matrix)) - 1.0) < atol


def test_random_transformations_include_inversions():
    """Sampling with inversions covers both O(3) cosets and sets is_improper."""
    transformations = random_transformations(
        20,
        device="cpu",
        dtype=torch.float64,
        include_inversions=True,
        generator=torch.Generator().manual_seed(20260718),
    )

    determinants = torch.stack([torch.det(t.matrix) for t in transformations])
    assert (determinants > 0).any() and (determinants < 0).any()
    for transformation in transformations:
        assert transformation.is_improper == bool(torch.det(transformation.matrix) < 0)


def test_o3_parity_factor():
    """Improper transformations scale spherical values by ``sigma * (-1)**ell``."""
    transformation = O3Transformation(
        -torch.eye(3, dtype=torch.float64),
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
                transformation.transform_spherical(values, ell, sigma),
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
    proper = O3Transformation(matrix, max_angular_momentum=3)
    improper = O3Transformation(-matrix, max_angular_momentum=3)

    for ell in range(4):
        assert torch.equal(proper.wigner_D_matrix(ell), improper.wigner_D_matrix(ell))


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
    transformation = O3Transformation(matrix, max_angular_momentum=1)

    C = CARTESIAN_TO_SPHERICAL_L1
    expected = C @ proper @ C.T
    assert torch.allclose(transformation.wigner_D_matrix(1), expected, atol=1e-12)

    # the spherical action of an ell=1, sigma=+1 vector equals the Cartesian
    # action, inversion included
    vectors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.2, -1.3, 0.7], [-0.5, 0.1, 2.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        transformation.transform_spherical(vectors @ C.T, ell=1, sigma=1),
        transformation.transform_cartesian(vectors) @ C.T,
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
    matrix = torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64)
    transformation = O3Transformation(matrix, max_angular_momentum=2)
    D2 = transformation.wigner_D_matrix(2)

    for v in torch.tensor(
        [[1.0, 0.0, 0.0], [0.2, -1.3, 0.7], [-0.5, 0.1, 2.0]],
        dtype=torch.float64,
    ):
        torch.testing.assert_close(
            D2 @ _real_solid_harmonics_l2(v),
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

        transformation = O3Transformation(matrix, max_angular_momentum=1)
        expected = basis @ proper @ basis.T
        actual = transformation.wigner_D_matrix(1)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)


def test_gradient_rows_follow_parent_system():
    """Gradient rows use their parent sample's transformation; inputs unchanged."""
    systems = [
        _make_system([1, 1]),
        _make_system([8, 8, 8]),
    ]
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

    transformed = transform_tensor(
        tensor,
        systems,
        [
            O3Transformation(R_92, max_angular_momentum=1),
            O3Transformation(R_38, max_angular_momentum=1),
        ],
        system_ids=[92, 38],
    )
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


def test_component_metadata_validation():
    """Hand-built blocks with wrong component metadata fail with clear errors."""
    systems = [_make_system([1])]
    transformations = [O3Transformation(torch.eye(3, dtype=torch.float64), 1)]

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
        transform_tensor(tensor, systems, transformations)

    # o3_sigma outside {-1, +1}
    tensor = _single_block_tensor_map(
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor([[0, 2]])),
        values=torch.ones((1, 1, 1), dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[0]])),
        components=[Labels(["o3_mu"], torch.tensor([[0]]))],
    )
    message = re.escape("sigma must be either -1 or +1, got 2.")
    with pytest.raises(ValueError, match=f"^{message}$"):
        transform_tensor(tensor, systems, transformations)

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
        transform_tensor(tensor, systems, transformations)

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
        transform_tensor(tensor, systems, transformations)


def test_insufficient_max_angular_momentum():
    """The default ``max_angular_momentum=0`` fails clearly on spherical data."""
    tensor = _single_block_tensor_map(
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor([[1, 1]])),
        values=torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[0]])),
        components=[Labels(["o3_mu"], torch.arange(-1, 2).reshape(-1, 1))],
    )
    systems = [_make_system([1])]

    transformations = random_transformations(
        1,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    message = re.escape("ell=1 exceeds max_angular_momentum=0.")
    with pytest.raises(ValueError, match=f"^{message}$"):
        transform_tensor(tensor, systems, transformations)

    transformations = random_transformations(
        1,
        max_angular_momentum=1,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    transformed = transform_tensor(tensor, systems, transformations)
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

        transformed = transform_tensor(
            tensor,
            [_make_system([1])],
            [O3Transformation(-torch.eye(3, dtype=torch.float64), 1)],
        )

        assert torch.allclose(
            transformed.block().values,
            sigma_9 * values,
            rtol=0.0,
            atol=1e-12,
        )


def test_rows_route_by_system_id():
    """Default, list, and tensor IDs route each row to its own transformation."""
    systems = [_make_system([1]), _make_system([8])]
    transformations = [
        O3Transformation(
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float64,
            ),
            max_angular_momentum=0,
        ),
        O3Transformation(
            _rotation_90_degrees_around_z(),
            max_angular_momentum=0,
        ),
    ]
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
        ([92, 38], [92, 38]),
        (torch.tensor([92, -7], dtype=torch.int32), [92, -7]),
    ]:
        kwargs = {} if system_ids is None else {"system_ids": system_ids}
        transformed = transform_tensor(
            TensorMap(keys, [make_block(sample_system_ids)]),
            systems,
            transformations,
            **kwargs,
        ).block()
        assert torch.equal(transformed.values, expected)

    # the public transform_block entry point agrees
    transformed_block = transform_block(
        keys[0],
        make_block([0, 1]),
        systems,
        transformations,
    )
    assert torch.equal(transformed_block.values, expected)


def _scalar_two_system_tensor(device="cpu"):
    """A minimal two-row scalar TensorMap with ``"system"`` labels 92 and 38."""
    return _single_block_tensor_map(
        values=torch.zeros((2, 1), dtype=torch.float64, device=device),
        samples=Labels(["system"], torch.tensor([[92], [38]], device=device)),
        components=[],
    )


def test_system_ids_validation():
    """Bad system assignments fail up front instead of misrouting rows."""
    systems = [_make_system([1]), _make_system([8])]
    transformations = [
        O3Transformation(torch.eye(3, dtype=torch.float64), 0),
        O3Transformation(torch.eye(3, dtype=torch.float64), 0),
    ]

    # one transformation per system, one distinct id per system
    message = re.escape(
        "Expected one transformation per system, but got len(systems)=2 and "
        "len(transformations)=1."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        transform_tensor(
            _scalar_two_system_tensor(), systems, transformations[:1], [92, -7]
        )
    message = re.escape(
        "system_ids must contain exactly one entry per system, but got "
        "len(system_ids)=1 and len(systems)=2."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        transform_tensor(_scalar_two_system_tensor(), systems, transformations, [92])
    message = re.escape(
        "system_ids must contain one distinct entry per system, but got [92, 92]."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        transform_tensor(
            _scalar_two_system_tensor(), systems, transformations, [92, 92]
        )

    # ids must live with the values ("meta" needs no accelerator hardware)
    message = re.escape(
        "system_ids are on device cpu, but the values to transform are on device meta."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        transform_tensor(
            _scalar_two_system_tensor(device="meta"),
            systems,
            transformations,
            torch.tensor([92, -7]),
        )

    # every "system" label present in a block must appear in system_ids
    message = re.escape(
        "Block samples contain system labels [38] that are not in "
        "system_ids=[92, 99]. Every sample must be assigned to a system in the "
        "transformation."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        transform_tensor(
            _scalar_two_system_tensor(), systems, transformations, [92, 99]
        )

    # with multiple systems, each row must identify its system
    no_system_column = _single_block_tensor_map(
        values=torch.zeros((1, 1), dtype=torch.float64),
        samples=Labels(["atom"], torch.tensor([[0]])),
        components=[],
    )
    message = re.escape("multiple transformations require a 'system' sample dimension")
    with pytest.raises(ValueError, match=f"^{message}$"):
        transform_tensor(no_system_column, systems, transformations)

    # a transformation with no assigned rows is still validated: a silent
    # mismatch would only surface once such rows appear in another batch
    block = TensorBlock(
        values=torch.ones((1, 1), dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[92]])),
        components=[],
        properties=Labels(["p"], torch.tensor([[0]])),
    )
    transformations[1] = O3Transformation(torch.eye(3, dtype=torch.float32), 0)
    message = re.escape(
        "Transformation at index 1 has dtype/device (torch.float32, cpu), "
        "differing from the values to transform (torch.float64, cpu)."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        transform_tensor(
            TensorMap(Labels(["_"], torch.tensor([[0]])), [block]),
            systems,
            transformations,
            system_ids=[92, 38],
        )


def test_transform_empty_inputs_are_no_ops():
    """Empty blocks, empty TensorMaps, and empty system lists pass through."""
    non_empty = _single_block_tensor_map(
        values=torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[0], [1]])),
        components=[],
    )
    passthrough = transform_tensor(non_empty, [], [])
    assert torch.equal(passthrough.block().values, non_empty.block().values)

    block = TensorBlock(
        values=torch.empty((0, 1), dtype=torch.float64),
        samples=Labels(
            ["system"],
            torch.empty((0, 1), dtype=torch.int64),
        ),
        components=[],
        properties=Labels(["p"], torch.tensor([[0]])),
    )

    transformed = transform_block(
        Labels(["_"], torch.tensor([[0]]))[0],
        block,
        [],
        [],
    )

    assert torch.equal(transformed.values, block.values)
    assert transformed.samples == block.samples

    empty = TensorMap(
        Labels(
            ["_"],
            torch.empty((0, 1), dtype=torch.int64),
        ),
        [],
    )
    # the mixed transformation dtypes check that no values dtype is imposed
    # when there is nothing to transform
    transformed_map = transform_tensor(
        empty,
        [_make_system([1]), _make_system([8])],
        [
            O3Transformation(
                torch.eye(3, dtype=torch.float64),
                max_angular_momentum=0,
            ),
            O3Transformation(
                torch.eye(3, dtype=torch.float32),
                max_angular_momentum=0,
            ),
        ],
    )

    assert len(transformed_map) == 0
    assert transformed_map.keys == empty.keys


def test_single_system_routes_all_rows():
    """With one system, all rows are transformed regardless of ``"system"`` labels."""
    systems = [_make_system([1, 1])]

    transformation = O3Transformation(_rotation_90_degrees_around_z(), 1)
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
        transformed = transform_tensor(tensor, systems, [transformation])
        assert torch.equal(transformed.block().values, expected)


def test_block_with_subset_of_systems():
    """Blocks may cover only some systems; ids must survive beyond int32."""
    systems = [_make_system([1]), _make_system([8])]

    transformation_92 = O3Transformation(_rotation_90_degrees_around_z(), 1)
    transformation_38 = O3Transformation(
        torch.eye(3, dtype=torch.float64),
        1,
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

    transformed = transform_tensor(
        tensor,
        systems,
        [transformation_92, transformation_38],
        system_ids=[92, 2**40],
    )

    expected = torch.tensor(
        [[[-2.0], [1.0], [3.0]]],
        dtype=torch.float64,
    )
    assert torch.equal(transformed.block().values, expected)


def test_pair_samples_routing():
    """Row routing by the "system" column also works for pair-sampled blocks (e.g.
    atom-pair targets), which carry extra sample columns beyond "system"/"atom"."""
    systems = [_make_system([1, 8]), _make_system([1, 8])]

    R0 = O3Transformation(torch.eye(3, dtype=torch.float64), 1)
    # 90-degree rotation around z: (x,y) -> (-y, x)
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    R1 = O3Transformation(
        torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64), 1
    )

    # row 0 (system 0) -> R0 (identity): unchanged
    # row 1 (system 1) -> R1 (z-90): (0,2,0) -> (-2,0,0)
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

    transformed = transform_tensor(tensor, systems, [R0, R1])
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
def test_batched_tensor_transform_matches_transform_tensor(
    device,
    dtype,
    parities,
):
    """The scripted batched kernel matches ``transform_tensor``, including for
    batches mixing proper and improper operations."""
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
    transformations = [
        O3Transformation(sign * matrix, max_angular_momentum=2)
        for sign, matrix in zip(parities, proper_matrices, strict=True)
    ]
    matrices, wigner_matrices, improper = _stack_o3_matrices(
        transformations,
        max_angular_momentum=2,
    )

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

    expected = transform_tensor(
        tensor,
        [
            _make_system([1], device=device, dtype=dtype),
            _make_system([8], device=device, dtype=dtype),
        ],
        transformations,
    )
    scripted_transform = torch.jit.script(_transform_tensormap_batched)
    result = scripted_transform(
        tensor,
        matrices,
        wigner_matrices,
        improper,
        None,
    )

    mts.allclose_raise(result, expected, rtol=0.0, atol=atol)
    assert result.info() == expected.info()
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


def test_batched_tensor_transform_single_transformation_is_scriptable():
    """The scripted singleton path should not require a ``system`` sample label."""
    transformation = O3Transformation(
        _rotation_90_degrees_around_z(),
        max_angular_momentum=1,
    )
    matrices, wigner_matrices, improper = _stack_o3_matrices(
        [transformation],
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

    scripted_transform = torch.jit.script(_transform_tensormap_batched)
    result = scripted_transform(
        tensor,
        matrices,
        wigner_matrices,
        improper,
        None,
    )
    expected = transform_tensor(
        tensor,
        [_make_system([1])],
        [transformation],
    )

    assert torch.allclose(
        result.block().values,
        expected.block().values,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_batched_tensor_transform_rejects_invalid_routing_and_wigner_rank():
    """Ambiguous routing or missing Wigner-D ranks fail instead of misrotating."""
    transformations = [
        O3Transformation(torch.eye(3, dtype=torch.float64), 1),
        O3Transformation(_rotation_90_degrees_around_z(), 1),
    ]
    matrices, wigner_matrices, improper = _stack_o3_matrices(
        transformations,
        max_angular_momentum=1,
    )

    missing_system = _single_block_tensor_map(
        values=torch.ones((1, 1), dtype=torch.float64),
        samples=Labels("sample", torch.tensor([[0]])),
        components=[],
    )
    message = re.escape("multiple transformations require a 'system' sample dimension")
    with pytest.raises(ValueError, match=f"^{message}$"):
        _transform_tensormap_batched(
            missing_system,
            matrices,
            wigner_matrices,
            improper,
            None,
        )

    for system_index in (-1, 2):
        out_of_range = _single_block_tensor_map(
            values=torch.ones((1, 1), dtype=torch.float64),
            samples=Labels("system", torch.tensor([[system_index]])),
            components=[],
        )
        message = re.escape("sample system indices exceed the transformation batch")
        with pytest.raises(ValueError, match=f"^{message}$"):
            _transform_tensormap_batched(
                out_of_range,
                matrices,
                wigner_matrices,
                improper,
                None,
            )

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
    message = re.escape("ell=1 exceeds max_angular_momentum=0.")
    with pytest.raises(ValueError, match=f"^{message}$"):
        _transform_tensormap_batched(
            unavailable_rank,
            matrices,
            wigner_matrices[:1],
            improper,
            None,
        )


def test_batched_transformation_matches_singles():
    """One batched O3Transformation behaves like its per-operation singles."""
    singles = random_transformations(
        4,
        max_angular_momentum=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
        include_inversions=True,
        generator=torch.Generator().manual_seed(20260805),
    )
    batch = O3Transformation(
        torch.cat([single.matrices for single in singles]),
        max_angular_momentum=2,
    )

    assert batch.matrices.shape == (4, 3, 3)
    determinant_signs = torch.stack(
        [torch.det(single.matrix) < 0 for single in singles]
    )
    assert torch.equal(batch.improper, determinant_signs)

    # Cartesian and spherical actions gain a leading batch axis
    vectors = torch.randn(5, 3, dtype=torch.float64)
    cartesian = batch.transform_cartesian(vectors)
    spherical = batch.transform_spherical(vectors, ell=1, sigma=-1)
    for index, single in enumerate(singles):
        assert torch.allclose(
            cartesian[index],
            single.transform_cartesian(vectors),
            atol=1e-12,
        )
        assert torch.allclose(
            spherical[index],
            single.transform_spherical(vectors, ell=1, sigma=-1),
            atol=1e-12,
        )

    # a batched System transformation matches transform_system per operation
    system = _make_system(
        [1, 8],
        positions=torch.randn(2, 3, dtype=torch.float64),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )
    batch_systems = batch.transform_systems(system)
    assert len(batch_systems) == 4
    for index, single in enumerate(singles):
        expected_system = transform_system(system, single)
        assert torch.allclose(
            batch_systems[index].positions,
            expected_system.positions,
            atol=1e-12,
        )
        assert torch.allclose(
            batch_systems[index].cell,
            expected_system.cell,
            atol=1e-12,
        )


def test_inverse_and_with_inversion_views():
    """``inverse`` and ``with_inversion`` return views composing correctly."""
    matrix = torch.tensor(_axis_angle([1.0, 2.0, 3.0], 0.7), dtype=torch.float64)
    transformation = O3Transformation(matrix, max_angular_momentum=2)

    inverse = transformation.inverse()
    assert torch.allclose(
        inverse.matrix @ transformation.matrix,
        torch.eye(3, dtype=torch.float64),
        atol=1e-12,
    )
    for ell in range(3):
        assert torch.allclose(
            inverse.wigner_D_matrix(ell),
            transformation.wigner_D_matrix(ell).T,
            atol=1e-12,
        )

    flipped = transformation.with_inversion()
    assert flipped.is_improper
    assert torch.equal(flipped.matrices, -transformation.matrices)
    # the proper part -- and with it the Wigner-D matrices -- is unchanged
    for ell in range(3):
        assert torch.equal(
            flipped.wigner_D_matrix(ell),
            transformation.wigner_D_matrix(ell),
        )

    # (-R)^-1 = -R^T, still improper
    inverse_flipped = flipped.inverse()
    assert inverse_flipped.is_improper
    assert torch.allclose(
        inverse_flipped.matrix,
        -matrix.T,
        atol=1e-12,
    )

    # values round-trip through a transformation and its inverse
    values = torch.randn(4, 5, dtype=torch.float64)
    roundtrip = inverse.transform_spherical(
        transformation.transform_spherical(values, ell=2, sigma=-1),
        ell=2,
        sigma=-1,
    )
    assert torch.allclose(roundtrip, values, atol=1e-12)


def test_o3_transformation_scriptable_in_forward():
    """A scripted model can build transformations from buffers inside forward
    and still be saved and loaded."""

    class BackRotate(torch.nn.Module):
        wigner_D: list[torch.Tensor]

        def __init__(self, matrices, wigner_D):
            super().__init__()
            self.register_buffer("matrices", matrices)
            self.wigner_D = list(wigner_D)

        def forward(self, tensor: TensorMap) -> TensorMap:
            proper = torch.zeros(
                self.matrices.size(0),
                dtype=torch.bool,
                device=self.matrices.device,
            )
            transformation = O3Transformation(
                self.matrices,
                len(self.wigner_D) - 1,
                _improper=proper,
                _wigner_D=self.wigner_D,
            )
            inverse = transformation.with_inversion().inverse()
            return inverse.transform_tensormap(tensor)

    singles = random_transformations(
        3,
        max_angular_momentum=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(3),
    )
    matrices, wigner_matrices, _improper = _stack_o3_matrices(
        singles,
        max_angular_momentum=2,
    )
    module = BackRotate(matrices, wigner_matrices)
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

    import io

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
