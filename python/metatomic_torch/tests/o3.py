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
from metatomic.torch.o3._tranformations import _validate_system_ids
from metatomic.torch.o3._wigner import (
    _complex_to_real_spherical_harmonics_transform,
    _rotation_to_angles,
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


@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_transform_system_rotates_geometry_neighbor_lists_and_custom_data(
    device,
    dtype,
):
    """Geometry, neighbor vectors, and custom TensorMaps receive one transformation."""
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
    scalar.set_info("custom", "preserved by transformation")

    # xyz vector per-atom data: must rotate by R on the component axis
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
    assert new_scalar.keys == scalar.keys
    assert new_scalar.info() == scalar.info()
    for block_id in range(len(scalar.keys)):
        assert torch.allclose(
            new_scalar.block_by_id(block_id).values,
            scalar.block_by_id(block_id).values,
        )

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


def test_transform_system_rejects_dtype_mismatch():
    """System positions and transformation matrix must have the same dtype."""
    system = _make_system([1], dtype=torch.float64)
    transformation = O3Transformation(
        torch.eye(3, dtype=torch.float32),
        max_angular_momentum=0,
    )

    with pytest.raises(ValueError, match="differing from the transformations"):
        transform_system(system, transformation)


@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_random_rotations_are_orthogonal(device, dtype):
    dtype = getattr(torch, dtype)
    atol = 1e-5 if dtype == torch.float32 else 1e-10

    assert (
        random_transformations(
            0,
            device=torch.device(device),
            dtype=dtype,
        )
        == []
    )

    transformations = random_transformations(
        20,
        device=torch.device(device),
        dtype=dtype,
    )

    assert len(transformations) == 20
    identity = torch.eye(3, device=device, dtype=dtype)

    for transformation in transformations:
        matrix = transformation.matrix

        assert transformation.device == identity.device
        assert transformation.dtype == dtype
        assert matrix.shape == (3, 3)
        assert torch.allclose(
            matrix @ matrix.T,
            identity,
            atol=atol,
        )
        assert abs(float(torch.det(matrix)) - 1.0) < atol


def test_random_transformations_include_inversions_and_are_reproducible():
    default_generated = random_transformations(
        100,
        device="cpu",
        dtype=torch.float64,
        include_inversions=True,
    )
    first = random_transformations(
        100,
        device="cpu",
        dtype=torch.float64,
        include_inversions=True,
        generator=torch.Generator().manual_seed(20260718),
    )
    second = random_transformations(
        100,
        device="cpu",
        dtype=torch.float64,
        include_inversions=True,
        generator=torch.Generator().manual_seed(20260718),
    )

    determinants = torch.stack(
        [torch.det(transformation.matrix) for transformation in default_generated]
    )
    assert torch.allclose(
        determinants.abs(),
        torch.ones(100, dtype=torch.float64),
        atol=1e-10,
    )
    assert (determinants > 0).any() and (determinants < 0).any()

    for first_transformation, second_transformation in zip(
        first,
        second,
        strict=True,
    ):
        assert torch.equal(
            first_transformation.matrix,
            second_transformation.matrix,
        )
        expected_is_improper = bool(torch.det(first_transformation.matrix) < 0)
        assert first_transformation.is_improper == expected_is_improper
        assert second_transformation.is_improper == expected_is_improper


@pytest.mark.parametrize("n", [0, 1])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.complex64,
        torch.int64,
    ],
)
def test_random_transformations_rejects_unsupported_dtype(n, dtype):
    with pytest.raises(ValueError, match="torch.float32 or torch.float64"):
        random_transformations(
            n,
            device=torch.device("cpu"),
            dtype=dtype,
        )


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_transformation_counts_and_limits_reject_invalid_values(value, error):
    with pytest.raises(error, match="max_angular_momentum"):
        O3Transformation(torch.eye(3, dtype=torch.float64), value)

    with pytest.raises(error, match="max_angular_momentum"):
        random_transformations(
            0,
            max_angular_momentum=value,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

    with pytest.raises(error, match="n must"):
        random_transformations(
            value,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )


def test_transformation_creation_rejects_invalid_matrix():
    with pytest.raises(ValueError, match="shape"):
        O3Transformation(
            torch.eye(2, dtype=torch.float64),
            max_angular_momentum=0,
        )

    matrix = torch.eye(3, dtype=torch.float64)
    matrix[0, 0] = 2.0
    with pytest.raises(ValueError, match="not orthogonal"):
        O3Transformation(
            matrix,
            max_angular_momentum=0,
        )


@pytest.mark.parametrize("is_improper", [False, True])
def test_create_from_internal_matrix_matches_constructor(is_improper):
    matrix = _rotation_90_degrees_around_z()
    if is_improper:
        matrix = -matrix

    expected = O3Transformation(
        matrix,
        max_angular_momentum=2,
    )
    actual = O3Transformation._create_from_internal_matrix(
        matrix,
        max_angular_momentum=2,
        is_improper=is_improper,
    )

    torch.testing.assert_close(actual.matrix, expected.matrix)
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert actual.is_improper == expected.is_improper

    for ell in range(3):
        torch.testing.assert_close(
            actual.wigner_D_matrix(ell),
            expected.wigner_D_matrix(ell),
        )


def test_constructor_copies_input_matrix():
    expected = torch.eye(3, dtype=torch.float64)
    source = expected.clone()
    transformation = O3Transformation(
        source,
        max_angular_momentum=0,
    )

    source[:] = -expected
    assert torch.equal(transformation.matrix, expected)
    assert transformation.is_improper is False
    assert torch.equal(
        transformation.transform_cartesian(expected),
        expected,
    )


def test_wigner_cache_is_built_only_when_needed():
    transformation = O3Transformation(
        torch.eye(3, dtype=torch.float64),
        max_angular_momentum=2,
    )
    assert transformation._wigner_D_cache is None

    transformation.transform_cartesian(
        torch.ones((4, 3), dtype=torch.float64),
    )
    assert transformation._wigner_D_cache is None

    transformation.wigner_D_matrix(1)
    cache = transformation._wigner_D_cache
    assert cache is not None

    transformation.wigner_D_matrix(2)
    assert transformation._wigner_D_cache is cache


@pytest.mark.parametrize(
    ("ell", "error"),
    [
        (-1, ValueError),
        (True, TypeError),
        (1.0, TypeError),
    ],
)
def test_wigner_D_matrix_and_transform_spherical_reject_invalid_ell(ell, error):
    transformation = O3Transformation(torch.eye(3, dtype=torch.float64), 1)

    with pytest.raises(error, match="ell must be a non-negative integer"):
        transformation.wigner_D_matrix(ell)

    with pytest.raises(error, match="ell must be a non-negative integer"):
        transformation.transform_spherical(
            torch.ones((1, 3), dtype=torch.float64),
            ell=ell,
            sigma=1,
        )


def test_wigner_D_matrix_and_transform_spherical_reject_ell_above_configured_maximum():
    transformation = O3Transformation(torch.eye(3, dtype=torch.float64), 0)

    with pytest.raises(ValueError, match="ell=1 exceeds max_angular_momentum=0"):
        transformation.wigner_D_matrix(1)

    with pytest.raises(ValueError, match="ell=1 exceeds max_angular_momentum=0"):
        transformation.transform_spherical(
            torch.ones((1, 3), dtype=torch.float64),
            ell=1,
            sigma=1,
        )


@pytest.mark.parametrize(
    ("matrix", "is_improper"),
    [
        (torch.eye(3), False),
        (-torch.eye(3), True),
    ],
)
@pytest.mark.parametrize("ell", [0, 1])
@pytest.mark.parametrize("sigma", [1, -1])
def test_transform_spherical_applies_o3_parity_factor(
    matrix,
    is_improper,
    ell,
    sigma,
):
    matrix = matrix.to(dtype=torch.float64)
    transformation = O3Transformation(
        matrix,
        max_angular_momentum=ell,
    )
    values = torch.arange(
        1,
        2 * (2 * ell + 1) + 1,
        dtype=torch.float64,
    ).reshape(2, 2 * ell + 1)

    parity_factor = sigma * (-1) ** ell if is_improper else 1
    expected = values * parity_factor

    assert torch.allclose(
        transformation.transform_spherical(values, ell, sigma),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        torch.eye(3),
        -torch.eye(3),
    ],
)
@pytest.mark.parametrize(
    ("sigma", "error"),
    [
        (2, ValueError),
        (-2, ValueError),
        (1.0, TypeError),
        (True, TypeError),
    ],
)
def test_transform_spherical_rejects_invalid_sigma(matrix, sigma, error):
    transformation = O3Transformation(
        matrix.to(dtype=torch.float64),
        max_angular_momentum=0,
    )

    with pytest.raises(
        error,
        match=r"sigma must be either -1 or \+1",
    ):
        transformation.transform_spherical(
            torch.ones((1, 1), dtype=torch.float64),
            ell=0,
            sigma=sigma,
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


@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_build_wigner_D_cache_returns_all_ranks_on_requested_dtype_and_device(
    device,
    dtype,
):
    """Return every rank on the requested dtype and device.

    A proper rotation and its improper counterpart must produce equal caches.
    """
    device = torch.device(device)
    dtype = getattr(torch, dtype)
    ell_max = 3
    proper = torch.tensor(
        _axis_angle([1.0, 2.0, 3.0], 0.7),
        device=device,
        dtype=dtype,
    )

    proper_cache = build_wigner_D_cache(
        ell_max,
        proper,
        device=device,
        dtype=dtype,
    )
    improper_cache = build_wigner_D_cache(
        ell_max,
        -proper,
        device=device,
        dtype=dtype,
    )

    expected_ranks = set(range(ell_max + 1))
    for cache in (proper_cache, improper_cache):
        assert set(cache) == expected_ranks
        for ell, matrix in cache.items():
            assert matrix.shape == (2 * ell + 1, 2 * ell + 1)
            assert matrix.dtype == dtype
            assert matrix.device == proper.device

    for ell in range(ell_max + 1):
        assert torch.equal(proper_cache[ell], improper_cache[ell])


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("is_improper", [False, True])
@pytest.mark.parametrize(
    "cos_beta",
    [
        pytest.param(1.0, id="beta-zero"),
        pytest.param(-1.0, id="beta-pi"),
    ],
)
def test_wigner_D_matrix_handles_roundoff_at_both_euler_poles(
    cos_beta,
    is_improper,
    dtype,
):
    """One-ULP matrix roundoff must preserve rotations at beta=0 and beta=pi."""
    alpha = 0.73
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
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
    if is_improper:
        matrix = -matrix

    transformation = O3Transformation(matrix, max_angular_momentum=1)
    basis = CARTESIAN_TO_SPHERICAL_L1.to(dtype=dtype)
    expected = basis @ proper @ basis.T
    actual = transformation.wigner_D_matrix(1)

    atol = 1e-6 if dtype == torch.float32 else 1e-14
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("matrix", TRANSFORMATIONS)
@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_spherical_rotation_matches_cartesian(matrix, device, dtype):
    dtype = getattr(torch, dtype)
    atol = 1e-6 if dtype == torch.float32 else 1e-12

    matrix = matrix.to(device=device, dtype=dtype)

    system = _make_system([1, 1], device=device, dtype=dtype)

    cartesian_vectors = torch.randn(2, 3, 1, dtype=dtype, device=device)

    samples = Labels(
        ["system", "atom"],
        torch.tensor([[0, 0], [0, 1]], device=device),
    )
    cartesian = _single_block_tensor_map(
        values=cartesian_vectors,
        samples=samples,
        components=[Labels(["xyz"], torch.arange(3, device=device).reshape(-1, 1))],
    )
    # spherical encoding: w = C @ v along the component axis
    cart_to_sph = CARTESIAN_TO_SPHERICAL_L1.to(device=device, dtype=dtype)
    spherical_values = torch.einsum(
        "Aa,iap->iAp", cart_to_sph, cartesian_vectors
    ).requires_grad_()
    spherical = _single_block_tensor_map(
        keys=Labels(
            ["o3_lambda", "o3_sigma"],
            torch.tensor([[1, 1]], device=device),
        ),
        values=spherical_values,
        samples=samples,
        components=[
            Labels(["o3_mu"], torch.arange(-1, 2, device=device).reshape(-1, 1))
        ],
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


def test_transform_tensor_routes_gradient_rows_by_parent_system():
    """Gradients follow parent systems while metadata and inputs are preserved."""
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

    # Position-gradient samples 0 and 2 use R_92; samples 1, 3, and 4 use R_38.
    expected_pos = pos_grad_before.clone()
    expected_pos[0] = R_92 @ pos_grad_before[0]
    expected_pos[1] = R_38 @ pos_grad_before[1]
    expected_pos[2] = R_92 @ pos_grad_before[2]
    expected_pos[3] = R_38 @ pos_grad_before[3]
    expected_pos[4] = R_38 @ pos_grad_before[4]

    transformed_pos = transformed_block.gradient("positions").values
    assert torch.allclose(transformed_pos, expected_pos)

    # Strain-gradient samples 0 and 1 use R_92 and R_38, respectively.
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

    assert transformed_block.samples == input_block.samples
    assert transformed_block.components == input_block.components
    assert transformed_block.properties == input_block.properties
    assert transformed_block.gradients_list() == input_block.gradients_list()
    for gradient_name in input_block.gradients_list():
        transformed_gradient = transformed_block.gradient(gradient_name)
        input_gradient = input_block.gradient(gradient_name)
        assert transformed_gradient.samples == input_gradient.samples
        assert transformed_gradient.components == input_gradient.components
        assert transformed_gradient.properties == input_gradient.properties

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


def test_unsupported_component_axis_raises():
    """Unknown component-axis names fail loudly."""
    systems = [_make_system([1])]
    tensor = _single_block_tensor_map(
        values=torch.zeros(1, 3, 1, dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[0]])),
        components=[Labels(["direction"], torch.arange(3).reshape(-1, 1))],
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


def test_transform_tensor_rejects_invalid_spherical_sigma():
    """TensorMap transformation rejects invalid spherical parity metadata."""
    tensor = _single_block_tensor_map(
        keys=Labels(
            ["o3_lambda", "o3_sigma"],
            torch.tensor([[0, 2]]),
        ),
        values=torch.ones((1, 1, 1), dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[0]])),
        components=[Labels(["o3_mu"], torch.tensor([[0]]))],
    )

    with pytest.raises(
        ValueError,
        match=re.escape("sigma must be either -1 or +1"),
    ):
        transform_tensor(
            tensor,
            [_make_system([1])],
            [
                O3Transformation(
                    torch.eye(3, dtype=torch.float64),
                    max_angular_momentum=0,
                )
            ],
        )


@pytest.mark.parametrize(
    ("location", "component", "keys", "message"),
    [
        pytest.param(
            "values",
            Labels(["xyz"], torch.tensor([[2], [0], [1]])),
            Labels(["_"], torch.tensor([[0]])),
            "Cartesian component axis 'xyz' must use labels",
            id="value-cartesian-label-order",
        ),
        pytest.param(
            "values",
            Labels(["o3_mu"], torch.tensor([[1], [-1], [0]])),
            Labels(
                ["o3_lambda", "o3_sigma"],
                torch.tensor([[1, 1]]),
            ),
            "Spherical component axis 'o3_mu' for ell=1 must use labels",
            id="value-spherical-label-order",
        ),
        pytest.param(
            "gradient",
            Labels(["xyz"], torch.tensor([[2], [0], [1]])),
            Labels(["_"], torch.tensor([[0]])),
            "Cartesian component axis 'xyz' must use labels",
            id="gradient-cartesian-label-order",
        ),
        pytest.param(
            "values",
            Labels(["o3_mu"], torch.tensor([[0]])),
            Labels(
                ["o3_lambda", "o3_sigma"],
                torch.tensor([[-1, 1]]),
            ),
            "ell must be a non-negative integer",
            id="negative-spherical-angular-momentum",
        ),
    ],
)
def test_transform_tensor_validates_empty_component_metadata(
    location,
    component,
    keys,
    message,
):
    """Empty values and gradients still require valid component metadata."""
    if location == "values":
        block = TensorBlock(
            values=torch.empty(
                (0, len(component), 1),
                dtype=torch.float64,
            ),
            samples=Labels(
                ["system"],
                torch.empty((0, 1), dtype=torch.int64),
            ),
            components=[component],
            properties=Labels(["p"], torch.tensor([[0]])),
        )
    else:
        block = TensorBlock(
            values=torch.ones((1, 1), dtype=torch.float64),
            samples=Labels(["system"], torch.tensor([[0]])),
            components=[],
            properties=Labels(["p"], torch.tensor([[0]])),
        )
        block.add_gradient(
            "positions",
            TensorBlock(
                values=torch.empty(
                    (0, len(component), 1),
                    dtype=torch.float64,
                ),
                samples=Labels(
                    ["sample"],
                    torch.empty((0, 1), dtype=torch.int64),
                ),
                components=[component],
                properties=block.properties,
            ),
        )

    tensor = TensorMap(keys, [block])

    with pytest.raises(ValueError, match=re.escape(message)):
        transform_tensor(
            tensor,
            [_make_system([1])],
            [
                O3Transformation(
                    torch.eye(3, dtype=torch.float64),
                    max_angular_momentum=1,
                )
            ],
        )


@pytest.mark.parametrize("sigma_9", [1, -1])
def test_transform_tensor_combines_spherical_axis_parities(sigma_9):
    """Improper parity factors multiply across spherical component axes."""
    values = torch.arange(
        1,
        10,
        dtype=torch.float64,
    ).reshape(1, 3, 3, 1)
    tensor = _single_block_tensor_map(
        keys=Labels(
            [
                "o3_lambda_1",
                "o3_lambda_9",
                "o3_sigma_1",
                "o3_sigma_9",
            ],
            torch.tensor([[1, 1, 1, sigma_9]]),
        ),
        values=values,
        samples=Labels(["system"], torch.tensor([[0]])),
        components=[
            Labels(
                ["o3_mu_1"],
                torch.arange(-1, 2).reshape(-1, 1),
            ),
            Labels(
                ["o3_mu_9"],
                torch.arange(-1, 2).reshape(-1, 1),
            ),
        ],
    )

    transformed = transform_tensor(
        tensor,
        [_make_system([1])],
        [
            O3Transformation(
                -torch.eye(3, dtype=torch.float64),
                max_angular_momentum=1,
            )
        ],
    )

    assert torch.allclose(
        transformed.block().values,
        sigma_9 * values,
        rtol=0.0,
        atol=1e-12,
    )


def test_transform_tensor_enforces_ten_component_axis_limit():
    """Ten component axes are transformed and an eleventh is rejected."""
    # The first spherical axis is unsuffixed by convention; the nine axes below
    # use `o3_mu` through `o3_mu_8`, followed by Cartesian `xyz_9`.
    spherical_suffixes = [""] + [f"_{index}" for index in range(1, 9)]
    components = [
        Labels([f"o3_mu{suffix}"], torch.tensor([[0]])) for suffix in spherical_suffixes
    ]
    components.append(Labels(["xyz_9"], torch.arange(3).reshape(-1, 1)))

    key_names = [f"o3_lambda{suffix}" for suffix in spherical_suffixes] + [
        f"o3_sigma{suffix}" for suffix in spherical_suffixes
    ]
    values = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float64,
    ).reshape([1] + [1] * 9 + [3, 1])
    tensor = _single_block_tensor_map(
        keys=Labels(key_names, torch.tensor([[0] * 9 + [1] * 9])),
        values=values,
        samples=Labels(["system"], torch.tensor([[0]])),
        components=components,
    )

    system = _make_system([1])
    transformation = O3Transformation(
        -torch.eye(3, dtype=torch.float64),
        max_angular_momentum=0,
    )
    transformed = transform_tensor(
        tensor,
        [system],
        [transformation],
    )
    assert torch.equal(transformed.block().values, -values)

    too_many_components = components + [Labels(["xyz"], torch.arange(3).reshape(-1, 1))]
    too_many_tensor = _single_block_tensor_map(
        keys=tensor.keys,
        values=torch.zeros(
            [1] + [1] * 9 + [3, 3, 1],
            dtype=torch.float64,
        ),
        samples=tensor.block().samples,
        components=too_many_components,
        properties=tensor.block().properties,
    )

    with pytest.raises(
        ValueError,
        match="can not transform a tensor with 11 component axes; at most 10",
    ):
        transform_tensor(
            too_many_tensor,
            [system],
            [transformation],
        )


def _system_ids_inputs():
    systems = [_make_system([1]), _make_system([8])]
    transformations = [
        O3Transformation(torch.eye(3, dtype=torch.float64), 0),
        O3Transformation(torch.eye(3, dtype=torch.float64), 0),
    ]
    return systems, transformations


@pytest.mark.parametrize(
    ("system_ids", "sample_system_ids"),
    [
        pytest.param(None, [0, 1], id="default"),
        pytest.param([92, 38], [92, 38], id="python-list"),
        pytest.param(
            torch.tensor([92, -7], dtype=torch.int32),
            [92, -7],
            id="integer-tensor",
        ),
    ],
)
@pytest.mark.parametrize("entry_point", ["block", "tensor"])
def test_transform_block_and_tensor_route_rows_by_system_id(
    entry_point,
    system_ids,
    sample_system_ids,
):
    """Default, list, and tensor IDs route rows through ``transform_block`` and
    ``transform_tensor``.
    """
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
    block = TensorBlock(
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

    kwargs = {} if system_ids is None else {"system_ids": system_ids}
    keys = Labels(["_"], torch.tensor([[0]]))
    if entry_point == "block":
        transformed = transform_block(
            keys[0],
            block,
            systems,
            transformations,
            **kwargs,
        )
    else:
        transformed = transform_tensor(
            TensorMap(keys, [block]),
            systems,
            transformations,
            **kwargs,
        ).block()

    expected = torch.tensor(
        [
            [[-1.0], [0.0], [0.0]],
            [[-2.0], [0.0], [0.0]],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(transformed.values, expected)


@pytest.mark.parametrize(
    ("system_ids", "message"),
    [
        pytest.param([92], "exactly one entry per system", id="wrong-count"),
        pytest.param([92, 92], "one distinct entry per system", id="duplicate"),
        pytest.param(
            torch.tensor([[92, -7]]),
            "system_ids must be one-dimensional",
            id="two-dimensional-tensor",
        ),
        pytest.param(
            torch.tensor([92.0, -7.0]),
            "system_ids must contain integers",
            id="floating-point-tensor",
        ),
        pytest.param(
            [92.0, -7.0],
            "system_ids must contain integers",
            id="floating-point-list",
        ),
        pytest.param(
            [True, False],
            "system_ids must contain integers",
            id="boolean-list",
        ),
    ],
)
def test_validate_system_ids_rejects_invalid_ids(system_ids, message):
    """Reject ambiguous or structurally invalid system identifiers."""
    systems, transformations = _system_ids_inputs()

    with pytest.raises(ValueError, match=message):
        _validate_system_ids(
            systems,
            transformations,
            system_ids,
            expected_device=torch.device("cpu"),
        )


def test_validate_system_ids_requires_one_transformation_per_system():
    """Each system must have exactly one transformation."""
    systems, transformations = _system_ids_inputs()

    with pytest.raises(ValueError, match="Expected one transformation per system"):
        _validate_system_ids(
            systems,
            transformations[:1],
            [92, -7],
            expected_device=torch.device("cpu"),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_validate_system_ids_checks_device_only_when_defined():
    """Reject device mismatches, while allowing IDs when no device is specified."""
    systems, transformations = _system_ids_inputs()

    with pytest.raises(
        ValueError,
        match="system_ids are on device cpu, but the values to transform are on device",
    ):
        _validate_system_ids(
            systems,
            transformations,
            torch.tensor([92, -7]),
            expected_device=torch.device("cuda"),
        )

    empty_cuda_ids = torch.empty(0, dtype=torch.long, device="cuda")
    validated = _validate_system_ids(
        [],
        [],
        empty_cuda_ids,
        expected_device=None,
    )
    assert validated.device == empty_cuda_ids.device


@pytest.mark.parametrize("entry_point", ["block", "tensor"])
@pytest.mark.parametrize(
    ("dtype", "device"),
    [
        pytest.param(torch.float32, torch.device("cpu"), id="dtype"),
        pytest.param(
            torch.float64,
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available",
            ),
            id="device",
        ),
    ],
)
def test_transform_block_and_tensor_reject_mismatched_transformation_with_no_rows(
    entry_point,
    dtype,
    device,
):
    """Every transformation must match the values even when it has no rows."""
    block = TensorBlock(
        values=torch.ones((1, 1), dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[92]])),
        components=[],
        properties=Labels(["p"], torch.tensor([[0]])),
    )
    systems, transformations = _system_ids_inputs()
    transformations[1] = O3Transformation(
        torch.eye(3, dtype=dtype, device=device),
        max_angular_momentum=0,
    )

    with pytest.raises(
        ValueError,
        match="Transformation at index 1 has dtype/device",
    ):
        if entry_point == "block":
            transform_block(
                Labels(["_"], torch.tensor([[0]]))[0],
                block,
                systems,
                transformations,
                system_ids=[92, 38],
            )
        else:
            transform_tensor(
                TensorMap(Labels(["_"], torch.tensor([[0]])), [block]),
                systems,
                transformations,
                system_ids=[92, 38],
            )


def test_transform_block_leaves_empty_block_unchanged_without_systems():
    """An empty block remains unchanged when no transformations are requested."""
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
    assert transformed.components == block.components
    assert transformed.properties == block.properties


def test_transform_tensor_preserves_empty_map_information():
    """An empty TensorMap retains its information without imposing a values dtype."""
    empty = TensorMap(
        Labels(
            ["_"],
            torch.empty((0, 1), dtype=torch.int64),
        ),
        [],
    )
    empty.set_info("unit", "eV")
    transformed = transform_tensor(
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

    assert len(transformed) == 0
    assert transformed.keys == empty.keys
    assert transformed.info() == empty.info()


@pytest.mark.parametrize(
    ("kwargs", "unknown_labels", "expected_system_ids"),
    [
        pytest.param(
            {"system_ids": [92, 99]},
            [38],
            [92, 99],
            id="one-unknown-label",
        ),
        pytest.param(
            {},
            [38, 92],
            [0, 1],
            id="default-system-ids",
        ),
    ],
)
def test_transform_tensor_rejects_unknown_system_labels(
    kwargs,
    unknown_labels,
    expected_system_ids,
):
    """Every value row must match one of the requested systems."""
    systems = [_make_system([1, 8]), _make_system([1, 8])]
    transformation = O3Transformation(torch.eye(3, dtype=torch.float64), 1)

    tensor = _single_block_tensor_map(
        values=torch.zeros(2, 3, 1, dtype=torch.float64),
        samples=Labels(
            ["system", "atom"],
            torch.tensor([[92, 0], [38, 0]]),
        ),
        components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
    )

    message = (
        f"Block samples contain system labels {unknown_labels} that are not in "
        f"system_ids={expected_system_ids}. Every sample must be assigned to a "
        "system in the transformation."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        transform_tensor(
            tensor,
            systems,
            [transformation, transformation],
            **kwargs,
        )


@pytest.mark.parametrize(
    "samples",
    [
        pytest.param(
            Labels(["atom"], torch.tensor([[0], [1]])),
            id="without-system-label",
        ),
        pytest.param(
            Labels(
                ["system", "atom"],
                torch.tensor([[4, 0], [4, 1]]),
            ),
            id="unrelated-system-label",
        ),
    ],
)
def test_transform_tensor_routes_all_rows_for_single_system(samples):
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
    tensor = _single_block_tensor_map(
        values=values,
        samples=samples,
        components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
    )

    transformed = transform_tensor(tensor, systems, [transformation])

    expected = torch.tensor(
        [
            [[-2.0], [1.0], [3.0]],
            [[-4.0], [-1.0], [0.0]],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(transformed.block().values, expected)


def test_transform_tensor_accepts_block_with_subset_of_systems():
    """Transform a block whose rows belong to only a subset of the input systems."""
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
        # The second system has no samples in this block. Its ID exceeds the
        # int32 range, checking that Python IDs are preserved as torch.long.
        system_ids=[92, 2**40],
    )

    expected = torch.tensor(
        [[[-2.0], [1.0], [3.0]]],
        dtype=torch.float64,
    )
    assert torch.equal(transformed.block().values, expected)


def test_transform_tensor_rejects_missing_system_column_for_multiple_systems():
    """With multiple systems, each row must identify which system it belongs to."""
    systems = [_make_system([1]), _make_system([8])]
    transformation = O3Transformation(
        torch.eye(3, dtype=torch.float64),
        max_angular_momentum=0,
    )

    tensor = _single_block_tensor_map(
        values=torch.zeros((1, 1), dtype=torch.float64),
        samples=Labels(["atom"], torch.tensor([[0]])),
        components=[],
    )

    message = (
        "Rotational augmentation expects output samples to include a 'system' "
        "dimension when transforming multiple systems."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        transform_tensor(
            tensor,
            systems,
            [transformation, transformation],
        )
