import re

import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    NeighborListOptions,
    System,
)
from metatomic.torch.o3 import (
    O3Transformation,
    random_transformations,
    transform_system,
    transform_tensor,
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

    # Apply a transformation to it
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
    for block_id in range(len(scalar.keys)):
        assert torch.allclose(
            new_scalar.block_by_id(block_id).values,
            scalar.block_by_id(block_id).values,
        )

    new_vector = rotated.get_data("custom::vector").block().values
    expected_vector = (vector_values.squeeze(-1) @ matrix.T).unsqueeze(-1)
    assert torch.allclose(new_vector, expected_vector, atol=atol)


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
    assert torch.allclose(transformation._wigner_D_cache[1], expected, atol=1e-12)


@pytest.mark.parametrize("matrix", TRANSFORMATIONS)
@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_spherical_rotation_matches_cartesian(matrix, device, dtype):
    dtype = getattr(torch, dtype)
    atol = 1e-6 if dtype == torch.float32 else 1e-12

    matrix = matrix.to(device=device, dtype=dtype)

    system = _make_system([1, 1], device=device, dtype=dtype)

    cartesian_vectors = torch.randn(2, 3, 1, dtype=dtype, device=device)

    cartesian = TensorMap(
        Labels(["_"], torch.tensor([[0]], device=device)),
        [
            TensorBlock(
                values=cartesian_vectors,
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, 0], [0, 1]], device=device),
                ),
                components=[
                    Labels(["xyz"], torch.arange(3, device=device).reshape(-1, 1))
                ],
                properties=Labels(["p"], torch.tensor([[0]], device=device)),
            )
        ],
    )
    # spherical encoding: w = C @ v along the component axis
    cart_to_sph = CARTESIAN_TO_SPHERICAL_L1.to(device=device, dtype=dtype)
    spherical_values = torch.einsum("Aa,iap->iAp", cart_to_sph, cartesian_vectors)
    spherical = TensorMap(
        Labels(["o3_lambda", "o3_sigma"], torch.tensor([[1, 1]], device=device)),
        [
            TensorBlock(
                values=spherical_values,
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, 0], [0, 1]], device=device),
                ),
                components=[
                    Labels(["o3_mu"], torch.arange(-1, 2, device=device).reshape(-1, 1))
                ],
                properties=Labels(["p"], torch.tensor([[0]], device=device)),
            )
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
    tensor = TensorMap(
        Labels(["_"], torch.tensor([[0]])),
        [
            TensorBlock(
                values=torch.zeros(1, 3, 1, dtype=torch.float64),
                samples=Labels(["system"], torch.tensor([[0]])),
                components=[Labels(["direction"], torch.arange(3).reshape(-1, 1))],
                properties=Labels(["p"], torch.tensor([[0]])),
            )
        ],
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


def test_system_ids_explicit_match():
    """Explicit ``system_ids`` correctly routes rows with non-contiguous labels."""
    systems = [_make_system([1, 8]), _make_system([1, 8])]

    R0 = O3Transformation(torch.eye(3, dtype=torch.float64), 1)
    # 90-degree rotation around z: (x,y) -> (-y, x)
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    R1 = O3Transformation(
        torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64), 1
    )

    # row 0 (label 92) -> R0 (identity): unchanged
    # row 1 (label 38) -> R1 (z-90): (0,2,0) -> (-2,0,0)
    vector_values = torch.tensor(
        [[[1.0], [0.0], [0.0]], [[0.0], [2.0], [0.0]]],
        dtype=torch.float64,
    )
    tensor = TensorMap(
        Labels(["_"], torch.tensor([[0]])),
        [
            TensorBlock(
                values=vector_values.clone(),
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[92, 0], [38, 0]]),
                ),
                components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
                properties=Labels(["p"], torch.tensor([[0]])),
            )
        ],
    )

    transformed = transform_tensor(tensor, systems, [R0, R1], system_ids=[92, 38])
    result = transformed.block().values
    # row 0 (system 92) rotated by R0 (identity)
    assert torch.allclose(result[0], vector_values[0])
    # row 1 (system 38) rotated by R1 (z-90)
    expected = torch.tensor([[[-2.0], [0.0], [0.0]]], dtype=torch.float64)
    assert torch.allclose(result[1], expected)


def test_system_ids_uncovered_samples_raises():
    """All distinct system indices in samples must appear in ``system_ids``."""
    systems = [_make_system([1, 8]), _make_system([1, 8])]
    R = O3Transformation(torch.eye(3, dtype=torch.float64), 1)

    tensor = TensorMap(
        Labels(["_"], torch.tensor([[0]])),
        [
            TensorBlock(
                values=torch.zeros(2, 3, 1, dtype=torch.float64),
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[92, 0], [38, 0]]),
                ),
                components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
                properties=Labels(["p"], torch.tensor([[0]])),
            )
        ],
    )

    message = (
        "Block samples contain system labels [38, 92] that are not in "
        "system_ids=[99, 100]. Every sample must be assigned to a system "
        "in the transformation."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        transform_tensor(tensor, systems, [R, R], system_ids=[99, 100])

    message = (
        "Block samples contain system labels [38, 92] that are not in "
        "system_ids=[0, 1]. Every sample must be assigned to a system "
        "in the transformation."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        # default system_ids=[0, 1] also misses labels 92 and 38
        transform_tensor(tensor, systems, [R, R])


def test_system_ids_single_system_ignores_label():
    """Single-system blocks return all rows regardless of the ``"system"`` label."""
    systems = [_make_system([1])]
    values = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float64)

    tensor = TensorMap(
        Labels(["_"], torch.tensor([[0]])),
        [
            TensorBlock(
                values=values.clone(),
                samples=Labels(["system", "atom"], torch.tensor([[4, 0]])),
                components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
                properties=Labels(["p"], torch.tensor([[0]])),
            )
        ],
    )

    transformation = O3Transformation(torch.eye(3, dtype=torch.float64), 1)
    transformed = transform_tensor(tensor, systems, [transformation])
    assert torch.allclose(transformed.block().values, values)


def _z_rotation(alpha):
    return torch.tensor(
        [
            [np.cos(alpha), -np.sin(alpha), 0.0],
            [np.sin(alpha), np.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def test_rank2_transform_with_missing_system_rows():
    """Rank-2 (``o3_mu_1``/``o3_mu_2``) blocks rotate correctly when one of the
    systems contributes no rows at all."""
    systems = [_make_system([1, 1]), _make_system([8, 8])]
    T0 = O3Transformation(_z_rotation(np.pi / 2), max_angular_momentum=1)
    T1 = O3Transformation(_z_rotation(np.pi), max_angular_momentum=1)

    components = [
        Labels(["o3_mu_1"], torch.arange(-1, 2, dtype=torch.int32).reshape(-1, 1)),
        Labels(["o3_mu_2"], torch.arange(-1, 2, dtype=torch.int32).reshape(-1, 1)),
    ]
    property_labels = Labels(["n_1", "n_2"], torch.tensor([[0, 0]], dtype=torch.int32))
    values = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3, 1)
    tensor = TensorMap(
        Labels(
            ["o3_lambda_1", "o3_lambda_2", "o3_sigma_1", "o3_sigma_2", "atom_type"],
            torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.int32),
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

    assert transformed.block().samples == tensor.block().samples
    assert torch.allclose(transformed.block().values, expected_values, atol=1e-12)
    assert not torch.allclose(transformed.block().values, values)
