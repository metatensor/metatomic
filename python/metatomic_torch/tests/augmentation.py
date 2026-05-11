import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import System
from metatomic.torch._augmentation import _apply_augmentations
from metatomic.torch.symmetrized_model import _compute_wigner_batch


def _make_system(types):
    n_atoms = len(types)
    return System(
        types=torch.tensor(types, dtype=torch.int32),
        positions=torch.zeros((n_atoms, 3), dtype=torch.float64),
        cell=torch.zeros((3, 3), dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
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
        for ell, matrix in _compute_wigner_batch(
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


def test_sparse_atomic_basis_rank1_augmentation():
    systems = [_make_system([1, 8]), _make_system([1, 8])]
    transformations, wigner_D_matrices = _rotation_batch([np.pi / 2, np.pi])

    component = Labels(
        ["o3_mu"],
        torch.arange(-1, 2, dtype=torch.int32).reshape(-1, 1),
    )
    property_labels = Labels(["n"], torch.tensor([[0]], dtype=torch.int32))
    tensor = TensorMap(
        Labels(
            ["o3_lambda", "o3_sigma", "atom_type"],
            torch.tensor([[1, 1, 1], [1, 1, 8]], dtype=torch.int32),
        ),
        [
            TensorBlock(
                values=torch.tensor(
                    [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
                    dtype=torch.float64,
                ),
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, 0], [1, 0]], dtype=torch.int32),
                ),
                components=[component],
                properties=property_labels,
            ),
            TensorBlock(
                values=torch.tensor(
                    [[[7.0], [8.0], [9.0]], [[10.0], [11.0], [12.0]]],
                    dtype=torch.float64,
                ),
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, 1], [1, 1]], dtype=torch.int32),
                ),
                components=[component],
                properties=property_labels,
            ),
        ],
    )

    _, augmented_targets, _ = _apply_augmentations(
        systems,
        {"target": tensor},
        transformations,
        wigner_D_matrices,
    )
    augmented = augmented_targets["target"]

    expected_blocks = []
    for block in tensor.blocks():
        expected_values = block.values.clone()
        for rows, wigner_D_matrix in zip(
            _row_indices(block.samples, len(systems)),
            wigner_D_matrices[1],
            strict=True,
        ):
            rotated = block.values[rows].clone().transpose(1, 2)
            rotated = rotated @ wigner_D_matrix.T
            expected_values[rows] = rotated.transpose(1, 2)

        expected_blocks.append(
            TensorBlock(
                values=expected_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )

    expected = TensorMap(tensor.keys, expected_blocks)
    assert augmented.keys == expected.keys
    for block_id in range(len(expected.keys)):
        assert (
            augmented.block_by_id(block_id).samples
            == expected.block_by_id(block_id).samples
        )
        assert torch.allclose(
            augmented.block_by_id(block_id).values,
            expected.block_by_id(block_id).values,
            atol=1e-12,
        )


def test_sparse_atomic_basis_rank2_augmentation_with_missing_system_rows():
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

    _, augmented_targets, _ = _apply_augmentations(
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
