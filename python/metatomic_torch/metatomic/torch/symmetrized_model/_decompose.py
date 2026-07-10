from typing import Dict

import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def _l0_components_from_matrices(A: torch.Tensor) -> torch.Tensor:
    """
    Extract the L=0 component (trace) from rank-2 Cartesian blocks.

    Expects ``A`` with shape ``(a, 3, 3, b)``; returns ``(a, 1, b)``.
    """
    # move (3, 3) axes to the end for the assert and indexing below
    A = A.permute(0, 3, 1, 2)
    assert A.shape[-2:] == (3, 3), "The last two dimensions of A must be (3, 3)."

    # Trace as L=0 component; unsqueeze preserves the autograd graph
    l0_A = (A[..., 0, 0] + A[..., 1, 1] + A[..., 2, 2]).unsqueeze(-1)

    l0_A = l0_A.permute(0, 2, 1)
    return l0_A


def _l2_components_from_matrices(A: torch.Tensor) -> torch.Tensor:
    """
    Extract the L=2 components (symmetric traceless part) from rank-2 Cartesian blocks.

    Expects ``A`` with shape ``(a, 3, 3, b)``; returns ``(a, 5, b)``.
    """
    # move (3, 3) axes to the end for the assert and indexing below
    A = A.permute(0, 3, 1, 2)
    assert A.shape[-2:] == (3, 3), "The last two dimensions of A must be (3, 3)."

    # Use torch.stack to preserve the autograd graph
    l2_A = torch.stack(
        [
            (A[..., 0, 1] + A[..., 1, 0]) / 2.0,
            (A[..., 1, 2] + A[..., 2, 1]) / 2.0,
            (2.0 * A[..., 2, 2] - A[..., 0, 0] - A[..., 1, 1]) / (2.0 * np.sqrt(3.0)),
            (A[..., 0, 2] + A[..., 2, 0]) / 2.0,
            (A[..., 0, 0] - A[..., 1, 1]) / 2.0,
        ],
        dim=-1,
    )

    l2_A = l2_A.permute(0, 2, 1)

    return l2_A


def _single_mu_labels(device: torch.device) -> Labels:
    return Labels(
        names=["o3_mu"],
        values=torch.tensor([[0]], device=device, dtype=torch.int32),
    )


def _mu_range_labels(ell: int, device: torch.device) -> Labels:
    return Labels(
        names=["o3_mu"],
        values=torch.tensor(
            [[mu] for mu in range(-ell, ell + 1)],
            device=device,
            dtype=torch.int32,
        ),
    )


def _decompose_output(name: str, tensor: TensorMap) -> Dict[str, TensorMap]:
    """
    Decompose a single model output into irreducible representations of O(3).

    Energies are relabeled as ``<name>_l0`` scalars, gaining a single m=0
    ``o3_mu`` component axis for consistency with higher-order decompositions.
    Forces (Cartesian vectors) become ``<name>_l1``, with the components
    reordered to spherical order (y, z, x) -> (m=-1, m=0, m=1) via a cyclic
    roll. Stresses (3x3 Cartesian matrices) are split into ``<name>_l0``
    (trace) and ``<name>_l2`` (symmetric traceless) parts; the antisymmetric
    L=1 part is zero for physical stress. Any other output is passed through
    unchanged.

    :param name: name of the output
    :param tensor: the output values
    :return: dictionary of decomposed TensorMaps, keyed by the new names
    """
    if name == "energy":
        return {
            name + "_l0": TensorMap(
                tensor.keys,
                [
                    TensorBlock(
                        values=block.values.unsqueeze(1),
                        samples=block.samples,
                        components=[_single_mu_labels(block.values.device)],
                        properties=block.properties,
                    )
                    for block in tensor
                ],
            )
        }

    if name in ("forces", "non_conservative_force", "non_conservative_forces"):
        return {
            name + "_l1": TensorMap(
                tensor.keys,
                [
                    TensorBlock(
                        values=block.values.roll(-1, 1),
                        samples=block.samples,
                        components=[_mu_range_labels(1, block.values.device)],
                        properties=block.properties,
                    )
                    for block in tensor
                ],
            )
        }

    if name in ("stress", "non_conservative_stress"):
        blocks_l0 = []
        blocks_l2 = []
        for block in tensor.blocks():
            blocks_l0.append(
                TensorBlock(
                    values=_l0_components_from_matrices(block.values),
                    samples=block.samples,
                    components=[_single_mu_labels(block.values.device)],
                    properties=block.properties,
                )
            )
            blocks_l2.append(
                TensorBlock(
                    values=_l2_components_from_matrices(block.values),
                    samples=block.samples,
                    components=[_mu_range_labels(2, block.values.device)],
                    properties=block.properties,
                )
            )
        return {
            name + "_l0": TensorMap(tensor.keys, blocks_l0),
            name + "_l2": TensorMap(tensor.keys, blocks_l2),
        }

    return {name: tensor}
