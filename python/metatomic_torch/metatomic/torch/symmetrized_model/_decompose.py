from typing import Dict, List

import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def _l0_components_from_matrices(A: torch.Tensor) -> torch.Tensor:
    """
    Extract the L=0 component (trace) from rank-2 Cartesian blocks.

    Expects ``A`` with shape ``(a, 3, 3, b)``; returns ``(a, 1, b)``.
    """
    assert A.shape[1:3] == (3, 3), "Cartesian component axes must have size (3, 3)."
    return (A[:, 0, 0, :] + A[:, 1, 1, :] + A[:, 2, 2, :]).unsqueeze(1)


def _l2_components_from_matrices(A: torch.Tensor) -> torch.Tensor:
    """
    Extract the L=2 components (symmetric traceless part) from rank-2 Cartesian blocks.

    Expects ``A`` with shape ``(a, 3, 3, b)``; returns ``(a, 5, b)``.
    """
    assert A.shape[1:3] == (3, 3), "Cartesian component axes must have size (3, 3)."
    return torch.stack(
        [
            (A[:, 0, 1, :] + A[:, 1, 0, :]) / 2.0,
            (A[:, 1, 2, :] + A[:, 2, 1, :]) / 2.0,
            (2.0 * A[:, 2, 2, :] - A[:, 0, 0, :] - A[:, 1, 1, :])
            / (2.0 * np.sqrt(3.0)),
            (A[:, 0, 2, :] + A[:, 2, 0, :]) / 2.0,
            (A[:, 0, 0, :] - A[:, 1, 1, :]) / 2.0,
        ],
        dim=1,
    )


def _single_mu_labels(device: torch.device) -> Labels:
    return Labels(
        names=["o3_mu"],
        values=torch.tensor([[0]], device=device, dtype=torch.int32),
    )


def _mu_range_labels(ell: int, device: torch.device) -> Labels:
    return Labels(
        names=["o3_mu"],
        values=torch.arange(-ell, ell + 1, device=device, dtype=torch.int32).reshape(
            -1, 1
        ),
    )


def _keys_with_irrep(keys: Labels, ell: int, sigma: int = 1) -> Labels:
    """Add or validate the key metadata required by an ``o3_mu`` axis."""
    names = list(keys.names)
    values = keys.values
    if names == ["_"]:
        # ``_`` is metatensor's placeholder for a map with no semantic keys.
        # Generated spherical targets use the canonical irrep-only schema.
        if len(keys) != 1 or int(values[0, 0]) != 0:
            raise ValueError(
                "the sole '_' key must be the canonical one-row, zero-valued "
                "placeholder before spherical decomposition"
            )
        names = []
        values = values[:, :0]
    for name, expected in (("o3_lambda", ell), ("o3_sigma", sigma)):
        if name in names:
            column = keys.column(name)
            if not bool(torch.all(column == expected).item()):
                raise ValueError(
                    f"can not decompose output as ({ell}, {sigma}): existing "
                    f"'{name}' key values are inconsistent"
                )
        else:
            names.append(name)
            values = torch.cat(
                [
                    values,
                    torch.full(
                        (len(keys), 1),
                        expected,
                        dtype=values.dtype,
                        device=values.device,
                    ),
                ],
                dim=1,
            )
    return Labels(names, values)


def _decomposed_output_names(name: str) -> List[str]:
    """Names produced by :func:`_decompose_output`, without inspecting values."""
    quantity = name.split("/", 1)[0]
    if quantity in ("energy", "energy_ensemble", "energy_uncertainty"):
        return [name + "_l0"]
    if quantity in (
        "forces",
        "non_conservative_force",
        "non_conservative_forces",
    ):
        return [name + "_l1"]
    if quantity in ("stress", "non_conservative_stress"):
        return [name + "_l0", name + "_l2"]
    return [name]


def _decompose_output(name: str, tensor: TensorMap) -> Dict[str, TensorMap]:
    """
    Decompose a single model output into irreducible representations of O(3).

    Energies are relabeled as ``<name>_l0`` scalars, gaining a single m=0
    ``o3_mu`` component axis for consistency with higher-order decompositions.
    Forces (Cartesian vectors) become ``<name>_l1``, with the components
    reordered to spherical order (y, z, x) -> (m=-1, m=0, m=1) via a cyclic
    roll. Stresses (3x3 Cartesian matrices) are split into ``<name>_l0``
    (trace) and ``<name>_l2`` (symmetric traceless) parts. Physical stress is
    assumed symmetric; any skew input component is intentionally omitted
    rather than returned as an L=1 diagnostic. Any other output is passed
    through unchanged.

    :param name: name of the output
    :param tensor: the output values
    :return: dictionary of decomposed TensorMaps, keyed by the new names
    """
    quantity = name.split("/", 1)[0]

    if quantity in ("energy", "energy_ensemble", "energy_uncertainty"):
        return {
            name + "_l0": TensorMap(
                _keys_with_irrep(tensor.keys, 0),
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

    if quantity in (
        "forces",
        "non_conservative_force",
        "non_conservative_forces",
    ):
        return {
            name + "_l1": TensorMap(
                _keys_with_irrep(tensor.keys, 1),
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

    if quantity in ("stress", "non_conservative_stress"):
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
            name + "_l0": TensorMap(_keys_with_irrep(tensor.keys, 0), blocks_l0),
            name + "_l2": TensorMap(_keys_with_irrep(tensor.keys, 2), blocks_l2),
        }

    return {name: tensor}
