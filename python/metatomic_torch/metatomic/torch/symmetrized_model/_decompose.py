import math
from typing import List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def _o3_mu_labels(o3_lambda: int, device: torch.device) -> Labels:
    """Return ``o3_mu`` labels from ``-o3_lambda`` through ``o3_lambda``."""
    return Labels(
        "o3_mu",
        torch.arange(
            -o3_lambda,
            o3_lambda + 1,
            dtype=torch.int32,
            device=device,
        ).reshape(-1, 1),
    )


def _cartesian_vectors_to_spherical(
    values: torch.Tensor,
    component_axis: int,
) -> torch.Tensor:
    """Reorder ``(x, y, z)`` as ``(mu=-1, 0, 1) = (y, z, x)``."""
    return values.roll(-1, dims=component_axis)


def _symmetric_matrices_to_spherical(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return orthonormal l=0 and l=2 components of the symmetric matrix part."""
    l0 = (values[:, 0, 0, :] + values[:, 1, 1, :] + values[:, 2, 2, :]).unsqueeze(
        1
    ) / math.sqrt(3.0)

    sqrt_two = math.sqrt(2.0)
    l2 = torch.stack(
        [
            (values[:, 0, 1, :] + values[:, 1, 0, :]) / sqrt_two,
            (values[:, 1, 2, :] + values[:, 2, 1, :]) / sqrt_two,
            (2.0 * values[:, 2, 2, :] - values[:, 0, 0, :] - values[:, 1, 1, :])
            / math.sqrt(6.0),
            (values[:, 0, 2, :] + values[:, 2, 0, :]) / sqrt_two,
            (values[:, 0, 0, :] - values[:, 1, 1, :]) / sqrt_two,
        ],
        dim=1,
    )

    return l0, l2


def _decompose_output(
    source_name: str,
    tensor: TensorMap,
) -> TensorMap:
    """Decompose standard outputs for variance and character projection."""
    quantity = source_name.split("/", 1)[0]
    is_energy = quantity in (
        "energy",
        "energy_ensemble",
        "energy_uncertainty",
    )
    is_force = quantity in (
        "non_conservative_force",
        "non_conservative_forces",
    )
    is_stress = quantity == "non_conservative_stress"
    if not (is_energy or is_force or is_stress):
        return tensor

    for block in tensor.blocks():
        if len(block.gradients_list()) != 0:
            raise ValueError(
                "O(3) diagnostic decomposition does not support gradients "
                "attached to '" + source_name + "'"
            )

    if is_energy:
        energy_blocks: List[TensorBlock] = []
        for block in tensor.blocks():
            if len(block.components) != 0:
                raise ValueError("energy-like outputs must not have components")
            energy_blocks.append(
                TensorBlock(
                    values=block.values.unsqueeze(1),
                    samples=block.samples,
                    components=[_o3_mu_labels(0, block.values.device)],
                    properties=block.properties,
                )
            )
        result = TensorMap(
            _add_o3_irrep_to_keys(tensor.keys, 0, 1),
            energy_blocks,
        )

    elif is_force:
        force_blocks: List[TensorBlock] = []
        for block in tensor.blocks():
            if (
                len(block.components) != 1
                or block.components[0].names != ["xyz"]
                or len(block.components[0]) != 3
            ):
                raise ValueError(
                    "non_conservative_force must have one 'xyz' component axis "
                    "of size 3"
                )
            force_blocks.append(
                TensorBlock(
                    values=_cartesian_vectors_to_spherical(block.values, 1),
                    samples=block.samples,
                    components=[_o3_mu_labels(1, block.values.device)],
                    properties=block.properties,
                )
            )
        result = TensorMap(
            _add_o3_irrep_to_keys(tensor.keys, 1, 1),
            force_blocks,
        )

    else:
        blocks_l0: List[TensorBlock] = []
        blocks_l2: List[TensorBlock] = []
        for block in tensor.blocks():
            if (
                len(block.components) != 2
                or block.components[0].names != ["xyz_1"]
                or block.components[1].names != ["xyz_2"]
                or len(block.components[0]) != 3
                or len(block.components[1]) != 3
            ):
                raise ValueError(
                    "non_conservative_stress must have 'xyz_1' and 'xyz_2' "
                    "component axes of size 3"
                )

            values_l0, values_l2 = _symmetric_matrices_to_spherical(block.values)
            blocks_l0.append(
                TensorBlock(
                    values=values_l0,
                    samples=block.samples,
                    components=[_o3_mu_labels(0, block.values.device)],
                    properties=block.properties,
                )
            )
            blocks_l2.append(
                TensorBlock(
                    values=values_l2,
                    samples=block.samples,
                    components=[_o3_mu_labels(2, block.values.device)],
                    properties=block.properties,
                )
            )

        keys_l0 = _add_o3_irrep_to_keys(tensor.keys, 0, 1)
        keys_l2 = _add_o3_irrep_to_keys(tensor.keys, 2, 1)
        result = TensorMap(
            Labels(
                list(keys_l0.names),
                torch.cat([keys_l0.values, keys_l2.values], dim=0),
            ),
            blocks_l0 + blocks_l2,
        )

    for info_name, info_value in tensor.info().items():
        result.set_info(info_name, info_value)
    return result


def _add_o3_irrep_to_keys(
    keys: Labels,
    o3_lambda: int,
    o3_sigma: int,
) -> Labels:
    """Add or validate the ``o3_lambda`` and ``o3_sigma`` key columns."""
    names = list(keys.names)
    values = keys.values

    if names == ["_"]:
        if len(keys) != 1 or int(values[0, 0]) != 0:
            raise ValueError(
                "the '_' placeholder must contain exactly one key with value 0"
            )
        names = []
        values = values[:, :0]

    for name, expected in (
        ("o3_lambda", o3_lambda),
        ("o3_sigma", o3_sigma),
    ):
        if name in names:
            column = values[:, names.index(name)]
            if not bool(torch.all(column == expected).item()):
                raise ValueError(
                    f"the existing '{name}' key column must contain only "
                    f"{expected} to assign O(3) irrep "
                    f"({o3_lambda}, {o3_sigma})"
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
