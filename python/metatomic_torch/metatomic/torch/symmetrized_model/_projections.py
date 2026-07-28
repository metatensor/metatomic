"""Character-projection helpers.

The projected quantity is defined in the :py:class:`SymmetrizedModel` class
docstring.
"""

from typing import List, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from ._utils import (
    _group_samples_by_rotated_copy,
    _restore_input_system_to_samples,
)


def _character_projection_coefficients_from_rotation_batch(
    values: torch.Tensor,
    weights: torch.Tensor,
    inverse_wigner_matrices: torch.Tensor,
) -> torch.Tensor:
    """Compute one rotation batch's character-projection coefficients."""
    weighted_wigner_matrices = weights.to(
        dtype=values.dtype,
        device=values.device,
    ).view(-1, 1, 1) * inverse_wigner_matrices.to(
        dtype=values.dtype,
        device=values.device,
    )
    return torch.einsum(
        "gmn,gs...->smn...",
        weighted_wigner_matrices,
        values,
    )


def _character_projections_from_proper_and_improper_coefficients(
    proper_coefficients: torch.Tensor,
    improper_coefficients: torch.Tensor,
    chi_lambda: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return squared character projections for ``chi_sigma=+1`` and ``-1``."""
    dimension = 2 * chi_lambda + 1
    parity = (-1) ** chi_lambda
    sigma_plus = proper_coefficients + parity * improper_coefficients
    sigma_minus = proper_coefficients - parity * improper_coefficients
    factor = float(dimension) / 4.0
    return (
        factor * sigma_plus.square().sum(dim=(1, 2)),
        factor * sigma_minus.square().sum(dim=(1, 2)),
    )


def _character_projection_coefficients_from_batch(
    tensor: TensorMap,
    weights: torch.Tensor,
    inverse_wigner_matrices: List[torch.Tensor],
    input_system_index: int,
) -> TensorMap:
    """Accumulate every character rank for one rotation batch."""
    key_names = list(tensor.keys.names)
    key_values = tensor.keys.values
    if key_names == ["_"]:
        if len(tensor.keys) != 1 or int(key_values[0, 0]) != 0:
            raise ValueError(
                "the '_' placeholder must contain exactly one key with value 0"
            )
        key_names = []
        key_values = key_values[:, :0]

    if "chi_lambda" in key_names or "chi_sigma" in key_names:
        raise ValueError(
            "source output keys must not contain 'chi_lambda' or 'chi_sigma'"
        )

    blocks: List[TensorBlock] = []
    output_key_values: List[torch.Tensor] = []
    n_rotated_copies = weights.numel()
    for key_index in range(len(tensor.keys)):
        block = tensor.block(key_index)
        values, sample_names, sample_values = _group_samples_by_rotated_copy(
            block,
            n_rotated_copies,
        )
        samples = _restore_input_system_to_samples(
            sample_names,
            sample_values,
            input_system_index,
            device=block.samples.device,
        )

        for chi_lambda in range(len(inverse_wigner_matrices)):
            coefficients = _character_projection_coefficients_from_rotation_batch(
                values,
                weights,
                inverse_wigner_matrices[chi_lambda],
            )
            dimension = 2 * chi_lambda + 1
            character_indices = torch.arange(
                dimension,
                dtype=torch.int32,
                device=coefficients.device,
            ).reshape(-1, 1)
            components = [
                Labels("chi_m", character_indices),
                Labels("chi_n", character_indices),
            ]
            for component in block.components:
                components.append(component)
            blocks.append(
                TensorBlock(
                    values=coefficients,
                    samples=samples,
                    components=components,
                    properties=block.properties,
                )
            )
            output_key_values.append(
                torch.cat(
                    [
                        key_values[key_index],
                        torch.tensor(
                            [chi_lambda],
                            dtype=key_values.dtype,
                            device=key_values.device,
                        ),
                    ]
                )
            )

    if len(output_key_values) == 0:
        values = key_values.new_empty((0, len(key_names) + 1))
    else:
        values = torch.stack(output_key_values)
    return TensorMap(Labels(key_names + ["chi_lambda"], values), blocks)


def _character_projection_tensormap_from_cosets(
    proper_coefficients: TensorMap,
    improper_coefficients: TensorMap,
) -> TensorMap:
    """Combine proper and improper coefficient TensorMaps into O(3) sectors."""
    key_names = list(proper_coefficients.keys.names)
    if "chi_lambda" not in key_names:
        raise ValueError("character coefficients must contain a 'chi_lambda' key")
    if "chi_sigma" in key_names:
        raise ValueError("source output keys must not contain 'chi_sigma'")
    chi_lambda_column = key_names.index("chi_lambda")

    blocks: List[TensorBlock] = []
    output_key_values: List[torch.Tensor] = []
    for key_index in range(len(proper_coefficients.keys)):
        proper_block = proper_coefficients.block(key_index)
        improper_block = improper_coefficients.block(key_index)
        chi_lambda = int(proper_coefficients.keys.values[key_index, chi_lambda_column])
        sigma_plus, sigma_minus = (
            _character_projections_from_proper_and_improper_coefficients(
                proper_block.values,
                improper_block.values,
                chi_lambda,
            )
        )
        target_components = proper_block.components[2:]
        for chi_sigma, values in ((1, sigma_plus), (-1, sigma_minus)):
            blocks.append(
                TensorBlock(
                    values=values,
                    samples=proper_block.samples,
                    components=target_components,
                    properties=proper_block.properties,
                )
            )
            output_key_values.append(
                torch.cat(
                    [
                        proper_coefficients.keys.values[key_index],
                        torch.tensor(
                            [chi_sigma],
                            dtype=proper_coefficients.keys.values.dtype,
                            device=proper_coefficients.keys.values.device,
                        ),
                    ]
                )
            )

    if len(output_key_values) == 0:
        values = proper_coefficients.keys.values.new_empty((0, len(key_names) + 1))
    else:
        values = torch.stack(output_key_values)
    return TensorMap(Labels(key_names + ["chi_sigma"], values), blocks)
