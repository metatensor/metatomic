import operator
from typing import List, Optional, Tuple

import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock


def _validate_integer(name: str, value, minimum: int) -> int:
    """Check that ``value`` is an integer at least ``minimum``.

    Return it as a Python ``int``.
    """
    if isinstance(value, (bool, np.bool_)) or (
        isinstance(value, torch.Tensor) and value.dtype == torch.bool
    ):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        integer_value = int(operator.index(value))
    except TypeError as error:
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}"
        ) from error
    if integer_value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}, got {integer_value}")
    return integer_value


def _map_selected_atoms_to_rotated_copies(
    selected_atoms: Optional[Labels],
    input_system_index: int,
    n_rotated_copies: int,
) -> Optional[Labels]:
    """Map one input system's selected atoms to each rotated copy."""
    if selected_atoms is None:
        return None

    input_system_mask = (
        selected_atoms.column("system").to(dtype=torch.long) == input_system_index
    )
    selected_atoms_for_input_system = selected_atoms.values[input_system_mask]
    if selected_atoms_for_input_system.shape[0] == 0:
        return Labels(
            list(selected_atoms.names),
            selected_atoms.values.new_empty((0, len(selected_atoms.names))),
        )

    rotated_values = selected_atoms_for_input_system.repeat((n_rotated_copies, 1))
    rotated_values[:, list(selected_atoms.names).index("system")] = torch.arange(
        n_rotated_copies,
        dtype=rotated_values.dtype,
        device=rotated_values.device,
    ).repeat_interleave(len(selected_atoms_for_input_system))
    return Labels(list(selected_atoms.names), rotated_values)


def _group_samples_by_rotated_copy(
    block: TensorBlock, n_rotated_copies: int
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Group samples from rotated copies along a leading copy axis."""
    sample_names = list(block.samples.names)
    system_column = sample_names.index("system")
    copy_indices = block.samples.column("system").to(dtype=torch.long)
    sample_values_without_system = torch.cat(
        [
            block.samples.values[:, :system_column],
            block.samples.values[:, system_column + 1 :],
        ],
        dim=1,
    )
    if len(copy_indices) != 0 and bool(
        torch.any((copy_indices < 0) | (copy_indices >= n_rotated_copies)).item()
    ):
        raise ValueError(
            "Encountered output samples with out-of-range rotated-copy indices."
        )

    # A single copy is already grouped; avoid sorting the common batch-size-one case.
    if n_rotated_copies == 1:
        return (
            block.values.unsqueeze(0),
            sample_names[:system_column] + sample_names[system_column + 1 :],
            sample_values_without_system,
        )

    if len(copy_indices) % n_rotated_copies != 0:
        raise ValueError(
            "SymmetrizedModel expects every rotated copy to produce the same "
            "sample labels in the same order."
        )
    n_samples_per_copy = len(copy_indices) // n_rotated_copies
    order = torch.argsort(copy_indices, stable=True)
    expected_copy_indices = torch.arange(
        n_rotated_copies,
        dtype=copy_indices.dtype,
        device=copy_indices.device,
    ).repeat_interleave(n_samples_per_copy)
    if not torch.equal(copy_indices[order], expected_copy_indices):
        raise ValueError(
            "SymmetrizedModel expects every rotated copy to produce the same "
            "sample labels in the same order."
        )

    values_shape = [n_rotated_copies, n_samples_per_copy]
    for axis in range(1, block.values.dim()):
        values_shape.append(block.values.shape[axis])
    values_by_copy = block.values[order].reshape(values_shape)
    sample_values_by_copy = sample_values_without_system[order].reshape(
        n_rotated_copies,
        n_samples_per_copy,
        sample_values_without_system.shape[1],
    )
    shared_sample_values = sample_values_by_copy[0]
    if not torch.equal(
        sample_values_by_copy,
        shared_sample_values.unsqueeze(0).expand_as(sample_values_by_copy),
    ):
        raise ValueError(
            "SymmetrizedModel expects every rotated copy to produce the same "
            "sample labels in the same order."
        )

    return (
        values_by_copy,
        sample_names[:system_column] + sample_names[system_column + 1 :],
        shared_sample_values,
    )


def _restore_input_system_to_samples(
    sample_names: List[str],
    sample_values: torch.Tensor,
    input_system_index: int,
    *,
    device: torch.device,
) -> Labels:
    """Restore the input-system label after reducing over rotated copies."""
    sample_values = sample_values.to(device=device)
    system_values = torch.full(
        (sample_values.shape[0], 1),
        input_system_index,
        dtype=sample_values.dtype,
        device=device,
    )
    return Labels(
        ["system"] + sample_names,
        torch.cat([system_values, sample_values], dim=1),
    )
