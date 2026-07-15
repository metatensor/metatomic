import operator
from typing import List, Optional, Tuple

import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def _validated_integer(name: str, value, minimum: int) -> int:
    """Normalize an integer-like public argument and enforce its lower bound."""
    if isinstance(value, (bool, np.bool_)) or (
        isinstance(value, torch.Tensor) and value.dtype == torch.bool
    ):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        normalized = int(operator.index(value))
    except TypeError as error:
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}"
        ) from error
    if normalized < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}, got {normalized}")
    return normalized


def _key_to_tuple(key_entry) -> Tuple[int, ...]:
    return tuple(int(v) for v in key_entry.values.tolist())


def _infer_n_systems(tensors: List[TensorMap]) -> int:
    """Largest ``system`` sample index + 1 across all blocks of the tensors."""
    n_systems = 0
    for tensor in tensors:
        for block in tensor.blocks():
            system = block.samples.column("system")
            if len(system) != 0:
                n_systems = max(n_systems, int(torch.max(system).item()) + 1)
    return n_systems


def _max_spherical_lambda(tensor: TensorMap) -> int:
    """Largest rank of any ``o3_mu``-like axis, or ``-1`` when absent."""
    max_lambda = -1
    for block in tensor.blocks():
        for component in block.components:
            if len(component.names) == 1 and component.names[0].startswith("o3_mu"):
                max_lambda = max(max_lambda, (len(component) - 1) // 2)
    return max_lambda


def _prepend_system_to_samples(
    sample_names: List[str],
    sample_values: torch.Tensor,
    system_index: int,
    *,
    device: torch.device,
) -> Labels:
    """Reattach the original ``system`` index after rotation-batch reduction."""
    system_values = torch.full(
        (sample_values.shape[0], 1),
        system_index,
        dtype=sample_values.dtype,
        device=device,
    )
    if len(sample_names) == 0:
        return Labels(["system"], system_values)
    return Labels(
        ["system"] + sample_names,
        torch.cat([system_values, sample_values.to(device=device)], dim=1),
    )


def _selected_atoms_for_local_systems(
    selected_atoms: Optional[Labels],
    system_index: int,
    n_local_systems: int,
) -> Optional[Labels]:
    """Replicate one global system's atom selection over its rotated copies."""
    if selected_atoms is None:
        return None

    system_mask = selected_atoms.column("system").to(dtype=torch.long) == system_index
    system_selected_atoms = selected_atoms.values[system_mask]
    if system_selected_atoms.shape[0] == 0:
        return Labels(
            list(selected_atoms.names),
            selected_atoms.values.new_empty((0, len(selected_atoms.names))),
        )

    local_values = system_selected_atoms.repeat((n_local_systems, 1))
    local_values[:, list(selected_atoms.names).index("system")] = torch.arange(
        n_local_systems,
        dtype=local_values.dtype,
        device=local_values.device,
    ).repeat_interleave(len(system_selected_atoms))
    return Labels(list(selected_atoms.names), local_values)


def _reshape_block_by_local_system(
    block: TensorBlock, n_local_systems: int
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Group equal per-copy samples along a leading local-system axis.

    The block must contain a ``system`` sample column, and every one of the
    ``n_local_systems`` rotated copies must produce identical non-``system``
    sample labels in the same order. Input rows may interleave local systems;
    they are stably grouped before validation.

    :return: values with shape ``(n_local_systems, n_samples, ...)``, sample names
        without ``system``, and the common non-``system`` sample-label values.
    """
    sample_names = list(block.samples.names)
    system_column = sample_names.index("system")
    local_ids = block.samples.column("system").to(dtype=torch.long)
    non_system_sample_values = torch.cat(
        [
            block.samples.values[:, :system_column],
            block.samples.values[:, system_column + 1 :],
        ],
        dim=1,
    )
    if len(local_ids) != 0 and bool(
        torch.any((local_ids < 0) | (local_ids >= n_local_systems)).item()
    ):
        raise ValueError("Encountered output samples with out-of-range system indices.")

    # Avoid sorting the overwhelmingly common scalar batch-size-one case.
    if n_local_systems == 1:
        return (
            block.values.unsqueeze(0),
            sample_names[:system_column] + sample_names[system_column + 1 :],
            non_system_sample_values,
        )

    if len(local_ids) % n_local_systems != 0:
        raise ValueError(
            "Streaming SymmetrizedModel expects each rotated copy of a system to "
            "produce the same sample labels in the same order."
        )
    n_samples = len(local_ids) // n_local_systems
    order = torch.argsort(local_ids, stable=True)
    expected_ids = torch.arange(
        n_local_systems, dtype=local_ids.dtype, device=local_ids.device
    ).repeat_interleave(n_samples)
    if not torch.equal(local_ids[order], expected_ids):
        raise ValueError(
            "Streaming SymmetrizedModel expects each rotated copy of a system to "
            "produce the same sample labels in the same order."
        )

    stacked_values = block.values[order].reshape(
        n_local_systems, n_samples, *block.values.shape[1:]
    )
    stacked_sample_values = non_system_sample_values[order].reshape(
        n_local_systems, n_samples, non_system_sample_values.shape[1]
    )
    base_sample_values = stacked_sample_values[0]
    if not torch.equal(
        stacked_sample_values,
        base_sample_values.unsqueeze(0).expand_as(stacked_sample_values),
    ):
        raise ValueError(
            "Streaming SymmetrizedModel expects each rotated copy of a system to "
            "produce the same sample labels in the same order."
        )

    return (
        stacked_values,
        sample_names[:system_column] + sample_names[system_column + 1 :],
        base_sample_values,
    )
