from typing import List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


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
    """Largest angular momentum among the ``o3_mu``-like component axes of the
    tensor's blocks (each axis has length ``2*lambda + 1``), or ``-1`` if no
    block has spherical components."""
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
    """Rebuild sample Labels with a leading ``system`` column.

    After reducing over the rotated copies of one system, the remaining samples
    no longer carry a system index; this re-attaches the index of the original
    system in the caller's list, so per-system results can be joined.
    """
    system_values = torch.full(
        (sample_values.shape[0], 1),
        system_index,
        dtype=torch.int32,
        device=device,
    )
    if len(sample_names) == 0:
        return Labels(["system"], system_values)

    return Labels(
        ["system"] + sample_names,
        torch.cat(
            [system_values, sample_values.to(device=device, dtype=torch.int32)], dim=1
        ),
    )


def _selected_atoms_for_local_systems(
    selected_atoms: Optional[Labels],
    system_index: int,
    n_local_systems: int,
) -> Optional[Labels]:
    """Map a global atom selection onto one system's batch of rotated copies.

    ``selected_atoms`` uses the caller's global system indices, but the base
    model is evaluated on ``n_local_systems`` rotated copies of the system at
    ``system_index``: the selection for that system is replicated once per
    copy, with the ``system`` column rewritten to the local copy index.
    """
    if selected_atoms is None:
        return None

    system_mask = selected_atoms.column("system").to(dtype=torch.long) == system_index
    system_selected_atoms = selected_atoms.values[system_mask]
    if system_selected_atoms.shape[0] == 0:
        return Labels(
            list(selected_atoms.names),
            selected_atoms.values.new_empty((0, len(selected_atoms.names))),
        )

    system_column = list(selected_atoms.names).index("system")
    local_selected_atoms: List[torch.Tensor] = []
    for local_system_index in range(n_local_systems):
        local_values = system_selected_atoms.clone()
        local_values[:, system_column] = local_system_index
        local_selected_atoms.append(local_values)

    return Labels(
        list(selected_atoms.names),
        torch.cat(local_selected_atoms, dim=0),
    )


def _reshape_block_by_local_system(
    block: TensorBlock, n_local_systems: int
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Split a block over batched rotated copies into a stacked per-copy tensor.

    Returns ``(values, sample_names, sample_values)``: ``values`` gains a
    leading axis of size ``n_local_systems`` (one entry per rotated copy), and
    ``sample_names``/``sample_values`` describe the per-copy samples shared by
    all copies, with the leading ``system`` column removed. Every copy must
    produce identical sample labels in the same order for the stacking (and
    any later reduction over the new axis) to be meaningful.
    """
    local_ids = block.samples.column("system").to(dtype=torch.long)
    if len(local_ids) != 0:
        min_local_id = int(torch.min(local_ids).item())
        max_local_id = int(torch.max(local_ids).item())
        if min_local_id < 0 or max_local_id >= n_local_systems:
            raise ValueError(
                "Encountered output samples with out-of-range system indices."
            )

    split_values: List[torch.Tensor] = []
    base_sample_values: Optional[torch.Tensor] = None
    for local_system_index in range(n_local_systems):
        local_mask = local_ids == local_system_index
        local_values = block.values[local_mask]
        local_sample_values = block.samples.values[local_mask][:, 1:]
        if base_sample_values is None:
            base_sample_values = local_sample_values
        elif not torch.equal(local_sample_values, base_sample_values):
            raise ValueError(
                "Streaming SymmetrizedModel expects each rotated copy of a system to "
                "produce the same sample labels in the same order."
            )
        split_values.append(local_values)

    assert base_sample_values is not None
    stacked_values = torch.stack(split_values, dim=0)
    return stacked_values, list(block.samples.names[1:]), base_sample_values
