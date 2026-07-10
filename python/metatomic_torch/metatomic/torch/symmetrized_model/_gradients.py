from typing import Callable, Dict, List, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    ModelOutput,
    System,
    register_autograd_neighbors,
)

from ..o3 import O3Transformation, transform_tensor


def _evaluate_with_gradients(
    model: Callable[
        [List[System], Dict[str, ModelOutput], Optional[Labels]],
        Dict[str, TensorMap],
    ],
    system: System,
    rotations: torch.Tensor,
    outputs: Dict[str, ModelOutput],
    selected_atoms: Optional[Labels],
    device: torch.device,
    dtype: torch.dtype,
    energy_name: str = "energy",
    ell_max: int = 0,
) -> Dict[str, TensorMap]:
    """
    Evaluate model on a batch of rotated copies of one system and compute conservative
    forces/stress via autograd.

    Forces are ``-dE/d(positions)`` in each rotated frame; stress is
    ``(1/V) dE/d(strain)`` via the strain trick. Both are packaged as Cartesian
    TensorMaps with one entry per rotation (sample axis = ``[system, atom]`` for
    forces, ``[system]`` for stress) so the downstream back-rotation pipeline can
    treat them like any other per-system output.

    :param model: callable evaluating the base model with the
        ``(systems, outputs, selected_atoms)`` signature
    :param system: input system (original frame)
    :param rotations: ``(N, 3, 3)`` rotation matrices (each may include inversion)
    :param outputs: model output specifications
    :param selected_atoms: optional atom selection (in the local batch index space)
    :param device: device for tensors
    :param dtype: dtype for tensors
    :param energy_name: name of the output forces/stress are derived from
    :param ell_max: maximum angular momentum used to transform custom system
        data; only needed when the system carries data with spherical components
    :return: model output dict with added ``"forces"`` and (if periodic) ``"stress"``
    """
    if rotations.dim() != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotations must have shape (N, 3, 3), got {tuple(rotations.shape)}"
        )

    n_rot = rotations.shape[0]
    n_atoms = system.positions.shape[0]
    has_cell = bool(torch.any(system.pbc).item())

    rotated_positions_list: List[torch.Tensor] = []
    strain_list: List[torch.Tensor] = []
    transformed_systems: List[System] = []

    detached_positions = system.positions.detach()
    detached_cell = system.cell.detach()
    # hoist device/dtype cast out of the per-rotation loop
    rotations = rotations.to(device=device, dtype=dtype)

    for i in range(n_rot):
        R = rotations[i]

        rotated_positions = (detached_positions @ R.T).requires_grad_(True)
        rotated_cell = detached_cell @ R.T
        rotated_positions_list.append(rotated_positions)

        if has_cell:
            strain = torch.eye(3, requires_grad=True, device=device, dtype=dtype)
            final_positions = rotated_positions @ strain
            final_cell = rotated_cell @ strain
            strain_list.append(strain)
        else:
            final_positions = rotated_positions
            final_cell = rotated_cell

        transformed = System(
            types=system.types,
            positions=final_positions,
            cell=final_cell,
            pbc=system.pbc,
        )

        # each rotated copy needs its own neighbor list block so autograd can
        # flow through the rotated positions independently per system; the
        # rotated values must be a fresh tensor (get_neighbor_list returns
        # storage shared with the system, and detach alone does not copy)
        for options in system.known_neighbor_lists():
            neighbors = system.get_neighbor_list(options)
            rotated_values = neighbors.values.detach().squeeze(-1) @ R.T
            rotated_neighbors = TensorBlock(
                values=rotated_values.unsqueeze(-1),
                samples=neighbors.samples,
                components=neighbors.components,
                properties=neighbors.properties,
            )
            register_autograd_neighbors(transformed, rotated_neighbors)
            transformed.add_neighbor_list(options, rotated_neighbors)

        # custom data rotates with the system, like in o3.transform_system
        if len(system.known_data()) > 0:
            transformation = O3Transformation(R, ell_max)
            for data_name in system.known_data():
                data = system.get_data(data_name)
                transformed.add_data(
                    data_name, transform_tensor(data, [system], [transformation])
                )

        transformed_systems.append(transformed)

    out = model(transformed_systems, outputs, selected_atoms)

    if energy_name not in out:
        raise ValueError(
            f"compute_gradients=True requires the model to output '{energy_name}'"
        )

    # The model treats the N systems independently, so d(sum)/d(rotated_positions[i])
    # equals dE_i/d(rotated_positions[i]): no cross-system contamination.
    energy_sum = out[energy_name].block().values.sum()

    grad_targets: List[torch.Tensor] = list(rotated_positions_list)
    if has_cell:
        grad_targets.extend(strain_list)
    grads = torch.autograd.grad(energy_sum, grad_targets, create_graph=False)

    position_grads = grads[:n_rot]
    strain_grads = grads[n_rot:] if has_cell else []

    forces_values = torch.cat([-g for g in position_grads], dim=0)  # (n_rot*n_atoms, 3)

    atom_range = torch.arange(n_atoms, dtype=torch.int64, device=device)
    system_indices = torch.arange(
        n_rot, dtype=torch.int64, device=device
    ).repeat_interleave(n_atoms)
    atom_indices = atom_range.repeat(n_rot)
    forces_samples = Labels(
        names=["system", "atom"],
        values=torch.stack([system_indices, atom_indices], dim=1),
    )

    key_labels = Labels(
        names=["_"],
        values=torch.tensor([[0]], dtype=torch.int64, device=device),
    )

    forces_block = TensorBlock(
        values=forces_values.unsqueeze(-1),  # (n_rot*n_atoms, 3, 1)
        samples=forces_samples,
        components=[
            Labels(
                "xyz",
                torch.arange(3, dtype=torch.int64, device=device).reshape(-1, 1),
            )
        ],
        properties=Labels(
            names=["force"],
            values=torch.tensor([[0]], dtype=torch.int64, device=device),
        ),
    )
    forces_tmap = TensorMap(key_labels, [forces_block])
    if selected_atoms is not None:
        forces_tmap = mts.slice(forces_tmap, axis="samples", selection=selected_atoms)
    out["forces"] = forces_tmap

    if has_cell:
        # volume is rotation-invariant, so the original cell volume is correct for
        # every rotated copy
        volume = torch.abs(torch.linalg.det(detached_cell))
        stress_values = torch.stack(strain_grads, dim=0) / volume  # (n_rot, 3, 3)

        stress_block = TensorBlock(
            values=stress_values.unsqueeze(-1),  # (n_rot, 3, 3, 1)
            samples=Labels(
                names=["system"],
                values=torch.arange(n_rot, dtype=torch.int64, device=device).reshape(
                    -1, 1
                ),
            ),
            components=[
                Labels(
                    "xyz_1",
                    torch.arange(3, dtype=torch.int64, device=device).reshape(-1, 1),
                ),
                Labels(
                    "xyz_2",
                    torch.arange(3, dtype=torch.int64, device=device).reshape(-1, 1),
                ),
            ],
            properties=Labels(
                names=["stress"],
                values=torch.tensor([[0]], dtype=torch.int64, device=device),
            ),
        )
        out["stress"] = TensorMap(key_labels, [stress_block])

    return out
