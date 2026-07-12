import warnings
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


def _range_labels(name: str, size: int, device: torch.device) -> Labels:
    values = torch.arange(size, dtype=torch.int64, device=device).reshape(-1, 1)
    return Labels(name, values)


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
    cell_volume: Optional[torch.Tensor] = None,
    quadrature_is_inverted: Optional[bool] = None,
) -> Dict[str, TensorMap]:
    """Evaluate rotated copies and derive conservative forces/stress with autograd.

    Forces are ``-dE/d(positions)`` in each rotated frame; stress is
    ``dE/d(strain)/V`` for fully periodic finite-volume cells. Returned Cartesian
    TensorMaps contain one local ``system`` entry per rotation. Supplying
    ``quadrature_is_inverted`` enables the validated internal transformation path.
    """
    if rotations.dim() != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotations must have shape (N, 3, 3), got {tuple(rotations.shape)}"
        )

    n_rot = rotations.shape[0]
    n_atoms = system.positions.shape[0]
    # A valid metatomic System has a zero cell vector along each non-periodic
    # direction. The 3D volume/stress convention is therefore available only
    # for fully periodic systems; lower-dimensional stress needs another schema.
    compute_stress = cell_volume is not None or bool(torch.all(system.pbc).item())

    rotated_positions_list: List[torch.Tensor] = []
    strain_list: List[torch.Tensor] = []
    transformed_systems: List[System] = []

    detached_positions = system.positions.detach()
    detached_cell = system.cell.detach()
    volume = cell_volume
    if compute_stress:
        if volume is None:
            volume = torch.abs(torch.linalg.det(detached_cell))
            invalid_volume = (~torch.isfinite(volume)) | (volume == 0)
            if bool(invalid_volume.item()):
                raise ValueError(
                    "can not compute 3D stress for a singular or non-finite "
                    "periodic cell"
                )
    # hoist device/dtype cast out of the per-rotation loop
    rotations = rotations.to(device=device, dtype=dtype)

    data_names = system.known_data()
    for R in rotations:
        rotated_positions = (detached_positions @ R.T).requires_grad_(True)
        rotated_cell = detached_cell @ R.T
        rotated_positions_list.append(rotated_positions)

        if compute_stress:
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
        if data_names:
            if quadrature_is_inverted is None:
                transformation = O3Transformation(R, ell_max)
            else:
                transformation = O3Transformation._from_validated_matrix(
                    R,
                    ell_max,
                    is_inverted=quadrature_is_inverted,
                )
            for data_name in data_names:
                transformed.add_data(
                    data_name,
                    transform_tensor(
                        system.get_data(data_name), [system], [transformation]
                    ),
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
    if compute_stress:
        grad_targets.extend(strain_list)
    # Metatrain composition models legitimately return detached, type-only
    # energies. Follow the suite convention: warn and treat any detached output
    # as constant (autograd can not distinguish it from a severed graph), while
    # graph-connected but unused targets are materialized as exact zeros.
    if energy_sum.requires_grad:
        grads = list(
            torch.autograd.grad(
                energy_sum,
                grad_targets,
                create_graph=False,
                materialize_grads=True,
            )
        )
    else:
        warnings.warn(
            f"'{energy_name}' is detached from autograd; treating it as constant "
            "and returning zero derivatives",
            RuntimeWarning,
            stacklevel=2,
        )
        grads = [torch.zeros_like(target) for target in grad_targets]

    position_grads = grads[:n_rot]
    strain_grads = grads[n_rot:]

    forces_values = -torch.cat(position_grads, dim=0)

    atom_range = torch.arange(n_atoms, dtype=torch.int64, device=device)
    system_indices = torch.arange(
        n_rot, dtype=torch.int64, device=device
    ).repeat_interleave(n_atoms)
    atom_indices = atom_range.repeat(n_rot)
    forces_samples = Labels(
        names=["system", "atom"],
        values=torch.stack([system_indices, atom_indices], dim=1),
    )

    key_labels = _range_labels("_", 1, device)

    forces_block = TensorBlock(
        values=forces_values.unsqueeze(-1),  # (n_rot*n_atoms, 3, 1)
        samples=forces_samples,
        components=[_range_labels("xyz", 3, device)],
        properties=_range_labels("force", 1, device),
    )
    forces_tmap = TensorMap(key_labels, [forces_block])
    if selected_atoms is not None:
        forces_tmap = mts.slice(forces_tmap, axis="samples", selection=selected_atoms)
    out["forces"] = forces_tmap

    if compute_stress:
        # volume is rotation-invariant, so the validated original cell volume is
        # correct for every rotated copy
        assert volume is not None
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
                _range_labels("xyz_1", 3, device),
                _range_labels("xyz_2", 3, device),
            ],
            properties=_range_labels("stress", 1, device),
        )
        out["stress"] = TensorMap(key_labels, [stress_block])

    return out
