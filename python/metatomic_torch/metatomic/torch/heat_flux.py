from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from vesin.metatomic import compute_requested_neighbors

from metatomic.torch import (
    AtomisticModel,
    ModelEvaluationOptions,
    ModelOutput,
    System,
)


def wrap_positions(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """
    Wrap positions into the periodic cell.
    """
    fractional_positions = torch.einsum("iv,kv->ik", positions, cell.inverse())
    fractional_positions -= torch.floor(fractional_positions)
    wrapped_positions = torch.einsum("iv,kv->ik", fractional_positions, cell)

    return wrapped_positions


def check_collisions(
    cell: torch.Tensor, positions: torch.Tensor, cutoff: float, skin: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Detect atoms that lie within a cutoff distance from the periodic cell boundaries,
    i.e. have interactions with atoms at the opposite end of the cell.
    """
    inv_cell = cell.inverse()
    norm_inv_cell = torch.linalg.norm(inv_cell, dim=1)
    inv_cell /= norm_inv_cell[:, None]
    cell_vec_lengths = torch.diag(cell @ inv_cell)
    if cell_vec_lengths.min() < (cutoff + skin):
        raise ValueError(
            "Cell is too small compared to (cutoff + skin) = "
            + str(cutoff + skin)
            + ". "
            "Ensure that all cell vectors are at least this length. Currently, the"
            " minimum cell vector length is " + str(cell_vec_lengths.min()) + "."
        )

    cutoff += skin
    norm_coords = torch.einsum("iv,kv->ik", positions, inv_cell)
    collisions = torch.hstack(
        [norm_coords <= cutoff, norm_coords >= cell_vec_lengths - cutoff],
    ).to(device=positions.device)

    return (
        collisions[
            :, [0, 3, 1, 4, 2, 5]  # reorder to (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
        ],
        norm_coords,
    )


def collisions_to_replicas(collisions: torch.Tensor) -> torch.Tensor:
    """
    Convert boundary-collision flags into a boolean mask over all periodic image
    displacements in {0, +1, -1}^3. e.g. for an atom colliding with the x_lo and y_hi
    boundaries, we need the replicas at (1, 0, 0), (0, -1, 0), (1, -1, 0) image cells.

    collisions: [N, 6]: has collisions with (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)

    returns: [N, 3, 3, 3] boolean mask over image displacements in {0, +1, -1}^3
        0: no replica needed along that axis
        1: +1 replica needed along that axis (i.e., near low boundary, a replica is
        placed just outside the high boundary)
        2: -1 replica needed along that axis (i.e., near high boundary, a replica is
        placed just outside the low boundary)
        axis order: x, y, z
    """
    origin = torch.full(
        (len(collisions),), True, dtype=torch.bool, device=collisions.device
    )
    axs = torch.vstack([origin, collisions[:, 0], collisions[:, 1]])
    ays = torch.vstack([origin, collisions[:, 2], collisions[:, 3]])
    azs = torch.vstack([origin, collisions[:, 4], collisions[:, 5]])
    # leverage broadcasting
    outs = axs[:, None, None] & ays[None, :, None] & azs[None, None, :]
    outs = torch.movedim(outs, -1, 0)
    outs[:, 0, 0, 0] = False  # not close to any boundary -> no replica needed
    return outs.to(device=collisions.device)


def generate_replica_atoms(
    types: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    replicas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For atoms near the low boundary (x_lo/y_lo/z_lo), generate their images shifted
    by +1 cell vector (i.e., placed just outside the high boundary).
    For atoms near the high boundary (x_hi/y_hi/z_hi), generate images shifted by −1
    cell vector.
    """
    replicas = torch.argwhere(replicas)
    replica_idx = replicas[:, 0]
    replica_offsets = torch.tensor(
        [0, 1, -1], device=positions.device, dtype=positions.dtype
    )[replicas[:, 1:]]
    replica_positions = positions[replica_idx]
    replica_positions += torch.einsum("aA,iA->ia", cell, replica_offsets)

    return replica_idx, types[replica_idx], replica_positions


def unfold_system(metatomic_system: System, cutoff: float, skin: float = 0.5) -> System:
    """
    Unfold a periodic system by generating replica atoms for those near the cell
    boundaries within the specified cutoff distance.
    The unfolded system has no periodic boundary conditions.
    """

    wrapped_positions = wrap_positions(
        metatomic_system.positions, metatomic_system.cell
    )
    collisions, _ = check_collisions(
        metatomic_system.cell, wrapped_positions, cutoff, skin
    )
    replicas = collisions_to_replicas(collisions)
    replica_idx, replica_types, replica_positions = generate_replica_atoms(
        metatomic_system.types, wrapped_positions, metatomic_system.cell, replicas
    )
    unfolded_types = torch.cat(
        [
            metatomic_system.types,
            replica_types,
        ]
    )
    unfolded_positions = torch.cat(
        [
            wrapped_positions,
            replica_positions,
        ]
    )
    unfolded_idx = torch.cat(
        [
            torch.arange(len(metatomic_system.types), device=metatomic_system.device),
            replica_idx,
        ]
    )
    unfolded_n_atoms = len(unfolded_types)
    masses_block = metatomic_system.get_data("masses").block()
    velocities_block = metatomic_system.get_data("velocities").block()
    unfolded_masses = masses_block.values[unfolded_idx]
    unfolded_velocities = velocities_block.values[unfolded_idx]
    unfolded_masses_block = TensorBlock(
        values=unfolded_masses,
        samples=Labels(
            ["atoms"],
            torch.arange(unfolded_n_atoms, device=metatomic_system.device).reshape(
                -1, 1
            ),
        ),
        components=masses_block.components,
        properties=masses_block.properties,
    )
    unfolded_velocities_block = TensorBlock(
        values=unfolded_velocities,
        samples=Labels(
            ["atoms"],
            torch.arange(unfolded_n_atoms, device=metatomic_system.device).reshape(
                -1, 1
            ),
        ),
        components=velocities_block.components,
        properties=velocities_block.properties,
    )
    unfolded_system = System(
        types=unfolded_types,
        positions=unfolded_positions,
        cell=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=unfolded_positions.dtype,
            device=metatomic_system.device,
        ),
        pbc=torch.tensor([False, False, False], device=metatomic_system.device),
    )
    unfolded_system.add_data(
        "masses",
        TensorMap(
            Labels("_", torch.tensor([[0]], device=metatomic_system.device)),
            [unfolded_masses_block],
        ),
    )
    unfolded_system.add_data(
        "velocities",
        TensorMap(
            Labels("_", torch.tensor([[0]], device=metatomic_system.device)),
            [unfolded_velocities_block],
        ),
    )
    return unfolded_system.to(metatomic_system.dtype, metatomic_system.device)


class HeatFluxWrapper(torch.nn.Module):
    """
    A wrapper around an AtomisticModel that computes the heat flux of a system using the
    unfolded system approach. The heat flux is computed using the atomic energies (eV),
    positions(Å), masses(u), velocities(Å/fs), and the energy gradients.

    The unfolded system is generated by creating replica atoms for those near the cell
    boundaries within the interaction range of the model wrapped. The wrapper adds the
    heat flux to the model's outputs under the key "extra::heat_flux".

    For more details on the heat flux calculation, see `Langer, M. F., et al., Heat flux
    for semilocal machine-learning potentials. (2023). Physical Review B, 108, L100302.`
    """

    def __init__(self, model: AtomisticModel, skin: float = 0.5):
        """
        :param model: the :py:class:`AtomisticModel` to wrap, which should be able to
        compute atomic energies and their gradients with respect to positions
        :param skin: the skin parameter for unfolding the system. The wrapper will
        generate replica atoms for those within (interaction_range + skin) distance from
        the cell boundaries. A skin results in more replica atoms and thus higher
        computational cost, but ensures that the heat flux is computed correctly.
        """
        super().__init__()

        self._model = model
        self.skin = skin
        self._interaction_range = model.capabilities().interaction_range

        self._requested_inputs = {
            "masses": ModelOutput(quantity="mass", unit="u", per_atom=True),
            "velocities": ModelOutput(quantity="velocity", unit="A/fs", per_atom=True),
        }

        hf_output = ModelOutput(
            quantity="heat_flux",
            unit="",
            explicit_gradients=[],
            per_atom=False,
        )
        outputs = self._model.capabilities().outputs.copy()
        outputs["extra::heat_flux"] = hf_output
        self._model.capabilities().outputs["extra::heat_flux"] = hf_output

        energies_output = ModelOutput(
            quantity="energy", unit=outputs["energy"].unit, per_atom=True
        )
        self._unfolded_run_options = ModelEvaluationOptions(
            length_unit=self._model.capabilities().length_unit,
            outputs={"energy": energies_output},
            selected_atoms=None,
        )

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return self._requested_inputs

    def barycenter_and_atomic_energies(self, system: System, n_atoms: int):
        atomic_e = self._model([system], self._unfolded_run_options, False)["energy"][
            0
        ].values.flatten()
        total_e = atomic_e[:n_atoms].sum()
        r_aux = system.positions.detach()
        barycenter = torch.einsum("i,ik->k", atomic_e[:n_atoms], r_aux[:n_atoms])

        return barycenter, atomic_e, total_e

    def calc_unfolded_heat_flux(self, system: System) -> torch.Tensor:
        n_atoms = len(system.positions)
        unfolded_system = unfold_system(system, self._interaction_range, self.skin).to(
            "cpu"
        )
        compute_requested_neighbors(
            unfolded_system, self._unfolded_run_options.length_unit, model=self._model
        )
        unfolded_system = unfolded_system.to(system.device)
        velocities: torch.Tensor = (
            unfolded_system.get_data("velocities").block().values.reshape(-1, 3)
        )
        masses: torch.Tensor = (
            unfolded_system.get_data("masses").block().values.reshape(-1)
        )
        barycenter, atomic_e, total_e = self.barycenter_and_atomic_energies(
            unfolded_system, n_atoms
        )

        term1 = torch.zeros(
            (3), device=system.positions.device, dtype=system.positions.dtype
        )
        for i in range(3):
            grad_i = torch.autograd.grad(
                [barycenter[i]],
                [unfolded_system.positions],
                retain_graph=True,
                create_graph=False,
            )[0]
            grad_i = torch.jit._unwrap_optional(grad_i)
            term1[i] = (grad_i * velocities).sum()

        go = torch.jit.annotate(
            Optional[List[Optional[torch.Tensor]]], [torch.ones_like(total_e)]
        )
        grads = torch.autograd.grad(
            [total_e],
            [unfolded_system.positions],
            grad_outputs=go,
        )[0]
        grads = torch.jit._unwrap_optional(grads)
        term2 = (
            unfolded_system.positions * (grads * velocities).sum(dim=1, keepdim=True)
        ).sum(dim=0)

        hf_pot = term1 - term2

        hf_conv = (
            (
                atomic_e[:n_atoms]
                + 0.5
                * masses[:n_atoms]
                * torch.linalg.norm(velocities[:n_atoms], dim=1) ** 2
                * 103.6427  # u*A^2/fs^2 to eV
            )[:, None]
            * velocities[:n_atoms]
        ).sum(dim=0)

        return hf_pot + hf_conv

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        run_options = ModelEvaluationOptions(
            length_unit=self._model.capabilities().length_unit,
            outputs=outputs,
            selected_atoms=None,
        )
        results = self._model(systems, run_options, False)

        if "extra::heat_flux" not in outputs:
            return results

        device = systems[0].device
        heat_fluxes: List[torch.Tensor] = []
        for system in systems:
            system.positions.requires_grad_(True)
            heat_fluxes.append(self.calc_unfolded_heat_flux(system))

        samples = Labels(
            ["system"], torch.arange(len(systems), device=device).reshape(-1, 1)
        )

        hf_block = TensorBlock(
            values=torch.vstack(heat_fluxes).reshape(-1, 3, 1).to(device=device),
            samples=samples,
            components=[Labels(["xyz"], torch.arange(3, device=device).reshape(-1, 1))],
            properties=Labels(["heat_flux"], torch.tensor([[0]], device=device)),
        )
        results["extra::heat_flux"] = TensorMap(
            Labels("_", torch.tensor([[0]], device=device)), [hf_block]
        )
        return results
