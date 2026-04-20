from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from vesin.metatomic import NeighborList

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelOutput,
    NeighborListOptions,
    System,
    unit_conversion_factor,
)


def _wrap_positions(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """
    Wrap positions into the periodic cell.
    """
    fractional_positions = positions @ cell.inverse()
    fractional_positions = fractional_positions - torch.floor(fractional_positions)
    wrapped_positions = fractional_positions @ cell

    return wrapped_positions


def _check_close_to_cell_boundary(
    cell: torch.Tensor, positions: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """
    Detect atoms that lie within a cutoff distance (in our context, the interaction
    range of the model) from the periodic cell boundaries,
    i.e. have interactions with atoms at the opposite end of the cell.
    """
    inv_cell = cell.inverse()
    recip = inv_cell.T
    norms = torch.linalg.norm(recip, dim=1)
    heights = 1.0 / norms
    if heights.min() < cutoff:
        raise ValueError(
            "Cell is too small compared to cutoff = " + str(cutoff) + ". "
            "Ensure that all cell vectors are at least this length. Currently, the"
            " minimum cell vector length is " + str(heights.min()) + "."
        )

    normals = recip / norms[:, None]
    norm_coords = positions @ normals.T
    collisions = torch.hstack(
        [norm_coords <= cutoff, norm_coords >= heights - cutoff],
    ).to(device=positions.device)

    return collisions[
        :, [0, 3, 1, 4, 2, 5]  # reorder to (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    ]


def _collisions_to_replicas(collisions: torch.Tensor) -> torch.Tensor:
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


def _generate_replica_atoms(
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
    replica_positions = positions[replica_idx] + replica_offsets @ cell

    return replica_idx, types[replica_idx], replica_positions


def _unfold_system(metatomic_system: System, cutoff: float) -> System:
    """
    Unfold a periodic system by generating replica atoms for those near the cell
    boundaries within the specified cutoff distance.
    The unfolded system has no periodic boundary conditions.
    """

    if not metatomic_system.pbc.any():
        raise ValueError("Unfolding systems is only supported for periodic systems.")
    wrapped_positions = _wrap_positions(
        metatomic_system.positions, metatomic_system.cell
    )
    collisions = _check_close_to_cell_boundary(
        metatomic_system.cell, wrapped_positions, cutoff
    )
    replicas = _collisions_to_replicas(collisions)
    replica_idx, replica_types, replica_positions = _generate_replica_atoms(
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


class HeatFlux(torch.nn.Module):
    """
    :py:class:`HeatFlux` is a wrapper around an :py:class:`AtomisticModel` that
    computes the heat flux of a system using the unfolded system approach.

    The unfolded system is generated by creating replica atoms for those near the cell
    boundaries within the interaction range of the model wrapped. The wrapper adds the
    heat flux to the model's outputs under the key "heat_flux".

    For more details on the heat flux calculation, see `Langer, M. F., et al., Heat flux
    for semilocal machine-learning potentials. (2023). Physical Review B, 108, L100302.`
    """

    def __init__(self, model: AtomisticModel):
        """
        :param model: the :py:class:`AtomisticModel` to wrap, which should be able to
            compute atomic energies and their gradients with respect to positions
        """
        super().__init__()

        assert isinstance(model, AtomisticModel)
        self._model = model.module
        self._interaction_range = model.capabilities().interaction_range
        if model.capabilities().length_unit.lower() not in ["angstrom", "a"]:
            raise NotImplementedError(
                f"HeatFluxWrapper only supports models with length unit 'angstrom' or "
                f"'A', but got {model.capabilities().length_unit}"
            )

        self._requested_neighbor_lists = model.requested_neighbor_lists()
        self._requested_inputs = {
            "masses": ModelOutput(quantity="mass", unit="u", per_atom=True),
            "velocities": ModelOutput(quantity="velocity", unit="A/fs", per_atom=True),
        }

        self._nl_calculators = [
            NeighborList(options, model.capabilities().length_unit, True, False)
            for options in self._requested_neighbor_lists
        ]

        self._wrapped_outputs = model.capabilities().outputs
        if not any(
            "energy" == key or key.startswith("energy/")
            for key in self._wrapped_outputs.keys()
        ):
            raise ValueError(
                "The wrapped model must be able to compute energy outputs to use "
                "HeatFluxWrapper."
            )

        mass_unit = self._requested_inputs["masses"].unit
        velocity_unit = self._requested_inputs["velocities"].unit
        self._kinetic_energy_conversion_factors = {}
        for key, output in self._wrapped_outputs.items():
            if key == "energy" or key.startswith("energy/"):
                variant = key.replace("energy", "", 1)
                energy_unit = output.unit
                if energy_unit == "":
                    # we don't know the energy unit, so we won't do any conversion
                    factor = 1.0
                else:
                    factor = unit_conversion_factor(
                        mass_unit + "*" + velocity_unit + "*" + velocity_unit,
                        energy_unit,
                    )
                self._kinetic_energy_conversion_factors[variant] = factor

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return self._requested_neighbor_lists

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return self._requested_inputs

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        outputs_heat_flux: Dict[str, ModelOutput] = {}
        for key, output in outputs.items():
            if key == "heat_flux" or key.startswith("heat_flux/"):
                outputs_heat_flux[key] = output

        # these are requested directly to the underlying model
        outputs_no_heat_flux: Dict[str, ModelOutput] = {}
        for key, output in outputs.items():
            if key != "heat_flux" and not key.startswith("heat_flux/"):
                outputs_no_heat_flux[key] = output

        if len(outputs_no_heat_flux) == 0:
            results = torch.jit.annotate(Dict[str, TensorMap], {})
        else:
            results = self._model(systems, outputs_no_heat_flux, selected_atoms)

        if len(outputs_heat_flux) == 0:
            return results

        energy_variants = [
            key.replace("heat_flux", "", 1) for key in outputs_heat_flux.keys()
        ]
        heat_fluxes_variants: Dict[str, List[torch.Tensor]] = {
            "heat_flux" + variant: [] for variant in energy_variants
        }
        for system in systems:
            unfolded_heat_flux = self._calc_unfolded_heat_flux(system, energy_variants)
            for variant in energy_variants:
                heat_fluxes_variants["heat_flux" + variant].append(
                    unfolded_heat_flux[variant]
                )

        device = systems[0].device
        samples = Labels(
            ["system"], torch.arange(len(systems), device=device).reshape(-1, 1)
        )
        components = [Labels(["xyz"], torch.arange(3, device=device).reshape(-1, 1))]
        properties = Labels(["heat_flux"], torch.tensor([[0]], device=device))

        for heat_flux_variant, heat_fluxes in heat_fluxes_variants.items():
            hf_block = TensorBlock(
                values=torch.vstack(heat_fluxes).reshape(-1, 3, 1).to(device=device),
                samples=samples,
                components=components,
                properties=properties,
            )
            results[heat_flux_variant] = TensorMap(
                Labels("_", torch.tensor([[0]], device=device)), [hf_block]
            )
        return results

    @staticmethod
    def wrap(model: AtomisticModel) -> AtomisticModel:
        """
        Wrap an existing model able to compute atomic energies (i.e. model with a
        per-atom ``"energy"`` output, or any energy variants like ``"energy/pbe"``),
        creating a new model that can compute the heat flux in addition to the original
        model's outputs.

        The returned model will have a ``"heat_flux[/variant]"`` output for each energy
        variant ``model`` can compute, where the same variant is used for the heat flux
        calculation. For example, if the original model can compute both an ``"energy"``
        output and an ``"energy/pbe"`` output, the wrapped model will have both a
        ``"heat_flux"`` output (using the default energy output) and a
        ``"heat_flux/pbe"`` output (using the ``"energy/pbe"`` output).

        :param model: the :py:class:`AtomisticModel` to wrap
        """
        wrapper = HeatFlux(model)
        capabilities = model.capabilities()
        outputs = capabilities.outputs

        heat_flux_outputs = {}
        for key, output in outputs.items():
            if key == "heat_flux" or key.startswith("heat_flux/"):
                raise ValueError(
                    "This model already has an output named " + key + ", which "
                    "conflicts with the heat flux output added by this function."
                )

            if key == "energy" or key.startswith("energy/"):
                variant = key.replace("energy", "", 1)

                energy_unit = output.unit
                velocity_unit = wrapper._requested_inputs["velocities"].unit
                heat_flux_unit = energy_unit + "*" + velocity_unit

                heat_flux_outputs["heat_flux" + variant] = ModelOutput(
                    quantity="heat_flux",
                    unit=heat_flux_unit,
                    explicit_gradients=[],
                    per_atom=False,
                    description=(
                        "Heat flux computed using the unfolded system approach based "
                        "on the '" + key + "' output of this model."
                    ),
                )

        outputs.update(heat_flux_outputs)

        new_capabilities = ModelCapabilities(
            outputs=outputs,
            atomic_types=capabilities.atomic_types,
            interaction_range=capabilities.interaction_range,
            length_unit=capabilities.length_unit,
            supported_devices=capabilities.supported_devices,
            dtype=capabilities.dtype,
        )

        return AtomisticModel(
            wrapper.eval(), model.metadata(), capabilities=new_capabilities
        )

    def _barycenter_and_atomic_energies(
        self, system: System, n_atoms: int, energy_variants: List[str]
    ) -> Dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        outputs = {
            "energy" + variant: self._wrapped_outputs["energy" + variant]
            for variant in energy_variants
        }

        energy_outputs = self._model(
            [system],
            outputs,
            selected_atoms=None,
        )

        results: Dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for variant in energy_variants:
            energy_block = energy_outputs["energy" + variant].block()

            atom_indices = energy_block.samples.column("atom").to(torch.long)
            sorted_order = torch.argsort(atom_indices)
            atomic_e = energy_block.values.flatten()[sorted_order]

            total_e = atomic_e[:n_atoms].sum()
            r_aux = system.positions.detach()
            barycenter = (atomic_e[:n_atoms, None] * r_aux[:n_atoms]).sum(dim=0)

            results[variant] = (barycenter, atomic_e, total_e)

        return results

    def _calc_unfolded_heat_flux(
        self, system: System, energy_variants: List[str]
    ) -> Dict[str, torch.Tensor]:
        n_atoms = len(system.positions)
        unfolded_system = _unfold_system(system, self._interaction_range).to(
            system.device
        )
        unfolded_system.positions.requires_grad_(True)
        for option, nl_calculator in zip(
            self._requested_neighbor_lists, self._nl_calculators, strict=True
        ):
            neighbors = nl_calculator.compute(unfolded_system)
            unfolded_system.add_neighbor_list(option, neighbors)

        velocities: torch.Tensor = (
            unfolded_system.get_data("velocities").block().values.reshape(-1, 3)
        )
        masses: torch.Tensor = (
            unfolded_system.get_data("masses").block().values.reshape(-1)
        )

        results: Dict[str, torch.Tensor] = {}

        barycenter_and_atomic_energies = self._barycenter_and_atomic_energies(
            unfolded_system, n_atoms, energy_variants
        )

        for variant in energy_variants:
            barycenter, atomic_e, total_e = barycenter_and_atomic_energies[variant]

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
                retain_graph=True,
                create_graph=False,
            )[0]
            grads = torch.jit._unwrap_optional(grads)
            term2 = (
                unfolded_system.positions
                * (grads * velocities).sum(dim=1, keepdim=True)
            ).sum(dim=0)

            hf_pot = term1 - term2
            hf_conv = (
                (
                    atomic_e[:n_atoms]
                    + 0.5
                    * masses[:n_atoms]
                    * torch.linalg.norm(velocities[:n_atoms], dim=1) ** 2
                    * self._kinetic_energy_conversion_factors[variant]
                )[:, None]
                * velocities[:n_atoms]
            ).sum(dim=0)

            results[variant] = hf_pot + hf_conv

        return results
