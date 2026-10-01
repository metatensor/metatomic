"""Run a metatomic model in OpenMM through OpenMM-ML.

Two routes: wrap MetatomicCalculator in the ASE backend, or use
MLPotential("metatomic"). Both should give the same energy.

Requires::

    pip install openmmml metatomic-torch metatomic-ase ase
"""

from typing import Dict, List, Optional

import ase
import ase.build
import matplotlib.pyplot as plt
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic_ase import MetatomicCalculator
from openmmml import MLPotential

import metatomic.torch as mta


class HarmonicEnergy(torch.nn.Module):
    """Harmonic restraint around the initial positions."""

    def __init__(self, k: float, equilibrium_positions: torch.Tensor):
        super().__init__()
        self.k = k
        self.register_buffer("equilibrium_positions", equilibrium_positions)

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            energy[i] += torch.sum(
                self.k * (system.positions - self.equilibrium_positions) ** 2
            )
        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {
            "energy": TensorMap(keys=Labels("_", torch.tensor([[0]])), blocks=[block])
        }


def atoms_to_topology(atoms: ase.Atoms) -> app.Topology:
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    for symbol in atoms.get_chemical_symbols():
        topology.addAtom(symbol, app.Element.getBySymbol(symbol), residue)
    return topology


def energy_kj(system, positions):
    context = mm.Context(system, mm.VerletIntegrator(0.001))
    context.setPositions(positions)
    return (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )


atoms = ase.build.molecule("CH4")
raw_model = HarmonicEnergy(
    k=1.0,
    equilibrium_positions=torch.tensor(atoms.get_positions(), dtype=torch.float64),
)
capabilities = mta.ModelCapabilities(
    outputs={"energy": mta.ModelOutput(unit="eV", sample_kind="system")},
    atomic_types=[1, 6],
    interaction_range=0.0,
    length_unit="Angstrom",
    supported_devices=["cpu"],
    dtype="float64",
)
wrapper = mta.AtomisticModel(raw_model.eval(), mta.ModelMetadata(), capabilities)
wrapper.save("exported-model.pt")

topology = atoms_to_topology(atoms)
positions = (atoms.get_positions() + 0.1) * unit.angstrom

calculator = MetatomicCalculator(
    "exported-model.pt",
    device="cpu",
    do_gradients_with_energy=True,
)
ase_system = MLPotential("ase").createSystem(topology, calculator=calculator)
energy_ase = energy_kj(ase_system, positions)

native = MLPotential("metatomic", modelPath="exported-model.pt", device="cpu")
native_system = native.createSystem(topology)
energy_native = energy_kj(native_system, positions)
print("ASE path energy / kJ/mol =", energy_ase)
print("native path energy / kJ/mol =", energy_native)
print("difference =", abs(energy_ase - energy_native))

integrator = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds
)
simulation = app.Simulation(topology, native_system, integrator)
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

energies = []
for _ in range(50):
    simulation.step(1)
    energies.append(
        simulation.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )

plt.plot(energies)
plt.xlabel("step")
plt.ylabel("potential energy / kJ/mol")
plt.tight_layout()
plt.show()
