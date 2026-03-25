"""
.. _torchsim-batched:

Batched simulations with TorchSim
=================================

TorchSim supports batching multiple systems into a single ``SimState``
for efficient parallel evaluation on GPU.
:py:class:`~metatomic_torchsim.MetatomicModel` handles this
transparently.
"""

# %%
#
# Setup
# -----
#
# We reuse the same minimal model from :ref:`torchsim-getting-started`.

import os
import shutil
import tempfile
from typing import Dict, List, Optional

import ase.build
import torch
import torch_sim as ts
from metatensor.torch import Labels, TensorBlock, TensorMap

import metatomic.torch as mta
from metatomic_torchsim import MetatomicModel


class ConstantEnergy(torch.nn.Module):
    """Assigns a constant energy per atom."""

    def __init__(self, energy_per_atom: float = -1.0):
        super().__init__()
        self.energy_per_atom = energy_per_atom

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energies = []
        for system in systems:
            energies.append(self.energy_per_atom * len(system))

        energy = torch.tensor(energies, dtype=systems[0].positions.dtype).reshape(-1, 1)
        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {
            "energy": TensorMap(keys=Labels("_", torch.tensor([[0]])), blocks=[block])
        }


tmpdir = tempfile.mkdtemp()
model_path = os.path.join(tmpdir, "constant-energy.pt")

capabilities = mta.ModelCapabilities(
    length_unit="Angstrom",
    atomic_types=[13, 29],  # Al, Cu
    interaction_range=0.0,
    outputs={"energy": mta.ModelOutput(quantity="energy", unit="eV")},
    supported_devices=["cpu"],
    dtype="float64",
)

atomistic_model = mta.AtomisticModel(
    ConstantEnergy(-1.5).eval(), mta.ModelMetadata(), capabilities
)
atomistic_model.save(model_path)

model = MetatomicModel(model_path, device="cpu")


# %%
#
# Creating a batched state
# ------------------------
#
# Pass a list of ASE ``Atoms`` objects to ``initialize_state``:

atoms_list = [
    ase.build.bulk("Cu", "fcc", a=3.6, cubic=True),
    ase.build.bulk("Cu", "fcc", a=3.65, cubic=True),
    ase.build.bulk("Al", "fcc", a=4.05, cubic=True),
]

sim_state = ts.initialize_state(atoms_list, device=model.device, dtype=model.dtype)
print("Total atoms in batch:", sim_state.n_atoms)

# %%
#
# Evaluating the batch
# --------------------
#
# A single forward call evaluates all systems:

results = model(sim_state)

print("Energy shape:", results["energy"].shape)  # [n_systems]
print("Forces shape:", results["forces"].shape)  # [n_total_atoms, 3]
print("Stress shape:", results["stress"].shape)  # [n_systems, 3, 3]

# %%
#
# The output shapes reflect the batch:
#
# - ``results["energy"]`` has shape ``[n_systems]`` -- one energy per system
# - ``results["forces"]`` has shape ``[n_total_atoms, 3]`` -- all atoms
#   concatenated
# - ``results["stress"]`` has shape ``[n_systems, 3, 3]`` -- one 3x3 tensor
#   per system

print("Per-system energies:", results["energy"])

# %%
#
# How ``system_idx`` works
# ------------------------
#
# ``SimState`` tracks which atom belongs to which system via the
# ``system_idx`` tensor. For three 4-atom systems, ``system_idx`` looks
# like:

print("system_idx:", sim_state.system_idx)

# %%
#
# ``MetatomicModel.forward`` uses this to split the batched positions and
# types into per-system ``System`` objects before calling the underlying
# model.
#
# Batch consistency
# -----------------
#
# Energies computed in a batch match those computed individually.
# This is guaranteed because each system gets its own neighbor list and
# independent evaluation:

individual_energies = []
for atoms in atoms_list:
    state = ts.initialize_state(atoms, device=model.device, dtype=model.dtype)
    res = model(state)
    individual_energies.append(res["energy"].item())

print("Batched:   ", [e.item() for e in results["energy"]])
print("Individual:", individual_energies)

# %%
#
# Performance considerations
# --------------------------
#
# Batching is most beneficial on GPU, where the neighbor list computation
# and model forward pass can run in parallel across systems. On CPU, the
# speedup comes from reduced Python overhead (one call instead of N).
#
# For very large systems or many small ones, adjust the batch size to fit
# in GPU memory. TorchSim does not impose a maximum batch size, but each
# system gets its own neighbor list, so memory scales with the sum of
# per-system sizes.

# %%
#
# Cleanup:

shutil.rmtree(tmpdir)
