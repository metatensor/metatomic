"""Legacy Torch compatibility with explicit gradients, as introduced by #367."""

from typing import Dict, List, Optional

import ase
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)
from metatomic_ase import MetatomicCalculator


class HarmonicModel(torch.nn.Module):
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        # Explicit-only outputs deliberately have no graph, like batched averaging
        # in SymmetrizedModel. Mixed requests keep a graph for the missing gradient.
        values = []
        position_values = []
        position_samples = torch.jit.annotate(List[List[int]], [])
        strain_values = []
        for i, system in enumerate(systems):
            positions = system.positions
            values.append((positions**2).sum())
            position_values.append(2 * positions.detach())
            for atom in range(len(system)):
                position_samples.append([i, i, atom])
            strain_values.append(2 * positions.detach().T @ positions.detach())
        energy = torch.stack(values).reshape(-1, 1)
        gradients = outputs["energy"].explicit_gradients
        if "positions" in gradients and "strain" in gradients:
            energy = energy.detach()
        block = TensorBlock(
            energy,
            Labels.range("system", len(systems)),
            [],
            Labels.range("energy", 1),
        )
        if "positions" in gradients:
            block.add_gradient(
                "positions",
                TensorBlock(
                    torch.cat(position_values).unsqueeze(-1),
                    Labels(
                        ["sample", "system", "atom"], torch.tensor(position_samples)
                    ),
                    [Labels.range("xyz", 3)],
                    block.properties,
                ),
            )
        if "strain" in gradients:
            block.add_gradient(
                "strain",
                TensorBlock(
                    torch.stack(strain_values).unsqueeze(-1),
                    Labels.range("sample", len(systems)),
                    [Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)],
                    block.properties,
                ),
            )
        return {"energy": TensorMap(Labels.single(), [block])}


@pytest.mark.parametrize(
    "gradients", [[], ["positions"], ["strain"], ["positions", "strain"]]
)
@pytest.mark.parametrize("scripted", [False, True])
def test_legacy_explicit_and_autograd(gradients, scripted):
    model = AtomisticModel(
        HarmonicModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy": ModelOutput(
                    unit="eV", sample_kind="system", explicit_gradients=gradients
                )
            },
            atomic_types=[1],
            interaction_range=0,
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        ),
    )
    if scripted:
        model = torch.jit.script(model)
    calculator = MetatomicCalculator(model, check_consistency=True)
    structures = [
        ase.Atoms(
            "H2", positions=[[0.1, 0.2, 0.3], [0.7, 0.5, 0.4]], cell=[3, 4, 5], pbc=True
        ),
        ase.Atoms("H", positions=[[0.4, 0.3, 0.2]], cell=[4, 5, 6], pbc=True),
    ]
    batch = calculator.compute_energy(structures, compute_forces_and_stresses=True)
    for i, atoms in enumerate(structures):
        positions = atoms.positions
        expected_energy = (positions**2).sum()
        expected_forces = -2 * positions
        expected_stress = 2 * positions.T @ positions / atoms.get_volume()
        atoms.calc = calculator
        assert atoms.get_potential_energy() == pytest.approx(expected_energy)
        np.testing.assert_allclose(atoms.get_forces(), expected_forces)
        np.testing.assert_allclose(atoms.get_stress(voigt=False), expected_stress)
        assert batch["energy"][i] == pytest.approx(expected_energy)
        np.testing.assert_allclose(batch["forces"][i], expected_forces)
        np.testing.assert_allclose(batch["stress"][i], expected_stress)
        assert calculator.compute_energy(atoms)["energy"] == pytest.approx(
            expected_energy
        )
