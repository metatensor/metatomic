from typing import Dict, List, Optional

import ase.build
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

import metatomic_lj_test
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)
from metatomic_ase import MetatomicCalculator


CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729


@pytest.fixture
def model():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=28,
        cutoff=CUTOFF,
        sigma=SIGMA,
        epsilon=EPSILON,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=False,
    )


@pytest.fixture
def atoms():
    np.random.seed(0xDEADBEEF)

    atoms = ase.build.make_supercell(
        ase.build.bulk("Ni", "fcc", a=3.6, cubic=True), 2 * np.eye(3)
    )
    atoms.positions += 0.2 * np.random.rand(*atoms.positions.shape)

    return atoms


def test_resolves_energy_ensemble_key(model, atoms):
    calc = MetatomicCalculator(
        model, device="cpu", check_consistency=True, uncertainty_threshold=None
    )
    assert calc._energy_ensemble_key == "energy_ensemble"

    # this reference model requires "energy" to be requested alongside
    # "energy_ensemble"; its value is unused here
    result = calc.run_model(
        atoms,
        {
            "energy_ensemble": ModelOutput(unit="eV", sample_kind="system"),
            "energy": ModelOutput(unit="eV", sample_kind="system"),
        },
    )
    assert result["energy_ensemble"].block().values.shape == (1, 16)


def test_variant(model, atoms):
    calc_doubled = MetatomicCalculator(
        model,
        device="cpu",
        check_consistency=True,
        variants={"energy": "doubled"},
        uncertainty_threshold=None,
    )
    assert calc_doubled._energy_ensemble_key == "energy_ensemble/doubled"

    calc = MetatomicCalculator(
        model, device="cpu", check_consistency=True, uncertainty_threshold=None
    )
    result = calc.run_model(
        atoms,
        {
            "energy_ensemble": ModelOutput(unit="eV", sample_kind="system"),
            "energy": ModelOutput(unit="eV", sample_kind="system"),
        },
    )
    result_doubled = calc_doubled.run_model(
        atoms,
        {
            "energy_ensemble/doubled": ModelOutput(unit="eV", sample_kind="system"),
            "energy/doubled": ModelOutput(unit="eV", sample_kind="system"),
        },
    )

    assert torch.allclose(
        2.0 * result["energy_ensemble"].block().values,
        result_doubled["energy_ensemble/doubled"].block().values,
        atol=1e-4,
    )


class _EnergyOnlyModel(torch.nn.Module):
    """Minimal model exposing only a plain 'energy' output, no 'energy_ensemble'."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        device = systems[0].device
        dtype = systems[0].positions.dtype
        samples = Labels(
            ["system"], torch.arange(len(systems), device=device).reshape(-1, 1)
        )
        block = TensorBlock(
            values=torch.zeros((len(systems), 1), dtype=dtype, device=device),
            samples=samples,
            components=[],
            properties=Labels("energy", torch.tensor([[0]], device=device)),
        )
        return {"energy": TensorMap(Labels("_", torch.tensor([[0]])), [block])}


@pytest.fixture
def model_without_ensemble():
    return AtomisticModel(
        _EnergyOnlyModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy": ModelOutput(sample_kind="system", unit="eV"),
            },
            atomic_types=[28],
            interaction_range=0.0,
            length_unit="Angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        ),
    )


def test_missing_energy_ensemble_output(model_without_ensemble):
    calc = MetatomicCalculator(
        model_without_ensemble,
        device="cpu",
        check_consistency=True,
        uncertainty_threshold=None,
    )
    assert calc._energy_ensemble_key is None
