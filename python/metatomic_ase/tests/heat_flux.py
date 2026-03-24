import numpy as np
import pytest
import torch
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import metatomic_lj_test
from metatomic.torch import ModelOutput
from metatomic.torch.heat_flux import (
    HeatFlux,
)
from metatomic_ase import MetatomicCalculator


@pytest.fixture
def model():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=18,
        cutoff=7.0,
        sigma=3.405,
        epsilon=0.01032,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=False,
    )


@pytest.fixture
def atoms(request):
    if hasattr(request, "param") and request.param == "atoms_triclinic":
        cell = np.array([[6.0, 3.0, 1.0], [2.0, 6.0, 0.0], [0.0, 0.0, 6.0]])
        positions = np.array([[0.0, 0.0, 0.0]])
    else:
        cell = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]])
        positions = np.array([[3.0, 3.0, 3.0]])
    atoms = Atoms("Ar", scaled_positions=positions, cell=cell, pbc=True).repeat(
        (2, 2, 2)
    )
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300, rng=np.random.default_rng(42)
    )
    return atoms


def test_wrap(model, atoms):
    wrapped_model = HeatFlux.wrap(model)
    calc = MetatomicCalculator(
        wrapped_model,
        device="cpu",
        additional_outputs={
            "heat_flux": ModelOutput(
                quantity="heat_flux",
                unit="eV*A/fs",
                explicit_gradients=[],
                sample_kind="system",
            )
        },
        check_consistency=True,
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    heat_flux = atoms.calc.additional_outputs["heat_flux"].block().values

    expected = torch.tensor(
        [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]], dtype=heat_flux.dtype
    )
    assert torch.allclose(heat_flux, expected)
