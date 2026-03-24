import numpy as np
import pytest
import torch
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import metatomic_lj_test
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    unit_conversion_factor,
)
from metatomic.torch.heat_flux import (
    HeatFlux,
)


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
def model_in_kcal_per_mol():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=18,
        cutoff=7.0,
        sigma=3.405,
        epsilon=0.2380,
        length_unit="Angstrom",
        energy_unit="kcal/mol",
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


def test_heat_flux_wrapper_requested_inputs(model):
    wrapper = HeatFlux(model)
    requested = wrapper.requested_inputs()
    assert set(requested.keys()) == {"masses", "velocities"}
