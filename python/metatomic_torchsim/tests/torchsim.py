"""Tests for the MetatomicModel TorchSim wrapper.

Uses the metatomic-lj-test model so that tests run without
downloading large model files.
"""

import numpy as np
import pytest
import torch
import torch_sim as ts

import metatomic_lj_test
from metatomic_torchsim import MetatomicModel


CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729

DEVICE = torch.device("cpu")
DTYPE = torch.float64


@pytest.fixture
def lj_model():
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
def ni_atoms():
    """Create a small perturbed Ni FCC supercell."""
    import ase.build

    np.random.seed(0xDEADBEEF)
    atoms = ase.build.make_supercell(
        ase.build.bulk("Ni", "fcc", a=3.6, cubic=True), 2 * np.eye(3)
    )
    atoms.positions += 0.2 * np.random.rand(*atoms.positions.shape)
    return atoms


@pytest.fixture
def metatomic_model(lj_model):
    return MetatomicModel(model=lj_model, device=DEVICE)


def test_initialization(lj_model):
    """MetatomicModel initializes with correct device and dtype."""
    model = MetatomicModel(model=lj_model, device=DEVICE)
    assert model.device == DEVICE
    assert model.dtype == DTYPE
    assert model.compute_forces is True
    assert model.compute_stress is True


def test_initialization_no_forces(lj_model):
    """Can disable force computation."""
    model = MetatomicModel(model=lj_model, device=DEVICE, compute_forces=False)
    assert model.compute_forces is False
    assert model.compute_stress is True


def test_forward_returns_energy(metatomic_model, ni_atoms):
    """Forward pass returns energy with correct shape."""
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = metatomic_model(sim_state)

    assert "energy" in output
    assert output["energy"].shape == (1,)
    assert output["energy"].dtype == DTYPE


def test_forward_returns_forces(metatomic_model, ni_atoms):
    """Forward pass returns forces with correct shape."""
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = metatomic_model(sim_state)

    assert "forces" in output
    n_atoms = len(ni_atoms)
    assert output["forces"].shape == (n_atoms, 3)
    assert output["forces"].dtype == DTYPE


def test_forward_returns_stress(metatomic_model, ni_atoms):
    """Forward pass returns stress with correct shape."""
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = metatomic_model(sim_state)

    assert "stress" in output
    assert output["stress"].shape == (1, 3, 3)
    assert output["stress"].dtype == DTYPE


def test_forward_no_stress(lj_model, ni_atoms):
    """Stress is not returned when compute_stress=False."""
    model = MetatomicModel(model=lj_model, device=DEVICE, compute_stress=False)
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = model(sim_state)

    assert "energy" in output
    assert "forces" in output
    assert "stress" not in output


def test_forward_no_forces(lj_model, ni_atoms):
    """Forces are not returned when compute_forces=False."""
    model = MetatomicModel(model=lj_model, device=DEVICE, compute_forces=False)
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = model(sim_state)

    assert "energy" in output
    assert "forces" not in output
    assert "stress" in output


@pytest.fixture
def ni_atoms_2():
    """Create a second Ni supercell (same size, different lattice parameter)."""
    import ase.build

    np.random.seed(0xCAFEBABE)
    atoms = ase.build.make_supercell(
        ase.build.bulk("Ni", "fcc", a=3.5, cubic=True), 2 * np.eye(3)
    )
    atoms.positions += 0.1 * np.random.rand(*atoms.positions.shape)
    return atoms


def test_batched_forward(metatomic_model, ni_atoms, ni_atoms_2):
    """Forward pass handles batched systems correctly."""
    sim_state = ts.io.atoms_to_state([ni_atoms, ni_atoms_2], DEVICE, DTYPE)
    output = metatomic_model(sim_state)

    assert output["energy"].shape == (2,)
    n_total = len(ni_atoms) + len(ni_atoms_2)
    assert output["forces"].shape == (n_total, 3)
    assert output["stress"].shape == (2, 3, 3)


def test_energy_consistency_single_vs_batch(metatomic_model, ni_atoms, ni_atoms_2):
    """Energy from single system matches the corresponding entry in a batch."""

    # single
    state_1 = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    out_1 = metatomic_model(state_1)

    state_2 = ts.io.atoms_to_state([ni_atoms_2], DEVICE, DTYPE)
    out_2 = metatomic_model(state_2)

    # batch
    state_batch = ts.io.atoms_to_state([ni_atoms, ni_atoms_2], DEVICE, DTYPE)
    out_batch = metatomic_model(state_batch)

    torch.testing.assert_close(out_1["energy"], out_batch["energy"][:1])
    torch.testing.assert_close(out_2["energy"], out_batch["energy"][1:])


def test_forces_sum_to_zero(metatomic_model, ni_atoms):
    """Net force on the system should be approximately zero (Newton's 3rd law)."""
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = metatomic_model(sim_state)

    net_force = output["forces"].sum(dim=0)
    torch.testing.assert_close(
        net_force, torch.zeros(3, dtype=DTYPE), atol=1e-6, rtol=0
    )


def test_validate_model_outputs(metatomic_model):
    """Model passes TorchSim's validate_model_outputs check."""
    try:
        from torch_sim.models.interface import validate_model_outputs
    except ImportError:
        pytest.skip("validate_model_outputs not available in this torch-sim version")

    # validate_model_outputs creates its own test systems (Si diamond + Fe FCC).
    # Our LJ model only knows atomic_type=28 (Ni), but the validator uses Si (14)
    # and Fe (26).  So we skip if the validator would fail for type reasons.
    try:
        validate_model_outputs(metatomic_model, DEVICE, DTYPE)
    except Exception as exc:
        if "atomic type" in str(exc).lower() or "species" in str(exc).lower():
            pytest.skip(f"LJ test model does not support Si/Fe types: {exc}")
        raise


def test_wrong_dtype_raises(metatomic_model, ni_atoms):
    """TypeError raised when positions have wrong dtype."""
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, torch.float32)
    with pytest.raises(TypeError, match="dtype"):
        metatomic_model(sim_state)


def test_single_atom_system(lj_model):
    """Model handles a single-atom system."""
    import ase

    atoms = ase.Atoms(
        symbols=["Ni"],
        positions=[[0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    model = MetatomicModel(model=lj_model, device=DEVICE)
    sim_state = ts.io.atoms_to_state([atoms], DEVICE, DTYPE)
    output = model(sim_state)

    assert output["energy"].shape == (1,)
    assert output["forces"].shape == (1, 3)
    assert output["stress"].shape == (1, 3, 3)


def test_energy_only_mode(lj_model, ni_atoms):
    """Model returns only energy when forces and stress are disabled."""
    model = MetatomicModel(
        model=lj_model, device=DEVICE, compute_forces=False, compute_stress=False
    )
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = model(sim_state)

    assert "energy" in output
    assert "forces" not in output
    assert "stress" not in output


def test_check_consistency_mode(lj_model, ni_atoms):
    """Model runs with consistency checking enabled."""
    model = MetatomicModel(model=lj_model, device=DEVICE, check_consistency=True)
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = model(sim_state)

    assert "energy" in output
    assert "forces" in output
    assert "stress" in output


def test_forces_match_finite_difference(lj_model, ni_atoms):
    """Autograd forces match finite-difference gradient of energy."""
    delta = 1e-4
    model = MetatomicModel(model=lj_model, device=DEVICE, compute_stress=False)
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = model(sim_state)
    autograd_forces = output["forces"]

    for i in range(3):
        for j in range(3):
            atoms_plus = ni_atoms.copy()
            atoms_minus = ni_atoms.copy()
            atoms_plus.positions[i, j] += delta
            atoms_minus.positions[i, j] -= delta

            state_plus = ts.io.atoms_to_state([atoms_plus], DEVICE, DTYPE)
            state_minus = ts.io.atoms_to_state([atoms_minus], DEVICE, DTYPE)

            e_plus = model(state_plus)["energy"][0]
            e_minus = model(state_minus)["energy"][0]

            numerical_force = -(e_plus - e_minus) / (2 * delta)
            torch.testing.assert_close(
                autograd_forces[i, j],
                numerical_force,
                atol=1e-4,
                rtol=0,
            )


def test_stress_is_symmetric(metatomic_model, ni_atoms):
    """Stress tensor is symmetric."""
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = metatomic_model(sim_state)
    stress = output["stress"]

    torch.testing.assert_close(stress, stress.transpose(-2, -1), atol=1e-10, rtol=0)
