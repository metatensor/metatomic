"""Tests for the MetatomicModel TorchSim wrapper.

Uses the metatomic-lj-test model so that tests run without
downloading large model files.  The pure-PyTorch LJ model
(``with_extension=False``) provides NC forces/stress, energy
uncertainty, and "/doubled" variants for full feature testing.
"""

import numpy as np
import pytest
import torch
import torch_sim as ts

import metatomic_lj_test
from metatomic.torch import ModelOutput
from metatomic_torchsim import MetatomicModel


CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729

DEVICE = torch.device("cpu")
DTYPE = torch.float64


@pytest.fixture
def lj_model():
    """Pure-PyTorch LJ model with NC, UQ, and variant outputs."""
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
def lj_model_ext():
    """Extension LJ model (no NC/UQ outputs)."""
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=28,
        cutoff=CUTOFF,
        sigma=SIGMA,
        epsilon=EPSILON,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=True,
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


# ---- Variants ----


def test_variants_default(lj_model, ni_atoms):
    """Default variant (None) selects the base energy output."""
    model = MetatomicModel(model=lj_model, device=DEVICE, variants={"energy": None})
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = model(sim_state)

    assert "energy" in output
    assert output["energy"].shape == (1,)


def test_variants_doubled(lj_model, ni_atoms):
    """Selecting the 'doubled' variant gives 2x the base energy."""
    model_base = MetatomicModel(model=lj_model, device=DEVICE)
    model_doubled = MetatomicModel(
        model=lj_model, device=DEVICE, variants={"energy": "doubled"}
    )

    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    e_base = model_base(sim_state)["energy"]
    e_doubled = model_doubled(sim_state)["energy"]

    torch.testing.assert_close(e_doubled, 2.0 * e_base, atol=1e-10, rtol=0)


# ---- Uncertainty ----


def test_uncertainty_warning_emitted(lj_model, ni_atoms):
    """Uncertainty warning fires when atoms exceed threshold."""
    # LJ test model's pseudo-uncertainty is 0.001 * n_atoms^2.
    # For 32 atoms: 0.001 * 32^2 = 1.024 per atom. Set threshold below that.
    model = MetatomicModel(model=lj_model, device=DEVICE, uncertainty_threshold=0.5)
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    with pytest.warns(UserWarning, match="uncertainty"):
        model(sim_state)


def test_uncertainty_no_warning_high_threshold(lj_model, ni_atoms):
    """No warning when threshold is above all uncertainties."""
    model = MetatomicModel(model=lj_model, device=DEVICE, uncertainty_threshold=1e6)
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    # Should not warn -- high threshold above all uncertainty values
    model(sim_state)


def test_uncertainty_threshold_none(lj_model, ni_atoms):
    """Setting uncertainty_threshold=None disables UQ entirely."""
    model = MetatomicModel(model=lj_model, device=DEVICE, uncertainty_threshold=None)
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    # Should not warn -- UQ disabled
    model(sim_state)


def test_negative_uncertainty_threshold_raises(lj_model):
    """Negative uncertainty_threshold raises ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        MetatomicModel(model=lj_model, device=DEVICE, uncertainty_threshold=-0.1)


# ---- Additional outputs ----


def test_additional_outputs_empty(lj_model, ni_atoms):
    """additional_outputs defaults to empty dict."""
    model = MetatomicModel(model=lj_model, device=DEVICE)
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    model(sim_state)
    assert model.additional_outputs == {}


def test_additional_outputs_requested(lj_model, ni_atoms):
    """Extra model outputs are stored in additional_outputs."""
    extra = {
        "energy_ensemble": ModelOutput(quantity="energy", unit="eV", per_atom=True),
    }
    model = MetatomicModel(model=lj_model, device=DEVICE, additional_outputs=extra)
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    model(sim_state)

    assert "energy_ensemble" in model.additional_outputs
    # energy_ensemble has 16 properties (ensemble members)
    block = model.additional_outputs["energy_ensemble"].block()
    assert block.values.shape[0] == len(ni_atoms)


# ---- Non-conservative ----


def test_non_conservative_forces(lj_model, ni_atoms):
    """NC forces are returned without autograd."""
    model = MetatomicModel(model=lj_model, device=DEVICE, non_conservative=True)
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = model(sim_state)

    assert "forces" in output
    assert output["forces"].shape == (len(ni_atoms), 3)
    # NC forces should have zero net force (mean-subtracted)
    net_force = output["forces"].sum(dim=0)
    torch.testing.assert_close(
        net_force, torch.zeros(3, dtype=DTYPE), atol=1e-6, rtol=0
    )


def test_non_conservative_stress(lj_model, ni_atoms):
    """NC stress is returned with correct shape."""
    model = MetatomicModel(model=lj_model, device=DEVICE, non_conservative=True)
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = model(sim_state)

    assert "stress" in output
    assert output["stress"].shape == (1, 3, 3)


def test_non_conservative_batched_forces(lj_model, ni_atoms):
    """NC net-force subtraction is per-system in batched mode."""
    model = MetatomicModel(model=lj_model, device=DEVICE, non_conservative=True)
    ni_atoms_2 = ni_atoms.copy()
    ni_atoms_2.positions += 0.3 * np.random.rand(*ni_atoms_2.positions.shape)

    sim_state = ts.io.atoms_to_state([ni_atoms, ni_atoms_2], DEVICE, DTYPE)
    output = model(sim_state)

    n1 = len(ni_atoms)
    n2 = len(ni_atoms_2)
    forces = output["forces"]
    assert forces.shape == (n1 + n2, 3)

    # Each system's forces should independently sum to zero
    net_1 = forces[:n1].sum(dim=0)
    net_2 = forces[n1:].sum(dim=0)
    torch.testing.assert_close(net_1, torch.zeros(3, dtype=DTYPE), atol=1e-6, rtol=0)
    torch.testing.assert_close(net_2, torch.zeros(3, dtype=DTYPE), atol=1e-6, rtol=0)


def test_non_conservative_missing_output_raises(lj_model_ext):
    """ValueError when model lacks NC outputs."""
    with pytest.raises((ValueError, RuntimeError), match="not found"):
        MetatomicModel(model=lj_model_ext, device=DEVICE, non_conservative=True)


def test_non_conservative_with_variants(lj_model, ni_atoms):
    """NC outputs respect variant selection."""
    model = MetatomicModel(
        model=lj_model,
        device=DEVICE,
        non_conservative=True,
        variants={
            "energy": "doubled",
            "non_conservative_forces": "doubled",
            "non_conservative_stress": "doubled",
        },
    )
    sim_state = ts.io.atoms_to_state([ni_atoms], DEVICE, DTYPE)
    output = model(sim_state)

    assert "energy" in output
    assert "forces" in output
    assert "stress" in output
