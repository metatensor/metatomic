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
from metatomic.torch.ase_calculator import MetatomicCalculator
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


@pytest.mark.parametrize("use_script", [True, False])
@pytest.mark.parametrize(
    "atoms, use_variant, expected",
    [
        ("atoms", True, [[9.0147e-05], [-2.6166e-04], [-1.9002e-04]]),
        ("atoms", False, [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]]),
        ("atoms_triclinic", True, [[1.0979e-04], [-2.7677e-04], [-1.7868e-04]]),
        ("atoms_triclinic", False, [[9.8061e-05], [-2.6314e-04], [-2.0004e-04]]),
    ],
    indirect=["atoms"],
)
def test_heat_flux_wrapper_calc_heat_flux(
    model, atoms, expected, use_script, use_variant
):
    metadata = ModelMetadata()
    wrapper = HeatFlux(
        model.eval(), variants=({"energy": "doubled"} if use_variant else None)
    )
    cap = model.capabilities()
    outputs = cap.outputs.copy()
    outputs[wrapper._hf_variant] = ModelOutput(
        quantity="heat_flux",
        unit="",
        explicit_gradients=[],
        per_atom=False,
    )

    new_cap = ModelCapabilities(
        outputs=outputs,
        atomic_types=cap.atomic_types,
        interaction_range=cap.interaction_range,
        length_unit=cap.length_unit,
        supported_devices=cap.supported_devices,
        dtype=cap.dtype,
    )

    heat_model = AtomisticModel(wrapper.eval(), metadata, capabilities=new_cap).to(
        device="cpu"
    )

    if use_script:
        heat_model = torch.jit.script(heat_model)

    calc = MetatomicCalculator(
        heat_model,
        device="cpu",
        additional_outputs={
            wrapper._hf_variant: ModelOutput(
                quantity="heat_flux",
                unit="",
                explicit_gradients=[],
                per_atom=False,
            )
        },
        check_consistency=True,
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    assert wrapper._hf_variant in atoms.calc.additional_outputs
    results = atoms.calc.additional_outputs[wrapper._hf_variant].block().values
    assert torch.allclose(
        results,
        torch.tensor(expected, dtype=results.dtype),
    )


@pytest.mark.parametrize("use_script", [True, False])
@pytest.mark.parametrize(
    "atoms, expected",
    [
        ("atoms", [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]]),
    ],
    indirect=["atoms"],
)
def test_wrap(model, atoms, expected, use_script):
    wrapped_model = HeatFlux.wrap(model, scripting=use_script)
    calc = MetatomicCalculator(
        wrapped_model,
        device="cpu",
        additional_outputs={
            "heat_flux": ModelOutput(
                quantity="heat_flux",
                unit="eV*A/fs",
                explicit_gradients=[],
                per_atom=False,
            )
        },
        check_consistency=True,
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    results = atoms.calc.additional_outputs["heat_flux"].block().values
    assert torch.allclose(
        results,
        torch.tensor(expected, dtype=results.dtype),
    )


@pytest.mark.parametrize("use_script", [True, False])
@pytest.mark.parametrize(
    "atoms, expected",
    [
        ("atoms", [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]]),
    ],
    indirect=["atoms"],
)
def test_input_energy_in_kcal_per_mol(model_in_kcal_per_mol, atoms, expected, use_script):
    wrapped_model = HeatFlux.wrap(model_in_kcal_per_mol, scripting=use_script)
    calc = MetatomicCalculator(
        wrapped_model,
        device="cpu",
        additional_outputs={
            "heat_flux": ModelOutput(
                quantity="heat_flux",
                unit="eV*A/fs",
                explicit_gradients=[],
                per_atom=False,
            )
        },
        check_consistency=True,
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    results = atoms.calc.additional_outputs["heat_flux"].block().values
    assert torch.allclose(results, torch.tensor(expected, dtype=results.dtype))


@pytest.mark.parametrize("use_script", [True, False])
@pytest.mark.parametrize(
    "atoms, expected",
    [
        ("atoms", [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]]),
    ],
    indirect=["atoms"],
)
def test_output_energy_in_kcal_per_mol(model, atoms, expected, use_script):
    wrapped_model = HeatFlux.wrap(model, scripting=use_script)
    calc = MetatomicCalculator(
        wrapped_model,
        device="cpu",
        additional_outputs={
            "heat_flux": ModelOutput(
                quantity="heat_flux",
                unit="kcal/mol*A/fs",
                explicit_gradients=[],
                per_atom=False,
            )
        },
        check_consistency=True,
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    results = atoms.calc.additional_outputs["heat_flux"].block().values
    expected_converted = torch.tensor(expected, dtype=results.dtype) * unit_conversion_factor(
        "eV*A/fs", "kcal/mol*A/fs"
    )
    assert torch.allclose(results, expected_converted, rtol=1e-3)