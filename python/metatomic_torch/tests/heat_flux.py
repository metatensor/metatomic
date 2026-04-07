import numpy as np
import pytest
import torch
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from metatensor.torch import Labels, TensorBlock, TensorMap
from vesin.metatomic import compute_requested_neighbors_from_options

import metatomic_lj_test
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    systems_to_torch,
    unit_conversion_factor,
)
from metatomic.torch.heat_flux import (
    HeatFlux,
)


@pytest.fixture
def model(capfd):
    m = metatomic_lj_test.lennard_jones_model(
        atomic_type=18,
        cutoff=7.0,
        sigma=3.405,
        epsilon=0.01032,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=False,
    )
    # consume quantity deprecation warning from C++
    captured = capfd.readouterr()
    if captured.err:
        assert "ModelOutput.quantity is deprecated" in captured.err
    return m


@pytest.fixture
def model_in_kcal_per_mol(capfd):
    m = metatomic_lj_test.lennard_jones_model(
        atomic_type=18,
        cutoff=7.0,
        sigma=3.405,
        epsilon=0.2380,
        length_unit="Angstrom",
        energy_unit="kcal/mol",
        with_extension=False,
    )
    captured = capfd.readouterr()
    if captured.err:
        assert "ModelOutput.quantity is deprecated" in captured.err
    return m


@pytest.fixture
def system(request):
    if hasattr(request, "param") and request.param == "system_triclinic":
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
    system = systems_to_torch(
        atoms,
        dtype=(torch.float64),
    )
    compute_requested_neighbors_from_options(
        system, [NeighborListOptions(7.0, False, True)], "Angstrom", True
    )

    # Add additional data for heat flux calculation
    n_atoms = len(atoms)
    masses = (
        torch.as_tensor(atoms.get_masses())
        .reshape(-1, 1)
        .to(device=system.device, dtype=system.positions.dtype)
    )
    velocities = (
        torch.as_tensor(atoms.get_velocities())
        .reshape(-1, 3, 1)
        .to(device=system.device, dtype=system.positions.dtype)
    )
    masses_block = TensorBlock(
        values=masses,
        samples=Labels(
            ["system", "atom"],
            torch.vstack([torch.full((n_atoms,), 0), torch.arange(n_atoms)]).T,
        ),
        components=[],
        properties=Labels(["mass"], torch.tensor([[0]])),
    )
    velocities_block = TensorBlock(
        values=velocities,
        samples=Labels(
            ["system", "atom"],
            torch.vstack([torch.full((n_atoms,), 0), torch.arange(n_atoms)]).T,
        ),
        components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
        properties=Labels(["velocity"], torch.tensor([[0]])),
    )
    masses_tensor = TensorMap(
        keys=Labels(["_"], torch.tensor([[0]])),
        blocks=[masses_block],
    )
    velocities_tensor = TensorMap(
        keys=Labels(["_"], torch.tensor([[0]])),
        blocks=[velocities_block],
    )
    masses_tensor.set_info("quantity", "mass")
    velocities_tensor.set_info("quantity", "velocity")
    masses_tensor.set_info("unit", "u")
    velocities_tensor.set_info("unit", "(eV/u)^(1/2)")
    system.add_data("masses", masses_tensor)
    system.add_data("velocities", velocities_tensor)
    return system


def test_heat_flux_wrapper_requested_inputs(model):
    wrapper = HeatFlux(model)
    requested = wrapper.requested_inputs()
    assert set(requested.keys()) == {"masses", "velocities"}


@pytest.mark.parametrize("use_script", [True, False])
@pytest.mark.parametrize(
    "system, use_variant, expected",
    [
        ("system", True, [[9.0147e-05], [-2.6166e-04], [-1.9002e-04]]),
        ("system", False, [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]]),
        ("system_triclinic", True, [[1.0979e-04], [-2.7677e-04], [-1.7868e-04]]),
        ("system_triclinic", False, [[9.8061e-05], [-2.6314e-04], [-2.0004e-04]]),
    ],
    indirect=["system"],
)
def test_heat_flux_wrapper_calc_heat_flux(
    model, system, expected, use_script, use_variant, capfd
):
    hf_variant = "heat_flux/doubled" if use_variant else "heat_flux"
    metadata = ModelMetadata()
    wrapper = HeatFlux(
        model.eval(), variants=({"energy": "doubled"} if use_variant else None)
    )
    if use_variant:
        evaulation_options = ModelEvaluationOptions(
            length_unit="Angstrom",
            outputs={
                hf_variant: ModelOutput(
                    quantity="heat_flux", unit="eV*A/fs", per_atom=False
                )
            },
        )
    else:
        evaulation_options = ModelEvaluationOptions(
            length_unit="Angstrom",
            outputs={
                "heat_flux": ModelOutput(
                    quantity="heat_flux", unit="eV*A/fs", per_atom=False
                )
            },
        )
    cap = model.capabilities()
    outputs = cap.outputs.copy()
    outputs[hf_variant] = ModelOutput(
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

    results = heat_model([system], evaulation_options, True)[hf_variant].block().values
    assert torch.allclose(
        results,
        torch.tensor(expected, dtype=results.dtype),
    )
    # consume quantity deprecation warnings from ModelOutput(quantity=...) calls
    captured = capfd.readouterr()
    if captured.err:
        assert "ModelOutput.quantity is deprecated" in captured.err


@pytest.mark.parametrize("use_script", [True, False])
@pytest.mark.parametrize(
    "system, expected",
    [
        ("system", [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]]),
    ],
    indirect=["system"],
)
def test_wrap(model, system, expected, use_script, capfd):
    wrapped_model = HeatFlux.wrap(model, scripting=use_script)
    evaulation_options = ModelEvaluationOptions(
        length_unit="Angstrom",
        outputs={
            "heat_flux": ModelOutput(
                quantity="heat_flux", unit="eV*A/fs", per_atom=False
            )
        },
    )
    results = (
        wrapped_model([system], evaulation_options, True)["heat_flux"].block().values
    )
    assert torch.allclose(
        results,
        torch.tensor(expected, dtype=results.dtype),
    )
    captured = capfd.readouterr()
    if captured.err:
        assert "ModelOutput.quantity is deprecated" in captured.err


@pytest.mark.parametrize("use_script", [True, False])
@pytest.mark.parametrize(
    "system, expected",
    [
        ("system", [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]]),
    ],
    indirect=["system"],
)
def test_input_energy_in_kcal_per_mol(
    model_in_kcal_per_mol, system, expected, use_script, capfd
):
    wrapped_model = HeatFlux.wrap(model_in_kcal_per_mol, scripting=use_script)
    evaulation_options = ModelEvaluationOptions(
        length_unit="Angstrom",
        outputs={
            "heat_flux": ModelOutput(
                quantity="heat_flux", unit="eV*A/fs", per_atom=False
            )
        },
    )
    results = (
        wrapped_model([system], evaulation_options, True)["heat_flux"].block().values
    )
    assert torch.allclose(results, torch.tensor(expected, dtype=results.dtype))
    captured = capfd.readouterr()
    if captured.err:
        assert "ModelOutput.quantity is deprecated" in captured.err


@pytest.mark.parametrize("use_script", [True, False])
@pytest.mark.parametrize(
    "system, expected",
    [
        ("system", [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]]),
    ],
    indirect=["system"],
)
def test_output_unit_conversion(model, system, expected, use_script, capfd):
    wrapped_model = HeatFlux.wrap(model, scripting=use_script)
    evaulation_options = ModelEvaluationOptions(
        length_unit="Angstrom",
        outputs={
            "heat_flux": ModelOutput(
                quantity="heat_flux", unit="kcal/mol*A/ps", per_atom=False
            )
        },
    )
    wrapped_model = HeatFlux.wrap(model, scripting=use_script)
    results = (
        wrapped_model([system], evaulation_options, True)["heat_flux"].block().values
    )
    expected_converted = torch.tensor(
        expected, dtype=results.dtype
    ) * unit_conversion_factor("eV*A/fs", "kcal/mol*A/ps")
    assert torch.allclose(results, expected_converted, rtol=1e-3)
    captured = capfd.readouterr()
    if captured.err:
        assert "ModelOutput.quantity is deprecated" in captured.err
