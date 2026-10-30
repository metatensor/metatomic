import math
from typing import Dict, List, Optional

import metatensor
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic_torch import LegacyModel

import metatomic
import metatomic_lj_test
from metatomic import Quantity, execute_model
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    register_autograd_neighbors,
    unit_conversion_factor,
)


CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729


def lj_model(length_unit="Angstrom", energy_unit="eV"):
    length = unit_conversion_factor(length_unit, "Angstrom")
    energy = unit_conversion_factor(energy_unit, "eV")
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=28,
        cutoff=CUTOFF / length,
        sigma=SIGMA / length,
        epsilon=EPSILON / energy,
        length_unit=length_unit,
        energy_unit=energy_unit,
        with_extension=False,
    )


@pytest.fixture(params=["eager", "script"])
def atomistic(request):
    model = lj_model()
    return model if request.param == "eager" else torch.jit.script(model)


@pytest.fixture
def system():
    # A fixed, distorted FCC cell avoids symmetry-induced zero gradients.
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    offsets = np.array(list(np.ndindex(2, 2, 2)))
    positions = 3.6 * (offsets[:, None, :] + basis).reshape(-1, 3)
    positions += 0.1 * np.sin(np.arange(positions.size).reshape(-1, 3))
    return system_with_pairs(
        np.full(len(positions), 28),
        positions,
        7.2 * np.eye(3),
        [True] * 3,
        [metatomic.PairListOptions(cutoff=CUTOFF, full_list=False, strict=True)],
    )


def system_with_pairs(
    types,
    positions,
    cell,
    pbc,
    pair_lists: list[metatomic.PairListOptions],
    length_unit: str = "angstrom",
    model_length_unit: Optional[str] = None,
) -> metatomic.System:
    """Build test systems and pair lists in the engine length unit."""
    import vesin

    types = np.asarray(types, dtype=np.int32)
    positions = np.asarray(positions)
    cell = np.asarray(cell, dtype=positions.dtype)
    pbc = np.asarray(pbc, dtype=bool)

    factor = 1.0
    if model_length_unit is not None:
        factor = unit_conversion_factor(model_length_unit, length_unit)

    system = metatomic.System(length_unit, types, positions, cell, pbc)
    for options in pair_lists:
        i, j, shifts, vectors = vesin.NeighborList(
            cutoff=options.cutoff * factor, full_list=options.full_list
        ).compute(
            points=positions.astype(np.float64),
            box=cell.astype(np.float64),
            periodic=pbc,
            quantities="ijSD",
        )
        samples = np.column_stack([i, j, shifts]).astype(np.int32)
        system.add_pairs(
            options,
            metatensor.TensorBlock(
                values=vectors.astype(positions.dtype).reshape(-1, 3, 1),
                samples=metatensor.Labels(
                    [
                        "first_atom",
                        "second_atom",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    samples.reshape(-1, 5),
                ),
                components=[metatensor.Labels.range("xyz", 3)],
                properties=metatensor.Labels.range("distance", 1),
            ),
        )
    return system


def legacy_reference(atomistic, systems, outputs, selected_atoms=None):
    """Evaluate the old API with independently constructed differentiable inputs."""
    legacy_systems = []
    positions = []
    strains = []
    for system in systems:
        position = (
            torch.as_tensor(system.positions).detach().clone().requires_grad_(True)
        )
        strain = torch.eye(3, dtype=torch.float64, requires_grad=True)
        legacy = System(
            torch.as_tensor(system.types),
            position @ strain,
            torch.as_tensor(system.cell) @ strain,
            torch.as_tensor(system.pbc),
        )
        for options in atomistic.requested_neighbor_lists():
            pairs = system.pairs(
                metatomic.PairListOptions(
                    cutoff=options.cutoff,
                    full_list=options.full_list,
                    strict=options.strict,
                )
            )
            block = TensorBlock(
                values=torch.as_tensor(pairs.values).detach().clone(),
                samples=Labels(pairs.samples.names, torch.tensor(pairs.samples.values)),
                components=[Labels("xyz", torch.arange(3).reshape(-1, 1))],
                properties=Labels("distance", torch.tensor([[0]])),
            )
            register_autograd_neighbors(legacy, block, check_consistency=True)
            legacy.add_neighbor_list(options, block)
        positions.append(position)
        strains.append(strain)
        legacy_systems.append(legacy)
    result = atomistic(
        legacy_systems,
        ModelEvaluationOptions(
            length_unit="Angstrom",
            outputs=outputs,
            selected_atoms=selected_atoms,
        ),
        check_consistency=True,
    )
    return result, positions, strains


def test_capabilities(atomistic):
    model = LegacyModel(atomistic)
    capabilities = model.capabilities()
    assert capabilities.length_unit == "Angstrom"
    assert capabilities.supported_devices == ["cpu", "cuda", "metal"]
    gradients = {(o.name, o.sample_kind): o.gradients for o in capabilities.outputs}
    assert gradients[("energy", "atom")] == []
    assert gradients[("energy", "system")] == ["positions", "strain"]
    assert gradients[("energy_ensemble", "system")] == ["positions", "strain"]
    assert gradients[("non_conservative_force", "atom")] == []
    assert gradients[("non_conservative_stress/doubled", "atom")] == []
    assert gradients[("non_conservative_stress/doubled", "system")] == []
    assert all(name != "non_conservative_forces" for name, _ in gradients)
    assert model.metadata().name == "Test Lennard-Jones"
    request = Quantity(name="energy", unit="eV", sample_kind="system")
    assert model.requested_pair_lists([request]) == [
        metatomic.PairListOptions(cutoff=CUTOFF, full_list=False, strict=True)
    ]
    assert model.requested_inputs([request]) == []


def test_against_legacy_model(atomistic, system):
    second = system_with_pairs(
        system.types,
        np.asarray(system.positions) * 1.03,
        np.asarray(system.cell) * 1.03,
        system.pbc,
        [metatomic.PairListOptions(cutoff=CUTOFF, full_list=False, strict=True)],
    )
    systems = [system, second]
    request = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    (energy,) = execute_model(
        LegacyModel(atomistic), systems, [request], check_consistency=True
    )
    reference, positions, strains = legacy_reference(
        atomistic,
        systems,
        {
            "energy": ModelOutput(unit="eV", sample_kind="system"),
        },
    )
    expected = reference["energy"].block()
    block = energy.block()
    np.testing.assert_array_equal(block.samples.values, expected.samples.values)
    np.testing.assert_allclose(block.values, expected.values.detach())
    derivatives = torch.autograd.grad(expected.values.sum(), positions + strains)
    np.testing.assert_allclose(
        block.gradient("positions").values[:, :, 0],
        torch.cat(derivatives[:2]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        block.gradient("strain").values[:, :, :, 0],
        torch.stack(derivatives[2:]),
        atol=1e-12,
    )


def test_per_atom_and_selected_atoms(atomistic, system):
    model = LegacyModel(atomistic)
    per_atom = Quantity(name="energy", unit="eV", sample_kind="atom")
    requests = [per_atom, Quantity(name="energy", unit="eV", sample_kind="system")]
    outputs = execute_model(model, [system], requests, check_consistency=True)
    reference, _, _ = legacy_reference(
        atomistic,
        [system],
        {
            "energy": ModelOutput(unit="eV", sample_kind="atom"),
        },
    )
    expected = reference["energy"].block()
    # The LJ model randomly shuffles per-atom samples on every evaluation.
    actual = outputs[0].block()
    actual_order = np.argsort(actual.samples.column("atom"))
    expected_order = torch.argsort(expected.samples.column("atom"))
    np.testing.assert_array_equal(
        actual.samples.values[actual_order], expected.samples.values[expected_order]
    )
    np.testing.assert_allclose(
        actual.values[actual_order], expected.values.detach()[expected_order]
    )
    np.testing.assert_allclose(
        outputs[1].block().values[0, 0], expected.values.sum().detach()
    )
    selected = metatensor.Labels(["system", "atom"], np.array([[0, 1], [0, 5]]))
    (output,) = execute_model(
        model, [system], [per_atom], selected_atoms=selected, check_consistency=True
    )
    reference, _, _ = legacy_reference(
        atomistic,
        [system],
        {
            "energy": ModelOutput(unit="eV", sample_kind="atom"),
        },
        Labels(selected.names, torch.tensor(selected.values)),
    )
    expected = reference["energy"].block()
    assert sorted(output.block().samples.column("atom")) == [1, 5]
    np.testing.assert_array_equal(
        output.block().samples.values, expected.samples.values
    )
    np.testing.assert_allclose(output.block().values, expected.values.detach())


def test_atom_declared_output_accepts_system_request(atomistic, system):
    model = LegacyModel(atomistic)
    name = "non_conservative_stress/doubled"
    assert atomistic.capabilities().outputs[name].sample_kind == "atom"
    assert {o.sample_kind for o in model.capabilities().outputs if o.name == name} == {
        "atom",
        "system",
    }
    request = Quantity(name=name, unit="eV/A^3", sample_kind="system")
    # The LJ stress output is random: replay the same RNG state for both calls.
    with torch.random.fork_rng(devices=[]):
        (stress,) = execute_model(model, [system], [request], check_consistency=True)
    with torch.random.fork_rng(devices=[]):
        reference, _, _ = legacy_reference(
            atomistic,
            [system],
            {name: ModelOutput(unit="eV/A^3", sample_kind="system")},
        )
    expected = reference[name].block()
    np.testing.assert_array_equal(
        stress.block().samples.values, expected.samples.values
    )
    np.testing.assert_allclose(stress.block().values, expected.values.detach())


@pytest.fixture
def units_results(system):
    reference = LegacyModel(lj_model())
    model = LegacyModel(lj_model(length_unit="Bohr", energy_unit="kJ/mol"))
    request = Quantity(
        name="energy",
        unit="kcal/mol",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    converted_pairs = system_with_pairs(
        system.types,
        system.positions,
        system.cell,
        system.pbc,
        model.requested_pair_lists([request]),
        model_length_unit="Bohr",
    )
    (expected,) = execute_model(reference, [system], [request])
    (actual,) = execute_model(
        model, [converted_pairs], [request], check_consistency=True
    )
    return actual.block(), expected.block()


def test_units(units_results):
    actual, expected = units_results
    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-6)
    np.testing.assert_allclose(
        actual.gradient("strain").values, expected.gradient("strain").values, rtol=1e-6
    )


def test_units_positions_gradients(units_results):
    actual, expected = units_results
    np.testing.assert_allclose(
        actual.gradient("positions").values,
        expected.gradient("positions").values,
        rtol=1e-6,
    )


@pytest.mark.parametrize("as_string", [False, True])
def test_load(tmp_path, system, as_string):
    path = tmp_path / "lj.pt"
    atomistic = lj_model()
    atomistic.save(str(path))
    model = LegacyModel(str(path) if as_string else path)
    request = Quantity(name="energy", unit="eV", sample_kind="system")
    (energy,) = execute_model(model, [system], [request], check_consistency=True)
    reference, _, _ = legacy_reference(
        atomistic,
        [system],
        {
            "energy": ModelOutput(unit="eV", sample_kind="system"),
        },
    )
    np.testing.assert_allclose(
        energy.block().values, reference["energy"].block().values.detach()
    )
    assert model.metadata().name == "Test Lennard-Jones"


def test_public_import():
    # Both the legacy import path and the package export expose the same wrapper.
    assert metatomic.torch.LegacyModel is LegacyModel


def test_errors():
    message = "`module` should be a torch.nn.Module"
    with pytest.raises(TypeError, match=message):
        LegacyModel(42)
    message = "`model` must be an AtomisticModel, got <class 'torch.nn.modules.*"
    with pytest.raises(TypeError, match=message):
        LegacyModel(torch.nn.Linear(2, 2))
    message = "can only override an empty model length unit"
    with pytest.raises(ValueError, match=message):
        LegacyModel(lj_model(), length_unit="Bohr")


class Harmonic(torch.nn.Module):
    """E = sum_i m_i |r_i|^2, with explicit positions gradients"""

    def __init__(self, input_name: str, gradient_offset: float):
        super().__init__()
        self.input_name = input_name
        # offset added to the explicit gradients, to check they are used as-is
        self.gradient_offset = gradient_offset
        self.requests: List[List[str]] = []

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return {self.input_name: ModelOutput(unit="u", sample_kind="atom")}

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        gradients = outputs["energy"].explicit_gradients
        self.requests.append(gradients)

        masses = [s.get_data(self.input_name).block().values for s in systems]
        energies = [
            (m * s.positions**2).sum() for m, s in zip(masses, systems, strict=True)
        ]
        block = TensorBlock(
            values=torch.stack(energies).reshape(-1, 1),
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )

        if "positions" in gradients:
            samples = []
            values = []
            for i, (m, s) in enumerate(zip(masses, systems, strict=True)):
                atoms = torch.arange(len(s))
                samples.append(
                    torch.stack([torch.full_like(atoms, i)] * 2 + [atoms], 1)
                )
                values.append(2 * m * s.positions.detach() + self.gradient_offset)

            block.add_gradient(
                "positions",
                TensorBlock(
                    values=torch.cat(values).reshape(-1, 3, 1),
                    samples=Labels(["sample", "system", "atom"], torch.cat(samples)),
                    components=[Labels("xyz", torch.tensor([[0], [1], [2]]))],
                    properties=block.properties,
                ),
            )

        return {"energy": TensorMap(Labels("_", torch.tensor([[0]])), [block])}


def harmonic_model(input_name="mass", gradient_offset=0.0):
    capabilities = ModelCapabilities(
        outputs={
            "energy": ModelOutput(
                unit="eV", sample_kind="system", explicit_gradients=["positions"]
            )
        },
        atomic_types=[1, 8],
        interaction_range=0.0,
        length_unit="Angstrom",
        supported_devices=["cpu"],
        dtype="float64",
    )
    module = Harmonic(input_name, gradient_offset).eval()
    return AtomisticModel(module, ModelMetadata(), capabilities), module


def harmonic_system(positions, masses):
    system = system_with_pairs(
        [8] + [1] * (len(positions) - 1), positions, np.zeros((3, 3)), [False] * 3, []
    )
    block = metatensor.TensorBlock(
        values=np.array(masses, dtype=np.float64).reshape(-1, 1),
        samples=metatensor.Labels(
            ["system", "atom"],
            np.array([[0, i] for i in range(len(positions))]),
        ),
        components=[],
        properties=metatensor.Labels.range("mass", 1),
    )
    system.add_custom_data(
        "mass", metatensor.TensorMap(metatensor.Labels.range("_", 1), [block])
    )
    return system


def test_explicit_gradients_and_inputs():
    atomistic, module = harmonic_model(gradient_offset=1.0)
    model = LegacyModel(atomistic)

    request = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    assert model.requested_inputs([request]) == [
        Quantity(name="mass", unit="u", sample_kind="atom")
    ]
    (declared,) = model.capabilities().outputs
    assert declared.gradients == ["positions", "strain"]
    assert model._raw_capabilities().outputs[0].gradients == ["positions"]

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.2, 1.5, 0.3]])
    masses = np.array([16.0, 1.0, 1.0])
    system = harmonic_system(positions, masses)
    (energy,) = execute_model(model, [system], [request], check_consistency=True)

    # only the explicit gradient is requested from the legacy model, the strain
    # gradient comes from autograd
    assert module.requests == [["positions"]]

    block = energy.block()
    np.testing.assert_allclose(
        block.values[0, 0], (masses[:, None] * positions**2).sum()
    )
    np.testing.assert_allclose(
        np.asarray(block.gradient("positions").values).reshape(-1, 3),
        2 * masses[:, None] * positions + 1.0,
    )
    np.testing.assert_allclose(
        np.asarray(block.gradient("strain").values).reshape(3, 3),
        2 * np.einsum("i,ia,ib->ab", masses, positions, positions),
        atol=1e-12,
    )


def test_empty_length_unit():
    atomistic, _ = harmonic_model()
    atomistic._capabilities.length_unit = ""

    message = "this model does not declare a length unit"
    with pytest.warns(UserWarning, match=message):
        model = LegacyModel(atomistic)
    assert model.capabilities().length_unit == "angstrom"

    model = LegacyModel(atomistic, length_unit="Bohr")
    assert model.capabilities().length_unit == "Bohr"

    # the system is converted to Bohr by `execute_model`, and not again by the model
    request = Quantity(name="energy", unit="eV", sample_kind="system")
    system = harmonic_system([[1.0, 0.0, 0.0]], [2.0])
    (energy,) = execute_model(model, [system], [request], check_consistency=True)
    np.testing.assert_allclose(
        energy.block().values[0, 0],
        2.0 / unit_conversion_factor("Bohr", "Angstrom") ** 2,
    )


def test_infinite_interaction_range():
    atomistic, _ = harmonic_model()
    atomistic._capabilities.interaction_range = math.inf
    model = LegacyModel(atomistic)
    assert model.capabilities().interaction_range == np.finfo(np.float64).max

    request = Quantity(name="energy", unit="eV", sample_kind="system")
    system = harmonic_system([[1.0, 0.0, 0.0]], [2.0])
    (energy,) = execute_model(model, [system], [request], check_consistency=True)
    np.testing.assert_allclose(energy.block().values[0, 0], 2.0)
