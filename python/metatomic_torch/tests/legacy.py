import math
from typing import Dict, List, Optional

import ase.build
import ase.units
import metatensor
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic_torch.legacy import LegacyModel, system_with_pairs

import metatomic
import metatomic_lj_test
from metatomic import Quantity, execute_model
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


def lj_model(length_unit="Angstrom", energy_unit="eV"):
    length = ase.units.Angstrom if length_unit == "Angstrom" else ase.units.Bohr
    energy = 1.0 if energy_unit == "eV" else ase.units.kJ / ase.units.mol
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


def nickel(seed):
    atoms = ase.build.make_supercell(
        ase.build.bulk("Ni", "fcc", a=3.6, cubic=True), 2 * np.eye(3)
    )
    atoms.positions += 0.2 * np.random.default_rng(seed).random(atoms.positions.shape)
    return atoms


def to_system(atoms, model, requests):
    return system_with_pairs(
        atoms.numbers,
        atoms.positions,
        atoms.cell[:],
        atoms.pbc,
        model.requested_pair_lists(requests),
        model_length_unit=model.capabilities().length_unit,
    )


def energy_request(gradients=("positions", "strain"), unit="eV"):
    return Quantity(
        name="energy", unit=unit, sample_kind="system", gradients=list(gradients)
    )


def test_capabilities(atomistic):
    model = LegacyModel(atomistic)
    capabilities = model.capabilities()

    assert capabilities.length_unit == "Angstrom"
    assert capabilities.supported_devices == ["cpu", "cuda", "metal"]
    outputs = {(o.name, o.sample_kind): o.gradients for o in capabilities.outputs}

    # per-atom outputs can also be requested per-system, as in the legacy API, and
    # get autograd gradients for energies
    assert outputs[("energy", "atom")] == []
    assert outputs[("energy", "system")] == ["positions", "strain"]
    assert outputs[("energy_ensemble", "system")] == ["positions", "strain"]
    assert outputs[("non_conservative_force", "atom")] == []
    assert outputs[("non_conservative_stress/doubled", "atom")] == []
    assert outputs[("non_conservative_stress/doubled", "system")] == []
    # AtomisticModel exposes the deprecated alias, but metatomic only knows the new
    assert all(name != "non_conservative_forces" for name, _ in outputs)

    assert model.metadata().name == "Test Lennard-Jones"
    assert model.requested_pair_lists([energy_request()]) == [
        metatomic.PairListOptions(cutoff=CUTOFF, full_list=False, strict=True)
    ]
    assert model.requested_inputs([energy_request()]) == []


def test_against_legacy_engine(atomistic):
    model = LegacyModel(atomistic)
    request = energy_request()
    # the LJ test model can not batch systems with different sizes
    all_atoms = [nickel(0), nickel(1)]

    systems = [to_system(atoms, model, [request]) for atoms in all_atoms]
    (energy,) = execute_model(model, systems, [request], check_consistency=True)
    block = energy.block()
    forces = -np.asarray(block.gradient("positions").values).reshape(-1, 3)
    strain = np.asarray(block.gradient("strain").values).reshape(-1, 3, 3)

    start = 0
    for i, atoms in enumerate(all_atoms):
        atoms.calc = MetatomicCalculator(
            atomistic, check_consistency=True, uncertainty_threshold=None
        )
        np.testing.assert_allclose(block.values[i, 0], atoms.get_potential_energy())
        np.testing.assert_allclose(
            forces[start : start + len(atoms)], atoms.get_forces(), atol=1e-12
        )
        np.testing.assert_allclose(
            strain[i] / atoms.cell.volume, atoms.get_stress(voigt=False), atol=1e-12
        )
        start += len(atoms)


def test_per_atom_and_selected_atoms(atomistic):
    model = LegacyModel(atomistic)
    atoms = nickel(0)
    atoms.calc = MetatomicCalculator(atomistic, uncertainty_threshold=None)
    reference = atoms.get_potential_energies()

    per_atom = Quantity(name="energy", unit="eV", sample_kind="atom")
    # the same name requested twice goes through two calls to the legacy model
    requests = [per_atom, energy_request(gradients=[])]
    outputs = execute_model(
        model, [to_system(atoms, model, requests)], requests, check_consistency=True
    )
    # the LJ test model does not sort the atoms in its outputs
    block = outputs[0].block()
    np.testing.assert_allclose(
        np.asarray(block.values)[:, 0], reference[block.samples.column("atom")]
    )
    np.testing.assert_allclose(outputs[1].block().values[0, 0], reference.sum())

    selected = metatensor.Labels(["system", "atom"], np.array([[0, 1], [0, 5]]))
    (output,) = execute_model(
        model,
        [to_system(atoms, model, [per_atom])],
        [per_atom],
        selected_atoms=selected,
        check_consistency=True,
    )
    block = output.block()
    assert sorted(block.samples.column("atom")) == [1, 5]
    np.testing.assert_allclose(
        np.asarray(block.values)[:, 0], reference[block.samples.column("atom")]
    )


def test_per_atom_output_per_system():
    # declared per-atom, but always computed per-system by the LJ model, like the
    # `non_conservative_stress` of pet-mad-s v1.5.0
    model = LegacyModel(lj_model())
    request = Quantity(
        name="non_conservative_stress/doubled", unit="eV/A^3", sample_kind="system"
    )
    (stress,) = execute_model(
        model,
        [to_system(nickel(0), model, [request])],
        [request],
        check_consistency=True,
    )
    assert stress.block().values.shape == (1, 3, 3, 1)


def units_results():
    # the systems and outputs are converted once by `execute_model`, and not again by
    # the AtomisticModel
    reference = LegacyModel(lj_model())
    model = LegacyModel(lj_model(length_unit="Bohr", energy_unit="kJ/mol"))
    assert model.capabilities().length_unit == "Bohr"

    atoms = nickel(0)
    request = energy_request(unit="kcal/mol")
    (expected,) = execute_model(
        reference, [to_system(atoms, reference, [request])], [request]
    )
    # the system is in angstrom, while the model and its pair lists use Bohr
    (energy,) = execute_model(
        model, [to_system(atoms, model, [request])], [request], check_consistency=True
    )
    return energy.block(), expected.block()


def test_units():
    actual, expected = units_results()
    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-6)
    np.testing.assert_allclose(
        actual.gradient("strain").values, expected.gradient("strain").values, rtol=1e-6
    )


@pytest.mark.xfail(
    strict=True,
    reason="metatomic-core scales positions gradients by the energy factor only, "
    "not by the length factor",
)
def test_units_positions_gradients():
    actual, expected = units_results()
    np.testing.assert_allclose(
        actual.gradient("positions").values,
        expected.gradient("positions").values,
        rtol=1e-6,
    )


def test_load(tmp_path):
    path = str(tmp_path / "lj.pt")
    lj_model().save(path)
    model = LegacyModel.load(path)

    atoms = nickel(0)
    atoms.calc = MetatomicCalculator(path, uncertainty_threshold=None)
    request = energy_request(gradients=[])
    (energy,) = execute_model(model, [to_system(atoms, model, [request])], [request])
    np.testing.assert_allclose(
        energy.block().values[0, 0], atoms.get_potential_energy()
    )


def test_errors():
    assert metatomic.torch.LegacyModel is LegacyModel

    message = "`model` must be an AtomisticModel, got <class 'torch.nn.modules.*"
    with pytest.raises(TypeError, match=message):
        LegacyModel(torch.nn.Linear(2, 2))

    # Changing a known unit without changing the model's parameters or cutoffs
    # would silently change its predictions.
    message = "can only override an empty model length unit"
    with pytest.raises(ValueError, match=message):
        LegacyModel(lj_model(), length_unit="Bohr")


# ============================================================================ #
# Small legacy models exercising the remaining parts of the API


def _single_key():
    return Labels("_", torch.tensor([[0]]))


def _energy_block(systems: List[System], values: torch.Tensor) -> TensorBlock:
    return TensorBlock(
        values=values.reshape(-1, 1),
        samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )


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
        block = _energy_block(systems, torch.stack(energies))

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

        return {"energy": TensorMap(_single_key(), [block])}


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

    request = energy_request()
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


def test_deprecated_names():
    # inputs requested with a deprecated name are exposed with the current one
    message = "the 'masses' input name is deprecated"
    with pytest.warns(UserWarning, match=message):
        atomistic, _ = harmonic_model(input_name="masses")

    model = LegacyModel(atomistic)
    request = energy_request(gradients=[])
    assert model.requested_inputs([request]) == [
        Quantity(name="mass", unit="u", sample_kind="atom")
    ]

    system = harmonic_system([[1.0, 0.0, 0.0]], [2.0])
    (energy,) = execute_model(model, [system], [request], check_consistency=True)
    np.testing.assert_allclose(energy.block().values[0, 0], 2.0)

    # outputs declared with a deprecated name are exposed with the current one
    class Forces(torch.nn.Module):
        def forward(
            self,
            systems: List[System],
            outputs: Dict[str, ModelOutput],
            selected_atoms: Optional[Labels],
        ) -> Dict[str, TensorMap]:
            positions = torch.cat([s.positions for s in systems])
            samples = torch.cat(
                [
                    torch.stack([torch.full((len(s),), i), torch.arange(len(s))], 1)
                    for i, s in enumerate(systems)
                ]
            )
            block = TensorBlock(
                values=-positions.reshape(-1, 3, 1),
                samples=Labels(["system", "atom"], samples),
                components=[Labels("xyz", torch.tensor([[0], [1], [2]]))],
                # older models used the deprecated name in the properties as well
                properties=Labels("non_conservative_forces", torch.tensor([[0]])),
            )
            return {"non_conservative_forces": TensorMap(_single_key(), [block])}

    capabilities = ModelCapabilities(
        outputs={
            "non_conservative_forces": ModelOutput(unit="eV/A", sample_kind="atom")
        },
        atomic_types=[1, 8],
        interaction_range=0.0,
        length_unit="Angstrom",
        supported_devices=["cpu"],
        dtype="float64",
    )
    message = "the 'non_conservative_forces' output name is deprecated"
    with pytest.warns(UserWarning, match=message):
        atomistic = AtomisticModel(Forces().eval(), ModelMetadata(), capabilities)

    model = LegacyModel(atomistic)
    assert {o.name for o in model.capabilities().outputs} == {"non_conservative_force"}

    request = Quantity(name="non_conservative_force", unit="eV/A", sample_kind="atom")
    (forces,) = execute_model(model, [system], [request], check_consistency=True)
    assert forces.property_names == ["non_conservative_force"]
    np.testing.assert_allclose(forces.block().values[:, :, 0], [[-1.0, 0.0, 0.0]])


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
    request = energy_request(gradients=[])
    system = harmonic_system([[1.0, 0.0, 0.0]], [2.0])
    (energy,) = execute_model(model, [system], [request], check_consistency=True)
    np.testing.assert_allclose(energy.block().values[0, 0], 2.0 / ase.units.Bohr**2)


def test_infinite_interaction_range():
    atomistic, _ = harmonic_model()
    atomistic._capabilities.interaction_range = math.inf
    model = LegacyModel(atomistic)
    assert model.capabilities().interaction_range == np.finfo(np.float64).max

    request = energy_request(gradients=[])
    system = harmonic_system([[1.0, 0.0, 0.0]], [2.0])
    (energy,) = execute_model(model, [system], [request], check_consistency=True)
    np.testing.assert_allclose(energy.block().values[0, 0], 2.0)
