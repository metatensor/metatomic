import math
from typing import Dict, List, Optional

import metatensor as mts
import numpy as np
import pytest
import torch
import vesin
from metatensor.torch import Labels, TensorBlock, TensorMap

import metatomic
from metatomic import Quantity, execute_model
from metatomic.torch import (
    AtomisticModel,
    LegacyModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    PairListOptions,
    System,
    unit_conversion_factor,
)

from ._tests_utils import prints_to_stderr


CUTOFF = 3.5
# spring constant of HarmonicPairs, in eV/A^2
K = 2.0
PAIRS = metatomic.PairListOptions(cutoff=CUTOFF, full_list=False, strict=True)


class HarmonicPairs(torch.nn.Module):
    """E_i = k/4 * sum_j |r_ij|^2, over the pairs i-j in a half neighbor list"""

    def __init__(self, k: float, cutoff: float):
        super().__init__()
        self.k = k
        self.options = PairListOptions(cutoff=cutoff, full_list=False, strict=True)

    def requested_neighbor_lists(self) -> List[PairListOptions]:
        return [self.options]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        samples: List[torch.Tensor] = []
        energies: List[torch.Tensor] = []
        for i, system in enumerate(systems):
            pairs = system.pairs(self.options)
            half = 0.25 * self.k * (pairs.values**2).sum(dim=(1, 2))
            energy = torch.zeros(len(system), dtype=half.dtype, device=half.device)
            for column in ["first_atom", "second_atom"]:
                energy = energy.index_add(0, pairs.samples.column(column), half)

            atoms = torch.arange(len(system), device=half.device)
            if selected_atoms is not None:
                selected = selected_atoms.values
                atoms = selected[selected[:, 0] == i][:, 1]

            samples.append(torch.stack([torch.full_like(atoms, i), atoms], dim=1))
            energies.append(energy[atoms])

        # `Labels.range` can not be used in models saved with TorchScript
        zero = torch.tensor([[0]])
        if outputs["energy"].sample_kind == "atom":
            values = torch.cat(energies).reshape(-1, 1)
            labels = Labels(["system", "atom"], torch.cat(samples))
        else:
            values = torch.stack([e.sum() for e in energies]).reshape(-1, 1)
            labels = Labels("system", torch.arange(len(systems)).reshape(-1, 1))

        block = TensorBlock(
            values=values,
            samples=labels,
            components=[],
            properties=Labels("energy", zero),
        )
        return {"energy": TensorMap(Labels("_", zero), [block])}


def harmonic_pairs_model(length_unit="angstrom", energy_unit="eV"):
    length = unit_conversion_factor("angstrom", length_unit)
    energy = unit_conversion_factor("eV", energy_unit)
    capabilities = ModelCapabilities(
        outputs={"energy": ModelOutput(unit=energy_unit, sample_kind="atom")},
        atomic_types=[1],
        interaction_range=CUTOFF * length,
        length_unit=length_unit,
        supported_devices=["cpu", "mps"],
        dtype="float64",
    )
    module = HarmonicPairs(K * energy / length**2, CUTOFF * length).eval()
    metadata = ModelMetadata(
        name="harmonic pairs",
        authors=["metatomic developers"],
        references={"implementation": ["https://github.com/metatensor/metatomic"]},
    )
    return AtomisticModel(module, metadata, capabilities)


def reference(system):
    """
    Per-atom energies of HarmonicPairs, and gradients of the total energy with
    respect to positions and strain, computed with numpy.
    """
    pairs = system.pairs(PAIRS)
    first = pairs.samples.column("first_atom")
    second = pairs.samples.column("second_atom")
    # r_ij = r_second - r_first + cell shift
    vectors = np.asarray(pairs.values).reshape(-1, 3)

    energies = np.zeros(len(system.types))
    half = 0.25 * K * (vectors**2).sum(axis=1)
    np.add.at(energies, first, half)
    np.add.at(energies, second, half)

    positions_gradient = np.zeros((len(energies), 3))
    np.add.at(positions_gradient, first, -K * vectors)
    np.add.at(positions_gradient, second, K * vectors)

    return energies, positions_gradient, K * vectors.T @ vectors


@pytest.fixture(params=["eager", "script"])
def atomistic(request):
    model = harmonic_pairs_model()
    return model if request.param == "eager" else torch.jit.script(model)


@pytest.fixture
def system():
    # a distorted FCC crystal, to avoid symmetry-induced zero gradients
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    offsets = np.array(list(np.ndindex(2, 2, 2)))
    positions = 3.6 * (offsets[:, None, :] + basis).reshape(-1, 3)
    positions += 0.1 * np.sin(np.arange(positions.size).reshape(-1, 3))
    return system_with_pairs(
        np.ones(len(positions)), positions, 7.2 * np.eye(3), [True] * 3, [PAIRS]
    )


def system_with_pairs(
    types,
    positions,
    cell,
    pbc,
    pair_lists: list[metatomic.PairListOptions],
    model_length_unit: str = "angstrom",
) -> metatomic.System:
    """
    Create a system in Angstrom, with the pairs requested by a model working in
    ``model_length_unit`` (the pair lists cutoffs are in the model unit).
    """
    types = np.asarray(types, dtype=np.int32)
    positions = np.asarray(positions)
    cell = np.asarray(cell, dtype=positions.dtype)
    pbc = np.asarray(pbc, dtype=bool)

    factor = unit_conversion_factor(model_length_unit, "angstrom")
    system = metatomic.System("angstrom", types, positions, cell, pbc)
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
            mts.TensorBlock(
                values=vectors.astype(positions.dtype).reshape(-1, 3, 1),
                samples=mts.Labels(
                    [
                        "first_atom",
                        "second_atom",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    samples.reshape(-1, 5),
                ),
                components=[mts.Labels.range("xyz", 3)],
                properties=mts.Labels.range("distance", 1),
            ),
        )
    return system


def test_capabilities(atomistic):
    model = LegacyModel(atomistic)
    capabilities = model.capabilities()
    assert capabilities.length_unit == "angstrom"
    assert capabilities.interaction_range == CUTOFF
    assert capabilities.supported_devices == ["cpu", "metal"]
    # the per-atom energy is also declared per-system, and TorchModel can compute
    # the gradients of the latter with autograd
    gradients = {(o.name, o.sample_kind): o.gradients for o in capabilities.outputs}
    assert gradients == {
        ("energy", "atom"): [],
        ("energy", "system"): ["positions", "strain"],
    }
    assert str(model.metadata()) == str(atomistic.metadata())

    request = Quantity(name="energy", unit="eV", sample_kind="system")
    assert model.requested_pair_lists([request]) == [PAIRS]
    assert model.requested_inputs([request]) == []

    atomistic, _ = harmonic_model()
    atomistic._capabilities.interaction_range = math.inf
    assert LegacyModel(atomistic).capabilities().interaction_range == math.inf


def test_deprecated_output_names(capfd):
    # AtomisticModel lists outputs under both their deprecated and current names,
    # LegacyModel only uses the name declared by the model
    torch.set_warn_always(True)
    message = "the 'features' quantity is deprecated"
    with prints_to_stderr(capfd, match=message):
        capabilities = ModelCapabilities(
            outputs={"features": ModelOutput(sample_kind="system")},
            atomic_types=[1],
            interaction_range=CUTOFF,
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        )

    message = "the 'features' output name is deprecated"
    with pytest.warns(UserWarning, match=message):
        module = HarmonicPairs(K, CUTOFF).eval()
        atomistic = AtomisticModel(module, ModelMetadata(), capabilities)
    torch.set_warn_always(False)

    assert sorted(atomistic.capabilities().outputs.keys()) == ["feature", "features"]
    outputs = LegacyModel(atomistic).capabilities().outputs
    assert [output.name for output in outputs] == ["features"]


def test_energy_and_gradients(atomistic, system):
    second = system_with_pairs(
        system.types,
        1.03 * np.asarray(system.positions),
        1.03 * np.asarray(system.cell),
        system.pbc,
        [PAIRS],
    )
    request = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    (energy,) = execute_model(
        LegacyModel(atomistic), [system, second], [request], check_consistency=True
    )

    block = energy.block()
    positions_gradient = block.gradient("positions")
    strain_gradient = block.gradient("strain")
    for i, s in enumerate([system, second]):
        energies, expected_positions_gradient, expected_strain_gradient = reference(s)
        np.testing.assert_allclose(block.values[i, 0], energies.sum())

        rows = positions_gradient.samples.column("system") == i
        np.testing.assert_array_equal(
            positions_gradient.samples.column("atom")[rows], np.arange(len(energies))
        )
        np.testing.assert_allclose(
            np.asarray(positions_gradient.values)[rows, :, 0],
            expected_positions_gradient,
        )
        np.testing.assert_allclose(
            strain_gradient.values[i, :, :, 0], expected_strain_gradient
        )


def test_per_atom_and_selected_atoms(atomistic, system):
    """
    With ``metatomic.torch``, outputs declared per-atom could also be requested
    per-system, and the model computed the per-system values. LegacyModel declares
    such outputs with both sample kinds, and passes ``sample_kind="system"`` requests
    to the legacy module.
    """
    model = LegacyModel(atomistic)
    energies, _, _ = reference(system)

    per_atom = Quantity(name="energy", unit="eV", sample_kind="atom")
    per_system = Quantity(name="energy", unit="eV", sample_kind="system")
    atom, total = execute_model(
        model, [system], [per_atom, per_system], check_consistency=True
    )
    np.testing.assert_array_equal(
        atom.block().samples.column("atom"), np.arange(len(energies))
    )
    np.testing.assert_allclose(atom.block().values[:, 0], energies)
    np.testing.assert_allclose(total.block().values[0, 0], energies.sum())

    selected = mts.Labels(["system", "atom"], np.array([[0, 1], [0, 5]]))
    (atom,) = execute_model(
        model, [system], [per_atom], selected_atoms=selected, check_consistency=True
    )
    np.testing.assert_array_equal(atom.block().samples.values, selected.values)
    np.testing.assert_allclose(atom.block().values[:, 0], energies[[1, 5]])


def test_units(system):
    """
    A model working in Bohr and kJ/mol gives the same results as one working in
    Angstrom and eV: execute_model converts the systems, pair lists and outputs
    (including gradients) between the engine and model units.
    """
    model = LegacyModel(harmonic_pairs_model(length_unit="Bohr", energy_unit="kJ/mol"))
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
    (energy,) = execute_model(
        model, [converted_pairs], [request], check_consistency=True
    )

    energies, positions_gradient, strain_gradient = reference(system)
    factor = unit_conversion_factor("eV", "kcal/mol")
    block = energy.block()
    np.testing.assert_allclose(block.values[0, 0], factor * energies.sum())
    np.testing.assert_allclose(
        block.gradient("positions").values[:, :, 0], factor * positions_gradient
    )
    np.testing.assert_allclose(
        block.gradient("strain").values[0, :, :, 0], factor * strain_gradient
    )


@pytest.mark.parametrize("as_string", [False, True])
def test_load(tmp_path, system, as_string):
    path = tmp_path / "harmonic-pairs.pt"
    harmonic_pairs_model().save(str(path))
    model = LegacyModel(str(path) if as_string else path)
    assert model.metadata().name == "harmonic pairs"

    request = Quantity(name="energy", unit="eV", sample_kind="system")
    (energy,) = execute_model(model, [system], [request], check_consistency=True)
    energies, _, _ = reference(system)
    np.testing.assert_allclose(energy.block().values[0, 0], energies.sum())


def test_errors():
    message = "`module` should be a torch.nn.Module"
    with pytest.raises(TypeError, match=message):
        LegacyModel(42)
    message = "`model` must be an AtomisticModel, got <class 'torch.nn.modules.*"
    with pytest.raises(TypeError, match=message):
        LegacyModel(torch.nn.Linear(2, 2))

    atomistic, _ = harmonic_model()
    atomistic._capabilities.length_unit = ""
    message = "this model does not declare a length unit"
    with pytest.raises(ValueError, match=message):
        LegacyModel(atomistic)


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

        masses = [s.custom_data(self.input_name).block().values for s in systems]
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
    block = mts.TensorBlock(
        values=np.array(masses, dtype=np.float64).reshape(-1, 1),
        samples=mts.Labels(
            ["system", "atom"],
            np.array([[0, i] for i in range(len(positions))]),
        ),
        components=[],
        properties=mts.Labels.range("mass", 1),
    )
    system.add_custom_data("mass", mts.TensorMap(mts.Labels.range("_", 1), [block]))
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
