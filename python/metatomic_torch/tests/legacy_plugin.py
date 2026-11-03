"""
Run models exported with ``AtomisticModel`` through the ``metatomic-torch`` plugin,
which wraps them in the C++ ``metatomic_torch::LegacyModel``. Results are compared to
numpy references, and to the Python ``LegacyModel`` when it is available.
"""

from typing import Dict, List, Optional

import metatensor as mts
import numpy as np
import pytest
import torch
import vesin
from metatensor.torch import Labels, TensorBlock, TensorMap

import metatomic
from metatomic import MetatomicError, Quantity, execute_model
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    PairListOptions,
    System,
    unit_conversion_factor,
)
from metatomic.torch.utils import plugin_path


PythonLegacyModel = getattr(metatomic, "LegacyModel", None) or getattr(
    metatomic.torch, "LegacyModel", None
)

CUTOFF = 3.5
# spring constant of HarmonicPairs, in eV/A^2
K = 2.0
PAIRS = metatomic.PairListOptions(cutoff=CUTOFF, full_list=False, strict=True)


@pytest.fixture(scope="module", autouse=True)
def plugin():
    metatomic.load_plugin(plugin_path)
    # outputs of the plugin are stored in C++ torch tensors, and the labels of
    # gradients computed with autograd in C++ arrays
    mts.register_external_data_wrapper(
        "metatensor_torch::TorchDataArray", mts.ExternalCpuArray
    )
    mts.register_external_data_wrapper(
        "metatensor::SimpleDataArray", mts.ExternalCpuArray
    )


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
    metadata = ModelMetadata(name="harmonic pairs", authors=["metatomic developers"])
    return AtomisticModel(module, metadata, capabilities)


def _pairs(masses: List[torch.Tensor], systems: List[System]):
    # zip(strict=True) is not supported by TorchScript
    assert len(masses) == len(systems)
    return [(masses[i], systems[i]) for i in range(len(systems))]


class Harmonic(torch.nn.Module):
    """E = sum_i m_i |r_i|^2, with explicit positions gradients"""

    def __init__(self, gradient_offset: float):
        super().__init__()
        # offset added to the explicit gradients, to check they are used as-is
        self.gradient_offset = gradient_offset

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return {"mass": ModelOutput(unit="u", sample_kind="atom")}

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        masses = [s.custom_data("mass").block().values for s in systems]
        energies = [(m * s.positions**2).sum() for m, s in _pairs(masses, systems)]
        block = TensorBlock(
            values=torch.stack(energies).reshape(-1, 1),
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )

        if "positions" in outputs["energy"].explicit_gradients:
            samples = []
            values = []
            for i, (m, s) in enumerate(_pairs(masses, systems)):
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


def harmonic_model(gradient_offset=0.0, length_unit="Angstrom"):
    capabilities = ModelCapabilities(
        outputs={
            "energy": ModelOutput(
                unit="eV", sample_kind="system", explicit_gradients=["positions"]
            )
        },
        atomic_types=[1, 8],
        interaction_range=0.0,
        length_unit=length_unit,
        supported_devices=["cpu"],
        dtype="float64",
    )
    module = Harmonic(gradient_offset).eval()
    return AtomisticModel(module, ModelMetadata(), capabilities)


def export(model, tmp_path, name="model.pt"):
    path = tmp_path / name
    model.save(str(path))
    return str(path)


def models(atomistic, tmp_path):
    """The C++ LegacyModel loaded through the plugin, and the Python one if any"""
    path = export(atomistic, tmp_path)
    result = {"cpp": metatomic.load_model(path, plugin_name="metatomic-torch")}
    if PythonLegacyModel is not None:
        result["python"] = PythonLegacyModel(path)
    return result


def system_with_pairs(types, positions, cell, pbc, pair_lists, length_unit="angstrom"):
    """
    Create a system in Angstrom, with the pairs requested by a model working in
    ``length_unit`` (the pair lists cutoffs are in the model unit).
    """
    types = np.asarray(types, dtype=np.int32)
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    pbc = np.asarray(pbc, dtype=bool)

    factor = unit_conversion_factor(length_unit, "angstrom")
    system = metatomic.System("angstrom", types, positions, cell, pbc)
    for options in pair_lists:
        i, j, shifts, vectors = vesin.NeighborList(
            cutoff=options.cutoff * factor, full_list=options.full_list
        ).compute(points=positions, box=cell, periodic=pbc, quantities="ijSD")
        samples = np.column_stack([i, j, shifts]).astype(np.int32)
        system.add_pairs(
            options,
            mts.TensorBlock(
                values=vectors.reshape(-1, 3, 1),
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


def crystal(scale=1.0, pair_lists=(PAIRS,), length_unit="angstrom"):
    # a distorted FCC crystal, to avoid symmetry-induced zero gradients
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    offsets = np.array(list(np.ndindex(2, 2, 2)))
    positions = 3.6 * (offsets[:, None, :] + basis).reshape(-1, 3)
    positions += 0.1 * np.sin(np.arange(positions.size).reshape(-1, 3))
    return system_with_pairs(
        np.ones(len(positions)),
        scale * positions,
        scale * 7.2 * np.eye(3),
        [True] * 3,
        pair_lists,
        length_unit,
    )


def reference(system):
    """
    Per-atom energies of HarmonicPairs, and gradients of the total energy with
    respect to positions and strain, computed with numpy.
    """
    pairs = system.pairs(PAIRS)
    first = pairs.samples.column("first_atom")
    second = pairs.samples.column("second_atom")
    vectors = np.asarray(pairs.values).reshape(-1, 3)

    energies = np.zeros(len(system.types))
    half = 0.25 * K * (vectors**2).sum(axis=1)
    np.add.at(energies, first, half)
    np.add.at(energies, second, half)

    positions_gradient = np.zeros((len(energies), 3))
    np.add.at(positions_gradient, first, -K * vectors)
    np.add.at(positions_gradient, second, K * vectors)

    return energies, positions_gradient, K * vectors.T @ vectors


def as_numpy(block):
    """Values, gradients and samples of ``block`` as numpy arrays"""
    result = {
        "values": np.asarray(block.values),
        "samples": np.asarray(block.samples.values),
    }
    for parameter, gradient in block.gradients():
        result[parameter] = np.asarray(gradient.values)
        result[f"{parameter}_samples"] = np.asarray(gradient.samples.values)
    return result


def assert_same(cpp, python):
    """Check that the outputs of the C++ and Python LegacyModel are the same"""
    assert len(cpp) == len(python)
    for cpp_output, python_output in zip(cpp, python, strict=True):
        assert cpp_output.keys == python_output.keys
        for cpp_block, python_block in zip(
            cpp_output.blocks(), python_output.blocks(), strict=True
        ):
            cpp_block = as_numpy(cpp_block)
            python_block = as_numpy(python_block)
            assert cpp_block.keys() == python_block.keys()
            for name, value in cpp_block.items():
                np.testing.assert_allclose(value, python_block[name], atol=1e-12)


def test_load(tmp_path):
    path = export(harmonic_pairs_model(), tmp_path)

    # the plugin is found without giving its name
    model = metatomic.load_model(path)
    assert model.metadata().name == "harmonic pairs"

    capabilities = model.capabilities()
    assert capabilities.length_unit == "angstrom"
    assert capabilities.interaction_range == CUTOFF
    assert capabilities.supported_devices == ["cpu", "metal"]
    gradients = {(o.name, o.sample_kind): o.gradients for o in capabilities.outputs}
    assert gradients == {
        ("energy", "atom"): [],
        ("energy", "system"): ["positions", "strain"],
    }

    request = Quantity(name="energy", unit="eV", sample_kind="system")
    assert model.requested_pair_lists([request]) == [PAIRS]
    assert model.requested_inputs([request]) == []

    if PythonLegacyModel is not None:
        python = PythonLegacyModel(path)
        assert str(model.metadata()) == str(python.metadata())
        assert capabilities.to_dict() == python.capabilities().to_dict()


def test_load_errors(tmp_path):
    not_a_model = tmp_path / "not-a-model.pt"
    not_a_model.write_text("this is not a TorchScript archive")
    message = "plugin 'metatomic-torch' could not load the model"
    with pytest.raises(MetatomicError, match=message):
        metatomic.load_model(str(not_a_model), plugin_name="metatomic-torch")

    path = export(harmonic_pairs_model(), tmp_path)
    message = "unknown option for metatomic-torch models: 'device'"
    with pytest.raises(MetatomicError, match=message):
        metatomic.load_model(path, {"device": "cpu"}, plugin_name="metatomic-torch")

    atomistic = harmonic_model()
    atomistic._capabilities.length_unit = ""
    message = "No length unit was provided"
    with pytest.warns(UserWarning, match=message):
        path = export(atomistic, tmp_path, "no-length-unit.pt")
    message = "this model does not declare a length unit"
    with pytest.raises(MetatomicError, match=message):
        metatomic.load_model(path, plugin_name="metatomic-torch")


def test_energy_and_gradients(tmp_path):
    systems = [crystal(), crystal(scale=1.03)]
    request = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )

    results = {
        name: execute_model(model, systems, [request], check_consistency=True)
        for name, model in models(harmonic_pairs_model(), tmp_path).items()
    }

    block = as_numpy(results["cpp"][0].block())
    for i, system in enumerate(systems):
        energies, positions_gradient, strain_gradient = reference(system)
        np.testing.assert_allclose(block["values"][i, 0], energies.sum())

        rows = block["positions_samples"][:, 1] == i
        np.testing.assert_array_equal(
            block["positions_samples"][rows, 2], np.arange(len(energies))
        )
        np.testing.assert_allclose(
            block["positions"][rows, :, 0], positions_gradient, atol=1e-12
        )
        np.testing.assert_allclose(
            block["strain"][i, :, :, 0], strain_gradient, atol=1e-12
        )

    if "python" in results:
        assert_same(results["cpp"], results["python"])


def test_per_atom_and_selected_atoms(tmp_path):
    system = crystal()
    energies, _, _ = reference(system)

    per_atom = Quantity(name="energy", unit="eV", sample_kind="atom")
    per_system = Quantity(name="energy", unit="eV", sample_kind="system")
    selected = mts.Labels(["system", "atom"], np.array([[0, 1], [0, 5]]))

    for model in models(harmonic_pairs_model(), tmp_path).values():
        # the same name with different sample kinds needs two calls to the model
        atom, total = execute_model(
            model, [system], [per_atom, per_system], check_consistency=True
        )
        np.testing.assert_allclose(np.asarray(atom.block().values)[:, 0], energies)
        np.testing.assert_allclose(total.block().values[0, 0], energies.sum())

        (atom,) = execute_model(
            model,
            [system],
            [per_atom],
            selected_atoms=selected,
            check_consistency=True,
        )
        np.testing.assert_array_equal(atom.block().samples.values, selected.values)
        np.testing.assert_allclose(
            np.asarray(atom.block().values)[:, 0], energies[[1, 5]]
        )


def test_units(tmp_path):
    """
    A model working in Bohr and kJ/mol gives the same results as one working in
    Angstrom and eV: execute_model converts the systems, pair lists and outputs.
    """
    atomistic = harmonic_pairs_model(length_unit="Bohr", energy_unit="kJ/mol")
    request = Quantity(
        name="energy",
        unit="kcal/mol",
        sample_kind="system",
        gradients=["positions", "strain"],
    )

    energies, positions_gradient, strain_gradient = reference(crystal())
    factor = unit_conversion_factor("eV", "kcal/mol")

    results = {}
    for name, model in models(atomistic, tmp_path).items():
        system = crystal(
            pair_lists=model.requested_pair_lists([request]), length_unit="Bohr"
        )
        results[name] = execute_model(
            model, [system], [request], check_consistency=True
        )

        block = as_numpy(results[name][0].block())
        np.testing.assert_allclose(block["values"][0, 0], factor * energies.sum())
        np.testing.assert_allclose(
            block["positions"][:, :, 0], factor * positions_gradient, atol=1e-10
        )
        np.testing.assert_allclose(
            block["strain"][0, :, :, 0], factor * strain_gradient, atol=1e-10
        )

    if "python" in results:
        assert_same(results["cpp"], results["python"])


def harmonic_system(positions, masses):
    system = system_with_pairs(
        [8] + [1] * (len(positions) - 1), positions, np.zeros((3, 3)), [False] * 3, []
    )
    block = mts.TensorBlock(
        values=np.array(masses, dtype=np.float64).reshape(-1, 1),
        samples=mts.Labels(
            ["system", "atom"], np.array([[0, i] for i in range(len(positions))])
        ),
        components=[],
        properties=mts.Labels.range("mass", 1),
    )
    system.add_custom_data("mass", mts.TensorMap(mts.Labels.range("_", 1), [block]))
    return system


def test_explicit_gradients_and_inputs(tmp_path):
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.2, 1.5, 0.3]])
    masses = np.array([16.0, 1.0, 1.0])
    system = harmonic_system(positions, masses)

    request = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )

    results = {}
    for name, model in models(harmonic_model(gradient_offset=1.0), tmp_path).items():
        assert model.requested_inputs([request]) == [
            Quantity(name="mass", unit="u", sample_kind="atom")
        ]
        (declared,) = model.capabilities().outputs
        assert declared.gradients == ["positions", "strain"]

        results[name] = execute_model(
            model, [system], [request], check_consistency=True
        )
        block = as_numpy(results[name][0].block())
        np.testing.assert_allclose(
            block["values"][0, 0], (masses[:, None] * positions**2).sum()
        )
        # the explicit gradient of the model is used as-is, including the offset
        np.testing.assert_allclose(
            block["positions"].reshape(-1, 3), 2 * masses[:, None] * positions + 1.0
        )
        # the strain gradient comes from autograd
        np.testing.assert_allclose(
            block["strain"].reshape(3, 3),
            2 * np.einsum("i,ia,ib->ab", masses, positions, positions),
            atol=1e-12,
        )

    if "python" in results:
        assert_same(results["cpp"], results["python"])
