import numpy as np
import pytest
import torch
from metatensor import Labels, TensorBlock, TensorMap

from metatomic import (
    ModelCapabilities,
    ModelMetadata,
    PairListOptions,
    Quantity,
    System,
    TorchModel,
    execute_model,
)


PAIRS_OPTIONS = PairListOptions(cutoff=5.0, full_list=False, strict=True)


class HarmonicPairs(torch.nn.Module):
    """E = 1/2 * k * sum_ij |r_ij|^2, over all pairs in the list"""

    def __init__(self):
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor(2.0))

    def requested_pair_lists(self, requested_outputs):
        return [PAIRS_OPTIONS]

    def requested_inputs(self, requested_outputs):
        return [Quantity(name="test::input", unit="eV", sample_kind="atom")]

    def forward(self, system):
        vectors = system.pairs(PAIRS_OPTIONS).values.reshape(-1, 3)
        return 0.5 * self.k * (vectors**2).sum()


class NestedModel(torch.nn.Module):
    def requested_pair_lists(self, requested_outputs):
        return [PairListOptions(cutoff=3.0, full_list=True, strict=True)]

    def forward(self, systems, selected_atoms, outputs):
        raise NotImplementedError()


class PairModel(TorchModel):
    def __init__(self):
        super().__init__()
        self.pairs = HarmonicPairs()

    def capabilities(self):
        return ModelCapabilities(
            atomic_types=[1, 6, 8],
            interaction_range=5.0,
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype="float32",
            outputs=[
                Quantity(name="energy", unit="eV", sample_kind="system"),
                Quantity(name="test::vector", unit="", sample_kind="system"),
                Quantity(name="test::per_atom", unit="", sample_kind="atom"),
                Quantity(name="charge", unit="e", sample_kind="system"),
            ],
        )

    def metadata(self):
        return ModelMetadata(name="pair model")

    def requested_pair_lists(self, requested_outputs):
        # same as the one requested by `HarmonicPairs`
        return [PAIRS_OPTIONS]

    def forward(self, systems, selected_atoms, outputs):
        assert selected_atoms is None

        results = []
        for request in outputs:
            # gradients are computed with autograd
            assert request.gradients == []

            for system in systems:
                assert system.arrays_backend == "torch"
                assert isinstance(system.positions, torch.Tensor)
                assert isinstance(system.cell, torch.Tensor)

            if request.name == "energy":
                energies = [self.pairs(system) for system in systems]
                block = TensorBlock(
                    values=torch.stack(energies).reshape(-1, 1),
                    samples=Labels.range("system", len(systems)),
                    components=[],
                    properties=Labels.range("energy", 1),
                )
            else:
                assert request.name == "test::vector"
                # sum of all the pair vectors, with two properties: x1 and x2
                sums = [
                    system.pairs(PAIRS_OPTIONS).values.reshape(-1, 3).sum(dim=0)
                    for system in systems
                ]
                sums = torch.stack(sums).reshape(-1, 3, 1)
                block = TensorBlock(
                    values=torch.concatenate([sums, 2 * sums], dim=2),
                    samples=Labels.range("system", len(systems)),
                    components=[Labels.range("direction", 3)],
                    properties=Labels.range("p", 2),
                )

            results.append(TensorMap(Labels.range("_", 1), [block]))

        return results


def make_system(positions, cell, pairs, vectors_offset=0.0):
    system = System(
        "angstrom",
        np.array([1, 6, 8], dtype=np.int32),
        np.array(positions, dtype=np.float32),
        np.array(cell, dtype=np.float32),
        np.array([np.any(c != 0.0) for c in np.array(cell)]),
    )

    pairs = np.array(pairs, dtype=np.int32)
    positions = np.array(positions, dtype=np.float32)
    cell = np.array(cell, dtype=np.float32)
    shifts = pairs[:, 2:].astype(np.float32)
    vectors = positions[pairs[:, 1]] - positions[pairs[:, 0]] + shifts @ cell
    vectors += vectors_offset

    block = TensorBlock(
        values=vectors.reshape(-1, 3, 1),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            pairs,
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    system.add_pairs(PAIRS_OPTIONS, block)

    custom = TensorBlock(
        values=np.zeros((3, 1), dtype=np.float32),
        samples=Labels(["system", "atom"], np.array([[0, 0], [0, 1], [0, 2]])),
        components=[],
        properties=Labels.range("_", 1),
    )
    system.add_custom_data("test::input", TensorMap(Labels.range("_", 1), [custom]))

    return system, vectors


def test_requested_pair_lists():
    model = PairModel()
    model.nested = NestedModel()

    requested = model.requested_pair_lists(model.capabilities().outputs)
    assert requested == [
        PairListOptions(cutoff=5.0, full_list=False, strict=True),
        PairListOptions(cutoff=3.0, full_list=True, strict=True),
    ]
    assert requested[0].requestors == ["PairModel", "PairModel.pairs"]
    assert requested[1].requestors == ["PairModel.nested"]


def test_requested_inputs():
    model = PairModel()
    assert model.requested_inputs(model.capabilities().outputs) == [
        Quantity(name="test::input", unit="eV", sample_kind="atom")
    ]

    class OtherInput(torch.nn.Module):
        def requested_inputs(self, requested_outputs):
            return [Quantity(name="test::input", unit="kcal/mol", sample_kind="atom")]

    model.other = OtherInput()
    message = (
        "the input 'test::input' is requested with different unit by "
        "'PairModel.other' \\(unit='kcal/mol'\\) and 'PairModel.pairs' "
        "\\(unit='eV'\\)"
    )
    with pytest.raises(ValueError, match=message):
        model.requested_inputs(model.capabilities().outputs)


def test_capabilities():
    model = PairModel()
    capabilities = model.capabilities()
    raw_capabilities = model._raw_capabilities()

    # gradients are added for per-system energy and custom outputs
    assert capabilities.outputs[0].name == "energy"
    assert raw_capabilities.outputs[0].name == "energy"
    assert capabilities.outputs[0].gradients == ["positions", "strain"]
    assert raw_capabilities.outputs[0].gradients == []

    assert capabilities.outputs[1].name == "test::vector"
    assert raw_capabilities.outputs[1].name == "test::vector"
    assert capabilities.outputs[1].gradients == ["positions", "strain"]
    assert raw_capabilities.outputs[1].gradients == []

    assert capabilities.outputs[2].name == "test::per_atom"
    assert raw_capabilities.outputs[2].name == "test::per_atom"
    assert capabilities.outputs[2].gradients == []
    assert raw_capabilities.outputs[2].gradients == []

    assert capabilities.outputs[3].name == "charge"
    assert raw_capabilities.outputs[3].name == "charge"
    assert capabilities.outputs[3].gradients == []
    assert raw_capabilities.outputs[3].gradients == []


def test_autograd():
    model = PairModel()

    positions = [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.2, 1.5, 0.3]]
    cell = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]]
    pairs = [[0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 2, 0, 0, 0], [0, 1, 1, 0, 0]]
    system, vectors = make_system(positions, cell, pairs)

    request = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    results = execute_model(model, [system], [request], check_consistency=True)

    block = results[0].block()
    expected_energy = (vectors**2).sum()
    np.testing.assert_allclose(np.asarray(block.values), [[expected_energy]])

    # dE/dr_i = -k sum_j r_ij for pairs i -> j, and k sum_i r_ij for pairs i -> j
    expected_positions = np.zeros((3, 3))
    for (i, j, *_), vector in zip(pairs, vectors, strict=True):
        expected_positions[i] -= 2.0 * vector
        expected_positions[j] += 2.0 * vector

    gradient = block.gradient("positions")
    assert gradient.samples == Labels(
        ["sample", "system", "atom"],
        np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
    )
    np.testing.assert_allclose(
        np.asarray(gradient.values).reshape(3, 3),
        expected_positions,
        rtol=1e-6,
        atol=1e-6,
    )

    # dE/dstrain = k sum_ij r_ij ⊗ r_ij
    expected_strain = 2.0 * np.einsum("pa,pb->ab", vectors, vectors)

    gradient = block.gradient("strain")
    assert gradient.samples == Labels(["sample"], np.array([[0]]))
    np.testing.assert_allclose(
        np.asarray(gradient.values).reshape(3, 3), expected_strain, rtol=1e-6, atol=1e-6
    )

    # we should only get positions gradients when we only request positions gradients
    request = Quantity(
        name="energy", unit="eV", sample_kind="system", gradients=["positions"]
    )
    results = execute_model(model, [system], [request], check_consistency=True)
    assert results[0].block().gradients_list() == ["positions"]


def test_autograd_components():
    model = PairModel()

    positions = [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.2, 1.5, 0.3]]
    cell = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]]
    pairs = [[0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 2, 0, 0, 0], [0, 1, 1, 0, 0]]
    system, vectors = make_system(positions, cell, pairs)

    request = Quantity(
        name="test::vector",
        unit="",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    # also request the energy with gradients, to check that the computational graph
    # shared by both outputs stays alive until all the gradients are computed
    energy = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    results = execute_model(model, [system], [energy, request], check_consistency=True)

    energy_block = results[0].block()
    np.testing.assert_allclose(np.asarray(energy_block.values), [[(vectors**2).sum()]])
    assert energy_block.gradients_list() == ["positions", "strain"]

    block = results[1].block()
    vector_sum = vectors.sum(axis=0)
    expected = np.stack([vector_sum, 2 * vector_sum], axis=1)
    np.testing.assert_allclose(np.asarray(block.values), [expected], rtol=1e-6)

    # d(sum_p r_p)_c / d(r_a)_b = (n_second(a) - n_first(a)) δ_bc, times the
    # factor for each property
    expected_positions = np.zeros((3, 3, 3, 2))
    for i, j, *_ in pairs:
        for factor, p in [(1.0, 0), (2.0, 1)]:
            expected_positions[i, :, :, p] -= factor * np.eye(3)
            expected_positions[j, :, :, p] += factor * np.eye(3)

    gradient = block.gradient("positions")
    assert gradient.components == [
        Labels.range("xyz", 3),
        Labels.range("direction", 3),
    ]
    np.testing.assert_allclose(
        np.asarray(gradient.values), expected_positions, rtol=1e-6, atol=1e-6
    )

    # d(sum_p r_p (1 + ε))_c / dε_ab = sum_p (r_p)_a δ_bc
    expected_strain = np.zeros((1, 3, 3, 3, 2))
    for factor, p in [(1.0, 0), (2.0, 1)]:
        expected_strain[0, :, :, :, p] = factor * np.einsum(
            "a,bc->abc", vector_sum, np.eye(3)
        )

    gradient = block.gradient("strain")
    assert gradient.components == [
        Labels.range("xyz_1", 3),
        Labels.range("xyz_2", 3),
        Labels.range("direction", 3),
    ]
    np.testing.assert_allclose(
        np.asarray(gradient.values), expected_strain, rtol=1e-6, atol=1e-6
    )


def test_reuse_pair_vectors():
    model = PairModel()

    # the pair vectors given by the engine are used as-is, not re-computed from the
    # positions. We check this by giving vectors that do not match the positions.
    positions = [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.2, 1.5, 0.3]]
    cell = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    pairs = [[0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 2, 0, 0, 0]]
    system, vectors = make_system(positions, cell, pairs, vectors_offset=0.25)

    request = Quantity(
        name="energy", unit="eV", sample_kind="system", gradients=["positions"]
    )
    results = execute_model(model, [system], [request], check_consistency=True)

    block = results[0].block()
    np.testing.assert_allclose(np.asarray(block.values), [[(vectors**2).sum()]])

    expected_positions = np.zeros((3, 3))
    for (i, j, *_), vector in zip(pairs, vectors, strict=True):
        expected_positions[i] -= 2.0 * vector
        expected_positions[j] += 2.0 * vector

    np.testing.assert_allclose(
        np.asarray(block.gradient("positions").values).reshape(3, 3),
        expected_positions,
        rtol=1e-6,
        atol=1e-6,
    )


def test_explicit_gradients():
    class ExplicitGradients(PairModel):
        def capabilities(self):
            capabilities = super().capabilities()
            outputs = capabilities.outputs
            outputs[0].gradients = ["positions"]
            capabilities.outputs = outputs
            return capabilities

        def forward(self, systems, selected_atoms, outputs):
            assert selected_atoms is None

            results = []
            for request in outputs:
                # strain gradients are computed with autograd
                assert request.gradients == ["positions"]

                n_atoms = len(systems[0])
                gradient = TensorBlock(
                    values=torch.full((n_atoms, 3, 1), 42.0),
                    samples=Labels(
                        ["sample", "system", "atom"],
                        np.array([[0, 0, i] for i in range(n_atoms)]),
                    ),
                    components=[Labels.range("xyz", 3)],
                    properties=Labels.range("energy", 1),
                )

                vectors = systems[0].pairs(PAIRS_OPTIONS).values.reshape(-1, 3)
                block = TensorBlock(
                    values=(vectors**2).sum().reshape(1, 1),
                    samples=Labels.range("system", 1),
                    components=[],
                    properties=Labels.range("energy", 1),
                )
                block.add_gradient("positions", gradient)
                results.append(TensorMap(Labels.range("_", 1), [block]))

            return results

    model = ExplicitGradients()
    assert model.capabilities().outputs[0].gradients == ["positions", "strain"]

    positions = [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.2, 1.5, 0.3]]
    cell = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    pairs = [[0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 2, 0, 0, 0]]
    system, vectors = make_system(positions, cell, pairs)

    request = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    results = execute_model(model, [system], [request], check_consistency=True)

    block = results[0].block()
    np.testing.assert_allclose(np.asarray(block.gradient("positions").values), 42.0)

    expected_strain = 2.0 * np.einsum("pa,pb->ab", vectors, vectors)
    np.testing.assert_allclose(
        np.asarray(block.gradient("strain").values).reshape(3, 3),
        expected_strain,
        rtol=1e-6,
        atol=1e-6,
    )


def test_super_calls():
    class Base(PairModel):
        def requested_pair_lists(self, requested_outputs):
            return [PairListOptions(cutoff=2.0, full_list=True, strict=True)]

    class Derived(Base):
        def capabilities(self):
            # the parent capabilities do not contain the autograd gradients
            capabilities = super().capabilities()
            assert capabilities.outputs[0].gradients == []
            return capabilities

        def requested_pair_lists(self, requested_outputs):
            # the parent requests do not contain the ones from sub-modules
            requested = super().requested_pair_lists(requested_outputs)
            assert requested == [
                PairListOptions(cutoff=2.0, full_list=True, strict=True)
            ]
            return requested + [
                PairListOptions(cutoff=1.0, full_list=True, strict=True)
            ]

    model = Derived()
    assert model.capabilities().outputs[0].gradients == ["positions", "strain"]
    assert model.requested_pair_lists(model.capabilities().outputs) == [
        PairListOptions(cutoff=2.0, full_list=True, strict=True),
        PairListOptions(cutoff=1.0, full_list=True, strict=True),
        PairListOptions(cutoff=5.0, full_list=False, strict=True),
    ]


# torch.compile reads `.grad` on the non-leaf tensors given to compiled functions, such
# as the pair vectors, which emits this warning
@pytest.mark.filterwarnings(
    "ignore:The \\.grad attribute of a Tensor that is not a leaf Tensor:UserWarning"
)
def test_torch_compile():
    @torch.compile(dynamic=True)
    def harmonic(k, vectors):
        return 0.5 * k * (vectors**2).sum()

    class CompiledPairs(HarmonicPairs):
        def forward(self, system):
            vectors = system.pairs(PAIRS_OPTIONS).values.reshape(-1, 3)
            return harmonic(self.k, vectors)

    model = PairModel()
    model.pairs = CompiledPairs()

    positions = [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.2, 1.5, 0.3]]
    cell = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]]
    pairs = [[0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 2, 0, 0, 0], [0, 1, 1, 0, 0]]

    request = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )

    system, _ = make_system(positions, cell, pairs)
    compiled = execute_model(model, [system], [request], check_consistency=True)

    system, _ = make_system(positions, cell, pairs)
    expected = execute_model(PairModel(), [system], [request])

    compiled = compiled[0].block()
    expected = expected[0].block()
    np.testing.assert_allclose(compiled.values, expected.values, rtol=1e-6)
    for parameter in ["positions", "strain"]:
        np.testing.assert_allclose(
            compiled.gradient(parameter).values,
            expected.gradient(parameter).values,
            rtol=1e-6,
        )
