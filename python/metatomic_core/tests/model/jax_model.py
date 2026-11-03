import metatensor as mts
import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

from metatomic import (
    ModelCapabilities,
    ModelMetadata,
    PairListOptions,
    Quantity,
    System,
    execute_model,
)


jax = pytest.importorskip("jax")
jnp = jax.numpy

from metatomic import JaxModel, JaxSystem, unit_conversion_factor  # noqa: E402


PAIRS_OPTIONS = PairListOptions(cutoff=5.0, full_list=False, strict=True)


@jax.jit
def harmonic(k, vectors):
    return 0.5 * k * (vectors**2).sum()


class HarmonicPairs:
    """E = 1/2 * k * sum_ij |r_ij|^2, over all pairs in the list"""

    def __init__(self):
        self.k = jnp.asarray(2.0)

    def requested_pair_lists(self, requested_outputs):
        return [PAIRS_OPTIONS]

    def requested_inputs(self, requested_outputs):
        return [Quantity(name="test::input", unit="eV", sample_kind="atom")]

    def __call__(self, system):
        return harmonic(self.k, system.pairs(PAIRS_OPTIONS).values.reshape(-1, 3))


class NestedModel:
    def requested_pair_lists(self, requested_outputs):
        return [PairListOptions(cutoff=3.0, full_list=True, strict=True)]


class PairModel(JaxModel):
    def __init__(self):
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
        return [PAIRS_OPTIONS]

    def forward(self, systems, selected_atoms, outputs):
        assert selected_atoms is None

        results = []
        for request in outputs:
            assert request.gradients == []
            for system in systems:
                assert isinstance(system.positions, jax.Array)

            if request.name == "energy":
                block = TensorBlock(
                    values=jnp.stack([self.pairs(s) for s in systems]).reshape(-1, 1),
                    samples=Labels.range("system", len(systems)),
                    components=[],
                    properties=Labels.range("energy", 1),
                )
            else:
                assert request.name == "test::vector"
                sums = jnp.stack(
                    [
                        s.pairs(PAIRS_OPTIONS).values.reshape(-1, 3).sum(axis=0)
                        for s in systems
                    ]
                ).reshape(-1, 3, 1)
                block = TensorBlock(
                    values=jnp.concatenate([sums, 2 * sums], axis=2),
                    samples=Labels.range("system", len(systems)),
                    components=[Labels.range("direction", 3)],
                    properties=Labels.range("p", 2),
                )

            results.append(TensorMap(Labels.range("_", 1), [block]))

        return results


def make_system(positions, cell, pairs, vectors_offset=0.0, length_unit="angstrom"):
    system = System(
        length_unit,
        np.array([1, 6, 8], dtype=np.int32),
        np.array(positions, dtype=np.float32),
        np.array(cell, dtype=np.float32),
        np.array([np.any(c != 0.0) for c in np.array(cell)]),
    )

    pairs = np.array(pairs, dtype=np.int32)
    positions = np.array(positions, dtype=np.float32)
    cell = np.array(cell, dtype=np.float32)
    vectors = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    vectors += pairs[:, 2:].astype(np.float32) @ cell + vectors_offset

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


POSITIONS = [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.2, 1.5, 0.3]]
CELL = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]]
PAIRS = [[0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 2, 0, 0, 0], [0, 1, 1, 0, 0]]
ENERGY = Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions", "strain"]
)


def expected_position_gradients(pairs, vectors):
    expected = np.zeros((3, 3))
    for (i, j, *_), vector in zip(pairs, vectors, strict=True):
        expected[i] -= 2.0 * vector
        expected[j] += 2.0 * vector
    return expected


def test_requested_pair_lists():
    model = PairModel()
    model.nested = [NestedModel()]

    requested = model.requested_pair_lists(model.capabilities().outputs)
    assert requested == [
        PairListOptions(cutoff=5.0, full_list=False, strict=True),
        PairListOptions(cutoff=3.0, full_list=True, strict=True),
    ]
    assert requested[0].requestors == ["PairModel", "PairModel.pairs"]
    assert requested[1].requestors == ["PairModel.nested.0"]


def test_requested_inputs():
    model = PairModel()
    assert model.requested_inputs(model.capabilities().outputs) == [
        Quantity(name="test::input", unit="eV", sample_kind="atom")
    ]

    class OtherInput:
        def requested_inputs(self, requested_outputs):
            return [Quantity(name="test::input", unit="kcal/mol", sample_kind="atom")]

    model.other = OtherInput()
    message = "requested with different unit"
    with pytest.raises(ValueError, match=message):
        model.requested_inputs(model.capabilities().outputs)


def test_capabilities():
    model = PairModel()
    gradients = [o.gradients for o in model.capabilities().outputs]
    assert gradients == [["positions", "strain"], ["positions", "strain"], [], []]
    assert all(o.gradients == [] for o in model._raw_capabilities().outputs)


def test_autograd():
    system, vectors = make_system(POSITIONS, CELL, PAIRS)
    (result,) = execute_model(PairModel(), [system], [ENERGY], check_consistency=True)

    block = result.block()
    np.testing.assert_allclose(block.values, [[(vectors**2).sum()]], rtol=1e-6)

    gradient = block.gradient("positions")
    assert gradient.samples == Labels(
        ["sample", "system", "atom"], np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    )
    np.testing.assert_allclose(
        gradient.values.reshape(3, 3),
        expected_position_gradients(PAIRS, vectors),
        rtol=1e-6,
        atol=1e-6,
    )

    gradient = block.gradient("strain")
    assert gradient.samples == Labels(["sample"], np.array([[0]]))
    np.testing.assert_allclose(
        gradient.values.reshape(3, 3),
        2.0 * np.einsum("pa,pb->ab", vectors, vectors),
        rtol=1e-6,
        atol=1e-6,
    )

    request = Quantity(
        name="energy", unit="eV", sample_kind="system", gradients=["positions"]
    )
    (result,) = execute_model(PairModel(), [system], [request], check_consistency=True)
    assert result.block().gradients_list() == ["positions"]


def test_autograd_components():
    system, vectors = make_system(POSITIONS, CELL, PAIRS)
    request = Quantity(
        name="test::vector",
        unit="",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    energy, result = execute_model(
        PairModel(), [system], [ENERGY, request], check_consistency=True
    )
    assert energy.block().gradients_list() == ["positions", "strain"]

    block = result.block()
    vector_sum = vectors.sum(axis=0)
    np.testing.assert_allclose(
        block.values, [np.stack([vector_sum, 2 * vector_sum], axis=1)], rtol=1e-6
    )

    expected_positions = np.zeros((3, 3, 3, 2))
    for i, j, *_ in PAIRS:
        for factor, p in [(1.0, 0), (2.0, 1)]:
            expected_positions[i, :, :, p] -= factor * np.eye(3)
            expected_positions[j, :, :, p] += factor * np.eye(3)

    gradient = block.gradient("positions")
    assert gradient.components == [Labels.range("xyz", 3), Labels.range("direction", 3)]
    np.testing.assert_allclose(gradient.values, expected_positions, atol=1e-6)

    expected_strain = np.zeros((1, 3, 3, 3, 2))
    for factor, p in [(1.0, 0), (2.0, 1)]:
        expected_strain[0, ..., p] = factor * np.einsum(
            "a,bc->abc", vector_sum, np.eye(3)
        )
    np.testing.assert_allclose(
        block.gradient("strain").values, expected_strain, rtol=1e-6, atol=1e-6
    )


def test_several_systems():
    system_1, vectors_1 = make_system(POSITIONS, CELL, PAIRS)
    system_2, vectors_2 = make_system(POSITIONS, CELL, PAIRS[:2], vectors_offset=0.1)
    (result,) = execute_model(
        PairModel(), [system_1, system_2], [ENERGY], check_consistency=True
    )

    block = result.block()
    np.testing.assert_allclose(
        block.values.ravel(), [(vectors_1**2).sum(), (vectors_2**2).sum()], rtol=1e-6
    )
    gradient = block.gradient("positions").values.reshape(2, 3, 3)
    np.testing.assert_allclose(
        gradient[1], expected_position_gradients(PAIRS[:2], vectors_2), atol=1e-6
    )


def test_reuse_pair_vectors():
    cell = np.zeros((3, 3))
    pairs = PAIRS[:3]
    system, vectors = make_system(POSITIONS, cell, pairs, vectors_offset=0.25)
    request = Quantity(
        name="energy", unit="eV", sample_kind="system", gradients=["positions"]
    )
    (result,) = execute_model(PairModel(), [system], [request], check_consistency=True)

    block = result.block()
    np.testing.assert_allclose(block.values, [[(vectors**2).sum()]], rtol=1e-6)
    np.testing.assert_allclose(
        block.gradient("positions").values.reshape(3, 3),
        expected_position_gradients(pairs, vectors),
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
            (request,) = outputs
            assert request.gradients == ["positions"]

            n_atoms = len(systems[0])
            vectors = systems[0].pairs(PAIRS_OPTIONS).values.reshape(-1, 3)
            block = TensorBlock(
                values=(vectors**2).sum().reshape(1, 1),
                samples=Labels.range("system", 1),
                components=[],
                properties=Labels.range("energy", 1),
            )
            block.add_gradient(
                "positions",
                TensorBlock(
                    values=jnp.full((n_atoms, 3, 1), 42.0),
                    samples=Labels(
                        ["sample", "system", "atom"],
                        np.array([[0, 0, i] for i in range(n_atoms)]),
                    ),
                    components=[Labels.range("xyz", 3)],
                    properties=Labels.range("energy", 1),
                ),
            )
            return [TensorMap(Labels.range("_", 1), [block])]

    model = ExplicitGradients()
    assert model.capabilities().outputs[0].gradients == ["positions", "strain"]

    system, vectors = make_system(POSITIONS, np.zeros((3, 3)), PAIRS[:3])
    (result,) = execute_model(model, [system], [ENERGY], check_consistency=True)

    block = result.block()
    np.testing.assert_allclose(block.gradient("positions").values, 42.0)
    np.testing.assert_allclose(
        block.gradient("strain").values.reshape(3, 3),
        2.0 * np.einsum("pa,pb->ab", vectors, vectors),
        rtol=1e-6,
        atol=1e-6,
    )


def test_super_calls():
    class Base(PairModel):
        def requested_pair_lists(self, requested_outputs):
            return [PairListOptions(cutoff=2.0, full_list=True, strict=True)]

    class Derived(Base):
        def capabilities(self):
            capabilities = super().capabilities()
            assert capabilities.outputs[0].gradients == []
            return capabilities

        def requested_pair_lists(self, requested_outputs):
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


def test_custom_data_and_no_gradients():
    class CustomData(PairModel):
        def forward(self, systems, selected_atoms, outputs):
            data = systems[0].custom_data("test::input").block()
            assert isinstance(data.values, jax.Array)
            return [
                TensorMap(
                    Labels.range("_", 1),
                    [
                        TensorBlock(
                            values=data.values.sum().reshape(1, 1) + 1.0,
                            samples=Labels.range("system", 1),
                            components=[],
                            properties=Labels.range("energy", 1),
                        )
                    ],
                )
            ]

    system, _ = make_system(POSITIONS, CELL, PAIRS)
    request = Quantity(name="energy", unit="eV", sample_kind="system")
    (result,) = execute_model(CustomData(), [system], [request], check_consistency=True)
    np.testing.assert_allclose(result.block().values, [[1.0]])


def test_float64_requires_x64():
    class Float64(PairModel):
        def capabilities(self):
            capabilities = super().capabilities()
            capabilities.dtype = "float64"
            return capabilities

    system, _ = make_system(POSITIONS, CELL, PAIRS)
    message = "jax_enable_x64"
    with pytest.raises(ValueError, match=message):
        Float64().execute_inner([system], None, [ENERGY])


def jax_system(positions, cell, pairs, length_unit="angstrom"):
    """the same system as ``make_system``, as a JaxSystem with computed pair vectors"""
    pairs = np.array(pairs, dtype=np.int32)
    names = [
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]
    custom = TensorBlock(
        values=jnp.zeros((3, 1), dtype=jnp.float32),
        samples=Labels(["system", "atom"], np.array([[0, 0], [0, 1], [0, 2]])),
        components=[],
        properties=Labels.range("_", 1),
    )
    return JaxSystem(
        [1, 6, 8],
        jnp.asarray(positions, dtype=jnp.float32),
        jnp.asarray(cell, dtype=jnp.float32),
        [np.any(c != 0.0) for c in np.array(cell)],
        length_unit=length_unit,
        pairs={PAIRS_OPTIONS: Labels(names, pairs)},
        custom_data={"test::input": TensorMap(Labels.range("_", 1), [custom])},
    )


def test_jax_system_pytree():
    system, vectors = make_system(POSITIONS, CELL, PAIRS)
    jax_system = JaxSystem.from_system(system)

    # positions, cell, pair vectors and custom data are leaves
    leaves, treedef = jax.tree.flatten(jax_system)
    assert len(leaves) == 4
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt.known_pairs() == [PAIRS_OPTIONS]
    assert rebuilt.known_custom_data() == ["test::input"]
    np.testing.assert_equal(rebuilt.types, [1, 6, 8])
    np.testing.assert_allclose(
        rebuilt.pairs(PAIRS_OPTIONS).values.reshape(-1, 3), vectors
    )

    out = jax.jit(lambda s: s.replace(positions=s.positions + 1.0))(jax_system)
    assert isinstance(out, JaxSystem)
    np.testing.assert_allclose(out.positions, np.array(POSITIONS) + 1.0)


def test_evaluate():
    model = PairModel()
    system, _ = make_system(POSITIONS, CELL, PAIRS)
    (expected,) = execute_model(model, [system], [ENERGY], check_consistency=True)

    def evaluate(systems):
        return model.evaluate(systems, [ENERGY])

    for function in [evaluate, jax.jit(evaluate)]:
        (result,) = function([jax_system(POSITIONS, CELL, PAIRS)])
        assert isinstance(result, TensorMap)
        assert isinstance(result.block().values, jax.Array)
        assert mts.equal_metadata(result, expected)
        assert mts.allclose(
            result.to(arrays="numpy"), expected.to(arrays="numpy"), rtol=1e-5, atol=1e-5
        )


def test_evaluate_units():
    # systems in nm, outputs in kcal/mol, both through execute_model and evaluate
    request = Quantity(
        name="energy",
        unit="kcal/mol",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    positions, cell = 0.1 * np.array(POSITIONS), 0.1 * np.array(CELL)
    system, _ = make_system(positions, cell, PAIRS, length_unit="nm")
    (expected,) = execute_model(PairModel(), [system], [request])

    nm = jax_system(positions, cell, PAIRS, length_unit="nm")
    (result,) = PairModel().evaluate([nm], [request])
    assert mts.allclose(
        result.to(arrays="numpy"), expected.to(arrays="numpy"), rtol=1e-4, atol=1e-4
    )

    (reference,) = PairModel().evaluate([jax_system(POSITIONS, CELL, PAIRS)], [ENERGY])
    factor = unit_conversion_factor("eV", "kcal/mol")
    np.testing.assert_allclose(
        result.block().gradient("positions").values,
        10 * factor * reference.block().gradient("positions").values,
        rtol=1e-4,
    )


def test_evaluate_transformations():
    model = PairModel()
    system = jax_system(POSITIONS, CELL, PAIRS)
    request = Quantity(name="energy", unit="eV", sample_kind="system")

    def energy(positions):
        (result,) = model.evaluate([system.replace(positions=positions)], [request])
        return result.block().values[0, 0]

    (expected,) = model.evaluate([system], [ENERGY])
    np.testing.assert_allclose(
        jax.jit(jax.grad(energy))(system.positions),
        expected.block().gradient("positions").values.reshape(3, 3),
        rtol=1e-5,
        atol=1e-5,
    )

    batch = jnp.stack([system.positions * (1 + 0.1 * k) for k in range(3)])
    np.testing.assert_allclose(
        jax.vmap(energy)(batch), [energy(p) for p in batch], rtol=1e-6
    )


def test_evaluate_errors():
    system = jax_system(POSITIONS, CELL, PAIRS)
    message = "can not compute 'test::missing'"
    with pytest.raises(ValueError, match=message):
        PairModel().evaluate(
            [system], [Quantity(name="test::missing", unit="", sample_kind="system")]
        )
