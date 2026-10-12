import numpy as np
from metatensor import Labels, TensorBlock, TensorMap

from metatomic import (
    ModelCapabilities,
    ModelMetadata,
    NumpyModel,
    Quantity,
    System,
    execute_model,
)


class SumOfPositions(NumpyModel):
    def capabilities(self):
        return ModelCapabilities(
            atomic_types=[1, 4, 7, 10],
            interaction_range=0.0,
            length_unit="nm",
            supported_devices=["cpu"],
            dtype="float32",
            outputs=[
                Quantity(name="test::sum_of_positions", unit="", sample_kind="system")
            ],
        )

    def metadata(self):
        return ModelMetadata(name="sum of positions")

    def execute_inner(self, systems, selected_atoms, requested_outputs):
        assert selected_atoms is None, "selected_atoms is not supported by this model"

        output = []
        for system in systems:
            assert isinstance(system.positions, np.ndarray)
            output.append([system.positions.sum()])

        block = TensorBlock(
            values=np.array(output, dtype=np.float32),
            samples=Labels.range("system", len(systems)),
            components=[],
            properties=Labels.range("energy", 1),
        )

        return [TensorMap(Labels.range("_", 1), [block]) for _ in requested_outputs]


def test_numpy_model():
    model = SumOfPositions()

    capabilities = model.capabilities()
    assert capabilities.length_unit == "nm"
    assert len(capabilities.outputs) == 1
    output = capabilities.outputs[0]
    assert output.name == "test::sum_of_positions"

    assert model.metadata().name == "sum of positions"
    assert model.requested_pair_lists(capabilities.outputs) == []
    assert model.requested_inputs(capabilities.outputs) == []

    system = System(
        "nm",
        np.array([i * 3 + 1 for i in range(4)], dtype=np.int32),
        np.arange(1, 3 * 4 + 1, dtype=np.float32).reshape(4, 3),
        np.array(
            [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.float32
        ),
        np.array([True, False, True]),
    )
    results = execute_model(model, [system], [output], check_consistency=True)
    assert len(results) == 1

    block = results[0].block(0)
    assert block.samples == Labels(["system"], np.array([[0]]))
    np.testing.assert_allclose(block.values, [[78.0]])
