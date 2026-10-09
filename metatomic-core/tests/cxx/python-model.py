# Python model used to test the Python plugin from C++.
import numpy as np
from metatensor import Labels, TensorBlock, TensorMap

from metatomic import BaseModel, ModelCapabilities, ModelMetadata, Quantity


class SumOfPositions(BaseModel):
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

    def requested_pair_lists(self):
        return []

    def requested_inputs(self):
        return []

    def execute_inner(self, systems, selected_atoms, requested_outputs):
        assert selected_atoms is None, "selected_atoms is not supported by this model"

        if len(requested_outputs) != 1:
            raise ValueError(
                f"expected exactly one requested output, got {len(requested_outputs)}"
            )

        assert requested_outputs[0].name == "test::sum_of_positions"
        assert requested_outputs[0].sample_kind == "system"

        energies = []
        for system in systems:
            system.arrays_backend = "numpy"
            energies.append([system.positions.sum()])

        block = TensorBlock(
            values=np.array(energies, dtype=np.float32),
            samples=Labels.range("system", len(systems)),
            components=[],
            properties=Labels.range("energy", 1),
        )

        return [TensorMap(Labels.range("_", 1), [block]) for _ in requested_outputs]


model = SumOfPositions()
