import numpy as np
import pytest
from metatensor import Labels

from metatomic import (
    ExternalModel,
    MetatomicError,
    System,
    execute_model,
    load_model,
)


PLUGIN = "python"

MODEL_SCRIPT = """
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

        outputs = []
        for output in requested_outputs:
            assert output.name == "test::sum_of_positions"
            assert output.sample_kind == "system"

            sums = []
            for system in systems:
                system.arrays_backend = "numpy"
                sums.append([system.positions.sum()])

            block = TensorBlock(
                values=np.array(sums, dtype=np.float32),
                samples=Labels.range("system", len(systems)),
                components=[],
                properties=Labels.range("energy", 1),
            )

            outputs.append(TensorMap(Labels.range("_", 1), [block]))
        return outputs


model = SumOfPositions()
"""


def test_python_model(tmp_path):
    path = tmp_path / "model.py"
    path.write_text(MODEL_SCRIPT)

    model = load_model(path, plugin_name=PLUGIN)
    assert isinstance(model, ExternalModel)

    capabilities = model.capabilities()
    assert capabilities.length_unit == "nm"
    assert len(capabilities.outputs) == 1
    output = capabilities.outputs[0]
    assert output.name == "test::sum_of_positions"

    assert model.metadata().name == "sum of positions"
    assert model.requested_pair_lists() == []
    assert model.requested_inputs() == []

    def make_system(n_atoms):
        return System(
            "nm",
            np.array([i * 3 + 1 for i in range(n_atoms)], dtype=np.int32),
            np.arange(1, 3 * n_atoms + 1, dtype=np.float32).reshape(n_atoms, 3),
            np.array(
                [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.float32
            ),
            np.array([True, False, True]),
        )

    systems = [make_system(4), make_system(2)]
    results = execute_model(model, systems, [output], check_consistency=True)
    assert len(results) == 1

    block = results[0].block(0)
    assert block.samples == Labels(["system"], np.array([[0], [1]]))
    np.testing.assert_allclose(block.values, [[78.0], [21.0]])

    # Python exceptions raised by the model are given back as-is
    selected_atoms = Labels(["system", "atom"], np.array([[0, 1]]))
    message = "selected_atoms is not supported by this model"
    with pytest.raises(AssertionError, match=message):
        execute_model(
            model,
            systems,
            [output],
            selected_atoms=selected_atoms,
            check_consistency=True,
        )


def test_not_python_script():
    message = (
        "failed to load model from 'model.txt': "
        f"plugin '{PLUGIN}' could not load the model"
    )
    with pytest.raises(MetatomicError, match=message):
        load_model("model.txt", plugin_name=PLUGIN)


def test_options(tmp_path):
    path = tmp_path / "model.py"
    path.write_text(MODEL_SCRIPT)

    # empty options are fine
    load_model(path, {}, plugin_name=PLUGIN)

    message = "the Python plugin does not support any options yet"
    with pytest.raises(MetatomicError, match=message):
        load_model(path, {"option": "value"}, plugin_name=PLUGIN)


def test_invalid_scripts(tmp_path):
    path = tmp_path / "model.py"
    path.write_text("x = 3\n")

    message = "does not define a 'model' variable"
    with pytest.raises(MetatomicError, match=message):
        load_model(path, plugin_name=PLUGIN)

    path = tmp_path / "model.py"
    path.write_text("model = 3\n")
    message = (
        "the 'model' variable in the Python script at '.*' must be an instance "
        "of metatomic\\.BaseModel"
    )
    with pytest.raises(MetatomicError, match=message):
        load_model(path, plugin_name=PLUGIN)

    # exceptions raised while running the script are given back as-is
    path = tmp_path / "model.py"
    path.write_text("raise ValueError('bad script')\n")
    message = "bad script"
    with pytest.raises(ValueError, match=message):
        load_model(path, plugin_name=PLUGIN)

    path = tmp_path / "model.py"
    path.write_text("def model(:\n")
    message = "invalid syntax"
    with pytest.raises(SyntaxError, match=message):
        load_model(path, plugin_name=PLUGIN)

    message = "No such file or directory"
    with pytest.raises(FileNotFoundError, match=message):
        load_model(str(tmp_path / "missing.py"), plugin_name=PLUGIN)
