import sys

import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

from metatomic import (
    BaseModel,
    ModelCapabilities,
    ModelMetadata,
    Quantity,
    System,
    execute_model,
)


@pytest.fixture
def system():
    """Simple ``System`` with 4 atoms"""
    types = np.array([1, 4, 7, 10], dtype=np.int32)
    positions = np.array(
        [[i * 3 + 1, i * 3 + 2, i * 3 + 3] for i in range(4)],
        dtype=np.float32,
    )
    cell = np.array(
        [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=np.float32,
    )
    pbc = np.array([True, False, True])
    return System("nm", types, positions, cell, pbc)


@pytest.fixture
def energy_output():
    return Quantity(name="energy", unit="eV", sample_kind="system")


class SimpleModel(BaseModel):
    def __init__(self, scale):
        self.scale = scale

    def capabilities(self):
        return ModelCapabilities(
            atomic_types=[1, 4, 7, 10],
            interaction_range=4.5,
            length_unit="nm",
            supported_devices=["cpu"],
            dtype="float32",
            outputs=[
                Quantity(name="energy", unit="eV", sample_kind="system"),
                Quantity(name="custom::output", unit="eV", sample_kind="atom"),
            ],
        )

    def metadata(self):
        return ModelMetadata(
            name="simple Python model",
            description="test model for BaseModel",
        )

    def requested_pair_lists(self):
        return []

    def requested_inputs(self):
        return []

    def execute_inner(self, systems, selected_atoms, requested_outputs):
        if selected_atoms is not None:
            atom_count = len(selected_atoms)
        else:
            atom_count = sum(len(system) for system in systems)

        outputs = []
        for output in requested_outputs:
            if output.name != "energy":
                raise ValueError(f"unknown output: {output.name}")

            block = TensorBlock(
                values=np.array([[self.scale * atom_count]], dtype=np.float64),
                samples=Labels(["system"], np.array([[0]])),
                components=[],
                properties=Labels(["energy"], np.array([[0]])),
            )
            outputs.append(TensorMap(Labels(["_"], np.array([[0]])), [block]))

        return outputs


def test_base_model(system):
    model = SimpleModel(2.5)

    capabilities = model.capabilities()
    assert len(capabilities.atomic_types) == 4

    outputs = capabilities.outputs
    assert len(outputs) == 2
    assert outputs[0].name == "energy"
    assert outputs[1].name == "custom::output"
    assert outputs[1].sample_kind == "atom"

    # NOTE: we call execute_inner directly only for testing, in practice the model
    # should be executed through `execute_model`
    results = model.execute_inner([system], None, [outputs[0]])

    assert len(results) == 1
    assert len(results[0].keys) == 1
    assert results[0].block(0).values[0, 0] == pytest.approx(10.0)


def test_execute_model(system, energy_output):
    model = SimpleModel(2.5)
    systems = [system]

    refcount = sys.getrefcount(model)

    # execute model twice to make sure the model remains valid
    for _ in range(2):
        outputs = execute_model(model, systems, [energy_output])
        assert len(outputs) == 1

        # 4 atoms * scale 2.5 = 10.0
        assert outputs[0].block(0).values[0, 0] == pytest.approx(10.0)

    # `execute_model` does not take ownership of the model
    assert sys.getrefcount(model) == refcount
    assert model.capabilities().length_unit == "nm"


def test_execute_model_selected_atoms(system, energy_output):
    model = SimpleModel(2.5)

    selected_atoms = Labels(
        ["system", "atom"], np.array([[0, 1], [0, 3]], dtype=np.int32)
    )
    outputs = execute_model(
        model, [system], [energy_output], selected_atoms=selected_atoms
    )

    # 2 selected atoms * scale 2.5 = 5.0
    assert outputs[0].block(0).values[0, 0] == pytest.approx(5.0)


def test_python_exceptions(system, energy_output):
    class ThrowingModel(SimpleModel):
        def capabilities(self):
            raise IndexError("ThrowingModel: intentional failure in capabilities")

    # the original Python exception is raised again by execute_model
    message = "ThrowingModel: intentional failure in capabilities"
    with pytest.raises(IndexError, match=message):
        execute_model(ThrowingModel(1.0), [system], [energy_output])

    # same for exceptions in execute_inner
    message = "unknown output: custom::output"
    with pytest.raises(ValueError, match=message):
        execute_model(
            SimpleModel(1.0),
            [system],
            [Quantity(name="custom::output", unit="eV", sample_kind="atom")],
        )


def test_invalid_return_types(system, energy_output):
    class WrongCapabilities(SimpleModel):
        def capabilities(self):
            return {}

    message = "capabilities\\(\\) must return a ModelCapabilities, got <class 'dict'>"
    with pytest.raises(TypeError, match=message):
        execute_model(WrongCapabilities(1.0), [system], [energy_output])

    class WrongInputs(SimpleModel):
        def requested_inputs(self):
            return "not a list"

    message = "requested_inputs\\(\\) must return a list, got <class 'str'>"
    with pytest.raises(TypeError, match=message):
        execute_model(WrongInputs(1.0), [system], [energy_output])

    class WrongPairLists(SimpleModel):
        def requested_pair_lists(self):
            return [energy_output]

    message = (
        "requested_pair_lists\\(\\) must return a list of PairListOptions, "
        "got an element of type <class 'metatomic.Quantity'>"
    )
    with pytest.raises(TypeError, match=message):
        # pair lists are only requested when checking consistency
        execute_model(
            WrongPairLists(1.0), [system], [energy_output], check_consistency=True
        )

    class WrongOutputs(SimpleModel):
        def execute_inner(self, systems, selected_atoms, requested_outputs):
            return [energy_output]

    message = (
        "execute_inner\\(\\) must return a list of TensorMap, "
        "got an element of type <class 'metatomic.Quantity'>"
    )
    with pytest.raises(TypeError, match=message):
        execute_model(WrongOutputs(1.0), [system], [energy_output])

    class WrongOutputsCount(SimpleModel):
        def execute_inner(self, systems, selected_atoms, requested_outputs):
            return []

    message = "model returned 0 outputs, but 1 were requested"
    with pytest.raises(ValueError, match=message):
        execute_model(WrongOutputsCount(1.0), [system], [energy_output])


def test_execute_model_invalid_arguments(system, energy_output):
    model = SimpleModel(1.0)

    message = "`model` must be a BaseModel, not <class 'str'>"
    with pytest.raises(TypeError, match=message):
        execute_model("model", [system], [energy_output])

    message = "`systems` must be a list of System, not <class 'str'>"
    with pytest.raises(TypeError, match=message):
        execute_model(model, ["system"], [energy_output])

    message = "`requested_outputs` must be a list of Quantity, not <class 'str'>"
    with pytest.raises(TypeError, match=message):
        execute_model(model, [system], ["energy"])

    message = "`selected_atoms` must be metatensor Labels or None, not <class 'list'>"
    with pytest.raises(TypeError, match=message):
        execute_model(model, [system], [energy_output], selected_atoms=[])
