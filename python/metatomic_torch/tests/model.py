import copy
import os
import re
import zipfile
from typing import Dict, List, Optional

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    check_atomistic_model,
    is_atomistic_model,
    load_atomistic_model,
    load_model_extensions,
    read_model_metadata,
)
from metatomic.torch.model import _convert_systems_units

from ._tests_utils import prints_to_stderr


class MinimalModel(torch.nn.Module):
    """The simplest possible metatomic model"""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if "tests::dummy::long_name" in outputs:
            block = TensorBlock(
                values=torch.tensor([[0.0]], dtype=torch.float64),
                samples=Labels("s", torch.tensor([[0]])),
                components=torch.jit.annotate(List[Labels], []),
                properties=Labels("p", torch.tensor([[0]])),
            )
            tensor = TensorMap(Labels("_", torch.tensor([[0]])), [block])
            return {
                "tests::dummy::long_name": tensor,
            }
        else:
            return {}

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(cutoff=1.2, full_list=False, strict=True),
            NeighborListOptions(cutoff=4.3, full_list=True, strict=True),
            NeighborListOptions(cutoff=1.2, full_list=False, strict=False),
        ]


class CustomOutputModel(torch.nn.Module):
    def __init__(self, outputs: List[str]):
        super().__init__()
        self._outputs = outputs

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        labels = Labels("_", torch.tensor([[0]]))
        block = TensorBlock(
            values=torch.zeros(1, 1),
            samples=labels,
            components=[],
            properties=labels,
        )
        result = TensorMap(keys=labels, blocks=[block])
        return {output: result for output in self._outputs}


class CustomInputModel(torch.nn.Module):
    def __init__(self, inputs: List[str]):
        super().__init__()
        self._inputs = inputs

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return {input: ModelOutput(sample_kind="atom") for input in self._inputs}

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        assert len(systems) == 1
        system = systems[0]

        results = {}
        for name in outputs.keys():
            input_name = name[7:]  # remove "input::" prefix
            assert input_name in self._inputs
            results[name] = system.get_data(input_name)

        return results


@pytest.fixture
def model():
    model = MinimalModel()
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 2, 3],
        interaction_range=4.3,
        outputs={
            "tests::dummy::long_name": ModelOutput(sample_kind="system"),
        },
        supported_devices=["cpu"],
        dtype="float64",
    )

    metadata = ModelMetadata()
    return AtomisticModel(model, metadata, capabilities)


@pytest.fixture
def system():
    return System(
        positions=torch.zeros((1, 3), dtype=torch.float64),
        types=torch.tensor([1]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )


def test_save(model, tmp_path):
    os.chdir(tmp_path)
    model.save("export.pt")

    with zipfile.ZipFile("export.pt") as file:
        assert "export/extra/metatomic-version" in file.namelist()
        assert "export/extra/torch-version" in file.namelist()

    check_atomistic_model("export.pt")


def test_recreate(model, tmp_path):
    os.chdir(tmp_path)
    model.save("export.pt")
    model_loaded = load_atomistic_model("export.pt")
    model_loaded.save("export_new.pt")

    with zipfile.ZipFile("export_new.pt") as file:
        assert "export_new/extra/metatomic-version" in file.namelist()
        assert "export_new/extra/torch-version" in file.namelist()

    check_atomistic_model("export_new.pt")


def test_torch_script():
    # make sure functions that have side effects are properly included in the
    # TorchScript code

    @torch.jit.script
    def test_function(path: str):
        check_atomistic_model(path)

    assert "ops.metatomic.check_atomistic_model" in test_function.code

    @torch.jit.script
    def test_function(path: str, extensions_directory: Optional[str]):
        load_model_extensions(path, extensions_directory)

    assert "ops.metatomic.load_model_extensions" in test_function.code


def test_training_mode():
    model = MinimalModel()
    model.train(True)
    capabilities = ModelCapabilities(supported_devices=["cpu"], dtype="float64")

    with pytest.raises(ValueError, match="module should not be in training mode"):
        AtomisticModel(model, ModelMetadata(), capabilities)


def test_save_warning_length_unit(model):
    model._capabilities.length_unit = ""
    match = r"No length unit was provided for the model."
    with pytest.warns(UserWarning, match=match):
        model.save("export.pt")


def test_export(model, tmp_path):
    os.chdir(tmp_path)
    match = r"`export\(\)` is deprecated, use `save\(\)` instead"
    with pytest.warns(DeprecationWarning, match=match):
        model.export("export.pt")


class ExampleModule(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self._name = name

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        return {}

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [NeighborListOptions(1.0, False, True, self._name)]


class OtherModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        return {}

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [NeighborListOptions(2.0, True, False, "other module")]


class FullModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = ExampleModule("first module")
        self.second = ExampleModule("second module")
        self.other = OtherModule()

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        result = self.first(systems, outputs, selected_atoms)
        result.update(self.second(systems, outputs, selected_atoms))
        result.update(self.other(systems, outputs, selected_atoms))

        return result


def test_requested_neighbor_lists(tmpdir):
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        interaction_range=0.0,
        length_unit="A",
        supported_devices=["cpu"],
        dtype="float64",
    )
    atomistic = AtomisticModel(model, ModelMetadata(), capabilities)
    requests = atomistic.requested_neighbor_lists()

    assert len(requests) == 2

    assert requests[0].cutoff == 1.0
    assert not requests[0].full_list
    assert requests[0].strict
    assert requests[0].requestors() == [
        "first module",
        "FullModel.first",
        "second module",
        "FullModel.second",
    ]

    assert requests[1].cutoff == 2.0
    assert requests[1].full_list
    assert not requests[1].strict
    assert requests[1].requestors() == [
        "other module",
        "FullModel.other",
    ]

    # check these are still around after serialization/reload
    atomistic.save(os.path.join(tmpdir, "model.pt"))
    loaded = torch.jit.load(os.path.join(tmpdir, "model.pt"))
    requests = loaded.requested_neighbor_lists()

    assert len(requests) == 2

    assert requests[0].cutoff == 1.0
    assert not requests[0].full_list
    assert requests[0].strict
    assert requests[0].requestors() == [
        "first module",
        "FullModel.first",
        "second module",
        "FullModel.second",
    ]

    assert requests[1].cutoff == 2.0
    assert requests[1].full_list
    assert not requests[1].strict
    assert requests[1].requestors() == [
        "other module",
        "FullModel.other",
    ]


def test_bad_capabilities():
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        supported_devices=["cpu"],
        dtype="float64",
    )
    message = (
        "`capabilities.interaction_range` was not set, "
        "but it is required to run simulations"
    )
    with pytest.raises(ValueError, match=message):
        AtomisticModel(model, ModelMetadata(), capabilities)

    capabilities = ModelCapabilities(
        interaction_range=12,
        dtype="float64",
    )
    message = (
        "`capabilities.supported_devices` was not set, "
        "but it is required to run simulations"
    )
    with pytest.raises(ValueError, match=message):
        AtomisticModel(model, ModelMetadata(), capabilities)

    capabilities = ModelCapabilities(
        interaction_range=float("nan"),
        supported_devices=["cpu"],
        dtype="float64",
    )
    message = (
        "`capabilities.interaction_range` should be a float between 0 and infinity"
    )
    with pytest.raises(ValueError, match=message):
        AtomisticModel(model, ModelMetadata(), capabilities)

    capabilities = ModelCapabilities(
        interaction_range=12.0,
        supported_devices=["cpu"],
    )
    message = "`capabilities.dtype` was not set, but it is required to run simulations"
    with pytest.raises(ValueError, match=message):
        AtomisticModel(model, ModelMetadata(), capabilities)

    message = (
        "invalid model output name 'not-a-standard': this is not a known quantity. "
        "Variant names should look like '<quantity>/<variant>'. "
        "Non-standard names should look like '<domain>::<quantity>[/<variant>]'"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        ModelCapabilities(outputs={"not-a-standard": ModelOutput()})


def test_annotation_check():
    class BadModel(torch.nn.Module):
        def forward(self, x: int) -> int:
            return x

    message = (
        "`module.forward()` takes unexpected arguments, expected signature is "
        "`forward(self, systems: List[System], outputs: Dict[str, ModelOutput], "
        "selected_atoms: Optional[Labels]) -> Dict[str, TensorMap]`, got "
        "`forward(self, x: int) -> int`"
    )
    model = BadModel().eval()
    with pytest.raises(TypeError, match=re.escape(message)):
        _ = AtomisticModel(model, ModelMetadata(), ModelCapabilities())

    model = torch.jit.script(model)
    with pytest.raises(TypeError, match=re.escape(message)):
        _ = AtomisticModel(model, ModelMetadata(), ModelCapabilities())

    # ================================================================================ #
    class BadModel(torch.nn.Module):
        def forward(self, systems: int, outputs: int, selected_atoms: int) -> int:
            return 0

    message = "`systems` argument must be a list of metatomic `System`, not `int`"
    model = BadModel().eval()
    with pytest.raises(TypeError, match=re.escape(message)):
        _ = AtomisticModel(model, ModelMetadata(), ModelCapabilities())

    model = torch.jit.script(model)
    with pytest.raises(TypeError, match=re.escape(message)):
        _ = AtomisticModel(model, ModelMetadata(), ModelCapabilities())

    # ================================================================================ #
    class BadModel(torch.nn.Module):
        def forward(
            self, systems: List[System], outputs: int, selected_atoms: int
        ) -> int:
            return 0

    message = "`outputs` argument must be `Dict[str, ModelOutput]`, not `int`"
    model = BadModel().eval()
    with pytest.raises(TypeError, match=re.escape(message)):
        _ = AtomisticModel(model, ModelMetadata(), ModelCapabilities())

    model = torch.jit.script(model)
    with pytest.raises(TypeError, match=re.escape(message)):
        _ = AtomisticModel(model, ModelMetadata(), ModelCapabilities())

    # ================================================================================ #
    class BadModel(torch.nn.Module):
        def forward(
            self,
            systems: List[System],
            outputs: Dict[str, ModelOutput],
            selected_atoms: int,
        ) -> int:
            return 0

    message = "`selected_atoms` argument must be `Optional[Labels]`, not `int`"
    model = BadModel().eval()
    with pytest.raises(TypeError, match=re.escape(message)):
        _ = AtomisticModel(model, ModelMetadata(), ModelCapabilities())

    model = torch.jit.script(model)
    with pytest.raises(TypeError, match=re.escape(message)):
        _ = AtomisticModel(model, ModelMetadata(), ModelCapabilities())

    # ================================================================================ #
    class BadModel(torch.nn.Module):
        def forward(
            self,
            systems: List[System],
            outputs: Dict[str, ModelOutput],
            selected_atoms: Optional[Labels],
        ) -> int:
            return 0

    message = "`forward()` must return a `Dict[str, TensorMap]`, not `int`"
    model = BadModel().eval()
    with pytest.raises(TypeError, match=re.escape(message)):
        _ = AtomisticModel(model, ModelMetadata(), ModelCapabilities())

    model = torch.jit.script(model)
    with pytest.raises(TypeError, match=re.escape(message)):
        _ = AtomisticModel(model, ModelMetadata(), ModelCapabilities())


def test_access_module(tmpdir):
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="nm",
        interaction_range=0.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    atomistic = AtomisticModel(model, ModelMetadata(), capabilities)

    # Access wrapped module
    assert atomistic.module is model

    atomistic.save(tmpdir / "export.pt")
    loaded_atomistic = load_atomistic_model(tmpdir / "export.pt")

    # Access wrapped module after loading
    loaded_atomistic.module

    # Verfify that it contains the original submodules
    loaded_atomistic.module.first
    loaded_atomistic.module.second
    loaded_atomistic.module.other


def test_is_atomistic_model(tmpdir):
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="A",
        interaction_range=0.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    atomistic = AtomisticModel(model, ModelMetadata(), capabilities)
    atomistic.save(tmpdir / "model.pt")

    scripted_atomistic = torch.jit.script(atomistic)
    loaded_atomistic = load_atomistic_model(tmpdir / "model.pt")

    assert is_atomistic_model(atomistic)
    assert is_atomistic_model(scripted_atomistic)
    assert is_atomistic_model(loaded_atomistic)

    match = "`module` should be a torch.nn.Module, not float"
    with pytest.raises(TypeError, match=match):
        is_atomistic_model(1.0)


def test_read_metadata(tmpdir):
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="nm",
        interaction_range=0.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    metadata = ModelMetadata(
        name="NEW_SOTA",
        description="A SOTA model",
        authors=["Alice", "Bob"],
        references={"implementation": ["doi:1234", "arXiv:1234"]},
    )
    atomistic = AtomisticModel(model, metadata, capabilities)
    atomistic.save(tmpdir / "model.pt")

    extracted_metadata = read_model_metadata(str(tmpdir / "model.pt"))

    assert str(extracted_metadata) == str(metadata)


@pytest.mark.parametrize("n_systems", [0, 1, 8])
@pytest.mark.parametrize("torch_scripted_model", [True, False])
def test_predictions(model, tmp_path, system, n_systems, torch_scripted_model):
    os.chdir(tmp_path)
    model.save("export.pt")
    model_loaded = load_atomistic_model("export.pt")

    # check re-wrapping and re-saving an already scripted model
    if torch_scripted_model:
        assert isinstance(model_loaded.module, torch.jit.RecursiveScriptModule)
        wrapper = AtomisticModel(
            model_loaded.module,
            model_loaded.metadata(),
            model_loaded.capabilities(),
        )
        wrapper.save("export_scripted.pt")
        model_loaded = load_atomistic_model("export_scripted.pt")

    requested_neighbor_lists = model_loaded.requested_neighbor_lists()
    for requested_neighbor_list in requested_neighbor_lists:
        system.add_neighbor_list(
            requested_neighbor_list,
            TensorBlock(
                values=torch.empty(0, 3, 1, dtype=torch.float64),
                samples=Labels(
                    [
                        "first_atom",
                        "second_atom",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    torch.empty(0, 5, dtype=torch.int32),
                ),
                components=[Labels.range("xyz", 3)],
                properties=Labels.range("distance", 1),
            ),
        )
    systems = [system] * n_systems

    outputs = {"tests::dummy::long_name": ModelOutput(sample_kind="system")}
    evaluation_options = ModelEvaluationOptions(length_unit="angstrom", outputs=outputs)

    result = model_loaded(systems, evaluation_options, check_consistency=True)
    assert "tests::dummy::long_name" in result
    assert isinstance(result["tests::dummy::long_name"], torch.ScriptObject)
    assert result["tests::dummy::long_name"]._type().name() == "TensorMap"


def test_consistent_requested_outputs(system):
    model = CustomOutputModel([])
    model.eval()

    outputs = {"energy": ModelOutput(unit="eV", sample_kind="system")}

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 2, 3],
        interaction_range=4.3,
        outputs=outputs,
        supported_devices=["cpu"],
        dtype="float64",
    )

    evaluation_options = ModelEvaluationOptions(length_unit="angstrom", outputs=outputs)
    atomistic = AtomisticModel(model, ModelMetadata(), capabilities)

    match = "the model did not produce the 'energy' output requested by the engine"
    with pytest.raises(ValueError, match=match):
        atomistic([system], evaluation_options, check_consistency=True)


def test_inconsistent_dtype(system):
    model = CustomOutputModel(["energy"])
    model.eval()

    outputs = {"energy": ModelOutput(unit="eV", sample_kind="system")}

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 2, 3],
        interaction_range=4.3,
        outputs=outputs,
        supported_devices=["cpu"],
        dtype="float64",
    )

    evaluation_options = ModelEvaluationOptions(length_unit="angstrom", outputs=outputs)
    atomistic = AtomisticModel(model, ModelMetadata(), capabilities)

    match = (
        "wrong dtype for 'energy': the model dtype is torch.float64 but "
        "the data uses torch.float32"
    )
    with pytest.raises(ValueError, match=match):
        atomistic([system], evaluation_options, check_consistency=True)


def test_not_requested_output(system):
    model = torch.jit.script(CustomOutputModel(["energy"]).eval())

    outputs = {
        "energy/scaled": ModelOutput(
            unit="eV",
            sample_kind="system",
            description="scaled energy",
        ),
        "energy": ModelOutput(
            unit="eV",
            sample_kind="system",
            description="energy without scaling",
        ),
    }

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 2, 3],
        interaction_range=4.3,
        outputs=outputs,
        supported_devices=["cpu"],
        dtype="float32",
    )

    atomistic = AtomisticModel(model, ModelMetadata(), capabilities)
    system = system.to(torch.float32)

    evaluation_options = ModelEvaluationOptions(length_unit="angstrom", outputs=outputs)
    # the model will be missing an output that was requested
    match = (
        "the model did not produce the 'energy/scaled' output requested by the engine"
    )
    with pytest.raises(ValueError, match=match):
        atomistic([system], evaluation_options, check_consistency=True)

    # make sure it does not crash with check_consistency=False
    atomistic([system], evaluation_options, check_consistency=False)

    # the model will create outputs that where not requested
    evaluation_options = ModelEvaluationOptions(length_unit="angstrom", outputs={})
    match = "the model produced an output named 'energy', which was not requested"
    with pytest.raises(ValueError, match=match):
        atomistic([system], evaluation_options, check_consistency=True)

    # make sure it does not crash with check_consistency=False
    atomistic([system], evaluation_options, check_consistency=False)


@pytest.mark.parametrize(
    "old_new_names", [("masses", "mass"), ("masses/variant", "mass/variant")]
)
def test_deprecated_outputs(system, old_new_names, capfd):
    torch.set_warn_always(True)
    old_name, new_name = old_new_names

    output = ModelOutput(unit="kg", sample_kind="atom")

    def make_capabilities(name):
        return ModelCapabilities(
            length_unit="angstrom",
            atomic_types=[1, 2, 3],
            interaction_range=4.3,
            outputs={name: output},
            supported_devices=["cpu"],
            dtype="float64",
        )

    ######### case 1: model and engine use the old name #########
    model = CustomOutputModel([old_name])

    if "/" in old_name:
        old_base = old_name.split("/")[0]
        new_base = new_name.split("/")[0]
        stderr_warning = (
            f"Warning: the '{old_base}' quantity in '{old_name}' is deprecated, "
            f"please update this code to use '{new_base}' instead."
        )
    else:
        stderr_warning = (
            f"Warning: the '{old_name}' quantity is deprecated, "
            f"please update this code to use '{new_name}' instead."
        )

    with prints_to_stderr(capfd, match=stderr_warning):
        capabilities = make_capabilities(old_name)

    message = (
        f"the '{old_name}' output name is deprecated, "
        f"please update the model to use '{new_name}' instead"
    )
    with pytest.warns(match=message):
        atomistic = AtomisticModel(model.eval(), ModelMetadata(), capabilities)

    # the model offers both the old and new name as output
    assert old_name in atomistic.capabilities().outputs
    assert new_name in atomistic.capabilities().outputs

    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom", outputs={old_name: output}
    )
    message = (
        f"the '{old_name}' output name is deprecated, "
        f"please update the engine to use '{new_name}' instead"
    )
    with pytest.warns(match=message):
        outputs = atomistic([system], evaluation_options, check_consistency=False)

    assert list(outputs.keys()) == [old_name]

    ######### case 2: model uses the old name, engine uses the new name #########
    model = CustomOutputModel([old_name])
    with prints_to_stderr(capfd, match=stderr_warning):
        capabilities = make_capabilities(old_name)

    message = (
        f"the '{old_name}' output name is deprecated, "
        f"please update the model to use '{new_name}' instead"
    )
    with pytest.warns(match=message):
        atomistic = AtomisticModel(model.eval(), ModelMetadata(), capabilities)

    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom", outputs={new_name: output}
    )
    # no warning at evaluation time
    outputs = atomistic([system], evaluation_options, check_consistency=False)
    assert list(outputs.keys()) == [new_name]

    ######### case 3: model uses the new name, engine uses the old name #########
    model = CustomOutputModel([new_name])
    capabilities = make_capabilities(new_name)
    atomistic = AtomisticModel(model.eval(), ModelMetadata(), capabilities)

    # the model offers both the old and new name as output
    assert old_name in atomistic.capabilities().outputs
    assert new_name in atomistic.capabilities().outputs

    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom", outputs={old_name: output}
    )
    message = (
        f"the '{old_name}' output name is deprecated, "
        f"please update the engine to use '{new_name}' instead"
    )
    with pytest.warns(match=message):
        outputs = atomistic([system], evaluation_options, check_consistency=False)
        assert list(outputs.keys()) == [old_name]

    ######### case 4: both model and engine use the new name #########
    model = CustomOutputModel([new_name])
    capabilities = make_capabilities(new_name)
    atomistic = AtomisticModel(model.eval(), ModelMetadata(), capabilities)

    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom", outputs={new_name: output}
    )
    # should not warn
    outputs = atomistic([system], evaluation_options, check_consistency=False)
    assert list(outputs.keys()) == [new_name]

    torch.set_warn_always(False)


@pytest.mark.parametrize(
    "old_new_names", [("masses", "mass"), ("masses/variant", "mass/variant")]
)
def test_deprecated_inputs(system, old_new_names, capfd):
    torch.set_warn_always(True)
    old_name, new_name = old_new_names

    output = ModelOutput(unit="kg", sample_kind="atom")

    labels = Labels("_", torch.tensor([[0]]))
    block = TensorBlock(
        values=torch.zeros(1, 1, dtype=torch.float64),
        samples=labels,
        components=[],
        properties=labels,
    )
    tensor = TensorMap(keys=labels, blocks=[block])

    system_without_data = system

    def make_capabilities(name):
        return ModelCapabilities(
            length_unit="angstrom",
            atomic_types=[1, 2, 3],
            interaction_range=4.3,
            outputs={"input::" + name: output},
            supported_devices=["cpu"],
            dtype="float64",
        )

    ######### case 1: model and engine use the old name #########
    model = CustomInputModel([old_name])

    capabilities = make_capabilities(old_name)

    message = (
        f"the '{old_name}' input name is deprecated, please update the model to "
        f"request and use '{new_name}' instead"
    )
    with pytest.warns(UserWarning, match=re.escape(message)):
        atomistic = AtomisticModel(model.eval(), ModelMetadata(), capabilities)

    message = (
        "calling Model.requested_inputs(use_new_names=False) is deprecated, "
        "please update your code to use the new names and call "
        "Model.requested_inputs(use_new_names=True) instead"
    )
    with pytest.warns(match=re.escape(message)):
        assert old_name in atomistic.requested_inputs()

    system = copy.deepcopy(system_without_data)

    if "/" in old_name:
        old_base = old_name.split("/")[0]
        new_base = new_name.split("/")[0]
        name_check_message = (
            f"the '{old_base}' quantity in '{old_name}' is deprecated, "
            f"please update this code to use '{new_base}' instead."
        )
    else:
        name_check_message = (
            f"the '{old_name}' quantity is deprecated, "
            f"please update this code to use '{new_name}' instead."
        )
    with pytest.warns(match=name_check_message):
        system.add_data(old_name, tensor)

    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom", outputs={"input::" + old_name: output}
    )
    # should not warn at this point
    atomistic([system], evaluation_options, check_consistency=False)

    ######### case 2: model uses the old name, engine uses the new name #########

    assert new_name in atomistic.requested_inputs(use_new_names=True)
    system = copy.deepcopy(system_without_data)
    system.add_data(new_name, tensor)

    with pytest.warns(DeprecationWarning, match=name_check_message):
        atomistic([system], evaluation_options, check_consistency=False)

    ######### case 3: model uses the new name, engine uses the old name #########
    model = CustomInputModel([new_name])
    capabilities = make_capabilities(new_name)

    atomistic = AtomisticModel(model.eval(), ModelMetadata(), capabilities)

    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom", outputs={"input::" + new_name: output}
    )

    message = (
        "calling Model.requested_inputs(use_new_names=False) is deprecated, "
        "please update your code to use the new names and call "
        "Model.requested_inputs(use_new_names=True) instead"
    )
    with pytest.warns(match=re.escape(message)):
        assert old_name in atomistic.requested_inputs()

    system = copy.deepcopy(system_without_data)
    with pytest.warns(match=name_check_message):
        system.add_data(old_name, tensor)

    atomistic([system], evaluation_options, check_consistency=False)

    ######### case 4: both model and engine use the new name #########
    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom", outputs={"input::" + new_name: output}
    )
    # should not warn
    system = copy.deepcopy(system_without_data)
    system.add_data(new_name, tensor)
    atomistic([system], evaluation_options, check_consistency=False)

    torch.set_warn_always(False)


def test_systems_unit_conversion(system):
    requested_inputs = {
        "mass": ModelOutput(unit="kg", sample_kind="atom"),
    }
    mass_block = TensorBlock(
        values=torch.tensor([[1.0]], dtype=torch.float64),
        samples=Labels("atom", torch.tensor([[0]])),
        components=[],
        properties=Labels("mass", torch.tensor([[0]])),
    )
    mass_tensor = TensorMap(Labels("atom", torch.tensor([[0]])), [mass_block])
    mass_tensor.set_info("unit", "u")
    mass_tensor.set_info("quantity", "mass")
    system.add_data("mass", mass_tensor)
    systems = [system, system]
    converted_systems = _convert_systems_units(
        systems, "angstrom", "nm", requested_inputs
    )

    # The systems are the same, so the converted systems should be the same as well
    assert torch.allclose(
        converted_systems[0].positions, converted_systems[1].positions
    )
    assert torch.allclose(
        converted_systems[0].get_data("mass").block().values,
        converted_systems[1].get_data("mass").block().values,
    )

    # To check if the conversion was correct
    assert torch.allclose(converted_systems[0].positions, systems[0].positions * 1e-1)
    assert torch.allclose(
        converted_systems[0].get_data("mass").block().values,
        systems[0].get_data("mass").block().values * 1.660539e-27,
    )
