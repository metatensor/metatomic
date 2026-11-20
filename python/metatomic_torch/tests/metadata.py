import math
import pickle
from typing import Dict, List, Optional

import pytest
import torch

from metatomic_torch import (
    ModelCapabilities,
    ModelMetadata,
    Quantity,
)


class QuantityWrap:
    def __init__(
        self,
        name: str,
        unit: str,
        sample_kind: str,
        gradients: Optional[List[str]],
        description: Optional[str],
    ):
        self._c = Quantity(
            name=name,
            unit=unit,
            sample_kind=sample_kind,
            gradients=gradients,
            description=description,
        )

    def get_name(self) -> str:
        return self._c.name

    def set_name(self, name: str):
        self._c.name = name

    def get_unit(self) -> str:
        return self._c.unit

    def set_unit(self, unit: str):
        self._c.unit = unit

    def get_sample_kind(self) -> str:
        return self._c.sample_kind

    def set_sample_kind(self, sample_kind: str):
        self._c.sample_kind = sample_kind

    def get_gradients(self) -> List[str]:
        return self._c.gradients

    def set_gradients(self, gradients: List[str]):
        self._c.gradients = gradients

    def get_description(self) -> Optional[str]:
        return self._c.description

    def set_description(self, description: Optional[str]):
        self._c.description = description

    def is_same(self, other: Quantity) -> bool:
        return self._c == other

    def string(self) -> str:
        return str(self._c)


def test_quantity_torchscript():
    class TestModule(torch.nn.Module):
        def forward(self, x: QuantityWrap) -> QuantityWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)


def test_quantity():
    quantity = Quantity(
        name="energy",
        unit="eV",
        sample_kind="atom",
        gradients=["positions"],
        description="total energy of the system",
    )

    assert quantity.name == "energy"
    assert quantity.unit == "eV"
    assert quantity.sample_kind == "atom"
    assert quantity.gradients == ["positions"]
    assert quantity.description == "total energy of the system"

    assert repr(quantity) == (
        "Quantity(name='energy', unit='eV', sample_kind='atom', "
        "gradients=['positions'], description='total energy of the system')"
    )

    # default values
    quantity = Quantity("energy", "eV", "system")
    assert quantity.gradients == []
    assert quantity.description is None

    quantity = Quantity("energy", "eV", "system", gradients=None, description=None)
    assert quantity.gradients == []
    assert quantity.description is None

    quantity.name = "mtt::energy/variant"
    assert quantity.name == "mtt::energy/variant"

    quantity.unit = "kcal/mol"
    assert quantity.unit == "kcal/mol"

    quantity.sample_kind = "atom_pair"
    assert quantity.sample_kind == "atom_pair"

    quantity.gradients = ["positions", "strain"]
    assert quantity.gradients == ["positions", "strain"]

    # the returned list of gradients is a copy
    quantity.gradients.append("strain")
    assert quantity.gradients == ["positions", "strain"]

    quantity.description = "some description"
    assert quantity.description == "some description"

    # an empty description is the same as no description
    quantity.description = ""
    assert quantity.description is None

    message = "sample_kind must be one of \\['system', 'atom', 'atom_pair'\\], got foo"
    with pytest.raises(ValueError, match=message):
        quantity.sample_kind = "foo"

    message = "gradient must be one of \\['positions', 'strain'\\], got foo"
    with pytest.raises(ValueError, match=message):
        quantity.gradients = ["positions", "foo"]

    with pytest.raises(ValueError, match=message):
        Quantity("energy", "eV", "atom", gradients=["foo"])

    other = Quantity("mtt::energy/variant", "kcal/mol", "atom_pair")
    assert quantity != other
    other.gradients = ["positions", "strain"]
    assert quantity == other

    quantity.description = "some description"
    loaded = pickle.loads(pickle.dumps(quantity))
    assert loaded == quantity
    assert loaded.description == "some description"


class ModelCapabilitiesWrap:
    def __init__(
        self,
        atomic_types: List[int],
        interaction_range: float,
        length_unit: str,
        supported_devices: List[str],
        dtype: str,
        outputs: Optional[List[Quantity]],
    ):
        self._c = ModelCapabilities(
            atomic_types=atomic_types,
            interaction_range=interaction_range,
            length_unit=length_unit,
            supported_devices=supported_devices,
            dtype=dtype,
            outputs=outputs,
        )

    def get_outputs(self) -> List[Quantity]:
        return self._c.outputs

    def set_outputs(self, outputs: List[Quantity]):
        self._c.outputs = outputs

    def get_atomic_types(self) -> List[int]:
        return self._c.atomic_types

    def set_atomic_types(self, atomic_types: List[int]):
        self._c.atomic_types = atomic_types

    def get_interaction_range(self) -> float:
        return self._c.interaction_range

    def set_interaction_range(self, interaction_range: float):
        self._c.interaction_range = interaction_range

    def get_length_unit(self) -> str:
        return self._c.length_unit

    def set_length_unit(self, length_unit: str):
        self._c.length_unit = length_unit

    def get_supported_devices(self) -> List[str]:
        return self._c.supported_devices

    def set_supported_devices(self, supported_devices: List[str]):
        self._c.supported_devices = supported_devices

    def get_dtype(self) -> str:
        return self._c.dtype

    def set_dtype(self, dtype: str):
        self._c.dtype = dtype

    def is_same(self, other: ModelCapabilities) -> bool:
        return self._c == other

    def string(self) -> str:
        return str(self._c)


def test_capabilities_torchscript():
    class TestModule(torch.nn.Module):
        def forward(self, x: ModelCapabilitiesWrap) -> ModelCapabilitiesWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)


def test_capabilities():
    energy = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions"],
        description="total energy",
    )
    charge = Quantity(name="custom::charge/variant", unit="e", sample_kind="atom")

    capabilities = ModelCapabilities(
        atomic_types=[1, 6, 8],
        interaction_range=5.0,
        length_unit="angstrom",
        supported_devices=["cpu", "cuda"],
        dtype="float32",
        outputs=[energy, charge],
    )

    assert capabilities.outputs == [energy, charge]
    assert capabilities.atomic_types == [1, 6, 8]
    assert capabilities.interaction_range == 5.0
    assert capabilities.length_unit == "angstrom"
    assert capabilities.supported_devices == ["cpu", "cuda"]
    assert capabilities.dtype == "float32"

    assert repr(capabilities) == (
        "ModelCapabilities(outputs=["
        "Quantity(name='energy', unit='eV', sample_kind='system', "
        "gradients=['positions'], description='total energy'), "
        "Quantity(name='custom::charge/variant', unit='e', sample_kind='atom', "
        "gradients=[], description=None)], atomic_types=[1, 6, 8], "
        "interaction_range=5.0, length_unit='angstrom', "
        "supported_devices=['cpu', 'cuda'], dtype='float32')"
    )

    # outputs is optional
    default = ModelCapabilities([1], 5.0, "angstrom", ["cpu"], "float64")
    assert default.outputs == []

    default = ModelCapabilities([1], 5.0, "angstrom", ["cpu"], "float64", None)
    assert default.outputs == []

    # setters
    capabilities.atomic_types = [1]
    assert capabilities.atomic_types == [1]

    capabilities.interaction_range = math.inf
    assert capabilities.interaction_range == math.inf

    capabilities.length_unit = "nm"
    assert capabilities.length_unit == "nm"

    capabilities.supported_devices = ["mps", "hip"]
    assert capabilities.supported_devices == ["metal", "rocm"]

    capabilities.supported_devices = ["metal"]
    assert capabilities.supported_devices == ["metal"]

    capabilities.dtype = "float64"
    assert capabilities.dtype == "float64"

    capabilities.outputs = [charge]
    assert capabilities.outputs == [charge]

    # the returned values are copies
    capabilities.atomic_types.append(16)
    assert capabilities.atomic_types == [1]
    capabilities.outputs.append(energy)
    assert capabilities.outputs == [charge]
    capabilities.outputs[0].unit = "C"
    assert capabilities.outputs == [charge]

    # errors
    message = "`interaction_range` must be non-negative"
    with pytest.raises(RuntimeError, match=message):
        capabilities.interaction_range = -1.0

    with pytest.raises(RuntimeError, match=message):
        capabilities.interaction_range = math.nan

    message = "atomic types must fit in a 32-bit integer, got 1099511627776"
    with pytest.raises(ValueError, match=message):
        capabilities.atomic_types = [2**40]

    message = (
        "device must be one of \\['cpu', 'cuda', 'rocm', 'hip', 'metal', 'mps'\\], "
        "got cuda:0"
    )
    with pytest.raises(ValueError, match=message):
        capabilities.supported_devices = ["cuda:0"]

    message = "dtype must be one of \\['float32', 'float64'\\], got float16"
    with pytest.raises(ValueError, match=message):
        capabilities.dtype = "float16"

    # equality & serialization
    other = ModelCapabilities([1], math.inf, "nm", ["metal"], "float64", [charge])
    assert capabilities == other
    other.dtype = "float32"
    assert capabilities != other

    loaded = pickle.loads(pickle.dumps(capabilities))
    assert loaded == capabilities


class ModelMetadataWrap:
    def __init__(
        self,
        name: str,
        description: str,
        authors: List[str],
        references: Dict[str, List[str]],
        extra: Dict[str, str],
    ):
        self._c = ModelMetadata(
            name=name,
            description=description,
            authors=authors,
            references=references,
            extra=extra,
        )

    def get_name(self) -> str:
        return self._c.name

    def set_name(self, name: str):
        self._c.name = name

    def get_description(self) -> str:
        return self._c.description

    def set_description(self, description: str):
        self._c.description = description

    def get_authors(self) -> List[str]:
        return self._c.authors

    def add_author(self, author: str):
        self._c.authors.append(author)

    def get_references(self) -> Dict[str, List[str]]:
        return self._c.references

    def add_reference(self, section: str, reference: str):
        self._c.references[section].append(reference)

    def string(self) -> str:
        return str(self._c)


def test_metadata_torchscript():
    class TestModule(torch.nn.Module):
        def forward(self, x: ModelMetadataWrap) -> ModelMetadataWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)


def test_metadata_repr():
    metadata = ModelMetadata(
        name="name",
        description="""Lorem ipsum dolor sit amet, consectetur
adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna
aliqua. Ut enim ad minim veniam, quis nostrud exercitation.""",
        authors=[
            "Short author",
            "Some extremely long author that will take more than one line "
            + "in the printed output",
        ],
        references={
            "model": [
                "a very long reference that will take more "
                + "than one line in the printed output"
            ],
            "architecture": ["ref-2", "ref-3"],
            "implementation": ["ref-4"],
        },
    )

    expected = """This is the name model
======================

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation.

Model authors
-------------

- Short author
- Some extremely long author that will take more than one line in the printed
  output

Model references
----------------

Please cite the following references when using this model:
- about this specific model:
  * a very long reference that will take more than one line in the printed
    output
- about the architecture of this model:
  * ref-2
  * ref-3
- about the implementation of this model:
  * ref-4
"""
    assert str(metadata) == expected


def test_extra_metadata(tmpdir):
    metadata = ModelMetadata(
        name="SOTA model",
        description="This is a state-of-the-art model",
        authors=["Author 1", "Author 2"],
        references={
            "model": ["ref-1", "ref-2"],
            "architecture": ["ref-3"],
            "implementation": ["ref-3"],
        },
        extra={
            "number_of_parameters": "1000",
            "foo": "bar",
            "GPU?": "Yes",
        },
    )

    assert metadata.extra["number_of_parameters"] == "1000"
    assert metadata.extra["foo"] == "bar"
    assert metadata.extra["GPU?"] == "Yes"

    torch.save(metadata, str(tmpdir.join("metadata.pt")))

    loaded_metadata = torch.load(str(tmpdir.join("metadata.pt")), weights_only=False)

    assert loaded_metadata.extra["number_of_parameters"] == "1000"
    assert loaded_metadata.extra["foo"] == "bar"
    assert loaded_metadata.extra["GPU?"] == "Yes"


def test_metadata():
    metadata = ModelMetadata(
        name="name",
        authors=["Author"],
        references={"model": ["ref-1"]},
        extra={"foo": "bar"},
    )

    # all the references sections are always present
    assert metadata.references == {
        "model": ["ref-1"],
        "architecture": [],
        "implementation": [],
    }

    metadata.name = "new name"
    assert metadata.name == "new name"

    metadata.description = "new description"
    assert metadata.description == "new description"

    metadata.authors = ["Author 1", "Author 2"]
    assert metadata.authors == ["Author 1", "Author 2"]

    metadata.references = {"implementation": ["ref-2"]}
    assert metadata.references == {
        "model": [],
        "architecture": [],
        "implementation": ["ref-2"],
    }

    metadata.extra = {"key": "value"}
    assert metadata.extra == {"key": "value"}

    message = "author can not be empty string in ModelMetadata"
    with pytest.raises(ValueError, match=message):
        metadata.authors = ["Author", ""]

    message = "unknown key in references: foo"
    with pytest.raises(ValueError, match=message):
        metadata.references = {"foo": ["ref"]}

    message = "reference can not be empty string \\(in 'model' section\\)"
    with pytest.raises(ValueError, match=message):
        metadata.references = {"model": [""]}

    loaded = pickle.loads(pickle.dumps(metadata))
    assert loaded.name == "new name"
    assert loaded.description == "new description"
    assert loaded.authors == ["Author 1", "Author 2"]
    assert loaded.references["implementation"] == ["ref-2"]
    assert loaded.extra == {"key": "value"}
