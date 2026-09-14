import math
import re

import pytest

from metatomic import ModelCapabilities, Quantity


@pytest.fixture
def capabilities():
    return ModelCapabilities(
        outputs=[
            Quantity(
                name="energy",
                unit="eV",
                sample_kind="system",
                description="total energy",
                gradients=["positions"],
            ),
            Quantity(
                name="custom::charge/with_variant",
                unit="e",
                sample_kind="atom",
            ),
        ],
        atomic_types=[1, 6, 8],
        interaction_range=5.0,
        length_unit="Angstrom",
        supported_devices=["cpu", "cuda"],
        dtype="float32",
    )


def test_model_capabilities(capabilities):
    assert len(capabilities.outputs) == 2
    assert capabilities.atomic_types == [1, 6, 8]
    assert capabilities.interaction_range == 5.0
    assert capabilities.length_unit == "Angstrom"
    assert capabilities.supported_devices == ["cpu", "cuda"]
    assert capabilities.dtype == "float32"

    # explicit long range models can use an infinite interaction range
    capabilities.interaction_range = math.inf
    assert capabilities.interaction_range == math.inf

    # the returned lists are copies
    capabilities.atomic_types.append(16)
    assert capabilities.atomic_types == [1, 6, 8]

    capabilities.supported_devices.append("metal")
    assert capabilities.supported_devices == ["cpu", "cuda"]

    capabilities.outputs.append(capabilities.outputs[0])
    assert len(capabilities.outputs) == 2


def test_model_capabilities_errors():
    def capabilities(**kwargs):
        parameters = {
            "atomic_types": [1],
            "interaction_range": 5.0,
            "length_unit": "angstrom",
            "supported_devices": ["cpu"],
            "dtype": "float32",
        }
        parameters.update(kwargs)
        return ModelCapabilities(**parameters)

    with pytest.raises(ValueError, match="interaction_range must be non-negative"):
        capabilities(interaction_range=-1.0)

    with pytest.raises(ValueError, match="atomic types must be integers"):
        capabilities(atomic_types=["1"])

    with pytest.raises(ValueError, match="length_unit must be a string"):
        capabilities(length_unit=42)

    message = "device must be one of ['cpu', 'cuda', 'rocm', 'metal'], got wat"
    with pytest.raises(ValueError, match=re.escape(message)):
        capabilities(supported_devices=["cpu", "wat"])

    message = "dtype must be one of ['float32', 'float64'], got float16"
    with pytest.raises(ValueError, match=re.escape(message)):
        capabilities(dtype="float16")

    with pytest.raises(ValueError, match="outputs must be Quantity"):
        capabilities(outputs=["energy"])


def test_model_capabilities_roundtrip(capabilities):
    data = capabilities.to_dict()

    assert data == {
        "type": "metatomic_model_capabilities",
        "outputs": [
            {
                "type": "metatomic_quantity",
                "name": "energy",
                "unit": "eV",
                "sample_kind": "system",
                "gradients": ["positions"],
                "description": "total energy",
            },
            {
                "type": "metatomic_quantity",
                "name": "custom::charge/with_variant",
                "unit": "e",
                "sample_kind": "atom",
                "gradients": [],
            },
        ],
        "atomic_types": [1, 6, 8],
        "interaction_range": 5.0,
        "length_unit": "Angstrom",
        "supported_devices": ["cpu", "cuda"],
        "dtype": "float32",
    }

    parsed = ModelCapabilities.from_dict(data)
    assert parsed == capabilities

    assert parsed.outputs[0].name == "energy"
    assert parsed.outputs[1].name == "custom::charge/with_variant"


def test_model_capabilities_from_dict_errors(capabilities):
    def corrupted(**kwargs):
        data = capabilities.to_dict()
        data.update(kwargs)
        return data

    def without(key):
        data = capabilities.to_dict()
        del data[key]
        return data

    cases = [
        (
            "not an object",
            "invalid JSON data for ModelCapabilities, expected an object",
        ),
        (
            corrupted(type="something-else"),
            "'type' in JSON for ModelCapabilities must be "
            "'metatomic_model_capabilities'",
        ),
        (
            corrupted(outputs="energy"),
            "'outputs' in JSON for ModelCapabilities must be an array",
        ),
        (
            corrupted(atomic_types="1"),
            "'atomic_types' in JSON for ModelCapabilities must be an array",
        ),
        (
            corrupted(atomic_types=[1, "x"]),
            "'atomic_types' in JSON for ModelCapabilities must be an array of integers",
        ),
        (
            without("interaction_range"),
            "'interaction_range' in JSON for ModelCapabilities must be a number",
        ),
        (
            corrupted(interaction_range=-1.0),
            "'interaction_range' in JSON for ModelCapabilities must be non-negative",
        ),
        (
            without("length_unit"),
            "'length_unit' in JSON for ModelCapabilities must be a string",
        ),
        (
            corrupted(supported_devices="cpu"),
            "'supported_devices' in JSON for ModelCapabilities must be an array",
        ),
        (
            corrupted(supported_devices=["cpu", "wat"]),
            "device must be one of ['cpu', 'cuda', 'rocm', 'metal'], got wat",
        ),
        (
            without("dtype"),
            "dtype in JSON for ModelCapabilities must be a string",
        ),
        (
            corrupted(dtype="float16"),
            "dtype must be one of ['float32', 'float64'], got float16",
        ),
    ]

    for data, message in cases:
        with pytest.raises(ValueError, match=re.escape(message)):
            ModelCapabilities.from_dict(data)
