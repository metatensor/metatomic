import re

import pytest

from metatomic import Quantity


@pytest.fixture
def quantity():
    return Quantity(
        name="energy",
        unit="eV",
        sample_kind="atom",
        description="total energy of the system",
        gradients=["positions"],
    )


def test_quantity(quantity):
    assert quantity.name == "energy"
    assert quantity.unit == "eV"
    assert quantity.sample_kind == "atom"
    assert quantity.description == "total energy of the system"
    assert quantity.gradients == ["positions"]

    # an empty description is the same as no description
    quantity.description = ""
    assert quantity.description is None

    # the returned list of gradients is a copy
    quantity.gradients.append("strain")
    assert quantity.gradients == ["positions"]


def test_quantity_errors():
    with pytest.raises(ValueError, match="name must be a string"):
        Quantity(name=42, unit="eV", sample_kind="atom")

    message = "sample_kind must be one of ['system', 'atom', 'atom_pair'], got foo"
    with pytest.raises(ValueError, match=re.escape(message)):
        Quantity(name="energy", unit="eV", sample_kind="foo")

    message = "sample_kind must be a string, got <class 'int'>"
    with pytest.raises(ValueError, match=message):
        Quantity(name="energy", unit="eV", sample_kind=42)

    message = "gradient must be one of ['positions', 'strain'], got foo"
    with pytest.raises(ValueError, match=re.escape(message)):
        Quantity(
            name="energy", unit="eV", sample_kind="atom", gradients=["positions", "foo"]
        )

    with pytest.raises(ValueError, match="unit must be a string"):
        Quantity(name="energy", unit=42, sample_kind="atom")

    with pytest.raises(ValueError, match="description must be a string"):
        Quantity(name="energy", unit="eV", sample_kind="atom", description=42)


def test_quantity_roundtrip(quantity):
    data = quantity.to_dict()

    assert data == {
        "type": "metatomic_quantity",
        "name": "energy",
        "unit": "eV",
        "sample_kind": "atom",
        "gradients": ["positions"],
        "description": "total energy of the system",
    }

    assert Quantity.from_dict(data) == quantity

    # `description` is left out when it is not set
    quantity.description = None
    assert "description" not in quantity.to_dict()
    assert Quantity.from_dict(quantity.to_dict()) == quantity


def test_quantity_from_dict_errors(quantity):
    def corrupted(**kwargs):
        data = quantity.to_dict()
        data.update(kwargs)
        return data

    def without(key):
        data = quantity.to_dict()
        del data[key]
        return data

    cases = [
        ("not an object", "invalid JSON data for Quantity, expected an object"),
        (
            corrupted(type="something-else"),
            "'type' in JSON for Quantity must be 'metatomic_quantity'",
        ),
        (without("name"), "'name' in JSON for Quantity must be a string"),
        (without("unit"), "'unit' in JSON for Quantity must be a string"),
        (corrupted(description=42), "'description' in JSON for Quantity must be a"),
        (without("gradients"), "'gradients' in JSON for Quantity must be an array"),
        (
            corrupted(gradients="positions"),
            "'gradients' in JSON for Quantity must be an array",
        ),
        (
            corrupted(gradients=["positions", "foo"]),
            "gradient must be one of ['positions', 'strain'], got foo",
        ),
        (
            without("sample_kind"),
            "'sample_kind' in JSON for Quantity must be a string",
        ),
        (
            corrupted(sample_kind="foo"),
            "sample_kind must be one of ['system', 'atom', 'atom_pair'], got foo",
        ),
    ]

    for data, message in cases:
        with pytest.raises(ValueError, match=re.escape(message)):
            Quantity.from_dict(data)
