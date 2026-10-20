import pytest
import torch

from metatomic.torch import (
    ModelOutput,
    unit_conversion_factor,
    unit_dimension_for_quantity,
)


def test_unit_conversion_factor():
    factor = unit_conversion_factor(from_unit="angstrom", to_unit="nm")
    assert factor == pytest.approx(0.1)

    message = "unknown unit 'foobar'"
    with pytest.raises(ValueError, match=message):
        unit_conversion_factor("foobar", "eV")


def test_3arg_is_an_error():
    """The 3-arg form has been removed, and should give an error in all the calling
    conventions it used to support."""
    message = (
        "the 3-argument unit_conversion_factor\\(quantity, from_unit, to_unit\\) "
        "has been removed"
    )

    # 3 positional arguments
    with pytest.raises(RuntimeError, match=message):
        unit_conversion_factor("energy", "eV", "meV")

    # 2 positional + to_unit keyword
    with pytest.raises(RuntimeError, match=message):
        unit_conversion_factor("energy", "eV", to_unit="meV")

    # 1 positional + from_unit + to_unit keywords
    with pytest.raises(RuntimeError, match=message):
        unit_conversion_factor("energy", from_unit="eV", to_unit="meV")

    # All keywords
    with pytest.raises(RuntimeError, match=message):
        unit_conversion_factor(quantity="energy", from_unit="eV", to_unit="meV")


def test_2arg_all_calling_conventions():
    """Test all supported calling conventions for the 2-arg form."""
    # 2 positional arguments
    assert unit_conversion_factor("eV", "meV") == pytest.approx(1000.0)
    assert unit_conversion_factor("angstrom", "nm") == pytest.approx(0.1)

    # All keywords
    assert unit_conversion_factor(from_unit="eV", to_unit="meV") == pytest.approx(
        1000.0
    )
    assert unit_conversion_factor(from_unit="angstrom", to_unit="nm") == pytest.approx(
        0.1
    )

    # Mixed: first positional, second keyword
    assert unit_conversion_factor("eV", to_unit="meV") == pytest.approx(1000.0)
    assert unit_conversion_factor("angstrom", to_unit="nm") == pytest.approx(0.1)


def test_3arg_error_cases():
    """Test error handling for calls that look like the removed 3-arg form."""
    # 2 positional args without to_unit keyword is treated as 2-arg form
    # This will fail with "unknown unit" since "energy" is not a valid unit expression
    message = "unknown unit 'energy'"
    with pytest.raises((ValueError, RuntimeError), match=message):
        unit_conversion_factor(
            "energy", "eV"
        )  # treated as 2-arg: from="energy", to="eV"

    # 1 positional arg is treated as 2-arg form with missing to_unit
    message = "unit_conversion_factor requires 2 arguments: from_unit and to_unit"
    with pytest.raises(RuntimeError, match=message):
        unit_conversion_factor("energy")  # treated as 2-arg: from="energy", to=None

    # quantity keyword alone without from_unit/to_unit is treated as 2-arg form
    # _0=None, from_unit="energy", to_unit=None → fails 2-arg validation
    with pytest.raises(RuntimeError, match=message):
        unit_conversion_factor(quantity="energy")


def test_2arg_error_cases():
    """Test error handling for the 2-arg form."""
    # Missing arguments in 2-arg form
    message = "unit_conversion_factor requires 2 arguments: from_unit and to_unit"
    with pytest.raises(RuntimeError, match=message):
        unit_conversion_factor("eV")  # missing to_unit

    with pytest.raises(RuntimeError, match=message):
        unit_conversion_factor()  # missing both

    # Partial kwargs in 2-arg form
    with pytest.raises(RuntimeError, match=message):
        unit_conversion_factor(from_unit="eV")  # missing to_unit


def test_model_output_validate_units():
    # these should work
    ModelOutput(unit="eV/Angstrom")
    ModelOutput(unit=" (eV*u )^(1/ 2 )")

    message = "unknown unit 'foobar'"
    with pytest.raises(ValueError, match=message):
        ModelOutput(unit="foobar")


def test_torchscript_unit_conversion():
    """unit_conversion_factor must be callable from TorchScript."""

    @torch.jit.script
    def convert(from_unit: str, to_unit: str) -> float:
        return torch.ops.metatomic.unit_conversion_factor(from_unit, to_unit)

    assert convert("eV", "meV") == pytest.approx(1000.0)
    assert convert("angstrom", "nm") == pytest.approx(0.1)


def test_unit_dimension_for_quantity():
    assert unit_dimension_for_quantity("energy") == "energy"
    assert unit_dimension_for_quantity("energy_uncertainty") == "energy"
    assert unit_dimension_for_quantity("energy_ensemble") == "energy"
    assert unit_dimension_for_quantity("energy/variant") == "energy"

    assert unit_dimension_for_quantity("heat_flux") == "heat_flux"
    assert unit_dimension_for_quantity("feature") == "none"
    assert unit_dimension_for_quantity("unknown::output") == ""
