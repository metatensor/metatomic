import math

import ase.units
import pytest
import torch

from metatomic.torch import ModelOutput, unit_conversion_factor


# 3-arg C++ op (deprecated, but still registered for backward compat)
_unit_conversion_factor_3arg = torch.ops.metatomic.unit_conversion_factor


# ---- Backward compat: 3-arg C++ op still works (with deprecation warning) ----


def test_conversion_length_3arg():
    length_angstrom = 1.0
    length_nm = (
        _unit_conversion_factor_3arg("length", "angstrom", "nm") * length_angstrom
    )
    assert length_nm == pytest.approx(0.1)


def test_conversion_energy_3arg():
    energy_ev = 1.0
    energy_mev = _unit_conversion_factor_3arg("energy", "ev", "mev") * energy_ev
    assert energy_mev == pytest.approx(1000.0)


# ---- 2-arg API ----


def test_conversion_length():
    assert unit_conversion_factor("angstrom", "nm") == pytest.approx(0.1)
    assert unit_conversion_factor("angstrom", "Bohr") == pytest.approx(
        1.8897259886, rel=1e-6
    )
    assert unit_conversion_factor("Angstrom", "meter") == pytest.approx(
        1e-10, rel=1e-10
    )


def test_conversion_energy():
    assert unit_conversion_factor("eV", "meV") == pytest.approx(1000.0)
    assert unit_conversion_factor("eV", "Hartree") == pytest.approx(0.0367493, rel=1e-3)


def test_units_vs_ase():
    assert unit_conversion_factor("angstrom", "bohr") == pytest.approx(
        ase.units.Ang / ase.units.Bohr, rel=1e-6
    )
    assert unit_conversion_factor("angstrom", "nm") == pytest.approx(
        ase.units.Ang / ase.units.nm
    )
    assert unit_conversion_factor("angstrom", "nanometer") == pytest.approx(
        ase.units.Ang / ase.units.nm
    )

    assert unit_conversion_factor("ev", "Hartree") == pytest.approx(
        ase.units.eV / ase.units.Hartree, rel=1e-6
    )
    kcal_mol = ase.units.kcal / ase.units.mol
    assert unit_conversion_factor("ev", "kcal/mol") == pytest.approx(
        ase.units.eV / kcal_mol, rel=1e-4
    )
    kJ_mol = ase.units.kJ / ase.units.mol
    assert unit_conversion_factor("ev", "kJ/mol") == pytest.approx(
        ase.units.eV / kJ_mol, rel=1e-4
    )


# ---- Compound expressions ----


def test_compound_expressions():
    # Force: eV/Angstrom -> Hartree/Bohr
    conv = unit_conversion_factor("eV/Angstrom", "Hartree/Bohr")
    assert conv == pytest.approx(0.0194469, rel=1e-3)

    # kJ/mol -> kcal/mol
    conv = unit_conversion_factor("kJ/mol", "kcal/mol")
    assert conv == pytest.approx(1000.0 / 4184.0, rel=1e-6)

    # Pressure identity
    conv = unit_conversion_factor("eV/Angstrom^3", "eV/A^3")
    assert conv == pytest.approx(1.0)


# ---- Fractional powers ----


def test_fractional_powers():
    # (eV*u)^(1/2) -> u*A/fs
    conv = unit_conversion_factor("(eV*u)^(1/2)", "u*A/fs")
    ev_si = 1.602176634e-19
    u_si = 1.66053906660e-27
    a_si = 1e-10
    fs_si = 1e-15
    expected = math.sqrt(ev_si * u_si) / (u_si * a_si / fs_si)
    assert conv == pytest.approx(expected, rel=1e-3)

    # (eV/u)^(1/2) -> A/fs
    conv = unit_conversion_factor("(eV/u)^(1/2)", "A/fs")
    expected = math.sqrt(ev_si / u_si) / (a_si / fs_si)
    assert conv == pytest.approx(expected, rel=1e-3)


# ---- Dimension mismatch ----


def test_dimension_mismatch():
    with pytest.raises((ValueError, RuntimeError), match="dimension mismatch"):
        unit_conversion_factor("eV", "Angstrom")


# ---- Unknown token ----


def test_unknown_token():
    with pytest.raises((ValueError, RuntimeError), match="unknown unit"):
        unit_conversion_factor("foobar", "eV")


# ---- Empty string ----


def test_empty_string():
    with pytest.raises((ValueError, RuntimeError), match="cannot be an empty string"):
        unit_conversion_factor("", "eV")
    with pytest.raises((ValueError, RuntimeError), match="cannot be an empty string"):
        unit_conversion_factor("eV", "")
    with pytest.raises((ValueError, RuntimeError), match="cannot be an empty string"):
        unit_conversion_factor("", "")


# ---- Overflow/underflow handling ----


def test_overflow_exponentiation():
    with pytest.raises((ValueError, RuntimeError), match="overflows"):
        unit_conversion_factor("angstrom^-100", "meter")
    with pytest.raises((ValueError, RuntimeError), match="overflows"):
        unit_conversion_factor("meter^100", "angstrom")


def test_overflow_multiplication():
    with pytest.raises((ValueError, RuntimeError), match="overflows"):
        unit_conversion_factor("meter^50 * meter^50", "angstrom")


def test_overflow_division():
    with pytest.raises((ValueError, RuntimeError), match="overflows"):
        unit_conversion_factor("meter", "meter^50")


# ---- Valid units (ModelOutput creation still works) ----


def test_valid_units():
    # just checking that all of these are valid
    ModelOutput(quantity="length", unit="A")
    ModelOutput(quantity="length", unit="Angstrom")
    ModelOutput(quantity="length", unit="Bohr")
    ModelOutput(quantity="length", unit="meter")
    ModelOutput(quantity="length", unit=" centimeter")
    ModelOutput(quantity="length", unit="cm")
    ModelOutput(quantity="length", unit="millimeter")
    ModelOutput(quantity="length", unit="mm")
    ModelOutput(quantity="length", unit=" micrometer")
    ModelOutput(quantity="length", unit="um")
    ModelOutput(quantity="length", unit="µm")
    ModelOutput(quantity="length", unit="nanometer")
    ModelOutput(quantity="length", unit="nm ")

    ModelOutput(quantity="energy", unit="eV")
    ModelOutput(quantity="energy", unit="meV")
    ModelOutput(quantity="energy", unit="Hartree")
    ModelOutput(quantity="energy", unit="kcal /  mol ")
    ModelOutput(quantity="energy", unit="kJ/mol")
    ModelOutput(quantity="energy", unit="Joule")
    ModelOutput(quantity="energy", unit="J")
    ModelOutput(quantity="energy", unit="Rydberg")
    ModelOutput(quantity="energy", unit="Ry")

    ModelOutput(quantity="force", unit="eV/Angstrom")
    ModelOutput(quantity="force", unit="eV/A")

    ModelOutput(quantity="pressure", unit="eV/Angstrom^3")
    ModelOutput(quantity="pressure", unit="eV/A^3")

    ModelOutput(quantity="momentum", unit="u * A/ fs")
    ModelOutput(quantity="momentum", unit=" (eV*u )^(1/ 2 )")

    ModelOutput(quantity="velocity", unit="A/fs")
    ModelOutput(quantity="velocity", unit="A/s")


# ---- Time units ----


def test_time_units():
    assert unit_conversion_factor("s", "fs") == pytest.approx(1e15)
    assert unit_conversion_factor("second", "ps") == pytest.approx(1e12)
    assert unit_conversion_factor("ns", "fs") == pytest.approx(1e6)
    assert unit_conversion_factor("us", "ns") == pytest.approx(1e3)
    assert unit_conversion_factor("ms", "us") == pytest.approx(1e3)


# ---- Micro sign for microsecond ----


def test_micro_sign_microsecond():
    # µs -> ns (microsecond via micro sign)
    assert unit_conversion_factor("µs", "ns") == pytest.approx(
        unit_conversion_factor("us", "ns")
    )


# ---- Quantity-unit mismatch ----


def test_quantity_unit_mismatch():
    # energy quantity with force unit
    with pytest.raises((ValueError, RuntimeError), match="incompatible with quantity"):
        ModelOutput(quantity="energy", unit="eV/A")

    # force quantity with energy unit
    with pytest.raises((ValueError, RuntimeError), match="incompatible with quantity"):
        ModelOutput(quantity="force", unit="eV")

    # length quantity with pressure unit
    with pytest.raises((ValueError, RuntimeError), match="incompatible with quantity"):
        ModelOutput(quantity="length", unit="eV/A^3")


# ---- Deprecation warning for 3-arg C++ op ----


def test_3arg_deprecation_warning():
    # The 3-arg C++ op emits a torch warning (not a Python DeprecationWarning)
    # on first call. We just verify the call succeeds and returns the right value.
    result = _unit_conversion_factor_3arg("energy", "eV", "meV")
    assert result == pytest.approx(1000.0)


# ---- TorchScript compatibility ----


def test_torchscript_unit_conversion():
    """unit_conversion_factor must be callable from TorchScript."""

    @torch.jit.script
    def convert(from_unit: str, to_unit: str) -> float:
        return torch.ops.metatomic.unit_conversion_factor_v2(from_unit, to_unit)

    assert convert("eV", "meV") == pytest.approx(1000.0)
    assert convert("angstrom", "nm") == pytest.approx(0.1)
