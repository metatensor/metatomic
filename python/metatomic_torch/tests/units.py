import ase.units

from metatomic.torch import ModelOutput, unit_conversion_factor


def test_conversion_length():
    length_angstrom = 1.0
    length_nm = unit_conversion_factor("length", "angstrom", "nm") * length_angstrom
    assert length_nm == 0.1


def test_conversion_energy():
    energy_ev = 1.0
    energy_mev = unit_conversion_factor("energy", "ev", "mev") * energy_ev
    assert energy_mev == 1000.0


def test_units():
    def length_conversion(unit):
        return unit_conversion_factor("length", "angstrom", unit)

    assert length_conversion("bohr") == ase.units.Ang / ase.units.Bohr
    assert length_conversion("nm") == ase.units.Ang / ase.units.nm
    assert length_conversion("nanometer") == ase.units.Ang / ase.units.nm

    def energy_conversion(unit):
        return unit_conversion_factor("energy", "ev", unit)

    assert energy_conversion("Hartree") == ase.units.eV / ase.units.Hartree
    kcal_mol = ase.units.kcal / ase.units.mol
    assert energy_conversion("kcal/mol") == ase.units.eV / kcal_mol
    kJ_mol = ase.units.kJ / ase.units.mol
    assert energy_conversion("kJ/mol") == ase.units.eV / kJ_mol


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
