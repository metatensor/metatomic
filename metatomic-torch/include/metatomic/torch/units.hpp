#ifndef METATOMIC_TORCH_UNITS_HPP
#define METATOMIC_TORCH_UNITS_HPP

#include <string>

#include "metatomic/torch/exports.h"

namespace metatomic_torch {

/// Check that a given physical quantity is valid and known. This is
/// intentionally not exported with `METATOMIC_TORCH_EXPORT`, and is only
/// intended for internal use.
///
/// Known quantities are: "length", "energy", "force", "pressure", "momentum",
/// "mass", "velocity", and "charge".
bool valid_quantity(const std::string& quantity);

/// Check that a given unit is valid and known for some physical quantity. This
/// is intentionally not exported with `METATOMIC_TORCH_EXPORT`, and is only
/// intended for internal use.
///
/// This function parses the unit expression and verifies that its physical
/// dimensions match the expected dimensions for the given quantity. For example,
/// `validate_unit("energy", "eV")` succeeds, but `validate_unit("energy", "eV/A")`
/// throws an error because eV/A has dimensions of force, not energy.
void validate_unit(const std::string& quantity, const std::string& unit);

/// Get the multiplicative conversion factor to use to convert from
/// `from_unit` to `to_unit`. Both units are parsed as expressions (e.g.
/// "kJ/mol/A^2", "(eV*u)^(1/2)") and their dimensions must match.
///
/// Unit expressions are built from base units combined with `*`, `/`, `^`,
/// and parentheses. Unit lookup is case-insensitive, and whitespace is
/// ignored. For example:
///
/// - `"kJ/mol"` -- energy per mole
/// - `"eV/Angstrom^3"` -- pressure
/// - `"(eV*u)^(1/2)"` -- momentum (fractional powers)
/// - `"Hartree/Bohr"` -- force in atomic units
///
/// Supported base units:
///
/// - **Length**: angstrom (A), bohr, nanometer (nm), meter (m),
///   centimeter (cm), millimeter (mm), micrometer (um, µm)
/// - **Energy**: eV, meV, Hartree, Ry (rydberg), Joule (J), kcal, kJ
///   (note: kcal and kJ are bare; write kcal/mol for per-mole)
/// - **Time**: second (s), millisecond (ms), microsecond (us, µs),
///   nanosecond (ns), picosecond (ps), femtosecond (fs)
/// - **Mass**: dalton (u), kilogram (kg), gram (g), electron_mass (m_e)
/// - **Charge**: e, coulomb (c)
/// - **Dimensionless**: mol
/// - **Derived**: hbar
///
/// Note on quantity validation:
/// The 2-argument form `unit_conversion_factor(from_unit, to_unit)` does not
/// take a quantity parameter. Dimensional compatibility is checked automatically
/// by comparing the parsed dimensions of both unit expressions. The deprecated
/// 3-argument form accepts a `quantity` parameter, but it is ignored for the
/// conversion calculation - it only emits a deprecation warning.
METATOMIC_TORCH_EXPORT double unit_conversion_factor(
    const std::string& from_unit,
    const std::string& to_unit
);

/// @deprecated Use the 2-argument overload instead. The `quantity` parameter
/// is ignored; dimensional compatibility is checked by the parser. Emits a
/// one-time runtime deprecation warning.
[[deprecated("use the 2-argument unit_conversion_factor(from_unit, to_unit) instead")]]
METATOMIC_TORCH_EXPORT double unit_conversion_factor(
    const std::string& quantity,
    const std::string& from_unit,
    const std::string& to_unit
);

}

#endif
