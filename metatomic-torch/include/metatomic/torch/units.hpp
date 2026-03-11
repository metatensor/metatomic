#ifndef METATOMIC_TORCH_UNITS_HPP
#define METATOMIC_TORCH_UNITS_HPP

#include <string>

#include "metatomic/torch/exports.h"

namespace metatomic_torch {

/// Check that a given physical quantity is valid and known. This is
/// intentionally not exported with `METATOMIC_TORCH_EXPORT`, and is only
/// intended for internal use.
bool valid_quantity(const std::string& quantity);

/// Check that a given unit is valid and known for some physical quantity. This
/// is intentionally not exported with `METATOMIC_TORCH_EXPORT`, and is only
/// intended for internal use.
void validate_unit(const std::string& quantity, const std::string& unit);

/// Get the multiplicative conversion factor to use to convert from
/// `from_unit` to `to_unit`. Both units are parsed as expressions (e.g.
/// "kJ/mol/A^2", "(eV*u)^(1/2)") and their dimensions must match.
///
/// Unit expressions are built from base tokens combined with `*`, `/`, `^`,
/// and parentheses. Token lookup is case-insensitive, and whitespace is
/// ignored. For example:
///
/// - `"kJ/mol"` -- energy per mole
/// - `"eV/Angstrom^3"` -- pressure
/// - `"(eV*u)^(1/2)"` -- momentum (fractional powers)
/// - `"Hartree/Bohr"` -- force in atomic units
///
/// Supported tokens:
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
METATOMIC_TORCH_EXPORT double unit_conversion_factor(
    const std::string& from_unit,
    const std::string& to_unit
);

/// @deprecated Use the 2-argument overload instead. The `quantity` parameter
/// is ignored; dimensional compatibility is checked by the parser. Emits a
/// one-time runtime deprecation warning.
METATOMIC_TORCH_EXPORT double unit_conversion_factor(
    const std::string& quantity,
    const std::string& from_unit,
    const std::string& to_unit
);

}

#endif
