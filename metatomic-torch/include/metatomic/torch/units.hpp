#ifndef METATOMIC_TORCH_UNITS_HPP
#define METATOMIC_TORCH_UNITS_HPP

#include <string>

#include "metatomic/torch/exports.h"

namespace metatomic_torch {

namespace details {
    /// Check that a given physical dimension is valid and known.
    ///
    /// This is intentionally not exported with `METATOMIC_TORCH_EXPORT`, and is
    /// only intended for internal use.
    bool valid_dimension(const std::string& dimension);

    /// Check that a given unit is valid and known for some physical dimension.
    ///
    /// This is intentionally not exported with `METATOMIC_TORCH_EXPORT`, and is
    /// only intended for internal use.
    ///
    /// This function parses the unit expression and verifies that its physical
    /// dimensions match the expected dimensions for the given quantity. For
    /// example, `validate_unit("energy", "eV")` succeeds, but
    /// `validate_unit("energy", "eV/A")` throws an error because eV/A has
    /// dimensions of force, not energy.
    void validate_unit(const std::string& dimension, const std::string& unit);
}

/// Get the multiplicative conversion factor to use to convert from
/// `from_unit` to `to_unit`. Both units are parsed as expressions (e.g.
/// `kJ / mol / A^2`, `(eV * u)^(1/2)`) and their dimensions must match.
///
/// @verbatim embed:rst:leading-slashes
///
/// .. seealso::
///
///     The general documentation for :ref:`units`, with the expression
///     syntax and list of supported base units.
///
/// @endverbatim
METATOMIC_TORCH_EXPORT double unit_conversion_factor(
    const std::string& from_unit,
    const std::string& to_unit
);

}

#endif
