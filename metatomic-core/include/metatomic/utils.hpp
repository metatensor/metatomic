#pragma once

#include <string>

#include <metatomic.h>
#include <metatomic/errors.hpp>

namespace metatomic {
    /// Get the multiplicative conversion factor to use to convert from
    /// `from_unit` to `to_unit`. Both units are parsed as expressions
    /// (e.g. `kJ / mol / A^2`, `(eV * u)^(1/2)`) and their dimensions must
    /// match.
    ///
    /// @verbatim embed:rst:leading-slashes
    ///
    /// .. seealso::
    ///
    ///     The general documentation for :ref:`units`, with the expression
    ///     syntax and list of supported base units.
    ///
    /// @endverbatim
    ///
    /// @param from_unit the unit to convert from
    /// @param to_unit the unit to convert to
    inline double unit_conversion_factor(
        const std::string& from_unit,
        const std::string& to_unit
    ) {
        double conversion = 0.0;

        auto status = mta_unit_conversion_factor(from_unit.c_str(), to_unit.c_str(), &conversion);
        details::check_status(status);

        return conversion;
    }
} // namespace metatomic
