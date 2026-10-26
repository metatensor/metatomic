#ifndef METATOMIC_TORCH_LEGACY_UNITS_HPP
#define METATOMIC_TORCH_LEGACY_UNITS_HPP

#include <string>

namespace metatomic_torch::details {
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

#endif
