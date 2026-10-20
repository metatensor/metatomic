#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/torch.h>

#include <metatomic/utils.hpp>

#include "metatomic/torch/units.hpp"

/// Reference SI unit for all known physical dimensions. A unit has a given
/// dimension if it can be converted to the corresponding reference unit.
static const auto DIMENSION_REFERENCE_UNITS = std::unordered_map<std::string, std::string>{
    {"length",    "m"},
    {"energy",    "J"},
    {"force",     "J/m"},
    {"pressure",  "Pa"},
    {"momentum",  "kg*m/s"},
    {"mass",      "kg"},
    {"velocity",  "m/s"},
    {"charge",    "C"},
    {"heat_flux", "J*m/s"},
};

double metatomic_torch::unit_conversion_factor(
    const std::string& from_unit,
    const std::string& to_unit
) {
    try {
        return metatomic::unit_conversion_factor(from_unit, to_unit);
    } catch (const metatomic::Error& e) {
        C10_THROW_ERROR(ValueError, e.what());
    }
}

bool metatomic_torch::details::valid_dimension(const std::string& dimension) {
    if (dimension.empty()) {
        return false;
    }

    if (DIMENSION_REFERENCE_UNITS.find(dimension) == DIMENSION_REFERENCE_UNITS.end()) {
        auto valid_dimensions = std::vector<std::string>();
        for (const auto& it: DIMENSION_REFERENCE_UNITS) {
            valid_dimensions.emplace_back(it.first);
        }
        std::sort(valid_dimensions.begin(), valid_dimensions.end());

        static std::unordered_set<std::string> ALREADY_WARNED = {};
        if (ALREADY_WARNED.insert(dimension).second) {
            TORCH_WARN(
                "unknown dimension '", dimension, "', only [",
                torch::str(valid_dimensions), "] are supported"
            );
        }
        return false;
    } else {
        return true;
    }
}

void metatomic_torch::details::validate_unit(const std::string& dimension, const std::string& unit) {
    if (unit.empty()) {
        return;
    }

    // Always try to parse the expression (catches syntax errors)
    metatomic_torch::unit_conversion_factor(unit, unit);

    // If the dimension is known, verify dimensions match
    auto it = DIMENSION_REFERENCE_UNITS.find(dimension);
    if (it != DIMENSION_REFERENCE_UNITS.end()) {
        try {
            metatomic::unit_conversion_factor(unit, it->second);
        } catch (const metatomic::Error& e) {
            C10_THROW_ERROR(ValueError,
                "unit '" + unit + "' is incompatible with '" + dimension + "' (" + e.what() + ")"
            );
        }
    }
}
