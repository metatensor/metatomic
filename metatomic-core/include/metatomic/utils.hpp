#pragma once

#include <string>

#include <metatomic.h>
#include <metatomic/errors.hpp>

namespace metatomic {

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
