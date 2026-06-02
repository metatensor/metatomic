#pragma once

#include <string_view>

#include <metatomic.h>
#include <metatomic/errors.hpp>

namespace metatomic {

    inline double unit_conversion_factor(
        std::string_view from_unit,
        std::string_view to_unit
    ) {
        double conversion = 0.0;

        auto status = mta_unit_conversion_factor(from_unit.data(), to_unit.data(), &conversion);
        details::check_status(status);

        return conversion;
    }

} // namespace metatomic
