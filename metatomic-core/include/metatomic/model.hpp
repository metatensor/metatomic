#pragma once

#include <string>

#include <metatomic.h>
#include <metatomic/errors.hpp>

namespace metatomic {
    /// Render model metadata as a human-readable string.
    ///
    /// @param metadata a JSON-serialized `ModelMetadata` object as produced by a
    ///     model's `metadata` callback
    /// @return a human-readable rendering of the metadata
    inline std::string format_metadata(const std::string& metadata) {
        mta_string_t printed = nullptr;
        auto status = mta_format_metadata(metadata.c_str(), &printed);
        details::check_status(status);

        auto result = std::string(mta_string_view(printed));
        mta_string_free(printed);

        return result;
    }
} // namespace metatomic
