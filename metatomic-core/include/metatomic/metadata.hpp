#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cstring>

#include <nlohmann/json.hpp>

namespace metatomic{
    struct PairListOptions{
        /// Cutoff radius for this pair list in the length unit of the model
        double cutoff;
        /// Whether the list is a full list (contains both the pair `i -> j` and `j -> i`)
        /// or a half list (contains only `i -> j`)
        bool full_list;
        /// Whether the list guarantees that only atoms within the cutoff are
        /// included (strict) or may also include pairs slightly beyond the cutoff
        /// (non-strict)
        bool strict;
        /// List of strings describing who requested this pair list
        std::vector<std::string> requestors;

        // Comparison operators (note: requestors are NOT included in comparisons)
        bool operator==(const PairListOptions& other) const {
            return cutoff == other.cutoff &&
                   full_list == other.full_list &&
                   strict == other.strict;
        }

        bool operator!=(const PairListOptions& other) const {
            return !(*this == other);
        }

        PairListOptions() = default;
        PairListOptions(double cutoff_, bool full_list_, bool strict_, const std::vector<std::string>& requestors_  = {})
            : cutoff(cutoff_), full_list(full_list_), strict(strict_), requestors(requestors_) {
            if (!std::isfinite(cutoff_) || cutoff_ <= 0.0) {
                throw std::invalid_argument("cutoff must be a finite positive number");
            }
        }
    };

    void to_json(nlohmann::json& j, const PairListOptions& p){
        // Store cutoff as hex-encoded bit pattern
        // Floating-point round-trip conversions is exact
        uint64_t bits;
        std::memcpy(&bits, &p.cutoff, sizeof(double));
        std::ostringstream oss;
        oss << "0x" << std::hex << bits;

        j = nlohmann::json{
            {"type", "metatomic_pair_options"},
            {"cutoff", oss.str()},
            {"full_list", p.full_list},
            {"strict", p.strict},
            {"requestors", p.requestors}
        };
    }

    void from_json(const nlohmann::json& j, PairListOptions& p){
        if (!j.is_object()) {
            throw std::invalid_argument("invalid JSON data for PairListOptions, expected an object");
        }

        // Validate type field
        if (!j.contains("type") || j["type"].get<std::string>() != "metatomic_pair_options") {
            throw std::invalid_argument("'type' in JSON for PairListOptions must be 'metatomic_pair_options'");
        }

        // Parse hex-encoded cutoff
        if (!j.contains("cutoff") || !j["cutoff"].is_string()) {
            throw std::invalid_argument("'cutoff' in JSON for PairListOptions must be a hex-encoded string");
        }
        std::string cutoff_str = j["cutoff"].get<std::string>();

        // Strip "0x" prefix if present
        if (cutoff_str.size() >= 2 && cutoff_str[0] == '0' && cutoff_str[1] == 'x') {
            cutoff_str = cutoff_str.substr(2);
        }

        uint64_t bits;
        try {
            bits = std::stoull(cutoff_str, nullptr, 16);
        } catch (...) {
            throw std::invalid_argument("'cutoff' in JSON for PairListOptions must be a hex-encoded string");
        }
        std::memcpy(&p.cutoff, &bits, sizeof(double));

        // Validate cutoff is finite and positive
        if (!std::isfinite(p.cutoff) || p.cutoff <= 0.0) {
            throw std::invalid_argument("'cutoff' in JSON for PairListOptions must be a finite positive number");
        }

        // Parse required boolean fields
        if (!j.contains("full_list") || !j["full_list"].is_boolean()) {
            throw std::invalid_argument("'full_list' in JSON for PairListOptions must be a boolean");
        }
        j["full_list"].get_to(p.full_list);

        if (!j.contains("strict") || !j["strict"].is_boolean()) {
            throw std::invalid_argument("'strict' in JSON for PairListOptions must be a boolean");
        }
        j["strict"].get_to(p.strict);

        // Parse optional requestors field, filtering empty strings and duplicates
        p.requestors.clear();
        if (j.contains("requestors")) {
            if (!j["requestors"].is_array()) {
                throw std::invalid_argument("'requestors' in JSON for PairListOptions must be an array");
            }

            for (const auto& requestor : j["requestors"]) {
                if (!requestor.is_string()) {
                    throw std::invalid_argument("'requestors' in JSON for PairListOptions must be an array of strings");
                }
                std::string req = requestor.get<std::string>();
                // Ignore empty strings and duplicates, keeping first-seen order
                if (!req.empty() && std::find(p.requestors.begin(), p.requestors.end(), req) == p.requestors.end()) {
                    p.requestors.push_back(req);
                }
            }
        }
    }
} // namespace metatomic
