#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cstring>
#include <map>
#include <cassert>

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

    struct References {
        /// The references about the model as a whole, e.g. a paper describing the
        /// model or a website presenting it.
        std::vector<std::string> model;
        /// The references about the architecture of the model, e.g. papers
        /// describing the mathematical form of the model.
        std::vector<std::string> architecture;
        /// The references about the implementation of the model, e.g. a link to
        /// the source code repository or a paper describing the software.
        std::vector<std::string> implementation;

        References() = default;
        References(
            const std::vector<std::string>& model_,
            const std::vector<std::string>& architecture_,
            const std::vector<std::string>& implementation_
        ) : model(model_), architecture(architecture_), implementation(implementation_) {}
    };

    namespace detail {

    inline std::vector<std::string> read_references(const nlohmann::json& j, const std::string& key) {
        if (!j.contains(key) || !j[key].is_array()) {
            throw std::invalid_argument("'" + key + "' in references of ModelMetadata must be an array");
        }

        std::vector<std::string> references;
        for (const auto& reference : j[key]) {
            if (!reference.is_string()) {
                throw std::invalid_argument("'" + key + "' in references of ModelMetadata must be an array of strings");
            }
            references.push_back(reference.get<std::string>());
        }
        return references;
    }

    } // namespace detail

    void to_json(nlohmann::json& j, const References& r) {
        j = nlohmann::json{
            {"model", r.model},
            {"architecture", r.architecture},
            {"implementation", r.implementation}
        };
    }

    void from_json(const nlohmann::json& j, References& r) {
        if (!j.is_object()) {
            throw std::invalid_argument("invalid JSON data for references in ModelMetadata, expected an object");
        }

        r.model = detail::read_references(j, "model");
        r.architecture = detail::read_references(j, "architecture");
        r.implementation = detail::read_references(j, "implementation");
    }

    namespace detail{

    inline std::vector<std::string> read_string_array(const nlohmann::json& j, const std::string& key) {
        if (!j.contains(key) || !j[key].is_array()) {
            throw std::invalid_argument("'" + key + "' in JSON for ModelMetadata must be an array");
        }

        std::vector<std::string> result;
        for (const auto& item : j[key]) {
            if (!item.is_string()) {
                throw std::invalid_argument("'" + key + "' in JSON for ModelMetadata must be an array of strings");
            }
            result.push_back(item.get<std::string>());
        }
        return result;
    }

    inline std::string normalize_whitespace(const std::string& data) {
        std::string result;
        result.reserve(data.size());
        for (char c : data) {
            if (c == '\n' || c == '\r' || c == '\t') {
                result.push_back(' ');
            } else {
                result.push_back(c);
            }
        }
        return result;
    }

    inline void wrap_80_chars(std::string& output, const std::string& data, size_t indent) {
        std::string normalized = normalize_whitespace(data);
        assert(indent < 30);
        size_t line_length = 80 - indent;
        assert(line_length > 50);
        bool first_line = true;
        size_t start = 0;

        while (true) {
            std::string remaining = normalized.substr(start);

            if (remaining.size() <= line_length) {
                if (!first_line) {
                    output += std::string(indent, ' ');
                }
                output += remaining;
                break;
            }

            std::string slice = remaining.substr(0, line_length);
            size_t space_pos = slice.rfind(' ');

            if (space_pos != std::string::npos) {
                if (!first_line) {
                    output += std::string(indent, ' ');
                }
                output += remaining.substr(0, space_pos);
                output += '\n';
                start += space_pos + 1;
                first_line = false;
            } else {
                size_t word_end = remaining.find(' ');
                if (word_end == std::string::npos) {
                    word_end = remaining.size();
                }
                if (!first_line) {
                    output += std::string(indent, ' ');
                }
                output += remaining.substr(0, word_end);
                output += '\n';
                first_line = false;
                if (word_end < remaining.size()) {
                    start += word_end + 1;
                } else {
                    break;
                }
            }
        }
    }

    } // namespace detail

    struct ModelMetadata {
        std::string name;
        std::vector<std::string> authors;
        std::string description;
        References references;
        std::map<std::string, std::string> extra;

        ModelMetadata() = default;
        ModelMetadata(
            const std::string& name_,
            const std::vector<std::string>& authors_,
            const std::string& description_,
            const References& references_,
            const std::map<std::string, std::string>& extra_
        ) : name(name_), authors(authors_), description(description_), references(references_), extra(extra_) {}

        std::string print() const {
            std::string output;
            if (name.empty()) {
                output += "This is an unnamed model\n";
                output += "========================\n";
            } else {
                output += "This is the " + name + " model\n";
                output += "============" + std::string(name.length(), '=') + "======\n";
            }

            if (!description.empty()) {
                output += "\n";
                detail::wrap_80_chars(output, description, 0);
                output += "\n";
            }

            if (!authors.empty()) {
                output += "\nModel authors\n-------------\n\n";
                for (const auto& author : authors) {
                    output += "- ";
                    detail::wrap_80_chars(output, author, 2);
                    output += "\n";
                }
            }

            std::string references_output;
            if (!references.model.empty()) {
                references_output += "- about this specific model:\n";
                for (const auto& reference : references.model) {
                    references_output += "  * ";
                    detail::wrap_80_chars(references_output, reference, 4);
                    references_output += "\n";
                }
            }

            if (!references.architecture.empty()) {
                references_output += "- about the architecture of this model:\n";
                for (const auto& reference : references.architecture) {
                    references_output += "  * ";
                    detail::wrap_80_chars(references_output, reference, 4);
                    references_output += "\n";
                }
            }

            if (!references.implementation.empty()) {
                references_output += "- about the implementation of this model:\n";
                for (const auto& reference : references.implementation) {
                    references_output += "  * ";
                    detail::wrap_80_chars(references_output, reference, 4);
                    references_output += "\n";
                }
            }

            if (!references_output.empty()) {
                output += "\nModel references\n----------------\n\n";
                output += "Please cite the following references when using this model:\n";
                output += references_output;
            }

            return output;
        }
    };

    void to_json(nlohmann::json& j, const ModelMetadata& m) {
        j = nlohmann::json{
            {"type", "metatomic_model_metadata"},
            {"name", m.name},
            {"authors", m.authors},
            {"description", m.description},
            {"references", m.references},
            {"extra", m.extra}
        };
    }

    void from_json(const nlohmann::json& j, ModelMetadata& m) {
        if (!j.is_object()) {
            throw std::invalid_argument("invalid JSON data for ModelMetadata, expected an object");
        }

        if (!j.contains("type") || j["type"].get<std::string>() != "metatomic_model_metadata") {
            throw std::invalid_argument("'type' in JSON for ModelMetadata must be 'metatomic_model_metadata'");
        }

        if (!j.contains("name") || !j["name"].is_string()) {
            throw std::invalid_argument("'name' in JSON for ModelMetadata must be a string");
        }
        j["name"].get_to(m.name);

        m.authors = detail::read_string_array(j, "authors");

        if (!j.contains("description") || !j["description"].is_string()) {
            throw std::invalid_argument("'description' in JSON for ModelMetadata must be a string");
        }
        j["description"].get_to(m.description);

        if (!j.contains("references")) {
            throw std::invalid_argument("invalid JSON data for references in ModelMetadata, expected an object");
        }
        j["references"].get_to(m.references);

        if (!j.contains("extra") || !j["extra"].is_object()) {
            throw std::invalid_argument("'extra' in JSON for ModelMetadata must be an object");
        }
        m.extra.clear();
        for (const auto& item : j["extra"].items()) {
            if (!item.value().is_string()) {
                throw std::invalid_argument("'extra' in JSON for ModelMetadata must be an object with string values");
            }
            m.extra[item.key()] = item.value().get<std::string>();
        }

        for (const auto& author : m.authors) {
            if (author.empty()) {
                throw std::invalid_argument("author can not be empty string in ModelMetadata");
            }
        }

        for (const auto& ref : m.references.model) {
            if (ref.empty()) {
                throw std::invalid_argument("reference can not be empty string (in 'model' section)");
            }
        }

        for (const auto& ref : m.references.architecture) {
            if (ref.empty()) {
                throw std::invalid_argument("reference can not be empty string (in 'architecture' section)");
            }
        }

        for (const auto& ref : m.references.implementation) {
            if (ref.empty()) {
                throw std::invalid_argument("reference can not be empty string (in 'implementation' section)");
            }
        }
    }
} // namespace metatomic
