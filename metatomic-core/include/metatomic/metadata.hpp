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
#include <cstdint>
#include <cctype>

#include <metatomic/errors.hpp>
#include <nlohmann/json.hpp>

namespace metatomic{
    struct PairListOptions{
        /// Cutoff radius for this pair list in the length unit of the model
        double cutoff = 0.0;
        /// Whether the list is a full list (contains both the pair `i -> j` and `j -> i`)
        /// or a half list (contains only `i -> j`)
        bool full_list = false;
        /// Whether the list guarantees that only atoms within the cutoff are
        /// included (strict) or may also include pairs slightly beyond the cutoff
        /// (non-strict)
        bool strict = false;
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

    namespace detail {

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

    inline bool is_valid_identifier(const std::string& s) {
        if (s.empty()) {
            return false;
        }

        // Check that the first character is a letter or underscore
        char first = s[0];
        if (!(std::isalpha(static_cast<unsigned char>(first)) || first == '_')) {
            return false;
        }

        // Check that all the characters are alphanumeric or underscore
        for (char c : s) {
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
                return false;
            }
        }

        return true;
    }

    inline void validate_quantity_name(const std::string& name) {
        static const std::vector<std::string> STANDARD_QUANTITIES = {
            "charge",
            "energy_ensemble",
            "energy_uncertainty",
            "energy",
            "feature",
            "heat_flux",
            "mass",
            "momentum",
            "non_conservative_force",
            "non_conservative_stress",
            "position",
            "spin_multiplicity",
            "velocity",
        };

        auto is_standard = [&](const std::string& candidate) {
            return std::find(STANDARD_QUANTITIES.begin(), STANDARD_QUANTITIES.end(), candidate)
                   != STANDARD_QUANTITIES.end();
        };

        if (is_standard(name)) {
            return;
        }

        std::string main_part = name;
        std::string variant;
        auto slash_pos = name.find('/');
        if (slash_pos != std::string::npos) {
            main_part = name.substr(0, slash_pos);
            variant = name.substr(slash_pos + 1);
        }

        if (main_part.empty()) {
            throw std::invalid_argument("quantity name cannot be empty in '" + name + "'");
        }

        if (!variant.empty() && !is_valid_identifier(variant)) {
            throw std::invalid_argument(
                "invalid quantity variant '" + variant + "' in '" + name +
                "': must be a valid identifier (alphanumeric or underscore, not starting with a digit)"
            );
        }

        if (is_standard(main_part)) {
            return;
        }

        std::vector<std::string> components;
        std::size_t start = 0;
        while (true) {
            auto pos = main_part.find("::", start);
            if (pos == std::string::npos) {
                components.push_back(main_part.substr(start));
                break;
            }
            components.push_back(main_part.substr(start, pos - start));
            start = pos + 2;
        }

        for (const auto& component : components) {
            if (!is_valid_identifier(component)) {
                throw std::invalid_argument(
                    "invalid quantity name component '" + component + "' in '" + name +
                    "': must be a valid identifier (alphanumeric or underscore, not starting with a digit)"
                );
            }
        }

        if (components.size() == 1) {
            throw std::invalid_argument(
                "'" + name + "' is not a standard quantity name; custom quantity names must use '<namespace>::<name>'"
            );
        }
    }

    } // namespace detail

    // Forward declarations
    // The ModelMetadata::print function uses to_json
    struct ModelMetadata;
    void to_json(nlohmann::json&, const ModelMetadata&);

    struct ModelMetadata {
        /// References for a model, divided into three categories: references about
        /// the model as a whole, references about the architecture of the model,
        /// and references about the implementation of the model.
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
            mta_string_t mta_string;
            nlohmann::json j;

            to_json(j, *this);

            mta_format_metadata(j.dump().c_str(), &mta_string);
            std::string output = mta_string_view(mta_string);
            mta_string_free(mta_string);

            return output;
        }
    };

    void to_json(nlohmann::json& j, const ModelMetadata::References& r) {
        j = nlohmann::json{
            {"model", r.model},
            {"architecture", r.architecture},
            {"implementation", r.implementation}
        };
    }

    void from_json(const nlohmann::json& j, ModelMetadata::References& r) {
        if (!j.is_object()) {
            throw std::invalid_argument("invalid JSON data for references in ModelMetadata, expected an object");
        }

        r.model = detail::read_references(j, "model");
        r.architecture = detail::read_references(j, "architecture");
        r.implementation = detail::read_references(j, "implementation");
    }

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

        if (!j.contains("references") || !j["references"].is_object()) {
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

    /// Capabilities of a model: which outputs it provides, which atoms it
    /// supports, etc.
    struct ModelCapabilities {
        /// The data type of a model, used for all inputs and outputs.
        enum class DType {
            /// 32-bit floating point, following the IEEE 754 standard
            Float32,
            /// 64-bit floating point, following the IEEE 754 standard
            Float64,
        };

        /// A device on which a model can run.
        enum class Device {
            CPU,
            CUDA,
            ROCM,
            Metal,
        };

        /// The kind of samples a quantity can be associated with
        enum class SampleKind {
            /// The quantity is defined for each atom (e.g. atomic energy, charge, ...)
            Atom,
            /// The quantity is defined for the whole system (e.g. total energy, ...)
            System,
            /// The quantity is defined for each pair of atoms (e.g. hamiltonian elements, ...)
            AtomPair,
        };

        /// The gradients a quantity can have
        enum class Gradients {
            /// Gradients with respect to atomic positions
            Positions,
            /// Gradients with respect to the strain (typically used for stress)
            Strain,
        };

        /// A quantity that a model can use as input or output
        struct Quantity {
            /// Name of the quantity, this can be a standard name from
            /// https://docs.metatensor.org/metatomic/latest/quantities/index.html, or
            /// a custom name of the form `<namespace>::<name>[/<variant>]`
            std::string name;
            /// Unit of the quantity
            std::string unit;
            /// Description of the quantity, used to provide more details about the
            /// quantity, especially when a model defines multiple variants of the same
            /// quantity. An empty string is treated as no description.
            std::string description;
            /// List of explicit gradients for this quantity
            std::vector<Gradients> gradients;
            /// The kind of samples this quantity is associated with
            SampleKind sample_kind = SampleKind::Atom;

            Quantity() = default;
            Quantity(
                const std::string& name_,
                const std::string& unit_,
                const std::string& description_,
                const std::vector<Gradients>& gradients_,
                const SampleKind& sample_kind_
            ) : name(name_), unit(unit_), description(description_), gradients(gradients_), sample_kind(sample_kind_) {}
        };

        /// The outputs this model can provide
        std::vector<Quantity> outputs;
        /// The atomic types this model supports. The meaning of the integers in
        /// this list is up to the model, and is not required to be the atomic
        /// numbers.
        std::vector<int64_t> atomic_types;
        /// The interaction range of the model (in the length unit of the model),
        /// i.e. the maximum distance between two atoms for which the model's output
        /// can depend on their relative position.
        double interaction_range = 0.0;
        /// The length unit of the model, e.g. "angstrom" or "nanometer". This is
        /// used to interpret the `interaction_range` and convert the inputs.
        std::string length_unit;
        /// The devices on which the model can run, e.g. `["cpu", "cuda"]`.
        std::vector<Device> supported_devices;
        /// The data type of the model, used for all inputs and outputs.
        DType dtype = DType::Float32;

        ModelCapabilities() = default;
        ModelCapabilities(
            const std::vector<Quantity>& outputs_,
            const std::vector<int64_t>& atomic_types_,
            double interaction_range_,
            const std::string& length_unit_,
            const std::vector<Device>& supported_devices_,
            DType dtype_
        ) : outputs(outputs_), atomic_types(atomic_types_), interaction_range(interaction_range_),
            length_unit(length_unit_), supported_devices(supported_devices_), dtype(dtype_) {}
    };

    void to_json(nlohmann::json& j, const ModelCapabilities::DType& dtype) {
        switch (dtype) {
            case ModelCapabilities::DType::Float32:
                j = "float32";
                break;
            case ModelCapabilities::DType::Float64:
                j = "float64";
                break;
        }
    }

    void from_json(const nlohmann::json& j, ModelCapabilities::DType& dtype) {
        if (!j.is_string()) {
            throw std::invalid_argument("dtype in JSON for ModelCapabilities must be a string");
        }

        std::string s = j.get<std::string>();
        if (s == "float32") {
            dtype = ModelCapabilities::DType::Float32;
        } else if (s == "float64") {
            dtype = ModelCapabilities::DType::Float64;
        } else {
            throw std::invalid_argument(
                "invalid string for dtype in JSON for ModelCapabilities, expected 'float32' or 'float64'"
            );
        }
    }

    void to_json(nlohmann::json& j, const ModelCapabilities::Device& device) {
        switch (device) {
            case ModelCapabilities::Device::CPU:
                j = "cpu";
                break;
            case ModelCapabilities::Device::CUDA:
                j = "cuda";
                break;
            case ModelCapabilities::Device::ROCM:
                j = "rocm";
                break;
            case ModelCapabilities::Device::Metal:
                j = "metal";
                break;
        }
    }

    void from_json(const nlohmann::json& j, ModelCapabilities::Device& device) {
        if (!j.is_string()) {
            throw std::invalid_argument("'supported_devices' in JSON for ModelCapabilities must be an array of strings");
        }

        std::string s = j.get<std::string>();
        if (s == "cpu") {
            device = ModelCapabilities::Device::CPU;
        } else if (s == "cuda") {
            device = ModelCapabilities::Device::CUDA;
        } else if (s == "rocm") {
            device = ModelCapabilities::Device::ROCM;
        } else if (s == "metal") {
            device = ModelCapabilities::Device::Metal;
        } else {
            throw std::invalid_argument(
                "invalid string for device in JSON for ModelCapabilities, expected 'cpu', 'cuda', 'rocm', or 'metal'"
            );
        }
    }

    void to_json(nlohmann::json& j, const ModelCapabilities::SampleKind& kind) {
        switch (kind) {
            case ModelCapabilities::SampleKind::Atom:
                j = "atom";
                break;
            case ModelCapabilities::SampleKind::System:
                j = "system";
                break;
            case ModelCapabilities::SampleKind::AtomPair:
                j = "atom_pair";
                break;
        }
    }

    void from_json(const nlohmann::json& j, ModelCapabilities::SampleKind& kind) {
        if (!j.is_string()) {
            throw std::invalid_argument("'sample_kind' in JSON for Quantity must be a string");
        }

        std::string s = j.get<std::string>();
        if (s == "atom") {
            kind = ModelCapabilities::SampleKind::Atom;
        } else if (s == "system") {
            kind = ModelCapabilities::SampleKind::System;
        } else if (s == "atom_pair") {
            kind = ModelCapabilities::SampleKind::AtomPair;
        } else {
            throw std::invalid_argument(
                "'sample_kind' in JSON for Quantity must be 'atom', 'system' or 'atom_pair', got '" + s + "'"
            );
        }
    }

    void to_json(nlohmann::json& j, const ModelCapabilities::Gradients& gradients) {
        switch (gradients) {
            case ModelCapabilities::Gradients::Positions:
                j = "positions";
                break;
            case ModelCapabilities::Gradients::Strain:
                j = "strain";
                break;
        }
    }

    void from_json(const nlohmann::json& j, ModelCapabilities::Gradients& gradients) {
        if (!j.is_string()) {
            throw std::invalid_argument("'gradients' in JSON for Quantity must be a string");
        }

        std::string s = j.get<std::string>();
        if (s == "positions") {
            gradients = ModelCapabilities::Gradients::Positions;
        } else if (s == "strain") {
            gradients = ModelCapabilities::Gradients::Strain;
        } else {
            throw std::invalid_argument(
                "'gradients' in JSON for Quantity must be 'positions' or 'strain', got '" + s + "'"
            );
        }
    }

    void to_json(nlohmann::json& j, const ModelCapabilities::Quantity& q) {
        j = nlohmann::json{
            {"type", "metatomic_quantity"},
            {"name", q.name},
            {"unit", q.unit},
            {"gradients", q.gradients},
            {"sample_kind", q.sample_kind}
        };

        if (!q.description.empty()) {
            j["description"] = q.description;
        }
    }

    void from_json(const nlohmann::json& j, ModelCapabilities::Quantity& q) {
        if (!j.is_object()) {
            throw std::invalid_argument("invalid JSON data for Quantity, expected an object");
        }

        if (!j.contains("type") || j["type"].get<std::string>() != "metatomic_quantity") {
            throw std::invalid_argument("'type' in JSON for Quantity must be 'metatomic_quantity'");
        }

        if (!j.contains("name") || !j["name"].is_string()) {
            throw std::invalid_argument("'name' in JSON for Quantity must be a string");
        }
        j["name"].get_to(q.name);
        detail::validate_quantity_name(q.name);

        if (!j.contains("unit") || !j["unit"].is_string()) {
            throw std::invalid_argument("'unit' in JSON for Quantity must be a string");
        }
        j["unit"].get_to(q.unit);

        q.description.clear();
        if (j.contains("description")) {
            if (!j["description"].is_string()) {
                throw std::invalid_argument("'description' in JSON for Quantity must be a string");
            }
            j["description"].get_to(q.description);
        }

        if (!j.contains("gradients") || !j["gradients"].is_array()) {
            throw std::invalid_argument("'gradients' in JSON for Quantity must be an array");
        }
        q.gradients.clear();
        for (const auto& gradient : j["gradients"]) {
            q.gradients.push_back(gradient.get<ModelCapabilities::Gradients>());
        }

        if (!j.contains("sample_kind") || !j["sample_kind"].is_string()) {
            throw std::invalid_argument("'sample_kind' in JSON for Quantity must be a string");
        }
        j["sample_kind"].get_to(q.sample_kind);
    }

    void to_json(nlohmann::json& j, const ModelCapabilities& c) {
        j = nlohmann::json{
            {"type", "metatomic_model_capabilities"},
            {"outputs", c.outputs},
            {"atomic_types", c.atomic_types},
            {"interaction_range", c.interaction_range},
            {"length_unit", c.length_unit},
            {"supported_devices", c.supported_devices},
            {"dtype", c.dtype}
        };
    }

    void from_json(const nlohmann::json& j, ModelCapabilities& c) {
        if (!j.is_object()) {
            throw std::invalid_argument("invalid JSON data for ModelCapabilities, expected an object");
        }

        if (!j.contains("type") || j["type"].get<std::string>() != "metatomic_model_capabilities") {
            throw std::invalid_argument("'type' in JSON for ModelCapabilities must be 'metatomic_model_capabilities'");
        }

        if (!j.contains("outputs") || !j["outputs"].is_array()) {
            throw std::invalid_argument("'outputs' in JSON for ModelCapabilities must be an array");
        }
        c.outputs.clear();
        for (const auto& output : j["outputs"]) {
            c.outputs.push_back(output.get<ModelCapabilities::Quantity>());
        }

        if (!j.contains("atomic_types") || !j["atomic_types"].is_array()) {
            throw std::invalid_argument("'atomic_types' in JSON for ModelCapabilities must be an array");
        }
        c.atomic_types.clear();
        for (const auto& atomic_type : j["atomic_types"]) {
            if (!atomic_type.is_number_integer()) {
                throw std::invalid_argument("'atomic_types' in JSON for ModelCapabilities must be an array of integers");
            }
            c.atomic_types.push_back(atomic_type.get<int64_t>());
        }

        if (!j.contains("interaction_range") || !j["interaction_range"].is_number()) {
            throw std::invalid_argument("'interaction_range' in JSON for ModelCapabilities must be a number");
        }
        j["interaction_range"].get_to(c.interaction_range);
        if (c.interaction_range < 0.0) {
            throw std::invalid_argument("'interaction_range' in JSON for ModelCapabilities must be non-negative");
        }

        if (!j.contains("length_unit") || !j["length_unit"].is_string()) {
            throw std::invalid_argument("'length_unit' in JSON for ModelCapabilities must be a string");
        }
        j["length_unit"].get_to(c.length_unit);

        // Validate that `length_unit` has the dimension of length by asking the
        // C API for a conversion factor to meters. The call only succeeds when
        // the dimensions match; otherwise `check_status` throws with the C API's
        // dimension-mismatch message.
        double conversion_factor = 0.0;
        auto status = mta_unit_conversion_factor(c.length_unit.c_str(), "m", &conversion_factor);
        details::check_status(status);

        if (!j.contains("supported_devices") || !j["supported_devices"].is_array()) {
            throw std::invalid_argument("'supported_devices' in JSON for ModelCapabilities must be an array");
        }
        c.supported_devices.clear();
        for (const auto& device : j["supported_devices"]) {
            c.supported_devices.push_back(device.get<ModelCapabilities::Device>());
        }

        if (!j.contains("dtype") || !j["dtype"].is_string()) {
            throw std::invalid_argument("dtype in JSON for ModelCapabilities must be a string");
        }
        j["dtype"].get_to(c.dtype);
    }
} // namespace metatomic
