#pragma once

#include <vector>
#include <sstream>
#include <map>
#include <string>
#include <optional>
#include <algorithm>
#include <cmath> // std::isfinite
#include <cstring> // std::memcpy
#include <cstdint> // std::uint64_t, std::int64_t
#include <cctype> // std::isxdigit

#include <metatomic/errors.hpp>
#include <nlohmann/json.hpp>

namespace metatomic{
    namespace detail {

    inline std::vector<std::string> read_string_array(
        const nlohmann::json& j, const std::string& key, const std::string& context
    ) {
        if (!j.contains(key) || !j[key].is_array()) {
            throw metatomic::Error("'" + key + "' in " + context + " must be an array");
        }

        std::vector<std::string> result;
        for (const auto& item : j[key]) {
            if (!item.is_string()) {
                throw metatomic::Error("'" + key + "' in " + context + " must be an array of strings");
            }
            result.push_back(item.get<std::string>());
        }
        return result;
    }

    } // namespace detail

    /// Options for the calculation of a pair list (neighbor list)
    struct PairListOptions{
    private:
        /// Cutoff radius for this pair list in the length unit of the model
        std::optional<double> cutoff_;
        /// Whether the list is a full list (contains both the pair `i -> j` and `j -> i`)
        /// or a half list (contains only `i -> j`)
        std::optional<bool> full_list_;
        /// Whether the list guarantees that only atoms within the cutoff are
        /// included (strict) or may also include pairs slightly beyond the cutoff
        /// (non-strict)
        bool strict_ = true;
        /// List of strings describing who requested this pair list
        std::vector<std::string> requestors_;

    public:
        /// Set the cutoff radius for this pair list.
        ///
        /// @throw metatomic::Error if the value is not a finite positive number.
        void cutoff(double value) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw metatomic::Error("cutoff must be a finite positive number");
            }
            cutoff_ = value;
        }

        /// Get the cutoff radius for this pair list.
        ///
        /// @throw metatomic::Error if the value has not been set.
        double cutoff() const {
            if (!cutoff_.has_value()) {
                throw metatomic::Error("cutoff is not set in PairListOptions");
            }
            return cutoff_.value();
        }

        /// Set whether this pair list is a full list.
        ///
        /// @throw metatomic::Error if the value has not been set.
        void full_list(bool value) {
            full_list_ = value;
        }

        /// Get whether this pair list is a full list.
        ///
        /// @throw metatomic::Error if the value has not been set.
        bool full_list() const {
            if (!full_list_.has_value()) {
                throw metatomic::Error("full_list is not set in PairListOptions");
            }
            return full_list_.value();
        }

        /// Set whether this pair list is strict.
        void strict(bool value) {
            strict_ = value;
        }

        /// Get whether this pair list is strict.
        bool strict() const {
            return strict_;
        }

        /// Set the list of requestors for this pair list.
        void requestors(const std::vector<std::string>& value) {
            requestors_ = value;
        }

        /// Get the list of requestors for this pair list.
        const std::vector<std::string>& requestors() const {
            return requestors_;
        }

        /// Add a requestor to the list.
        ///
        /// Empty strings and duplicates are ignored, keeping first-seen order.
        void add_requestor(const std::string& requestor) {
            if (!requestor.empty() && std::find(requestors_.begin(), requestors_.end(), requestor) == requestors_.end()) {
                requestors_.push_back(requestor);
            }
        }

        /// Clear the list of requestors.
        void clear_requestors() {
            requestors_.clear();
        }

        bool operator==(const PairListOptions& other) const {
            return cutoff_ == other.cutoff_ &&
                   full_list_ == other.full_list_ &&
                   strict_ == other.strict_;
        }

        bool operator!=(const PairListOptions& other) const {
            return !(*this == other);
        }

        PairListOptions() = default;

        PairListOptions(
            double cutoff_,
            bool full_list_,
            bool strict_ = true,
            const std::vector<std::string>& requestors_ = {}
        ) {
            this->cutoff(cutoff_);
            this->full_list(full_list_);
            this->strict(strict_);
            this->requestors(requestors_);
        }
    };

    inline void to_json(nlohmann::json& j, const PairListOptions& p){
        // Store cutoff as hex-encoded bit pattern
        // Floating-point round-trip conversions is exact
        double cutoff = p.cutoff();
        uint64_t bits;
        std::memcpy(&bits, &cutoff, sizeof(double));
        std::ostringstream oss;
        oss << "0x" << std::hex << bits;

        j = nlohmann::json{
            {"type", "metatomic_pair_options"},
            {"cutoff", oss.str()},
            {"full_list", p.full_list()},
            {"strict", p.strict()},
            {"requestors", p.requestors()}
        };
    }

    inline void from_json(const nlohmann::json& j, PairListOptions& p) {
        if (!j.is_object()) {
            throw metatomic::Error("invalid JSON data for PairListOptions, expected an object");
        }

        if (!j.contains("type") || !j["type"].is_string() || j["type"].get<std::string>() != "metatomic_pair_options") {
            throw metatomic::Error("'type' in JSON for PairListOptions must be 'metatomic_pair_options'");
        }

        // Cutoff is an hex-encoded string
        if (!j.contains("cutoff") || !j["cutoff"].is_string()) {
            throw metatomic::Error("'cutoff' in JSON for PairListOptions must be a hex-encoded string");
        }
        std::string cutoff_str = j["cutoff"].get<std::string>();

        // Strip "0x" prefix if present
        if (cutoff_str.size() >= 2 && cutoff_str[0] == '0' && cutoff_str[1] == 'x') {
            cutoff_str = cutoff_str.substr(2);
        }

        uint64_t bits;
        try {
            // std::isxdigit checks for hex digits
            if (cutoff_str.empty() || !std::all_of(cutoff_str.begin(), cutoff_str.end(), [](unsigned char c) { return std::isxdigit(c); })) {
                throw metatomic::Error("'cutoff' in JSON for PairListOptions must be a hex-encoded string");
            }

            std::size_t pos = 0;
            bits = std::stoull(cutoff_str, &pos, 16);
            if (pos != cutoff_str.size()) {
                throw metatomic::Error("'cutoff' in JSON for PairListOptions must be a hex-encoded string");
            }
        } catch (...) {
            throw metatomic::Error("'cutoff' in JSON for PairListOptions must be a hex-encoded string");
        }
        double cutoff;
        std::memcpy(&cutoff, &bits, sizeof(double));

        if (!std::isfinite(cutoff) || cutoff <= 0.0) {
            throw metatomic::Error("'cutoff' in JSON for PairListOptions must be a finite positive number");
        }

        if (!j.contains("full_list") || !j["full_list"].is_boolean()) {
            throw metatomic::Error("'full_list' in JSON for PairListOptions must be a boolean");
        }
        bool full_list = j["full_list"].get<bool>();

        if (!j.contains("strict") || !j["strict"].is_boolean()) {
            throw metatomic::Error("'strict' in JSON for PairListOptions must be a boolean");
        }
        bool strict = j["strict"].get<bool>();

        p = PairListOptions(cutoff, full_list, strict, {});
        if (j.contains("requestors")) {
            if (!j["requestors"].is_array()) {
                throw metatomic::Error("'requestors' in JSON for PairListOptions must be an array");
            }

            for (const auto& requestor : j["requestors"]) {
                if (!requestor.is_string()) {
                    throw metatomic::Error("'requestors' in JSON for PairListOptions must be an array of strings");
                }
                p.add_requestor(requestor.get<std::string>());
            }
        }
    }

    // Forward declarations
    // The ModelMetadata::print function uses to_json
    struct ModelMetadata;
    void to_json(nlohmann::json&, const ModelMetadata&);

    struct ModelMetadata {
        /// References for a model, divided into three categories: references about
        /// the model as a whole, references about the architecture of the model,
        /// and references about the implementation of the model.
        struct References {
        private:
            /// The references about the model as a whole, e.g. a paper describing the
            /// model or a website presenting it.
            std::vector<std::string> model_;
            /// The references about the architecture of the model, e.g. papers
            /// describing the mathematical form of the model.
            std::vector<std::string> architecture_;
            /// The references about the implementation of the model, e.g. a link to
            /// the source code repository or a paper describing the software.
            std::vector<std::string> implementation_;

        public:
            void model(const std::vector<std::string>& value) {
                model_ = value;
            }

            const std::vector<std::string>& model() const {
                return model_;
            }

            void add_model(const std::string& reference) {
                model_.push_back(reference);
            }

            void clear_model() {
                model_.clear();
            }

            void architecture(const std::vector<std::string>& value) {
                architecture_ = value;
            }

            const std::vector<std::string>& architecture() const {
                return architecture_;
            }

            void add_architecture(const std::string& reference) {
                architecture_.push_back(reference);
            }

            void clear_architecture() {
                architecture_.clear();
            }

            void implementation(const std::vector<std::string>& value) {
                implementation_ = value;
            }

            const std::vector<std::string>& implementation() const {
                return implementation_;
            }

            void add_implementation(const std::string& reference) {
                implementation_.push_back(reference);
            }

            void clear_implementation() {
                implementation_.clear();
            }

            References(
                const std::vector<std::string>& model_ = {},
                const std::vector<std::string>& architecture_ = {},
                const std::vector<std::string>& implementation_ = {}
            ) {
                this->model(model_);
                this->architecture(architecture_);
                this->implementation(implementation_);
            }
        };

    private:
        std::string name_;
        std::vector<std::string> authors_;
        std::string description_;
        References references_;
        // BTreeMap in Rust is an ordered map
        std::map<std::string, std::string> extra_;

    public:
        void name(const std::string& value) {
            name_ = value;
        }

        const std::string& name() const {
            return name_;
        }

        void authors(const std::vector<std::string>& value) {
            authors_ = value;
        }

        const std::vector<std::string>& authors() const {
            return authors_;
        }

        void add_author(const std::string& author) {
            authors_.push_back(author);
        }

        void clear_authors() {
            authors_.clear();
        }

        void description(const std::string& value) {
            description_ = value;
        }

        const std::string& description() const {
            return description_;
        }

        void references(const References& value) {
            references_ = value;
        }

        const References& references() const {
            return references_;
        }

        void add_reference(const std::string& section, const std::string& reference) {
            if (section == "model") {
                references_.add_model(reference);
            } else if (section == "architecture") {
                references_.add_architecture(reference);
            } else if (section == "implementation") {
                references_.add_implementation(reference);
            } else {
                throw metatomic::Error(
                    "reference section must be 'model', 'architecture', or 'implementation', got '" + section + "'"
                );
            }
        }

        void clear_reference(const std::string& section) {
            if (section == "model") {
                references_.clear_model();
            } else if (section == "architecture") {
                references_.clear_architecture();
            } else if (section == "implementation") {
                references_.clear_implementation();
            } else {
                throw metatomic::Error(
                    "reference section must be 'model', 'architecture', or 'implementation', got '" + section + "'"
                );
            }
        }

        void clear_references() {
            references_.clear_model();
            references_.clear_architecture();
            references_.clear_implementation();
        }

        void extra(const std::map<std::string, std::string>& value) {
            extra_ = value;
        }

        const std::map<std::string, std::string>& extra() const {
            return extra_;
        }

        void add_extra(const std::string& key, const std::string& value) {
            extra_[key] = value;
        }

        void clear_extra() {
            extra_.clear();
        }

        ModelMetadata(
            const std::string& name_ = "",
            const std::vector<std::string>& authors_ = {},
            const std::string& description_ = "",
            const References& references_ = {},
            const std::map<std::string, std::string>& extra_ = {}
        ) {
            this->name(name_);
            this->authors(authors_);
            this->description(description_);
            this->references(references_);
            this->extra(extra_);
        }

        std::string print() const {
            // Re-use C API to avoid re-implementing 'normalize_withespace' and 'wrap_80_chars'
            mta_string_t mta_string;
            nlohmann::json j;

            to_json(j, *this);
            auto status = mta_format_metadata(j.dump().c_str(), &mta_string);
            details::check_status(status);

            std::string output = mta_string_view(mta_string);
            mta_string_free(mta_string);

            return output;
        }
    };

    inline void to_json(nlohmann::json& j, const ModelMetadata::References& r) {
        j = nlohmann::json{
            {"model", r.model()},
            {"architecture", r.architecture()},
            {"implementation", r.implementation()}
        };
    }

    inline void from_json(const nlohmann::json& j, ModelMetadata::References& r) {
        if (!j.is_object()) {
            throw metatomic::Error("invalid JSON data for references in ModelMetadata, expected an object");
        }

        r = ModelMetadata::References(
            detail::read_string_array(j, "model", "references of ModelMetadata"),
            detail::read_string_array(j, "architecture", "references of ModelMetadata"),
            detail::read_string_array(j, "implementation", "references of ModelMetadata")
        );
    }

    inline void to_json(nlohmann::json& j, const ModelMetadata& m) {
        j = nlohmann::json{
            {"type", "metatomic_model_metadata"},
            {"name", m.name()},
            {"authors", m.authors()},
            {"description", m.description()},
            {"references", m.references()},
            {"extra", m.extra()}
        };
    }

    inline void from_json(const nlohmann::json& j, ModelMetadata& m) {
        if (!j.is_object()) {
            throw metatomic::Error("invalid JSON data for ModelMetadata, expected an object");
        }

        if (!j.contains("type") || !j["type"].is_string() || j["type"].get<std::string>() != "metatomic_model_metadata") {
            throw metatomic::Error("'type' in JSON for ModelMetadata must be 'metatomic_model_metadata'");
        }

        if (!j.contains("name") || !j["name"].is_string()) {
            throw metatomic::Error("'name' in JSON for ModelMetadata must be a string");
        }
        std::string name = j["name"].get<std::string>();

        auto authors = metatomic::detail::read_string_array(j, "authors", "JSON for ModelMetadata");

        if (!j.contains("description") || !j["description"].is_string()) {
            throw metatomic::Error("'description' in JSON for ModelMetadata must be a string");
        }
        std::string description = j["description"].get<std::string>();

        if (!j.contains("references") || !j["references"].is_object()) {
            throw metatomic::Error("invalid JSON data for references in ModelMetadata, expected an object");
        }
        auto references = j["references"].get<ModelMetadata::References>();

        if (!j.contains("extra") || !j["extra"].is_object()) {
            throw metatomic::Error("'extra' in JSON for ModelMetadata must be an object");
        }
        std::map<std::string, std::string> extra;
        for (const auto& item : j["extra"].items()) {
            if (!item.value().is_string()) {
                throw metatomic::Error("'extra' in JSON for ModelMetadata must be an object with string values");
            }
            extra[item.key()] = item.value().get<std::string>();
        }

        // Validate authors content
        for (const auto& author : authors) {
            if (author.empty()) {
                throw metatomic::Error("author can not be empty string in ModelMetadata");
            }
        }

        // Validate references content
        for (const auto& ref : references.model()) {
            if (ref.empty()) {
                throw metatomic::Error("reference can not be empty string (in 'model' section)");
            }
        }

        for (const auto& ref : references.architecture()) {
            if (ref.empty()) {
                throw metatomic::Error("reference can not be empty string (in 'architecture' section)");
            }
        }

        for (const auto& ref : references.implementation()) {
            if (ref.empty()) {
                throw metatomic::Error("reference can not be empty string (in 'implementation' section)");
            }
        }

        m = ModelMetadata(name, authors, description, references, extra);
    }

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
    private:
        /// Name of the quantity, this can be a standard name from
        /// https://docs.metatensor.org/metatomic/latest/quantities/index.html, or
        /// a custom name of the form `<namespace>::<name>[/<variant>]`
        std::optional<std::string> name_;
        /// Unit of the quantity
        std::optional<std::string> unit_;
        /// Description of the quantity, used to provide more details about the
        /// quantity, especially when a model defines multiple variants of the same
        /// quantity. An empty string is treated as no description.
        std::string description_;
        /// List of explicit gradients for this quantity
        std::vector<Gradients> gradients_;
        /// The kind of samples this quantity is associated with
        std::optional<SampleKind> sample_kind_;

    public:
        void name(const std::string& value) {
            name_ = value;
        }

        const std::string& name() const {
            if (!name_.has_value()) {
                throw metatomic::Error("name is not set in Quantity");
            }
            return name_.value();
        }

        void unit(const std::string& value) {
            unit_ = value;
        }

        const std::string& unit() const {
            if (!unit_.has_value()) {
                throw metatomic::Error("unit is not set in Quantity");
            }
            return unit_.value();
        }

        void description(const std::string& value) {
            description_ = value;
        }

        const std::string& description() const {
            return description_;
        }

        void gradients(const std::vector<Gradients>& value) {
            gradients_ = value;
        }

        const std::vector<Gradients>& gradients() const {
            return gradients_;
        }

        void add_gradient(Gradients gradient) {
            gradients_.push_back(gradient);
        }

        void clear_gradients() {
            gradients_.clear();
        }

        void sample_kind(const SampleKind& value) {
            sample_kind_ = value;
        }

        SampleKind sample_kind() const {
            if (!sample_kind_.has_value()) {
                throw metatomic::Error("sample_kind is not set in Quantity");
            }
            return sample_kind_.value();
        }

        Quantity() = default;

        Quantity(
            const std::string& name_,
            const std::string& unit_,
            const SampleKind& sample_kind_,
            const std::string& description_ = "",
            const std::vector<Gradients>& gradients_ = {}
        ) {
            this->name(name_);
            this->unit(unit_);
            this->sample_kind(sample_kind_);
            this->description(description_);
            this->gradients(gradients_);
        }
    };

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

        using SampleKind = metatomic::SampleKind;
        using Gradients = metatomic::Gradients;
        using Quantity = metatomic::Quantity;

    private:
        /// The outputs this model can provide
        std::vector<Quantity> outputs_;
        /// The atomic types this model supports. The meaning of the integers in
        /// this list is up to the model, and is not required to be the atomic
        /// numbers.
        std::optional<std::vector<int64_t>> atomic_types_;
        /// The interaction range of the model (in the length unit of the model),
        /// i.e. the maximum distance between two atoms for which the model's output
        /// can depend on their relative position.
        std::optional<double> interaction_range_;
        /// The length unit of the model, e.g. "angstrom" or "nanometer". This is
        /// used to interpret the `interaction_range` and convert the inputs.
        std::optional<std::string> length_unit_;
        /// The devices on which the model can run, e.g. `["cpu", "cuda"]`.
        std::optional<std::vector<Device>> supported_devices_;
        /// The data type of the model, used for all inputs and outputs.
        std::optional<DType> dtype_;

    public:
        void outputs(const std::vector<Quantity>& value) {
            outputs_ = value;
        }

        const std::vector<Quantity>& outputs() const {
            return outputs_;
        }

        void add_output(const Quantity& output) {
            outputs_.push_back(output);
        }

        void clear_outputs() {
            outputs_.clear();
        }

        void atomic_types(const std::vector<int64_t>& value) {
            atomic_types_ = value;
        }

        const std::vector<int64_t>& atomic_types() const {
            if (!atomic_types_.has_value()) {
                throw metatomic::Error("atomic_types is not set in ModelCapabilities");
            }
            return atomic_types_.value();
        }

        void add_atomic_type(int64_t atomic_type) {
            if (!atomic_types_.has_value()) {
                atomic_types_ = std::vector<int64_t>();
            }
            atomic_types_->push_back(atomic_type);
        }

        void clear_atomic_types() {
            atomic_types_ = std::vector<int64_t>();
        }

        void interaction_range(double value) {
            if (value < 0.0) {
                throw metatomic::Error("interaction_range must be non-negative");
            }
            interaction_range_ = value;
        }

        double interaction_range() const {
            if (!interaction_range_.has_value()) {
                throw metatomic::Error("interaction_range is not set in ModelCapabilities");
            }
            return interaction_range_.value();
        }

        void length_unit(const std::string& value) {
            length_unit_ = value;
        }

        const std::string& length_unit() const {
            if (!length_unit_.has_value()) {
                throw metatomic::Error("length_unit is not set in ModelCapabilities");
            }
            return length_unit_.value();
        }

        void supported_devices(const std::vector<Device>& value) {
            supported_devices_ = value;
        }

        const std::vector<Device>& supported_devices() const {
            if (!supported_devices_.has_value()) {
                throw metatomic::Error("supported_devices is not set in ModelCapabilities");
            }
            return supported_devices_.value();
        }

        void add_supported_device(Device device) {
            if (!supported_devices_.has_value()) {
                supported_devices_ = std::vector<Device>();
            }
            supported_devices_->push_back(device);
        }

        void clear_supported_devices() {
            supported_devices_ = std::vector<Device>();
        }

        void dtype(DType value) {
            dtype_ = value;
        }

        DType dtype() const {
            if (!dtype_.has_value()) {
                throw metatomic::Error("dtype is not set in ModelCapabilities");
            }
            return dtype_.value();
        }

        ModelCapabilities() = default;

        ModelCapabilities(
            const std::vector<int64_t>& atomic_types_,
            double interaction_range_,
            const std::string& length_unit_,
            const std::vector<Device>& supported_devices_,
            DType dtype_,
            const std::vector<Quantity>& outputs_ = {}
        ) {
            this->atomic_types(atomic_types_);
            this->interaction_range(interaction_range_);
            this->length_unit(length_unit_);
            this->supported_devices(supported_devices_);
            this->dtype(dtype_);
            this->outputs(outputs_);
        }
    };

    inline void to_json(nlohmann::json& j, const ModelCapabilities::DType& dtype) {
        switch (dtype) {
            case ModelCapabilities::DType::Float32:
                j = "float32";
                break;
            case ModelCapabilities::DType::Float64:
                j = "float64";
                break;
        }
    }

    inline void from_json(const nlohmann::json& j, ModelCapabilities::DType& dtype) {
        if (!j.is_string()) {
            throw metatomic::Error("dtype in JSON for ModelCapabilities must be a string");
        }

        std::string s = j.get<std::string>();
        if (s == "float32") {
            dtype = ModelCapabilities::DType::Float32;
        } else if (s == "float64") {
            dtype = ModelCapabilities::DType::Float64;
        } else {
            throw metatomic::Error(
                "invalid string for dtype in JSON for ModelCapabilities, expected 'float32' or 'float64'"
            );
        }
    }

    inline void to_json(nlohmann::json& j, const ModelCapabilities::Device& device) {
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

    inline void from_json(const nlohmann::json& j, ModelCapabilities::Device& device) {
        if (!j.is_string()) {
            throw metatomic::Error("device in JSON for ModelCapabilities must be a string");
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
            throw metatomic::Error(
                "invalid string for device in JSON for ModelCapabilities, expected 'cpu', 'cuda', 'rocm', or 'metal'"
            );
        }
    }

    inline void to_json(nlohmann::json& j, const SampleKind& kind) {
        switch (kind) {
            case SampleKind::Atom:
                j = "atom";
                break;
            case SampleKind::System:
                j = "system";
                break;
            case SampleKind::AtomPair:
                j = "atom_pair";
                break;
        }
    }

    inline void from_json(const nlohmann::json& j, SampleKind& kind) {
        if (!j.is_string()) {
            throw metatomic::Error("'sample_kind' in JSON for Quantity must be a string");
        }

        std::string s = j.get<std::string>();
        if (s == "atom") {
            kind = SampleKind::Atom;
        } else if (s == "system") {
            kind = SampleKind::System;
        } else if (s == "atom_pair") {
            kind = SampleKind::AtomPair;
        } else {
            throw metatomic::Error(
                "'sample_kind' in JSON for Quantity must be 'atom', 'system' or 'atom_pair', got '" + s + "'"
            );
        }
    }

    inline void to_json(nlohmann::json& j, const Gradients& gradients) {
        switch (gradients) {
            case Gradients::Positions:
                j = "positions";
                break;
            case Gradients::Strain:
                j = "strain";
                break;
        }
    }

    inline void from_json(const nlohmann::json& j, Gradients& gradients) {
        if (!j.is_string()) {
            throw metatomic::Error("'gradients' in JSON for Quantity must be a string");
        }

        std::string s = j.get<std::string>();
        if (s == "positions") {
            gradients = Gradients::Positions;
        } else if (s == "strain") {
            gradients = Gradients::Strain;
        } else {
            throw metatomic::Error(
                "'gradients' in JSON for Quantity must be 'positions' or 'strain', got '" + s + "'"
            );
        }
    }

    inline void to_json(nlohmann::json& j, const Quantity& q) {
        j = nlohmann::json{
            {"type", "metatomic_quantity"},
            {"name", q.name()},
            {"unit", q.unit()},
            {"gradients", q.gradients()},
            {"sample_kind", q.sample_kind()}
        };

        if (!q.description().empty()) {
            j["description"] = q.description();
        }
    }

    inline void from_json(const nlohmann::json& j, Quantity& q) {
        if (!j.is_object()) {
            throw metatomic::Error("invalid JSON data for Quantity, expected an object");
        }

        if (!j.contains("type") || !j["type"].is_string() || j["type"].get<std::string>() != "metatomic_quantity") {
            throw metatomic::Error("'type' in JSON for Quantity must be 'metatomic_quantity'");
        }

        if (!j.contains("name") || !j["name"].is_string()) {
            throw metatomic::Error("'name' in JSON for Quantity must be a string");
        }
        std::string name = j["name"].get<std::string>();

        if (!j.contains("unit") || !j["unit"].is_string()) {
            throw metatomic::Error("'unit' in JSON for Quantity must be a string");
        }
        std::string unit = j["unit"].get<std::string>();

        std::string description;
        if (j.contains("description")) {
            if (!j["description"].is_string()) {
                throw metatomic::Error("'description' in JSON for Quantity must be a string");
            }
            description = j["description"].get<std::string>();
        }

        if (!j.contains("gradients") || !j["gradients"].is_array()) {
            throw metatomic::Error("'gradients' in JSON for Quantity must be an array");
        }
        std::vector<Gradients> gradients;
        for (const auto& gradient : j["gradients"]) {
            gradients.push_back(gradient.get<Gradients>());
        }

        if (!j.contains("sample_kind") || !j["sample_kind"].is_string()) {
            throw metatomic::Error("'sample_kind' in JSON for Quantity must be a string");
        }
        auto sample_kind = j["sample_kind"].get<SampleKind>();

        q = Quantity(name, unit, sample_kind, description, gradients);
    }

    inline void to_json(nlohmann::json& j, const ModelCapabilities& c) {
        j = nlohmann::json{
            {"type", "metatomic_model_capabilities"},
            {"outputs", c.outputs()},
            {"atomic_types", c.atomic_types()},
            {"interaction_range", c.interaction_range()},
            {"length_unit", c.length_unit()},
            {"supported_devices", c.supported_devices()},
            {"dtype", c.dtype()}
        };
    }

    inline void from_json(const nlohmann::json& j, ModelCapabilities& c) {
        if (!j.is_object()) {
            throw metatomic::Error("invalid JSON data for ModelCapabilities, expected an object");
        }

        if (!j.contains("type") || !j["type"].is_string() || j["type"].get<std::string>() != "metatomic_model_capabilities") {
            throw metatomic::Error("'type' in JSON for ModelCapabilities must be 'metatomic_model_capabilities'");
        }

        if (!j.contains("outputs") || !j["outputs"].is_array()) {
            throw metatomic::Error("'outputs' in JSON for ModelCapabilities must be an array");
        }
        std::vector<Quantity> outputs;
        for (const auto& output : j["outputs"]) {
            outputs.push_back(output.get<Quantity>());
        }

        if (!j.contains("atomic_types") || !j["atomic_types"].is_array()) {
            throw metatomic::Error("'atomic_types' in JSON for ModelCapabilities must be an array");
        }
        std::vector<int64_t> atomic_types;
        for (const auto& atomic_type : j["atomic_types"]) {
            if (!atomic_type.is_number_integer()) {
                throw metatomic::Error("'atomic_types' in JSON for ModelCapabilities must be an array of integers");
            }
            atomic_types.push_back(atomic_type.get<int64_t>());
        }

        if (!j.contains("interaction_range") || !j["interaction_range"].is_number()) {
            throw metatomic::Error("'interaction_range' in JSON for ModelCapabilities must be a number");
        }
        double interaction_range = j["interaction_range"].get<double>();
        if (interaction_range < 0.0) {
            throw metatomic::Error("'interaction_range' in JSON for ModelCapabilities must be non-negative");
        }

        if (!j.contains("length_unit") || !j["length_unit"].is_string()) {
            throw metatomic::Error("'length_unit' in JSON for ModelCapabilities must be a string");
        }
        std::string length_unit = j["length_unit"].get<std::string>();

        // Validate that `length_unit` has the dimension of length by asking the
        // C API for a conversion factor to meters. The call only succeeds when
        // the dimensions match; otherwise `check_status` throws with the C API's
        // dimension-mismatch message.
        double conversion_factor = 0.0;
        auto status = mta_unit_conversion_factor(length_unit.c_str(), "m", &conversion_factor);
        metatomic::details::check_status(status);

        if (!j.contains("supported_devices") || !j["supported_devices"].is_array()) {
            throw metatomic::Error("'supported_devices' in JSON for ModelCapabilities must be an array");
        }
        std::vector<ModelCapabilities::Device> supported_devices;
        for (const auto& device : j["supported_devices"]) {
            supported_devices.push_back(device.get<ModelCapabilities::Device>());
        }

        if (!j.contains("dtype") || !j["dtype"].is_string()) {
            throw metatomic::Error("dtype in JSON for ModelCapabilities must be a string");
        }
        auto dtype = j["dtype"].get<ModelCapabilities::DType>();

        c = ModelCapabilities(atomic_types, interaction_range, length_unit, supported_devices, dtype, outputs);
    }

} // namespace metatomic
