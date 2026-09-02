#pragma once

#include <vector>
#include <sstream>
#include <map>
#include <string>
#include <optional>
#include <algorithm>
#include <utility> // std::move
#include <cmath> // std::isfinite
#include <cstring> // std::memcpy
#include <cstdint> // std::uint64_t, std::int64_t
#include <cctype> // std::isxdigit

#include <metatomic/errors.hpp>
#include <nlohmann/json.hpp>

namespace metatomic {
    namespace detail {

    inline std::vector<std::string> read_string_array(
        const nlohmann::json& j, const std::string& key, const char* context
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
    class PairListOptions final {
    private:
        /// Cutoff radius for this pair list in the length unit of the model
        double cutoff_;
        /// Whether the list is a full list (contains both the pair `i -> j` and `j -> i`)
        /// or a half list (contains only `i -> j`)
        bool full_list_;
        /// Whether the list guarantees that only atoms within the cutoff are
        /// included (strict) or may also include pairs slightly beyond the cutoff
        /// (non-strict)
        bool strict_ = true;
        /// List of strings describing who requested this pair list
        std::vector<std::string> requestors_;

        /// Validate that `value` is a finite positive number, throwing
        /// `metatomic::Error` otherwise. Used by both the class setters and the
        /// `Builder` setters so validation lives in a single place.
        static void validate_cutoff(double value) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw metatomic::Error("cutoff must be a finite positive number");
            }
        }

        /// Add `requestor` to `requestors_`, ignoring empty strings and
        /// duplicates, keeping first-seen order. Shared by the class and the
        /// `Builder` to avoid duplicating the deduplication logic.
        static void add_requestor_to(std::vector<std::string>& requestors, const std::string& requestor) {
            if (!requestor.empty() && std::find(requestors.begin(), requestors.end(), requestor) == requestors.end()) {
                requestors.push_back(requestor);
            }
        }

        /// Private constructor — only `Builder::build()` calls this. The object
        /// is always fully initialized after construction.
        PairListOptions(
            double cutoff,
            bool full_list,
            bool strict,
            std::vector<std::string> requestors
        ) : cutoff_(cutoff), full_list_(full_list),
            strict_(strict), requestors_(std::move(requestors)) {}

    public:
        /// Set the cutoff radius for this pair list.
        ///
        /// @throw metatomic::Error if the value is not a finite positive number.
        PairListOptions& cutoff(double value) {
            validate_cutoff(value);
            cutoff_ = value;
            return *this;
        }

        /// Get the cutoff radius for this pair list.
        double cutoff() const {
            return cutoff_;
        }

        /// Set whether this pair list is a full list.
        PairListOptions& full_list(bool value) {
            full_list_ = value;
            return *this;
        }

        /// Get whether this pair list is a full list.
        bool full_list() const {
            return full_list_;
        }

        /// Set whether this pair list is strict.
        PairListOptions& strict(bool value) {
            strict_ = value;
            return *this;
        }

        /// Get whether this pair list is strict.
        bool strict() const {
            return strict_;
        }

        /// Set the list of requestors for this pair list.
        ///
        /// Empty strings and duplicates are ignored, keeping first-seen order.
        PairListOptions& requestors(std::vector<std::string> value) {
            requestors_.clear();
            for (const auto& requestor : value) {
                add_requestor_to(requestors_, requestor);
            }
            return *this;
        }

        /// Get the list of requestors for this pair list.
        const std::vector<std::string>& requestors() const {
            return requestors_;
        }

        /// Add a requestor to the list.
        ///
        /// Empty strings and duplicates are ignored, keeping first-seen order.
        PairListOptions& add_requestor(const std::string& requestor) {
            add_requestor_to(requestors_, requestor);
            return *this;
        }

        /// Clear the list of requestors.
        PairListOptions& clear_requestors() {
            requestors_.clear();
            return *this;
        }

        /// Check if two `PairListOptions` are equal.
        ///
        /// The list of requestors is ignored when checking for equality.
        bool operator==(const PairListOptions& other) const {
            return cutoff_ == other.cutoff_ &&
                   full_list_ == other.full_list_ &&
                   strict_ == other.strict_;
        }

        /// Check if two `PairListOptions` are different.
        ///
        /// The list of requestors is ignored when checking for equality.
        bool operator!=(const PairListOptions& other) const {
            return !(*this == other);
        }

        /// Builder for `PairListOptions`.
        ///
        /// Use `PairListOptions::builder()` to create a new builder, set the
        /// required fields via the fluent setters, and call `build()` to obtain
        /// a fully-initialized `PairListOptions`.
        ///
        /// `cutoff` and `full_list` are required and must be set before calling
        /// `build()`, otherwise `build()` throws `metatomic::Error`. `strict`
        /// defaults to `true` and `requestors` defaults to an empty list.
        class Builder {
        private:
            std::optional<double> cutoff_;
            std::optional<bool> full_list_;
            bool strict_ = true;
            std::vector<std::string> requestors_;

        public:
            /// Set the cutoff radius for this pair list.
            ///
            /// @throw metatomic::Error if the value is not a finite positive number.
            Builder& cutoff(double value) {
                PairListOptions::validate_cutoff(value);
                cutoff_ = value;
                return *this;
            }

            /// Set whether this pair list is a full list.
            Builder& full_list(bool value) {
                full_list_ = value;
                return *this;
            }

            /// Set whether this pair list is strict. Defaults to `true`.
            Builder& strict(bool value) {
                strict_ = value;
                return *this;
            }

            /// Set the list of requestors for this pair list.
            ///
            /// Empty strings and duplicates are ignored, keeping first-seen order.
            Builder& requestors(std::vector<std::string> value) {
                requestors_.clear();
                for (const auto& requestor : value) {
                    PairListOptions::add_requestor_to(requestors_, requestor);
                }
                return *this;
            }

            /// Add a requestor to the list.
            ///
            /// Empty strings and duplicates are ignored, keeping first-seen order.
            Builder& add_requestor(const std::string& requestor) {
                PairListOptions::add_requestor_to(requestors_, requestor);
                return *this;
            }

            /// Clear the list of requestors.
            Builder& clear_requestors() {
                requestors_.clear();
                return *this;
            }

            /// Build a fully-initialized `PairListOptions`.
            ///
            /// @throw metatomic::Error if `cutoff` or `full_list` has not been set.
            ///
            /// This moves the builder's fields; calling `build()` a second time
            /// produces an object with moved-from values. Builders are intended
            /// as one-shot temporaries.
            [[nodiscard]] PairListOptions build() {
                if (!cutoff_.has_value()) {
                    throw metatomic::Error("cutoff must be set before building PairListOptions");
                }
                if (!full_list_.has_value()) {
                    throw metatomic::Error("full_list must be set before building PairListOptions");
                }
                return PairListOptions(
                    cutoff_.value(),
                    full_list_.value(),
                    strict_,
                    std::move(requestors_)
                );
            }
        };

        /// Create a new `Builder` for `PairListOptions`.
        [[nodiscard]] static Builder builder() {
            return Builder{};
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

    inline PairListOptions from_json(
        const nlohmann::json& j, nlohmann::detail::identity_tag<PairListOptions>
    ) {
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

        auto p = PairListOptions::builder()
            .cutoff(cutoff)
            .full_list(full_list)
            .strict(strict);

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

        return p.build();
    }

    // Forward declarations
    // The ModelMetadata::print function uses to_json
    class ModelMetadata;
    void to_json(nlohmann::json&, const ModelMetadata&);

    class ModelMetadata final {
    public:
        /// References for a model, divided into three categories: references about
        /// the model as a whole, references about the architecture of the model,
        /// and references about the implementation of the model.
        class References final {
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

            /// Private constructor — only `Builder::build()` calls this. The
            /// object is always fully initialized after construction.
            References(
                std::vector<std::string> model,
                std::vector<std::string> architecture,
                std::vector<std::string> implementation
            ) : model_(std::move(model)), architecture_(std::move(architecture)),
                implementation_(std::move(implementation)) {}

        public:
            /// Set the references about the model as a whole.
            References& model(std::vector<std::string> value) {
                model_ = std::move(value);
                return *this;
            }

            /// Get the references about the model as a whole.
            const std::vector<std::string>& model() const {
                return model_;
            }

            /// Add a reference about the model as a whole.
            References& add_model(const std::string& reference) {
                model_.push_back(reference);
                return *this;
            }

            /// Clear the references about the model as a whole.
            References& clear_model() {
                model_.clear();
                return *this;
            }

            /// Set the references about the architecture of the model.
            References& architecture(std::vector<std::string> value) {
                architecture_ = std::move(value);
                return *this;
            }

            /// Get the references about the architecture of the model.
            const std::vector<std::string>& architecture() const {
                return architecture_;
            }

            /// Add a reference about the architecture of the model.
            References& add_architecture(const std::string& reference) {
                architecture_.push_back(reference);
                return *this;
            }

            /// Clear the references about the architecture of the model.
            References& clear_architecture() {
                architecture_.clear();
                return *this;
            }

            /// Set the references about the implementation of the model.
            References& implementation(std::vector<std::string> value) {
                implementation_ = std::move(value);
                return *this;
            }

            /// Get the references about the implementation of the model.
            const std::vector<std::string>& implementation() const {
                return implementation_;
            }

            /// Add a reference about the implementation of the model.
            References& add_implementation(const std::string& reference) {
                implementation_.push_back(reference);
                return *this;
            }

            /// Clear the references about the implementation of the model.
            References& clear_implementation() {
                implementation_.clear();
                return *this;
            }

            /// Builder for `References`.
            ///
            /// Use `References::builder()` to create a new builder, set the
            /// fields via the fluent setters, and call `build()` to obtain a
            /// fully-initialized `References`. All fields default to empty lists,
            /// so `build()` always succeeds.
            class Builder {
            private:
                std::vector<std::string> model_;
                std::vector<std::string> architecture_;
                std::vector<std::string> implementation_;

            public:
                /// Set the references about the model as a whole.
                Builder& model(std::vector<std::string> value) {
                    model_ = std::move(value);
                    return *this;
                }

                /// Add a reference about the model as a whole.
                Builder& add_model(const std::string& reference) {
                    model_.push_back(reference);
                    return *this;
                }

                /// Clear the references about the model as a whole.
                Builder& clear_model() {
                    model_.clear();
                    return *this;
                }

                /// Set the references about the architecture of the model.
                Builder& architecture(std::vector<std::string> value) {
                    architecture_ = std::move(value);
                    return *this;
                }

                /// Add a reference about the architecture of the model.
                Builder& add_architecture(const std::string& reference) {
                    architecture_.push_back(reference);
                    return *this;
                }

                /// Clear the references about the architecture of the model.
                Builder& clear_architecture() {
                    architecture_.clear();
                    return *this;
                }

                /// Set the references about the implementation of the model.
                Builder& implementation(std::vector<std::string> value) {
                    implementation_ = std::move(value);
                    return *this;
                }

                /// Add a reference about the implementation of the model.
                Builder& add_implementation(const std::string& reference) {
                    implementation_.push_back(reference);
                    return *this;
                }

                /// Clear the references about the implementation of the model.
                Builder& clear_implementation() {
                    implementation_.clear();
                    return *this;
                }

                /// Build a fully-initialized `References`.
                ///
                /// This moves the builder's fields; calling `build()` a second
                /// time produces an object with moved-from values. Builders are
                /// intended as one-shot temporaries.
                [[nodiscard]] References build() {
                    return References(
                        std::move(model_),
                        std::move(architecture_),
                        std::move(implementation_)
                    );
                }
            };

            /// Create a new `Builder` for `References`.
            [[nodiscard]] static Builder builder() {
                return Builder{};
            }
        };

    private:
        std::string name_;
        std::vector<std::string> authors_;
        std::string description_;
        References references_;
        // BTreeMap in Rust is an ordered map
        std::map<std::string, std::string> extra_;

        /// Add a reference to the given `section` of `refs`. Shared by the
        /// class and the `Builder` to avoid duplicating the section dispatch.
        ///
        /// @param refs the `References` to mutate
        /// @param section reference section, one of "model", "architecture", or
        ///     "implementation"
        /// @param reference the reference to add
        /// @throw metatomic::Error if `section` is not one of the allowed values
        static void add_reference_to(References& refs, const std::string& section, const std::string& reference) {
            if (section == "model") {
                refs.add_model(reference);
            } else if (section == "architecture") {
                refs.add_architecture(reference);
            } else if (section == "implementation") {
                refs.add_implementation(reference);
            } else {
                throw metatomic::Error(
                    "reference section must be 'model', 'architecture', or 'implementation', got '" + section + "'"
                );
            }
        }

        /// Clear a single `section` of `refs`. Shared by the class and the
        /// `Builder` to avoid duplicating the section dispatch.
        ///
        /// @param refs the `References` to mutate
        /// @param section reference section, one of "model", "architecture", or
        ///     "implementation"
        /// @throw metatomic::Error if `section` is not one of the allowed values
        static void clear_reference_from(References& refs, const std::string& section) {
            if (section == "model") {
                refs.clear_model();
            } else if (section == "architecture") {
                refs.clear_architecture();
            } else if (section == "implementation") {
                refs.clear_implementation();
            } else {
                throw metatomic::Error(
                    "reference section must be 'model', 'architecture', or 'implementation', got '" + section + "'"
                );
            }
        }

        /// Private constructor — only `Builder::build()` calls this. The object
        /// is always fully initialized after construction.
        ModelMetadata(
            std::string name,
            std::vector<std::string> authors,
            std::string description,
            References references,
            std::map<std::string, std::string> extra
        ) : name_(std::move(name)), authors_(std::move(authors)),
            description_(std::move(description)), references_(std::move(references)),
            extra_(std::move(extra)) {}

    public:
        /// Set the name of the model.
        ModelMetadata& name(std::string value) {
            name_ = std::move(value);
            return *this;
        }

        /// Get the name of the model.
        const std::string& name() const {
            return name_;
        }

        /// Set the list of authors of the model.
        ModelMetadata& authors(std::vector<std::string> value) {
            authors_ = std::move(value);
            return *this;
        }

        /// Get the list of authors of the model.
        const std::vector<std::string>& authors() const {
            return authors_;
        }

        /// Add an author to the list of authors.
        ModelMetadata& add_author(const std::string& author) {
            authors_.push_back(author);
            return *this;
        }

        /// Clear the list of authors.
        ModelMetadata& clear_authors() {
            authors_.clear();
            return *this;
        }

        /// Set the description of the model.
        ModelMetadata& description(std::string value) {
            description_ = std::move(value);
            return *this;
        }

        /// Get the description of the model.
        const std::string& description() const {
            return description_;
        }

        /// Set the references for the model.
        ModelMetadata& references(References value) {
            references_ = std::move(value);
            return *this;
        }

        /// Get the references for the model.
        const References& references() const {
            return references_;
        }

        /// Add a reference to the given section.
        ///
        /// @param section reference section, one of "model", "architecture", or
        ///     "implementation"
        /// @param reference the reference to add
        /// @throw metatomic::Error if `section` is not one of the allowed values
        ModelMetadata& add_reference(const std::string& section, const std::string& reference) {
            add_reference_to(references_, section, reference);
            return *this;
        }

        /// Clear a single reference section.
        ///
        /// @param section reference section, one of "model", "architecture", or
        ///     "implementation"
        /// @throw metatomic::Error if `section` is not one of the allowed values
        ModelMetadata& clear_reference(const std::string& section) {
            clear_reference_from(references_, section);
            return *this;
        }

        /// Clear all references for the model.
        ModelMetadata& clear_references() {
            references_.clear_model();
            references_.clear_architecture();
            references_.clear_implementation();
            return *this;
        }

        /// Set the extra metadata for the model.
        ModelMetadata& extra(std::map<std::string, std::string> value) {
            extra_ = std::move(value);
            return *this;
        }

        /// Get the extra metadata for the model.
        const std::map<std::string, std::string>& extra() const {
            return extra_;
        }

        /// Add a key/value pair to the extra metadata.
        ///
        /// If the key already exists, its value is overwritten.
        ///
        /// @param key key for the extra metadata entry
        /// @param value value for the extra metadata entry
        ModelMetadata& add_extra(const std::string& key, const std::string& value) {
            extra_[key] = value;
            return *this;
        }

        /// Clear the extra metadata.
        ModelMetadata& clear_extra() {
            extra_.clear();
            return *this;
        }

        /// Builder for `ModelMetadata`.
        ///
        /// Use `ModelMetadata::builder()` to create a new builder, set the
        /// fields via the fluent setters, and call `build()` to obtain a
        /// fully-initialized `ModelMetadata`. All fields have defaults, so
        /// `build()` always succeeds.
        class Builder {
        private:
            std::string name_;
            std::vector<std::string> authors_;
            std::string description_;
            References references_;
            std::map<std::string, std::string> extra_;

        public:
            /// Default constructor — initializes `references` to an empty
            /// `References` since `References` is not default-constructible.
            Builder() : references_(References::builder().build()) {}

            /// Set the name of the model.
            Builder& name(std::string value) {
                name_ = std::move(value);
                return *this;
            }

            /// Get the name of the model.
            const std::string& name() const {
                return name_;
            }

            /// Set the list of authors of the model.
            Builder& authors(std::vector<std::string> value) {
                authors_ = std::move(value);
                return *this;
            }

            /// Get the list of authors of the model.
            const std::vector<std::string>& authors() const {
                return authors_;
            }

            /// Add an author to the list of authors.
            Builder& add_author(const std::string& author) {
                authors_.push_back(author);
                return *this;
            }

            /// Clear the list of authors.
            Builder& clear_authors() {
                authors_.clear();
                return *this;
            }

            /// Set the description of the model.
            Builder& description(std::string value) {
                description_ = std::move(value);
                return *this;
            }

            /// Get the description of the model.
            const std::string& description() const {
                return description_;
            }

            /// Set the references for the model.
            Builder& references(References value) {
                references_ = std::move(value);
                return *this;
            }

            /// Get the references for the model.
            const References& references() const {
                return references_;
            }

            /// Add a reference to the given section.
            ///
            /// @param section reference section, one of "model", "architecture", or
            ///     "implementation"
            /// @param reference the reference to add
            /// @throw metatomic::Error if `section` is not one of the allowed values
            Builder& add_reference(const std::string& section, const std::string& reference) {
                ModelMetadata::add_reference_to(references_, section, reference);
                return *this;
            }

            /// Clear a single reference section.
            ///
            /// @param section reference section, one of "model", "architecture", or
            ///     "implementation"
            /// @throw metatomic::Error if `section` is not one of the allowed values
            Builder& clear_reference(const std::string& section) {
                ModelMetadata::clear_reference_from(references_, section);
                return *this;
            }

            /// Clear all references for the model.
            Builder& clear_references() {
                references_.clear_model();
                references_.clear_architecture();
                references_.clear_implementation();
                return *this;
            }

            /// Set the extra metadata for the model.
            Builder& extra(std::map<std::string, std::string> value) {
                extra_ = std::move(value);
                return *this;
            }

            /// Get the extra metadata for the model.
            const std::map<std::string, std::string>& extra() const {
                return extra_;
            }

            /// Add a key/value pair to the extra metadata.
            ///
            /// If the key already exists, its value is overwritten.
            ///
            /// @param key key for the extra metadata entry
            /// @param value value for the extra metadata entry
            Builder& add_extra(const std::string& key, const std::string& value) {
                extra_[key] = value;
                return *this;
            }

            /// Clear the extra metadata.
            Builder& clear_extra() {
                extra_.clear();
                return *this;
            }

            /// Build a fully-initialized `ModelMetadata`.
            ///
            /// This moves the builder's fields; calling `build()` a second
            /// time produces an object with moved-from values. Builders are
            /// intended as one-shot temporaries.
            [[nodiscard]] ModelMetadata build() {
                return ModelMetadata(
                    std::move(name_),
                    std::move(authors_),
                    std::move(description_),
                    std::move(references_),
                    std::move(extra_)
                );
            }
        };

        /// Create a new `Builder` for `ModelMetadata`.
        [[nodiscard]] static Builder builder() {
            return Builder{};
        }

        /// Print the metadata as a human-readable string.
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

    inline ModelMetadata::References from_json(
        const nlohmann::json& j, nlohmann::detail::identity_tag<ModelMetadata::References>
    ) {
        if (!j.is_object()) {
            throw metatomic::Error("invalid JSON data for references in ModelMetadata, expected an object");
        }

        return ModelMetadata::References::builder()
            .model(detail::read_string_array(j, "model", "references of ModelMetadata"))
            .architecture(detail::read_string_array(j, "architecture", "references of ModelMetadata"))
            .implementation(detail::read_string_array(j, "implementation", "references of ModelMetadata"))
            .build();
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

    inline ModelMetadata from_json(
        const nlohmann::json& j, nlohmann::detail::identity_tag<ModelMetadata>
    ) {
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

        return ModelMetadata::builder()
            .name(std::move(name))
            .authors(std::move(authors))
            .description(std::move(description))
            .references(std::move(references))
            .extra(std::move(extra))
            .build();
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
    class Quantity final {
    private:
        /// Name of the quantity, this can be a standard name from
        /// https://docs.metatensor.org/metatomic/latest/quantities/index.html, or
        /// a custom name of the form `<namespace>::<name>[/<variant>]`
        std::string name_;
        /// Unit of the quantity
        std::string unit_;
        /// The kind of samples this quantity is associated with
        SampleKind sample_kind_;
        /// Description of the quantity, used to provide more details about the
        /// quantity, especially when a model defines multiple variants of the same
        /// quantity. An empty string is treated as no description.
        std::string description_;
        /// List of explicit gradients for this quantity
        std::vector<Gradients> gradients_;

        /// Private constructor — only `Builder::build()` calls this. The object
        /// is always fully initialized after construction.
        Quantity(
            std::string name,
            std::string unit,
            SampleKind sample_kind,
            std::string description,
            std::vector<Gradients> gradients
        ) : name_(std::move(name)), unit_(std::move(unit)),
            sample_kind_(sample_kind), description_(std::move(description)),
            gradients_(std::move(gradients)) {}

    public:
        /// Set the name of this quantity.
        Quantity& name(std::string value) {
            name_ = std::move(value);
            return *this;
        }

        /// Get the name of this quantity.
        const std::string& name() const {
            return name_;
        }

        /// Set the unit of this quantity.
        Quantity& unit(std::string value) {
            unit_ = std::move(value);
            return *this;
        }

        /// Get the unit of this quantity.
        const std::string& unit() const {
            return unit_;
        }

        /// Set the description of this quantity.
        Quantity& description(std::string value) {
            description_ = std::move(value);
            return *this;
        }

        /// Get the description of this quantity.
        const std::string& description() const {
            return description_;
        }

        /// Set the list of explicit gradients for this quantity.
        Quantity& gradients(std::vector<Gradients> value) {
            gradients_ = std::move(value);
            return *this;
        }

        /// Get the list of explicit gradients for this quantity.
        const std::vector<Gradients>& gradients() const {
            return gradients_;
        }

        /// Add an explicit gradient to this quantity.
        Quantity& add_gradient(Gradients gradient) {
            gradients_.push_back(gradient);
            return *this;
        }

        /// Clear the list of explicit gradients for this quantity.
        Quantity& clear_gradients() {
            gradients_.clear();
            return *this;
        }

        /// Set the kind of samples this quantity is associated with.
        Quantity& sample_kind(SampleKind value) {
            sample_kind_ = value;
            return *this;
        }

        /// Get the kind of samples this quantity is associated with.
        SampleKind sample_kind() const {
            return sample_kind_;
        }

        /// Builder for `Quantity`.
        ///
        /// Use `Quantity::builder()` to create a new builder, set the required
        /// fields via the fluent setters, and call `build()` to obtain a fully
        /// initialized `Quantity`.
        ///
        /// `name`, `unit`, and `sample_kind` are required and must be set before
        /// calling `build()`, otherwise `build()` throws `metatomic::Error`.
        /// `description` defaults to an empty string and `gradients` defaults to
        /// an empty list.
        class Builder {
        private:
            std::optional<std::string> name_;
            std::optional<std::string> unit_;
            std::string description_;
            std::vector<Gradients> gradients_;
            std::optional<SampleKind> sample_kind_;

        public:
            /// Set the name of this quantity.
            Builder& name(std::string value) {
                name_ = std::move(value);
                return *this;
            }

            /// Set the unit of this quantity.
            Builder& unit(std::string value) {
                unit_ = std::move(value);
                return *this;
            }

            /// Set the description of this quantity.
            Builder& description(std::string value) {
                description_ = std::move(value);
                return *this;
            }

            /// Set the list of explicit gradients for this quantity.
            Builder& gradients(std::vector<Gradients> value) {
                gradients_ = std::move(value);
                return *this;
            }

            /// Add an explicit gradient to this quantity.
            Builder& add_gradient(Gradients gradient) {
                gradients_.push_back(gradient);
                return *this;
            }

            /// Clear the list of explicit gradients for this quantity.
            Builder& clear_gradients() {
                gradients_.clear();
                return *this;
            }

            /// Set the kind of samples this quantity is associated with.
            Builder& sample_kind(SampleKind value) {
                sample_kind_ = value;
                return *this;
            }

            /// Build a fully-initialized `Quantity`.
            ///
            /// @throw metatomic::Error if `name`, `unit`, or `sample_kind` has
            ///     not been set.
            ///
            /// This moves the builder's fields; calling `build()` a second time
            /// produces an object with moved-from values. Builders are intended
            /// as one-shot temporaries.
            [[nodiscard]] Quantity build() {
                if (!name_.has_value()) {
                    throw metatomic::Error("name must be set before building Quantity");
                }
                if (!unit_.has_value()) {
                    throw metatomic::Error("unit must be set before building Quantity");
                }
                if (!sample_kind_.has_value()) {
                    throw metatomic::Error("sample_kind must be set before building Quantity");
                }
                return Quantity(
                    std::move(name_).value(),
                    std::move(unit_).value(),
                    sample_kind_.value(),
                    std::move(description_),
                    std::move(gradients_)
                );
            }
        };

        /// Create a new `Builder` for `Quantity`.
        [[nodiscard]] static Builder builder() {
            return Builder{};
        }
    };

    /// Capabilities of a model: which outputs it provides, which atoms it
    /// supports, etc.
    class ModelCapabilities final {
    public:
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

        using SampleKind = metatomic::SampleKind;     ///< Alias for top-level `metatomic::SampleKind`
        using Gradients = metatomic::Gradients;       ///< Alias for top-level `metatomic::Gradients`
        using Quantity = metatomic::Quantity;         ///< Alias for top-level `metatomic::Quantity`

    private:
        /// The atomic types this model supports. The meaning of the integers in
        /// this list is up to the model, and is not required to be the atomic
        /// numbers.
        std::vector<int64_t> atomic_types_;
        /// The interaction range of the model (in the length unit of the model),
        /// i.e. the maximum distance between two atoms for which the model's output
        /// can depend on their relative position.
        double interaction_range_;
        /// The length unit of the model, e.g. "angstrom" or "nanometer". This is
        /// used to interpret the `interaction_range` and convert the inputs.
        std::string length_unit_;
        /// The devices on which the model can run, e.g. `["cpu", "cuda"]`.
        std::vector<Device> supported_devices_;
        /// The data type of the model, used for all inputs and outputs.
        DType dtype_;
        /// The outputs this model can provide
        std::vector<Quantity> outputs_;

        /// Validate that `value` is non-negative, throwing `metatomic::Error`
        /// otherwise. Used by both the class setters and the `Builder` setters so
        /// validation lives in a single place.
        static void validate_interaction_range(double value) {
            if (value < 0.0) {
                throw metatomic::Error("interaction_range must be non-negative");
            }
        }

        /// Private constructor — only `Builder::build()` calls this. The object
        /// is always fully initialized after construction.
        ModelCapabilities(
            std::vector<int64_t> atomic_types,
            double interaction_range,
            std::string length_unit,
            std::vector<Device> supported_devices,
            DType dtype,
            std::vector<Quantity> outputs
        ) : atomic_types_(std::move(atomic_types)),
            interaction_range_(interaction_range),
            length_unit_(std::move(length_unit)),
            supported_devices_(std::move(supported_devices)),
            dtype_(dtype), outputs_(std::move(outputs)) {}

    public:
        /// Set the list of outputs this model can provide.
        ModelCapabilities& outputs(std::vector<Quantity> value) {
            outputs_ = std::move(value);
            return *this;
        }

        /// Get the list of outputs this model can provide.
        const std::vector<Quantity>& outputs() const {
            return outputs_;
        }

        /// Add an output to the list of outputs this model can provide.
        ModelCapabilities& add_output(const Quantity& output) {
            outputs_.push_back(output);
            return *this;
        }

        /// Clear the list of outputs this model can provide.
        ModelCapabilities& clear_outputs() {
            outputs_.clear();
            return *this;
        }

        /// Set the atomic types this model supports.
        ModelCapabilities& atomic_types(std::vector<int64_t> value) {
            atomic_types_ = std::move(value);
            return *this;
        }

        /// Get the atomic types this model supports.
        const std::vector<int64_t>& atomic_types() const {
            return atomic_types_;
        }

        /// Add an atomic type to the list of atomic types this model supports.
        ModelCapabilities& add_atomic_type(int64_t atomic_type) {
            atomic_types_.push_back(atomic_type);
            return *this;
        }

        /// Clear the list of atomic types this model supports.
        ModelCapabilities& clear_atomic_types() {
            atomic_types_.clear();
            return *this;
        }

        /// Set the interaction range of the model.
        ///
        /// @throw metatomic::Error if the value is negative.
        ModelCapabilities& interaction_range(double value) {
            validate_interaction_range(value);
            interaction_range_ = value;
            return *this;
        }

        /// Get the interaction range of the model.
        double interaction_range() const {
            return interaction_range_;
        }

        /// Set the length unit of the model.
        ModelCapabilities& length_unit(std::string value) {
            length_unit_ = std::move(value);
            return *this;
        }

        /// Get the length unit of the model.
        const std::string& length_unit() const {
            return length_unit_;
        }

        /// Set the devices on which this model can run.
        ModelCapabilities& supported_devices(std::vector<Device> value) {
            supported_devices_ = std::move(value);
            return *this;
        }

        /// Get the devices on which this model can run.
        const std::vector<Device>& supported_devices() const {
            return supported_devices_;
        }

        /// Add a device to the list of devices on which this model can run.
        ModelCapabilities& add_supported_device(Device device) {
            supported_devices_.push_back(device);
            return *this;
        }

        /// Clear the list of devices on which this model can run.
        ModelCapabilities& clear_supported_devices() {
            supported_devices_.clear();
            return *this;
        }

        /// Set the data type of the model.
        ModelCapabilities& dtype(DType value) {
            dtype_ = value;
            return *this;
        }

        /// Get the data type of the model.
        DType dtype() const {
            return dtype_;
        }

        /// Builder for `ModelCapabilities`.
        ///
        /// Use `ModelCapabilities::builder()` to create a new builder, set the
        /// required fields via the fluent setters, and call `build()` to obtain
        /// a fully-initialized `ModelCapabilities`.
        ///
        /// `atomic_types`, `interaction_range`, `length_unit`,
        /// `supported_devices`, and `dtype` are required and must be set before
        /// calling `build()`, otherwise `build()` throws `metatomic::Error`.
        /// `outputs` defaults to an empty list.
        class Builder {
        private:
            std::vector<Quantity> outputs_;
            std::optional<std::vector<int64_t>> atomic_types_;
            std::optional<double> interaction_range_;
            std::optional<std::string> length_unit_;
            std::optional<std::vector<Device>> supported_devices_;
            std::optional<DType> dtype_;

        public:
            /// Set the list of outputs this model can provide.
            Builder& outputs(std::vector<Quantity> value) {
                outputs_ = std::move(value);
                return *this;
            }

            /// Add an output to the list of outputs this model can provide.
            Builder& add_output(const Quantity& output) {
                outputs_.push_back(output);
                return *this;
            }

            /// Clear the list of outputs this model can provide.
            Builder& clear_outputs() {
                outputs_.clear();
                return *this;
            }

            /// Set the atomic types this model supports.
            Builder& atomic_types(std::vector<int64_t> value) {
                atomic_types_ = std::move(value);
                return *this;
            }

            /// Add an atomic type to the list of atomic types this model supports.
            Builder& add_atomic_type(int64_t atomic_type) {
                if (!atomic_types_.has_value()) {
                    atomic_types_ = std::vector<int64_t>();
                }
                atomic_types_->push_back(atomic_type);
                return *this;
            }

            /// Clear the list of atomic types this model supports.
            Builder& clear_atomic_types() {
                if (atomic_types_.has_value()) {
                    atomic_types_->clear();
                }
                return *this;
            }

            /// Set the interaction range of the model.
            ///
            /// @throw metatomic::Error if the value is negative.
            Builder& interaction_range(double value) {
                ModelCapabilities::validate_interaction_range(value);
                interaction_range_ = value;
                return *this;
            }

            /// Set the length unit of the model.
            Builder& length_unit(std::string value) {
                length_unit_ = std::move(value);
                return *this;
            }

            /// Set the devices on which this model can run.
            Builder& supported_devices(std::vector<Device> value) {
                supported_devices_ = std::move(value);
                return *this;
            }

            /// Add a device to the list of devices on which this model can run.
            Builder& add_supported_device(Device device) {
                if (!supported_devices_.has_value()) {
                    supported_devices_ = std::vector<Device>();
                }
                supported_devices_->push_back(device);
                return *this;
            }

            /// Clear the list of devices on which this model can run.
            Builder& clear_supported_devices() {
                if (supported_devices_.has_value()) {
                    supported_devices_->clear();
                }
                return *this;
            }

            /// Set the data type of the model.
            Builder& dtype(DType value) {
                dtype_ = value;
                return *this;
            }

            /// Build a fully-initialized `ModelCapabilities`.
            ///
            /// @throw metatomic::Error if `atomic_types`, `interaction_range`,
            ///     `length_unit`, `supported_devices`, or `dtype` has not been set.
            ///
            /// This moves the builder's fields; calling `build()` a second time
            /// produces an object with moved-from values. Builders are intended
            /// as one-shot temporaries.
            [[nodiscard]] ModelCapabilities build() {
                if (!atomic_types_.has_value()) {
                    throw metatomic::Error("atomic_types must be set before building ModelCapabilities");
                }
                if (!interaction_range_.has_value()) {
                    throw metatomic::Error("interaction_range must be set before building ModelCapabilities");
                }
                if (!length_unit_.has_value()) {
                    throw metatomic::Error("length_unit must be set before building ModelCapabilities");
                }
                if (!supported_devices_.has_value()) {
                    throw metatomic::Error("supported_devices must be set before building ModelCapabilities");
                }
                if (!dtype_.has_value()) {
                    throw metatomic::Error("dtype must be set before building ModelCapabilities");
                }
                return ModelCapabilities(
                    std::move(atomic_types_).value(),
                    interaction_range_.value(),
                    std::move(length_unit_).value(),
                    std::move(supported_devices_).value(),
                    dtype_.value(),
                    std::move(outputs_)
                );
            }
        };

        /// Create a new `Builder` for `ModelCapabilities`.
        [[nodiscard]] static Builder builder() {
            return Builder{};
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
            default:
                throw metatomic::Error("invalid dtype in ModelCapabilities");
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
            default:
                throw metatomic::Error("invalid device in ModelCapabilities");
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
            default:
                throw metatomic::Error("invalid sample_kind in Quantity");
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
            default:
                throw metatomic::Error("invalid gradients in Quantity");
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

    inline Quantity from_json(
        const nlohmann::json& j, nlohmann::detail::identity_tag<Quantity>
    ) {
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

        return Quantity::builder()
            .name(std::move(name))
            .unit(std::move(unit))
            .sample_kind(sample_kind)
            .description(std::move(description))
            .gradients(std::move(gradients))
            .build();
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

    inline ModelCapabilities from_json(
        const nlohmann::json& j, nlohmann::detail::identity_tag<ModelCapabilities>
    ) {
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

        return ModelCapabilities::builder()
            .atomic_types(std::move(atomic_types))
            .interaction_range(interaction_range)
            .length_unit(std::move(length_unit))
            .supported_devices(std::move(supported_devices))
            .dtype(dtype)
            .outputs(std::move(outputs))
            .build();
    }

} // namespace metatomic
