#pragma once

#include <string>
#include <utility>
#include <vector>

#include <metatomic.h>
#include <metatensor.hpp>

#include "./system.hpp"
#include "./utils.hpp"

namespace metatomic {

/// RAII wrapper around a `mta_model_t`.
class Model final {
public:
    /// Create an empty, invalid model.
    Model() {
        model_ = empty_model();
    }

    /// Take ownership of a raw `mta_model_t`.
    explicit Model(mta_model_t model): model_(model) {}

    ~Model() {
        this->reset_noexcept();
    }

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    Model(Model&& other) noexcept: Model() {
        *this = std::move(other);
    }

    Model& operator=(Model&& other) noexcept {
        if (this != &other) {
            this->reset_noexcept();
            model_ = other.model_;
            other.model_ = empty_model();
        }
        return *this;
    }

    /// Does this wrapper contain a model?
    bool is_valid() const {
        return model_.data != nullptr;
    }

    /// Unload the model.
    void unload() {
        if (model_.data != nullptr && model_.unload != nullptr) {
            details::check_status(model_.unload(model_.data));
        }
        model_ = empty_model();
    }

    /// Get model metadata as a JSON string.
    std::string metadata() const {
        this->check_callback(model_.metadata, "metadata");

        mta_string_t metadata = nullptr;
        details::check_status(model_.metadata(model_.data, &metadata));
        return String(metadata).str();
    }

    /// Get supported outputs as a JSON string.
    std::string supported_outputs() const {
        this->check_callback(model_.supported_outputs, "supported_outputs");

        mta_string_t outputs = nullptr;
        details::check_status(model_.supported_outputs(model_.data, &outputs));
        return String(outputs).str();
    }

    /// Get all pair lists requested by this model, each one serialized as JSON.
    std::vector<std::string> requested_pair_lists() const {
        this->check_callback(model_.requested_pair_lists_count, "requested_pair_lists_count");
        this->check_callback(model_.requested_pair_list, "requested_pair_list");

        uintptr_t count = 0;
        details::check_status(model_.requested_pair_lists_count(model_.data, &count));

        auto result = std::vector<std::string>();
        result.reserve(count);
        for (uintptr_t i=0; i<count; i++) {
            mta_string_t options = nullptr;
            details::check_status(model_.requested_pair_list(model_.data, i, &options));
            result.push_back(String(options).str());
        }

        return result;
    }

    /// Get all custom inputs requested by this model, each one serialized as JSON.
    std::vector<std::string> requested_inputs() const {
        this->check_callback(model_.requested_inputs_count, "requested_inputs_count");
        this->check_callback(model_.requested_input, "requested_input");

        uintptr_t count = 0;
        details::check_status(model_.requested_inputs_count(model_.data, &count));

        auto result = std::vector<std::string>();
        result.reserve(count);
        for (uintptr_t i=0; i<count; i++) {
            mta_string_t input = nullptr;
            details::check_status(model_.requested_input(model_.data, i, &input));
            result.push_back(String(input).str());
        }

        return result;
    }

    /// Execute this model.
    ///
    /// The number of returned tensor maps is `requested_outputs.size()`.
    std::vector<metatensor::TensorMap> execute(
        const std::vector<const System*>& systems,
        const metatensor::Labels* selected_atoms,
        const std::vector<std::string>& requested_outputs
    ) {
        this->check_valid();

        auto c_systems = std::vector<const mta_system_t*>();
        c_systems.reserve(systems.size());
        for (const auto* system: systems) {
            details::check_pointer(system);
            c_systems.push_back(system->as_mta_system_t());
        }

        auto c_requested_outputs = std::vector<const char*>();
        c_requested_outputs.reserve(requested_outputs.size());
        for (const auto& output: requested_outputs) {
            c_requested_outputs.push_back(output.c_str());
        }

        auto raw_outputs = std::vector<mts_tensormap_t*>(requested_outputs.size(), nullptr);
        details::check_status(mta_execute_model(
            model_,
            c_systems.data(),
            c_systems.size(),
            selected_atoms == nullptr ? nullptr : selected_atoms->as_mts_labels_t(),
            c_requested_outputs.data(),
            c_requested_outputs.size(),
            raw_outputs.data(),
            raw_outputs.size()
        ));

        auto outputs = std::vector<metatensor::TensorMap>();
        outputs.reserve(raw_outputs.size());

        try {
            for (auto*& output: raw_outputs) {
                details::check_pointer(output);
                outputs.emplace_back(output);
                output = nullptr;
            }
        } catch (...) {
            for (auto* output: raw_outputs) {
                if (output != nullptr) {
                    (void)mts_tensormap_free(output);
                }
            }
            throw;
        }

        return outputs;
    }

    /// Execute this model on all atoms.
    std::vector<metatensor::TensorMap> execute(
        const std::vector<const System*>& systems,
        const std::vector<std::string>& requested_outputs
    ) {
        return this->execute(systems, nullptr, requested_outputs);
    }

    /// Get the underlying `mta_model_t`.
    const mta_model_t& as_mta_model_t() const & {
        return model_;
    }

    const mta_model_t& as_mta_model_t() && = delete;

    /// Release the underlying `mta_model_t` without unloading it.
    mta_model_t release() {
        auto model = model_;
        model_ = empty_model();
        return model;
    }

private:
    static mta_model_t empty_model() {
        mta_model_t model;
        model.data = nullptr;
        model.unload = nullptr;
        model.metadata = nullptr;
        model.supported_outputs = nullptr;
        model.requested_pair_lists_count = nullptr;
        model.requested_pair_list = nullptr;
        model.requested_inputs_count = nullptr;
        model.requested_input = nullptr;
        model.execute_inner = nullptr;
        return model;
    }

    void reset_noexcept() noexcept {
        if (model_.data != nullptr && model_.unload != nullptr) {
            (void)model_.unload(model_.data);
        }
        model_ = empty_model();
    }

    void check_valid() const {
        if (model_.data == nullptr) {
            throw Error("can not use an empty metatomic::Model");
        }
    }

    template<typename Callback>
    void check_callback(Callback callback, const char* name) const {
        this->check_valid();
        if (callback == nullptr) {
            throw Error("metatomic::Model does not implement " + std::string(name));
        }
    }

    mta_model_t model_;
};

} // namespace metatomic
