#pragma once

#include <string>
#include <memory>
#include <utility>
#include <vector>

#include <metatomic.h>
#include <metatensor.hpp>
#include <nlohmann/json.hpp>

#include "./metadata.hpp"
#include "./system.hpp"
#include "./utils.hpp"

namespace metatomic {

/// Abstract base class for atomistic models implemented in C++.
class ModelInterface {
public:
    virtual ~ModelInterface() = default;

    /// Get metadata about this model.
    virtual ModelMetadata metadata() const = 0;

    /// Get all quantities this model can compute.
    virtual std::vector<Quantity> supported_outputs() const = 0;

    /// Get all pair lists this model requires.
    virtual std::vector<PairListOptions> requested_pair_lists() const {
        return {};
    }

    /// Get all custom inputs this model requires.
    virtual std::vector<Quantity> requested_inputs() const {
        return {};
    }

    /// Execute this model.
    virtual std::vector<metatensor::TensorMap> execute(
        const std::vector<const System*>& systems,
        const mts_labels_t* selected_atoms,
        const std::vector<Quantity>& requested_outputs
    ) = 0;
};

/// RAII wrapper around a `mta_model_t`.
class AtomisticModel final {
public:
    /// Create an empty, invalid model.
    AtomisticModel() {
        model_ = empty_model();
    }

    /// Take ownership of a raw `mta_model_t`.
    explicit AtomisticModel(mta_model_t model): model_(model) {}

    /// Create a C API model wrapping a C++ model implementation.
    explicit AtomisticModel(std::unique_ptr<ModelInterface> model) {
        if (model == nullptr) {
            throw Error("can not create a metatomic::AtomisticModel from a null ModelInterface");
        }

        model_ = empty_model();
        model_.data = model.release();
        model_.unload = &AtomisticModel::unload_callback;
        model_.metadata = &AtomisticModel::metadata_callback;
        model_.supported_outputs = &AtomisticModel::supported_outputs_callback;
        model_.requested_pair_lists = &AtomisticModel::requested_pair_lists_callback;
        model_.requested_inputs = &AtomisticModel::requested_inputs_callback;
        model_.execute_inner = &AtomisticModel::execute_callback;
    }

    ~AtomisticModel() {
        this->reset_noexcept();
    }

    AtomisticModel(const AtomisticModel&) = delete;
    AtomisticModel& operator=(const AtomisticModel&) = delete;

    AtomisticModel(AtomisticModel&& other) noexcept: AtomisticModel() {
        *this = std::move(other);
    }

    AtomisticModel& operator=(AtomisticModel&& other) noexcept {
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

    /// Get model metadata serialized as JSON.
    std::string metadata_json() const {
        this->check_callback(model_.metadata, "metadata");

        mta_string_t metadata = nullptr;
        details::check_status(model_.metadata(model_.data, &metadata));
        return String(metadata).str();
    }

    /// Get model metadata.
    ModelMetadata metadata() const {
        return ModelMetadata::from_json(this->metadata_json());
    }

    /// Get supported outputs serialized as JSON.
    std::string supported_outputs_json() const {
        this->check_callback(model_.supported_outputs, "supported_outputs");

        mta_string_t outputs = nullptr;
        details::check_status(model_.supported_outputs(model_.data, &outputs));
        return String(outputs).str();
    }

    /// Get all quantities this model can compute.
    std::vector<Quantity> supported_outputs() const {
        auto outputs = std::vector<Quantity>();
        for (const auto& output: nlohmann::json::parse(this->supported_outputs_json())) {
            outputs.push_back(Quantity::from_json(output.dump()));
        }
        return outputs;
    }

    /// Get all pair lists requested by this model serialized as a JSON array.
    std::string requested_pair_lists_json() const {
        this->check_callback(model_.requested_pair_lists, "requested_pair_lists");

        mta_string_t options = nullptr;
        details::check_status(model_.requested_pair_lists(model_.data, &options));
        return String(options).str();
    }

    /// Get all pair lists requested by this model.
    std::vector<PairListOptions> requested_pair_lists() const {
        auto result = std::vector<PairListOptions>();
        for (const auto& options: nlohmann::json::parse(this->requested_pair_lists_json())) {
            result.push_back(PairListOptions::from_json(options.dump()));
        }
        return result;
    }

    /// Get all custom inputs requested by this model serialized as a JSON array.
    std::string requested_inputs_json() const {
        this->check_callback(model_.requested_inputs, "requested_inputs");

        mta_string_t inputs = nullptr;
        details::check_status(model_.requested_inputs(model_.data, &inputs));
        return String(inputs).str();
    }

    /// Get all custom inputs requested by this model.
    std::vector<Quantity> requested_inputs() const {
        auto result = std::vector<Quantity>();
        for (const auto& input: nlohmann::json::parse(this->requested_inputs_json())) {
            result.push_back(Quantity::from_json(input.dump()));
        }
        return result;
    }

    /// Execute this model.
    ///
    /// The number of returned tensor maps is `requested_outputs.size()`.
    std::vector<metatensor::TensorMap> execute(
        const std::vector<const System*>& systems,
        const metatensor::Labels* selected_atoms,
        const std::vector<Quantity>& requested_outputs,
        bool check_consistency = true
    ) {
        this->check_valid();

        auto c_systems = std::vector<const mta_system_t*>();
        c_systems.reserve(systems.size());
        for (const auto* system: systems) {
            details::check_pointer(system);
            c_systems.push_back(system->as_mta_system_t());
        }

        auto c_requested_outputs = std::vector<const char*>();
        auto requested_outputs_json = std::vector<std::string>();
        requested_outputs_json.reserve(requested_outputs.size());
        c_requested_outputs.reserve(requested_outputs.size());
        for (const auto& output: requested_outputs) {
            requested_outputs_json.push_back(output.to_json());
            c_requested_outputs.push_back(requested_outputs_json.back().c_str());
        }

        auto raw_outputs = std::vector<mts_tensormap_t*>(requested_outputs.size(), nullptr);
        details::check_status(mta_execute_model(
            model_,
            c_systems.data(),
            c_systems.size(),
            selected_atoms == nullptr ? nullptr : selected_atoms->as_mts_labels_t(),
            c_requested_outputs.data(),
            c_requested_outputs.size(),
            check_consistency,
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
        const std::vector<Quantity>& requested_outputs,
        bool check_consistency = true
    ) {
        return this->execute(systems, nullptr, requested_outputs, check_consistency);
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
        model.requested_pair_lists = nullptr;
        model.requested_inputs = nullptr;
        model.execute_inner = nullptr;
        return model;
    }

    static ModelInterface* model_base(const void* data) {
        details::check_pointer(data);
        return static_cast<ModelInterface*>(const_cast<void*>(data));
    }

    static mta_status_t unload_callback(void* data) {
        return details::catch_exceptions([&]() {
            delete model_base(data);
        });
    }

    static mta_status_t metadata_callback(const void* data, mta_string_t* metadata_json) {
        return details::catch_exceptions([&]() {
            details::check_pointer(metadata_json);
            *metadata_json = mta_string_create(model_base(data)->metadata().to_json().c_str());
            details::check_pointer(*metadata_json);
        });
    }

    static mta_status_t supported_outputs_callback(const void* data, mta_string_t* outputs_json) {
        return details::catch_exceptions([&]() {
            details::check_pointer(outputs_json);
            auto outputs = nlohmann::json::array();
            for (const auto& output: model_base(data)->supported_outputs()) {
                outputs.push_back(nlohmann::json::parse(output.to_json()));
            }

            *outputs_json = mta_string_create(outputs.dump().c_str());
            details::check_pointer(*outputs_json);
        });
    }

    static mta_status_t requested_pair_lists_callback(const void* data, mta_string_t* pair_options_json) {
        return details::catch_exceptions([&]() {
            details::check_pointer(pair_options_json);
            auto options = nlohmann::json::array();
            for (const auto& option: model_base(data)->requested_pair_lists()) {
                options.push_back(nlohmann::json::parse(option.to_json()));
            }

            *pair_options_json = mta_string_create(options.dump().c_str());
            details::check_pointer(*pair_options_json);
        });
    }

    static mta_status_t requested_inputs_callback(const void* data, mta_string_t* inputs_json) {
        return details::catch_exceptions([&]() {
            details::check_pointer(inputs_json);
            auto inputs = nlohmann::json::array();
            for (const auto& input: model_base(data)->requested_inputs()) {
                inputs.push_back(nlohmann::json::parse(input.to_json()));
            }

            *inputs_json = mta_string_create(inputs.dump().c_str());
            details::check_pointer(*inputs_json);
        });
    }

    static mta_status_t execute_callback(
        void* data,
        const mta_system_t* const* systems,
        uintptr_t systems_count,
        const mts_labels_t* selected_atoms,
        const char* const* requested_outputs_json,
        uintptr_t requested_outputs_count,
        mts_tensormap_t** outputs,
        uintptr_t outputs_count
    ) {
        return details::catch_exceptions([&]() {
            if (systems_count != 0) {
                details::check_pointer(systems);
            }
            if (requested_outputs_count != 0) {
                details::check_pointer(requested_outputs_json);
            }
            if (outputs_count != 0) {
                details::check_pointer(outputs);
            }
            if (requested_outputs_count != outputs_count) {
                throw Error("expected one output storage slot for each requested output");
            }

            auto system_views = std::vector<System>();
            system_views.reserve(systems_count);
            for (uintptr_t i=0; i<systems_count; i++) {
                details::check_pointer(systems[i]);
                system_views.push_back(System::unsafe_view_from_ptr(systems[i]));
            }

            auto cxx_systems = std::vector<const System*>();
            cxx_systems.reserve(system_views.size());
            for (const auto& system: system_views) {
                cxx_systems.push_back(&system);
            }

            auto requested_outputs = std::vector<Quantity>();
            requested_outputs.reserve(requested_outputs_count);
            for (uintptr_t i=0; i<requested_outputs_count; i++) {
                details::check_pointer(requested_outputs_json[i]);
                requested_outputs.push_back(Quantity::from_json(requested_outputs_json[i]));
            }

            auto result = model_base(data)->execute(cxx_systems, selected_atoms, requested_outputs);
            if (result.size() != outputs_count) {
                throw Error("model returned the wrong number of outputs");
            }

            for (uintptr_t i=0; i<outputs_count; i++) {
                outputs[i] = mts_tensormap_copy(result[i].as_mts_tensormap_t());
                details::check_pointer(outputs[i]);
            }
        });
    }

    void reset_noexcept() noexcept {
        if (model_.data != nullptr && model_.unload != nullptr) {
            (void)model_.unload(model_.data);
        }
        model_ = empty_model();
    }

    void check_valid() const {
        if (model_.data == nullptr) {
            throw Error("can not use an empty metatomic::AtomisticModel");
        }
    }

    template<typename Callback>
    void check_callback(Callback callback, const char* name) const {
        this->check_valid();
        if (callback == nullptr) {
            throw Error("metatomic::AtomisticModel does not implement " + std::string(name));
        }
    }

    mta_model_t model_;
};

} // namespace metatomic
