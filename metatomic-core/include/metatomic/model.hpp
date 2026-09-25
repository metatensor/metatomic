#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <metatensor.hpp>
#include <metatomic.h>

#include <metatomic/errors.hpp>
#include <metatomic/metadata.hpp>
#include <metatomic/system.hpp>
#include <metatomic/utils.hpp>

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

        return details::string_from_mta(printed);
    }

    /// Abstract base class for atomistic models.
    ///
    /// This class provides a C++ interface for implementing custom models. Users
    /// can inherit from this class, override the virtual methods, and then
    /// convert the model to a `mta_model_t` with `BaseModel::to_mta_model`.
    class BaseModel {
    public:
        virtual ~BaseModel() = default;

        /// Get the capabilities of this model.
        virtual ModelCapabilities capabilities() const = 0;

        /// Get metadata describing this model.
        virtual ModelMetadata metadata() const = 0;

        /// List the pair lists (neighbor lists) this model needs as input.
        virtual std::vector<PairListOptions> requested_pair_lists() const = 0;

        /// List the additional per-system inputs this model needs.
        virtual std::vector<Quantity> requested_inputs() const = 0;

        /// Run the model and compute the requested outputs.
        ///
        /// This method should not be used directly. It is intended to be used
        /// through `mta_execute_model`.
        ///
        /// @param systems systems to run the model on
        /// @param selected_atoms optional selection of atoms to compute outputs
        ///     for, or `std::nullopt` to use all atoms
        /// @param requested_outputs outputs the model should compute
        /// @return the computed outputs, one tensor map per requested output
        virtual std::vector<metatensor::TensorMap> execute_inner(
            const std::vector<System>& systems,
            const std::optional<metatensor::Labels>& selected_atoms,
            const std::vector<Quantity>& requested_outputs
        ) = 0;

        /// Convert a C++ model to a `mta_model_t`.
        ///
        /// The returned `mta_model_t` takes ownership of the model and will
        /// delete it when the `unload` callback is called.
        ///
        /// @param model model to convert
        /// @return a `mta_model_t` model
        static mta_model_t to_mta_model(std::unique_ptr<BaseModel> model);

        /// Build a `mta_model_t` pointing at `model`, without taking
        /// ownership of it.
        ///
        /// The `unload` callback of the returned `mta_model_t` is left as
        /// `nullptr`, and none of the other callbacks free the model. This is
        /// the counterpart of `BaseModel::to_mta_model` for cases where the
        /// model must stay owned by the caller, such as `execute_model`.
        ///
        /// @warning The returned `mta_model_t` is a view of `model`: it stores
        ///     a plain pointer to it and does nothing to keep it alive. It is
        ///     the caller's responsibility to ensure `model` outlives every use
        ///     of the returned `mta_model_t`, and to never pass the result to
        ///     an API that takes ownership of the model (i.e. one that would
        ///     call `unload`).
        ///     `BaseModel::to_mta_model` whenever ownership can be transferred.
        ///
        /// @param model model to take a view of
        /// @return a non-owning `mta_model_t` view of `model`
        static mta_model_t mta_model_view(BaseModel& model);
    };

    /// RAII wrapper around an existing `mta_model_t`.
    ///
    /// This class wraps a model loaded from a plugin and exposes it through the
    /// same `BaseModel` interface. It owns the underlying `mta_model_t` and
    /// calls its `unload` callback on destruction.
    class ExternalModel final: public BaseModel {
    public:
        /// Wrap an existing `mta_model_t`, taking ownership of it.
        ///
        /// @param model model to wrap
        explicit ExternalModel(mta_model_t model):
            model_(model) {}

        ~ExternalModel() override {
            if (model_.unload != nullptr) {
                model_.unload(model_.data);
            }
        }

        ExternalModel(const ExternalModel&) = delete;
        ExternalModel& operator=(const ExternalModel&) = delete;

        ExternalModel(ExternalModel&& other) noexcept {
            *this = std::move(other);
        }

        ExternalModel& operator=(ExternalModel&& other) noexcept {
            if (this != &other) {
                if (model_.unload != nullptr) {
                    model_.unload(model_.data);
                }

                model_ = other.model_;

                other.model_ = mta_model_t{};
            }

            return *this;
        }

        /// Get the capabilities of this model.
        ModelCapabilities capabilities() const override {
            this->check_callback("capabilities", model_.capabilities);

            mta_string_t output = nullptr;
            auto status = model_.capabilities(model_.data, &output);
            details::check_status(status);

            auto json_str = details::string_from_mta(output);
            return nlohmann::json::parse(json_str).get<ModelCapabilities>();
        }

        /// Get metadata describing this model.
        ModelMetadata metadata() const override {
            this->check_callback("metadata", model_.metadata);

            mta_string_t output = nullptr;
            auto status = model_.metadata(model_.data, &output);
            details::check_status(status);

            auto json_str = details::string_from_mta(output);
            return nlohmann::json::parse(json_str).get<ModelMetadata>();
        }

        /// List the pair lists (neighbor lists) this model needs as input.
        std::vector<PairListOptions> requested_pair_lists() const override {
            this->check_callback("requested_pair_lists", model_.requested_pair_lists);

            mta_string_t output = nullptr;
            auto status = model_.requested_pair_lists(model_.data, &output);
            details::check_status(status);

            auto json_str = details::string_from_mta(output);
            return nlohmann::json::parse(json_str).get<std::vector<PairListOptions>>();
        }

        /// List the additional per-system inputs this model needs.
        std::vector<Quantity> requested_inputs() const override {
            this->check_callback("requested_inputs", model_.requested_inputs);

            mta_string_t output = nullptr;
            auto status = model_.requested_inputs(model_.data, &output);
            details::check_status(status);

            auto json_str = details::string_from_mta(output);
            return nlohmann::json::parse(json_str).get<std::vector<Quantity>>();
        }

        /// Run the model and compute the requested outputs.
        std::vector<metatensor::TensorMap> execute_inner(
            const std::vector<System>& systems,
            const std::optional<metatensor::Labels>& selected_atoms,
            const std::vector<Quantity>& requested_outputs
        ) override {
            this->check_callback("execute_inner", model_.execute_inner);

            std::vector<const mta_system_t*> systems_ptrs;
            systems_ptrs.reserve(systems.size());
            for (const auto& system: systems) {
                systems_ptrs.push_back(system.as_mta_system_t());
            }

            const mts_labels_t* selected_atoms_ptr = selected_atoms.has_value()
                ? selected_atoms->as_mts_labels_t()
                : nullptr;

            nlohmann::json json = requested_outputs;
            auto requested_outputs_str = json.dump();

            std::vector<mts_tensormap_t*> outputs(requested_outputs.size(), nullptr);

            auto status = model_.execute_inner(
                model_.data,
                systems_ptrs.data(),
                static_cast<uintptr_t>(systems_ptrs.size()),
                selected_atoms_ptr,
                requested_outputs_str.c_str(),
                outputs.data(),
                static_cast<uintptr_t>(outputs.size())
            );
            details::check_status(status);

            std::vector<metatensor::TensorMap> result;
            result.reserve(outputs.size());
            for (auto* output: outputs) {
                result.push_back(metatensor::TensorMap::unsafe_from_ptr(output));
            }

            return result;
        }

        /// Get a pointer to the raw `mta_model_t` backing this wrapper.
        ///
        /// The `ExternalModel` keeps ownership of the underlying model.
        mta_model_t* as_mta_model_t() & {
            return &model_;
        }

        /// Get a pointer to the raw `mta_model_t` backing this wrapper.
        ///
        /// The `ExternalModel` keeps ownership of the underlying model.
        const mta_model_t* as_mta_model_t() const & {
            return &model_;
        }

        /// Getting the raw pointer from a temporary `ExternalModel` is forbidden,
        /// as it would immediately dangle.
        mta_model_t* as_mta_model_t() && = delete;

        /// Release ownership of the underlying `mta_model_t`.
        ///
        /// After this call, the `ExternalModel` is empty and will not call
        /// the `unload` callback on destruction. The caller is responsible
        /// for calling the `unload` callback.
        mta_model_t release() {
            auto model = model_;
            model_ = mta_model_t{};
            return model;
        }

    private:
        template<typename Callback>
        void check_callback(const char* name, Callback callback) const {
            if (callback == nullptr) {
                throw Error(
                    "model is missing a '" + std::string(name) + "' callback"
                );
            }
        }

        mta_model_t model_ = mta_model_t{};
    };

    inline mta_model_t BaseModel::mta_model_view(BaseModel& model) {
        // Short-circuit if the model is already an ExternalModel, to avoid
        // double wrapping. The `ExternalModel` keeps ownership of the
        // underlying model, so we clear `unload`.
        if (auto* ext = dynamic_cast<ExternalModel*>(&model)) {
            auto m = *ext->as_mta_model_t();
            m.unload = nullptr;
            return m;
        }

        mta_model_t m = mta_model_t{};

        m.data = &model;

        m.capabilities = [](const void* model_data, mta_string_t* capabilities_json) -> mta_status_t {
            return details::catch_exceptions([](const void* model_data, mta_string_t* capabilities_json) {
                const auto* model = static_cast<const BaseModel*>(model_data);
                nlohmann::json json = model->capabilities();
                *capabilities_json = mta_string_create(json.dump().c_str());
            }, model_data, capabilities_json);
        };

        m.metadata = [](const void* model_data, mta_string_t* metadata_json) -> mta_status_t {
            return details::catch_exceptions([](const void* model_data, mta_string_t* metadata_json) {
                const auto* model = static_cast<const BaseModel*>(model_data);
                nlohmann::json json = model->metadata();
                *metadata_json = mta_string_create(json.dump().c_str());
            }, model_data, metadata_json);
        };

        m.requested_pair_lists = [](const void* model_data, mta_string_t* pair_options_json) -> mta_status_t {
            return details::catch_exceptions([](const void* model_data, mta_string_t* pair_options_json) {
                const auto* model = static_cast<const BaseModel*>(model_data);
                nlohmann::json json = model->requested_pair_lists();
                *pair_options_json = mta_string_create(json.dump().c_str());
            }, model_data, pair_options_json);
        };

        m.requested_inputs = [](const void* model_data, mta_string_t* inputs_json) -> mta_status_t {
            return details::catch_exceptions([](const void* model_data, mta_string_t* inputs_json) {
                const auto* model = static_cast<const BaseModel*>(model_data);
                nlohmann::json json = model->requested_inputs();
                *inputs_json = mta_string_create(json.dump().c_str());
            }, model_data, inputs_json);
        };

        m.execute_inner = [](
            void* model_data,
            const struct mta_system_t* const* systems,
            uintptr_t systems_count,
            const mts_labels_t* selected_atoms,
            const char* requested_outputs_json,
            mts_tensormap_t** outputs,
            uintptr_t outputs_count
        ) -> mta_status_t {
            return details::catch_exceptions([](
                void* model_data,
                const struct mta_system_t* const* systems,
                uintptr_t systems_count,
                const mts_labels_t* selected_atoms,
                const char* requested_outputs_json,
                mts_tensormap_t** outputs,
                uintptr_t outputs_count
            ) {
                auto* model = static_cast<BaseModel*>(model_data);

                std::vector<System> cpp_systems;
                cpp_systems.reserve(systems_count);
                for (uintptr_t i = 0; i < systems_count; ++i) {
                    cpp_systems.push_back(System::unsafe_view_from_ptr(systems[i]));
                }

                std::optional<metatensor::Labels> selected_atoms_cpp;
                if (selected_atoms != nullptr) {
                    selected_atoms_cpp = metatensor::Labels::unsafe_from_ptr(
                        mts_labels_clone(selected_atoms)
                    );
                }

                nlohmann::json json = nlohmann::json::parse(requested_outputs_json);
                auto requested_outputs = json.get<std::vector<Quantity>>();

                auto cpp_outputs = model->execute_inner(
                    cpp_systems, selected_atoms_cpp, requested_outputs
                );

                if (cpp_outputs.size() != outputs_count) {
                    throw Error(
                        "model returned " + std::to_string(cpp_outputs.size()) +
                        " outputs, but " + std::to_string(outputs_count) +
                        " were requested"
                    );
                }

                for (uintptr_t i = 0; i < outputs_count; ++i) {
                    outputs[i] = cpp_outputs[i].release();
                }
            }, model_data, systems, systems_count, selected_atoms, requested_outputs_json, outputs, outputs_count);
        };

        return m;
    }

    inline mta_model_t BaseModel::to_mta_model(std::unique_ptr<BaseModel> model) {
        // Short-circuit if the model is already an ExternalModel
        // to avoid double wrapping
        if (auto* ext = dynamic_cast<ExternalModel*>(model.get())) {
            return ext->release();
        }

        auto m = BaseModel::mta_model_view(*model);

        // mta_model_view returns a non-owning view of the model
        // Here we add an `unload` callback to take ownership
        m.unload = [](void* model_data) -> mta_status_t {
            return details::catch_exceptions([](void* model_data) {
                delete static_cast<BaseModel*>(model_data);
            }, model_data);
        };

        model.release();

        return m;
    }

    /// Execute a model to compute the requested outputs for a set of systems.
    ///
    /// @param model the model to execute. A view of the model is created,
    ///     so the ownership of `model` remains with the caller.
    /// @param systems systems to run the model on
    /// @param selected_atoms optional selection of atoms to compute outputs
    ///     for, or `nullptr` to use all atoms
    /// @param requested_outputs outputs the model should compute, one per
    ///     requested output
    /// @param check_consistency if `true`, run additional checks on the inputs
    ///     and on the data produced by the model
    /// @return the computed outputs, one tensor map per requested output
    inline std::vector<metatensor::TensorMap> execute_model(
        BaseModel& model,
        const std::vector<System>& systems,
        const std::optional<metatensor::Labels>& selected_atoms,
        const std::vector<Quantity>& requested_outputs,
        bool check_consistency
    ) {
        // non-owning view of the model
        // `model` is kept alive by the caller
        auto raw_model = BaseModel::mta_model_view(model);

        std::vector<const mta_system_t*> systems_ptrs;
        systems_ptrs.reserve(systems.size());
        for (const auto& system: systems) {
            systems_ptrs.push_back(system.as_mta_system_t());
        }

        const mts_labels_t* selected_atoms_ptr = selected_atoms.has_value() ? selected_atoms->as_mts_labels_t() : nullptr;

        nlohmann::json json = requested_outputs;
        auto requested_outputs_str = json.dump();

        std::vector<mts_tensormap_t*> outputs(requested_outputs.size(), nullptr);

        auto status = mta_execute_model(
            raw_model,
            systems_ptrs.data(),
            static_cast<uintptr_t>(systems_ptrs.size()),
            selected_atoms_ptr,
            requested_outputs_str.c_str(),
            check_consistency,
            outputs.data(),
            static_cast<uintptr_t>(outputs.size())
        );
        details::check_status(status);

        std::vector<metatensor::TensorMap> result;
        result.reserve(outputs.size());
        for (auto* output: outputs) {
            result.push_back(metatensor::TensorMap::unsafe_from_ptr(output));
        }

        return result;
    }
} // namespace metatomic
