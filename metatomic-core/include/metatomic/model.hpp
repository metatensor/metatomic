#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
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
    /// convert the model to a C-compatible `mta_model_t` with
    /// `ModelBase::to_mta_model`.
    class ModelBase {
    public:
        virtual ~ModelBase() = default;

        /// Get the capabilities of this model.
        virtual ModelCapabilities capabilities() const = 0;

        /// Get metadata describing this model.
        virtual ModelMetadata metadata() const = 0;

        /// List the outputs this model is able to compute.
        ///
        /// The default implementation returns the outputs declared in
        /// `capabilities()`.
        virtual std::vector<Quantity> supported_outputs() const {
            return capabilities().outputs();
        }

        /// List the pair lists (neighbor lists) this model needs as input.
        virtual std::vector<PairListOptions> requested_pair_lists() const = 0;

        /// List the additional per-system inputs this model needs.
        virtual std::vector<Quantity> requested_inputs() const = 0;

        /// Run the model and compute the requested outputs.
        ///
        /// @param systems systems to run the model on
        /// @param selected_atoms optional selection of atoms to compute outputs
        ///     for, or `nullptr` to use all atoms
        /// @param requested_outputs outputs the model should compute
        /// @return the computed outputs, one tensor map per requested output
        virtual std::vector<metatensor::TensorMap> execute(
            const std::vector<System>& systems,
            const metatensor::Labels* selected_atoms,
            const std::vector<Quantity>& requested_outputs
        ) = 0;

        /// Convert a C++ model to a C-compatible `mta_model_t`.
        ///
        /// The returned `mta_model_t` takes ownership of the model and will
        /// delete it when the `unload` callback is called.
        ///
        /// @param model model to convert
        /// @return `mta_model_t`
        static mta_model_t to_mta_model(std::unique_ptr<ModelBase> model) {
            mta_model_t m = mta_model_t{};
            auto* ptr = model.release();

            m.data = ptr;

            m.unload = [](void* model_data) -> mta_status_t {
                return details::catch_exceptions([](void* model_data) {
                    delete static_cast<ModelBase*>(model_data);
                }, model_data);
            };

            m.capabilities = [](const void* model_data, mta_string_t* capabilities_json) -> mta_status_t {
                return details::catch_exceptions([](const void* model_data, mta_string_t* capabilities_json) {
                    const auto* model = static_cast<const ModelBase*>(model_data);
                    nlohmann::json json = model->capabilities();
                    *capabilities_json = mta_string_create(json.dump().c_str());
                }, model_data, capabilities_json);
            };

            m.metadata = [](const void* model_data, mta_string_t* metadata_json) -> mta_status_t {
                return details::catch_exceptions([](const void* model_data, mta_string_t* metadata_json) {
                    const auto* model = static_cast<const ModelBase*>(model_data);
                    nlohmann::json json = model->metadata();
                    *metadata_json = mta_string_create(json.dump().c_str());
                }, model_data, metadata_json);
            };

            m.supported_outputs = [](const void* model_data, mta_string_t* outputs_json) -> mta_status_t {
                return details::catch_exceptions([](const void* model_data, mta_string_t* outputs_json) {
                    const auto* model = static_cast<const ModelBase*>(model_data);
                    nlohmann::json json = model->supported_outputs();
                    *outputs_json = mta_string_create(json.dump().c_str());
                }, model_data, outputs_json);
            };

            m.requested_pair_lists = [](const void* model_data, mta_string_t* pair_options_json) -> mta_status_t {
                return details::catch_exceptions([](const void* model_data, mta_string_t* pair_options_json) {
                    const auto* model = static_cast<const ModelBase*>(model_data);
                    nlohmann::json json = model->requested_pair_lists();
                    *pair_options_json = mta_string_create(json.dump().c_str());
                }, model_data, pair_options_json);
            };

            m.requested_inputs = [](const void* model_data, mta_string_t* inputs_json) -> mta_status_t {
                return details::catch_exceptions([](const void* model_data, mta_string_t* inputs_json) {
                    const auto* model = static_cast<const ModelBase*>(model_data);
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
                    auto* model = static_cast<ModelBase*>(model_data);

                    std::vector<System> cpp_systems;
                    cpp_systems.reserve(systems_count);
                    for (uintptr_t i = 0; i < systems_count; ++i) {
                        cpp_systems.push_back(System::unsafe_view_from_ptr(systems[i]));
                    }

                    std::optional<metatensor::Labels> selected_atoms_copy;
                    const metatensor::Labels* selected_atoms_cpp = nullptr;
                    if (selected_atoms != nullptr) {
                        selected_atoms_copy = metatensor::Labels::unsafe_from_ptr(
                            mts_labels_clone(selected_atoms)
                        );
                        selected_atoms_cpp = &*selected_atoms_copy;
                    }

                    nlohmann::json json = nlohmann::json::parse(requested_outputs_json);
                    auto requested_outputs = json.get<std::vector<Quantity>>();

                    auto cpp_outputs = model->execute(
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
    };

    /// RAII wrapper around an existing `mta_model_t`.
    ///
    /// This class wraps a model loaded from a plugin and exposes it through the
    /// same `ModelBase` interface. It owns the underlying `mta_model_t` and
    /// calls its `unload` callback on destruction.
    class ModelWrapper final: public ModelBase {
    public:
        /// Wrap an existing `mta_model_t`, taking ownership of it.
        ///
        /// @param model model to wrap
        explicit ModelWrapper(mta_model_t model):
            model_(model), is_view_(false) {}

        /// Create a non-owning view of an existing `mta_model_t`.
        ///
        /// The `mta_model_t` must outlive the returned view.
        ///
        /// @param model model to view
        static ModelWrapper unsafe_view_from_ptr(const mta_model_t& model) {
            return ModelWrapper(model, /*is_view*/ true);
        }

        ~ModelWrapper() override {
            if (!is_view_ && model_.unload != nullptr) {
                model_.unload(model_.data);
            }
        }

        ModelWrapper(const ModelWrapper&) = delete;
        ModelWrapper& operator=(const ModelWrapper&) = delete;

        ModelWrapper(ModelWrapper&& other) noexcept {
            *this = std::move(other);
        }

        ModelWrapper& operator=(ModelWrapper&& other) noexcept {
            if (!is_view_ && model_.unload != nullptr) {
                model_.unload(model_.data);
            }

            model_ = other.model_;
            is_view_ = other.is_view_;

            other.model_ = mta_model_t{};
            other.is_view_ = true;

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

        /// List the outputs this model is able to compute.
        std::vector<Quantity> supported_outputs() const override {
            if (model_.supported_outputs == nullptr) {
                return ModelBase::supported_outputs();
            }

            mta_string_t output = nullptr;
            auto status = model_.supported_outputs(model_.data, &output);
            details::check_status(status);

            auto json_str = details::string_from_mta(output);
            return nlohmann::json::parse(json_str).get<std::vector<Quantity>>();
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
        std::vector<metatensor::TensorMap> execute(
            const std::vector<System>& systems,
            const metatensor::Labels* selected_atoms,
            const std::vector<Quantity>& requested_outputs
        ) override {
            this->check_callback("execute_inner", model_.execute_inner);

            std::vector<const mta_system_t*> systems_ptrs;
            systems_ptrs.reserve(systems.size());
            for (const auto& system: systems) {
                systems_ptrs.push_back(system.as_mta_system_t());
            }

            const mts_labels_t* selected_atoms_ptr = nullptr;
            if (selected_atoms != nullptr) {
                selected_atoms_ptr = selected_atoms->as_mts_labels_t();
            }

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
        /// The `ModelWrapper` keeps ownership of the underlying model.
        mta_model_t* as_mta_model_t() & {
            return &model_;
        }

        /// Get a pointer to the raw `mta_model_t` backing this wrapper.
        ///
        /// The `ModelWrapper` keeps ownership of the underlying model.
        const mta_model_t* as_mta_model_t() const & {
            return &model_;
        }

        // Prevent getting the raw pointer from a temporary `ModelWrapper`
        mta_model_t* as_mta_model_t() && = delete;

        /// Release ownership of the underlying `mta_model_t`.
        ///
        /// After this call, the `ModelWrapper` becomes a non-owning view and
        /// the caller is responsible for calling the `unload` callback.
        mta_model_t release() {
            this->check_not_view("release");
            is_view_ = true;
            auto model = model_;
            model_ = mta_model_t{};
            return model;
        }

    private:
        /// Wrap an existing `mta_model_t` pointer, see `unsafe_view_from_ptr`.
        explicit ModelWrapper(mta_model_t model, bool is_view):
            model_(model), is_view_(is_view) {}


        void check_not_view(const std::string& method_name) const {
            if (is_view_) {
                    throw Error(
                        "can not call ModelWrapper::" + method_name +
                        " on this system since it is a view of a system owned elsewhere."
                    );
                }
            }

        template<typename Callback>
        void check_callback(const std::string& name, Callback callback) const {
            if (callback == nullptr) {
                throw Error(
                    "model is missing a '" + name + "' callback"
                );
            }
        }

        mta_model_t model_ = mta_model_t{};
        bool is_view_ = true;
    };
} // namespace metatomic
