#ifndef METATOMIC_TORCH_LEGACY_CAPABILITIES_HPP
#define METATOMIC_TORCH_LEGACY_CAPABILITIES_HPP

#include <vector>
#include <string>

#include <torch/script.h>

#include <metatensor/torch.hpp>
#include <metatomic.hpp>

#include "metatomic/torch/legacy/output.hpp"
#include "metatomic/torch/exports.h"


namespace metatomic_torch {

class ModelCapabilitiesHolder;
/// TorchScript will always manipulate `ModelCapabilitiesHolder` through a `torch::intrusive_ptr`
using ModelCapabilities = torch::intrusive_ptr<ModelCapabilitiesHolder>;

/// Description of a model's capabilities, i.e. everything a model can do.
class METATOMIC_TORCH_EXPORT ModelCapabilitiesHolder: public torch::CustomClassHolder {
public:
    ModelCapabilitiesHolder() = default;

    /// Initialize `ModelCapabilities` with the given data
    ModelCapabilitiesHolder(
        torch::Dict<std::string, ModelOutput> outputs,
        std::vector<int64_t> atomic_types_,
        double interaction_range_,
        std::string length_unit,
        std::vector<std::string> supported_devices_,
        std::string dtype
    ):
        atomic_types(std::move(atomic_types_)),
        interaction_range(interaction_range_),
        supported_devices(std::move(supported_devices_))
    {
        this->set_outputs(outputs);
        this->set_length_unit(std::move(length_unit));

        if (!dtype.empty()) {
            this->set_dtype(std::move(dtype));
        }
    }

    ~ModelCapabilitiesHolder() override = default;

    /// all possible outputs from this model and corresponding settings
    torch::Dict<std::string, ModelOutput> outputs() const {
        return outputs_;
    }
    /// set the outputs for this model
    void set_outputs(torch::Dict<std::string, ModelOutput> outputs);

    /// which types the model can handle
    std::vector<int64_t> atomic_types;

    /// How far a given atom needs to know about other atoms, in the length unit
    /// of the model.
    ///
    /// This is used to properly implement domain decomposition with this model.
    ///
    /// For a short range model, this is the same as the largest neighbor list
    /// cutoff. For a message passing model, this is the cutoff of one
    /// environment times the number of message passing steps. For an explicit
    /// long range model, this should be set to infinity.
    ///
    /// This will default to -1 if not explicitly set by the user.
    double interaction_range = -1.0;

    /// unit of lengths the model expects as input
    const std::string& length_unit() const {
        return length_unit_;
    }
    /// set the unit of length for this model
    void set_length_unit(std::string unit);

    /// Get the `interaction_range` in the length unit of the engine
    double engine_interaction_range(const std::string& engine_length_unit) const;

    /// What devices can this model run on? This should only contain the
    /// `device_type` part of the device, and not the device number (i.e. this
    /// should be `"cuda"`, not `"cuda:0"`).
    ///
    /// Devices should be ordered in order of preference: first one should be
    /// the best device for this model, and so on.
    std::vector<std::string> supported_devices;

    /// Get the dtype of this model. This can be "float32" or "float64", and
    /// must be used by the engine as the dtype of all inputs and outputs for
    /// this model.
    const std::string& dtype() const {
        return dtype_;
    }

    /// Set the dtype of this model.
    void set_dtype(std::string dtype);

    /// Serialize a `ModelCapabilities` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelCapabilities` from a JSON string.
    static ModelCapabilities from_json(std::string_view json);

private:
    void set_outputs(torch::Dict<std::string, ModelOutput> outputs, bool warn_on_deprecated);

    torch::Dict<std::string, ModelOutput> outputs_;
    std::string length_unit_;
    std::string dtype_;
};

}

#endif
