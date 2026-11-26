#ifndef METATOMIC_TORCH_PLUGIN_LEGACY_MODEL_HPP
#define METATOMIC_TORCH_PLUGIN_LEGACY_MODEL_HPP

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <torch/script.h>

#include <metatensor/torch.hpp>
#include <metatomic.hpp>

#include <metatomic/torch/system.hpp>

namespace metatomic_torch {

/// Run a model exported with `metatomic.torch.AtomisticModel` (typically a
/// `.pt` file) with `metatomic::execute_model`. This behaves like the Python
/// `metatomic.LegacyModel`.
class LegacyModel final: public metatomic::BaseModel {
public:
    /// Load the model exported at `path`, loading the TorchScript extensions
    /// it requires from `extensions_directory`.
    explicit LegacyModel(
        const std::string& path,
        c10::optional<std::string> extensions_directory = c10::nullopt
    );

    /// Wrap an `AtomisticModel` that was already loaded, e.g. with
    /// `load_atomistic_model`.
    explicit LegacyModel(torch::jit::Module model);

    /// Configure this model with `options`, like `set_options` in Python
    /// models. The only option is `device`: when given, the model always runs
    /// on this device, the systems are moved to it, and the outputs are moved
    /// back to the device of the systems, which can then also be the CPU.
    /// Otherwise, the model runs on the device of the systems.
    void set_options(const std::map<std::string, std::string>& options);

    ~LegacyModel() override = default;

    LegacyModel(const LegacyModel&) = delete;
    LegacyModel& operator=(const LegacyModel&) = delete;
    LegacyModel(LegacyModel&&) = default;
    LegacyModel& operator=(LegacyModel&&) = default;

    metatomic::ModelCapabilities capabilities() const override;

    metatomic::ModelMetadata metadata() const override {
        return metadata_;
    }

    std::vector<metatomic::PairListOptions> requested_pair_lists(
        const std::vector<metatomic::Quantity>& requested_outputs
    ) const override;

    std::vector<metatomic::Quantity> requested_inputs(
        const std::vector<metatomic::Quantity>& requested_outputs
    ) const override;

    std::vector<metatensor::TensorMap> execute_inner(
        const std::vector<metatomic::System>& systems,
        std::optional<metatensor::Labels> selected_atoms,
        const std::vector<metatomic::Quantity>& requested_outputs
    ) override;

private:
    /// The `AtomisticModel.module` of the legacy model, which is called
    /// directly: `execute_model` already converts units and checks the
    /// inputs and outputs
    metatensor_torch::Module module_;
    /// Device on which `module_` currently is
    torch::Device device_ = torch::kCPU;
    /// Device on which the model always runs, if the user gave one
    c10::optional<torch::Device> requested_device_;
    /// Devices supported by the legacy model, with their legacy names
    std::vector<std::string> legacy_devices_;

    /// Capabilities of the legacy model, with its explicit gradients only
    metatomic::ModelCapabilities declared_;
    metatomic::ModelMetadata metadata_;

    /// Pair lists requested by the legacy model, used to add the pairs given
    /// by the engine to the systems
    std::vector<PairListOptions> pair_lists_;
    std::vector<metatomic::Quantity> inputs_;
};

}

#endif
