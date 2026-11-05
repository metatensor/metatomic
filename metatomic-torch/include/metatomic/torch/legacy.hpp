#ifndef METATOMIC_TORCH_LEGACY_HPP
#define METATOMIC_TORCH_LEGACY_HPP

#include <optional>
#include <string>
#include <vector>

#include <torch/script.h>

#include <metatensor/torch.hpp>
#include <metatomic.hpp>

#include "metatomic/torch/exports.h"
#include "metatomic/torch/system.hpp"

namespace metatomic_torch {

/// Run a model exported with `metatomic.torch.AtomisticModel` (typically a
/// `.pt` file) with `metatomic::execute_model`, or any engine using the
/// metatomic C API through `metatomic::BaseModel::to_mta_model`.
///
/// This is the C++ counterpart of the Python `metatomic.LegacyModel`, with
/// the same behavior:
///
/// - outputs keep the names declared by the model, deprecated names are not
///   translated;
/// - outputs declared with `sample_kind="atom"` are also declared with
///   `sample_kind="system"`, since the legacy API allowed requesting both;
/// - the `"mps"` device is renamed to `"metal"`;
/// - the explicit gradients of the model are used as-is, and the gradients
///   with respect to `"positions"` and `"strain"` of per-system energies and
///   custom quantities are computed with autograd.
///
/// The systems given by the engine are not copied: the model sees new
/// `System` objects sharing the types, positions, cell, pair lists and
/// custom data arrays of the engine's systems. In the same way, the outputs
/// share their data with the tensors created by the model.
class METATOMIC_TORCH_EXPORT LegacyModel final: public metatomic::BaseModel {
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
