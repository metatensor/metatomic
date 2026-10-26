#ifndef METATOMIC_TORCH_LEGACY_MODEL_HPP
#define METATOMIC_TORCH_LEGACY_MODEL_HPP

#include <string>

#include <torch/script.h>

#include <metatensor/torch.hpp>
#include <metatomic.hpp>

#include "metatomic/torch/legacy/output.hpp"
#include "metatomic/torch/metadata.hpp"
#include "metatomic/torch/exports.h"

namespace metatomic_torch {

class ModelEvaluationOptionsHolder;
/// TorchScript will always manipulate `ModelEvaluationOptionsHolder` through a `torch::intrusive_ptr`
using ModelEvaluationOptions = torch::intrusive_ptr<ModelEvaluationOptionsHolder>;


/// Options requested by the simulation engine when running with a model
class METATOMIC_TORCH_EXPORT ModelEvaluationOptionsHolder: public torch::CustomClassHolder {
public:
    ModelEvaluationOptionsHolder() = default;

    /// Initialize `ModelEvaluationOptions` with the given data
    ModelEvaluationOptionsHolder(
        std::string length_unit,
        torch::Dict<std::string, ModelOutput> outputs,
        torch::optional<metatensor_torch::Labels> selected_atoms
    );

    ~ModelEvaluationOptionsHolder() override = default;

    /// unit of lengths the engine uses in the data it calls the model with
    const std::string& length_unit() const {
        return length_unit_;
    }
    /// set the unit of length used by the engine
    void set_length_unit(std::string unit);

    /// requested outputs for this run and corresponding settings

    // FIXME: it would be nice to also check that the units are properly set
    // here, but since this is a field and not a `set_outputs` function it will
    // be hard to do, so we will wait until the next set of breaking changes.
    torch::Dict<std::string, ModelOutput> outputs;

    /// Only run the calculation for a selected subset of atoms. If this is set
    /// to `None`, run the calculation on all atoms. If this is a set of
    /// `Labels`, it will have two dimensions named `"system"` and `"atom"`,
    /// containing the 0-based indices of all the atoms in the selected subset.
    torch::optional<metatensor_torch::Labels> get_selected_atoms() const {
        return selected_atoms_;
    }

    /// Setter for `selected_atoms`
    void set_selected_atoms(torch::optional<metatensor_torch::Labels> selected_atoms);

    /// Serialize a `ModelEvaluationOptions` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelEvaluationOptions` from a JSON string.
    static ModelEvaluationOptions from_json(std::string_view json);

private:
    std::string length_unit_;
    torch::optional<metatensor_torch::Labels> selected_atoms_ = torch::nullopt;
};

/// Read the metadata of the exported model at the given `path` and return it
/// as a string. This is useful to check the metadata of a model without
/// loading it.
METATOMIC_TORCH_EXPORT ModelMetadata read_model_metadata(std::string path);

/// Check the exported atomistic model at the given `path`, and warn/error as
/// required. This should be called after `load_model_extensions`
METATOMIC_TORCH_EXPORT void check_atomistic_model(std::string path);

/// Load all extensions and extensions dependencies for the model at the given
/// `path`, trying to find extensions and dependencies in the given
/// `extensions`. Users can set the `METATENSOR_DEBUG_EXTENSIONS_LOADING`
/// environment variable to get more information when loading fails.
METATOMIC_TORCH_EXPORT void load_model_extensions(
    std::string path,
    c10::optional<std::string> extensions_directory
);

/// Check and then load the atomistic model at the given `path`.
///
/// This function calls `check_atomistic_model(path)` and
/// `load_model_extensions(path, extension_directory)` before attempting to load
/// the model.
METATOMIC_TORCH_EXPORT metatensor_torch::Module load_atomistic_model(
    std::string path,
    c10::optional<std::string> extensions_directory = c10::nullopt
);

}

#endif
