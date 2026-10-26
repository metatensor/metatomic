#ifndef METATOMIC_TORCH_METADATA_HPP
#define METATOMIC_TORCH_METADATA_HPP

#include <vector>
#include <string>

#include <torch/script.h>

#include <metatensor/torch.hpp>
#include <metatomic.hpp>

#include "metatomic/torch/exports.h"

namespace metatomic_torch {

class ModelMetadataHolder;
/// TorchScript will always manipulate `ModelMetadataHolder` through a `torch::intrusive_ptr`
using ModelMetadata = torch::intrusive_ptr<ModelMetadataHolder>;


/// Metadata about a specific exported model
class METATOMIC_TORCH_EXPORT ModelMetadataHolder final: public torch::CustomClassHolder {
public:
    /// Create an empty `ModelMetadata`
    ModelMetadataHolder();

    /// Initialize `ModelMetadata` with the given information
    ModelMetadataHolder(
        std::string name,
        std::string description,
        std::vector<std::string> authors,
        torch::Dict<std::string, std::vector<std::string>> references,
        torch::Dict<std::string, std::string> extra
    );

    /// Create `ModelMetadata` from an existing `metatomic::ModelMetadata`
    explicit ModelMetadataHolder(metatomic::ModelMetadata metadata);

    ~ModelMetadataHolder() override = default;

    /// Name of this model
    const std::string& name() const {
        return metadata_.name();
    }
    /// Set the name of this model
    void set_name(std::string name);

    /// Description of this model
    const std::string& description() const {
        return metadata_.description();
    }
    /// Set the description of this model
    void set_description(std::string description);

    /// List of authors for this model
    std::vector<std::string> authors() const {
        return metadata_.authors();
    }
    /// Set the list of authors for this model
    void set_authors(std::vector<std::string> authors);

    /// References for this model. The top level dict contains three keys:
    ///
    /// - "implementation": for reference to software and libraries used in the
    ///   implementation of the model (i.e. for PyTorch
    ///   https://dl.acm.org/doi/10.5555/3454287.3455008)
    /// - "architecture": for reference that introduced the general architecture
    ///   used by this model
    /// - "model": for reference specific to this exact model
    torch::Dict<std::string, std::vector<std::string>> references() const;
    /// Set the references for this model. The dict can only contain the keys
    /// "implementation", "architecture" and "model". Missing keys are treated
    /// as an empty list of references.
    void set_references(torch::Dict<std::string, std::vector<std::string>> references);

    /// Extra metadata about this model. This can be anything, and it is intended
    /// to be used by models to store data they need.
    torch::Dict<std::string, std::string> extra() const;
    /// Set the extra metadata about this model.
    void set_extra(torch::Dict<std::string, std::string> extra);

    /// Get the underlying `metatomic::ModelMetadata` object
    const metatomic::ModelMetadata& as_metatomic() const {
        return metadata_;
    }

    /// Implementation of Python's `__repr__` and `__str__`, printing all
    /// metadata about this model.
    std::string print() const;

    /// Serialize `ModelMetadata` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelMetadata` from a JSON string.
    static ModelMetadata from_json(std::string_view json);

private:
    metatomic::ModelMetadata metadata_;
};

}

#endif
