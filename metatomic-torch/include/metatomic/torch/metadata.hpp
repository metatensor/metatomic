#ifndef METATOMIC_TORCH_METADATA_HPP
#define METATOMIC_TORCH_METADATA_HPP

#include <vector>
#include <string>

#include <torch/script.h>

#include <metatensor/torch.hpp>
#include <metatomic.hpp>

#include "metatomic/torch/exports.h"

namespace metatomic_torch {

class QuantityHolder;
/// TorchScript will always manipulate `QuantityHolder` through a `torch::intrusive_ptr`
using Quantity = torch::intrusive_ptr<QuantityHolder>;

/// A physical quantity that a model can take as input or produce as output
class METATOMIC_TORCH_EXPORT QuantityHolder final: public torch::CustomClassHolder {
public:
    /// Initialize `Quantity` with the given information.
    ///
    /// `sample_kind` must be one of "system", "atom" or "atom_pair", and each
    /// entry in `gradients` must be one of "positions" or "strain". Setting
    /// `gradients` to `torch::nullopt` is the same as an empty list.
    QuantityHolder(
        std::string name,
        std::string unit,
        const std::string& sample_kind,
        const torch::optional<std::vector<std::string>>& gradients,
        torch::optional<std::string> description
    );

    /// Create `Quantity` from an existing `metatomic::Quantity`
    explicit QuantityHolder(metatomic::Quantity quantity);

    ~QuantityHolder() override = default;

    /// Name of this quantity, this can be a standard name from
    /// https://docs.metatensor.org/metatomic/latest/quantities/index.html, or
    /// a custom name of the form `<namespace>::<name>[/<variant>]`
    const std::string& name() const {
        return quantity_.name();
    }
    /// Set the name of this quantity
    void set_name(std::string name);

    /// Unit of this quantity
    const std::string& unit() const {
        return quantity_.unit();
    }
    /// Set the unit of this quantity
    void set_unit(std::string unit);

    /// Kind of samples this quantity is associated with, one of "system",
    /// "atom" or "atom_pair"
    std::string sample_kind() const;
    /// Set the kind of samples this quantity is associated with
    void set_sample_kind(const std::string& sample_kind);

    /// List of gradients stored explicitly for this quantity, each entry is
    /// one of "positions" or "strain"
    std::vector<std::string> gradients() const;
    /// Set the list of gradients stored explicitly for this quantity
    void set_gradients(const std::vector<std::string>& gradients);

    /// Description of this quantity, `torch::nullopt` if there is no
    /// description
    torch::optional<std::string> description() const;
    /// Set the description of this quantity. An empty string is the same as
    /// no description.
    void set_description(torch::optional<std::string> description);

    /// Get the underlying `metatomic::Quantity` object
    const metatomic::Quantity& as_metatomic() const {
        return quantity_;
    }

    /// Implementation of Python's `__repr__` and `__str__`
    std::string repr() const;

    /// Serialize `Quantity` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `Quantity` from a JSON string.
    static Quantity from_json(std::string_view json);

private:
    metatomic::Quantity quantity_;
};

/// Check `Quantity` for equality
inline bool operator==(const Quantity& lhs, const Quantity& rhs) {
    return lhs->as_metatomic() == rhs->as_metatomic();
}

/// Check `Quantity` for inequality
inline bool operator!=(const Quantity& lhs, const Quantity& rhs) {
    return !(lhs == rhs);
}

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
