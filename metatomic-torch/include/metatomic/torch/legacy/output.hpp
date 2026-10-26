#ifndef METATOMIC_TORCH_LEGACY_OUTPUT_HPP
#define METATOMIC_TORCH_LEGACY_OUTPUT_HPP

#include <vector>
#include <string>

#include <torch/script.h>

#include <metatensor/torch.hpp>
#include <metatomic.hpp>

#include "metatomic/torch/exports.h"

namespace metatomic_torch {

class ModelOutputHolder;
/// TorchScript will always manipulate `ModelOutputHolder` through a `torch::intrusive_ptr`
using ModelOutput = torch::intrusive_ptr<ModelOutputHolder>;

/// Information about one of the quantity a model can compute
class METATOMIC_TORCH_EXPORT ModelOutputHolder: public torch::CustomClassHolder {
public:
    ModelOutputHolder();

    /// Initialize `ModelOutput` with the given data
    ModelOutputHolder(
        std::string quantity,
        std::string unit,
        std::string sample_kind,
        std::vector<std::string> explicit_gradients_,
        std::string description_
    );

    /// Overload to use the `sample_kind` constructor with a `const char*`,
    /// otherwise this would default to calling the `per_atom` constructor.
    ModelOutputHolder(
        std::string quantity,
        std::string unit,
        const char* sample_kind,
        std::vector<std::string> explicit_gradients_,
        std::string description_
    ): ModelOutputHolder(
        std::move(quantity),
        std::move(unit),
        std::string(sample_kind),
        std::move(explicit_gradients_),
        std::move(description_)
    ) {}

    /// For backward compatibility in the C++ API (per_atom argument)
    ModelOutputHolder(
        std::string quantity,
        std::string unit,
        bool per_atom_,
        std::vector<std::string> explicit_gradients_,
        std::string description_
    );

    /// For backward compatibility in the Python API
    ModelOutputHolder(
        std::string quantity,
        std::string unit,
        torch::IValue per_atom_or_sample_kind,
        std::vector<std::string> explicit_gradients_,
        std::string description_,
        torch::optional<bool> per_atom = torch::nullopt,
        torch::optional<std::string> sample_kind = torch::nullopt
    );

    ~ModelOutputHolder() override = default;

    /// description of this output, defaults to empty string of not set by the user
    std::string description;

    /// quantity of the output (e.g. energy, dipole, ...).  If this is an empty
    /// string, no unit conversion will be performed.
    /// @deprecated This field is no longer required for unit conversion.
    ///             The unit parser determines dimensions from the expression itself.
    [[deprecated("quantity is no longer required for unit conversion, use unit directly")]]
    const std::string& quantity() const {
        return quantity_;
    }

    /// set the quantity of the output
    [[deprecated("quantity is no longer required for unit conversion, use unit directly")]]
    void set_quantity(std::string quantity);

    /// unit of the output. If this is an empty string, no unit conversion will
    /// be performed.
    const std::string& unit() const {
        return unit_;
    }
    /// set the unit of the output
    void set_unit(std::string unit);

    /// The setter and getter for `per_atom` that are used in TorchBind, which
    /// allow us to raise an error if `sample_kind` can't be mapped to a boolean
    /// value for `per_atom`.
    void set_per_atom(bool per_atom);
    /// Getter for `per_atom`, see above
    bool get_per_atom() const;

    /// This is deprecated in favor of `sample_kind`, and kept for backward compatibility reasons only.
    [[deprecated("use sample_kind instead")]]
    bool per_atom = false;

    /// Get the sample kind of the output. TODO: explain
    std::string sample_kind() const;

    /// Set the `sample_kind` of the output.
    void set_sample_kind(std::string sample_kind);

    /// Which gradients should be computed eagerly and stored inside the output
    /// `TensorMap`
    std::vector<std::string> explicit_gradients;

    /// Serialize a `ModelOutput` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelOutput` from a JSON string.
    static ModelOutput from_json(std::string_view json);

private:
    void set_per_atom_no_deprecation(bool per_atom);
    bool get_per_atom_no_deprecation() const;

    std::string quantity_;
    std::string unit_;
    torch::optional<std::string> sample_kind_;
};

/// Pick the output for the given `requested_output` from the availabilities of the
/// model's `outputs`, according to the optional `desired_variant`.
METATOMIC_TORCH_EXPORT std::string pick_output(
	std::string requested_output,
	torch::Dict<std::string, ModelOutput> outputs,
	torch::optional<std::string> desired_variant = torch::nullopt
);

namespace details {
    nlohmann::json model_output_to_json(const ModelOutputHolder& self);
}

}

#endif
