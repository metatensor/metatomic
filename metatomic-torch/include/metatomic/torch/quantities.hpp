#ifndef METATOMIC_TORCH_QUANTITIES_HPP
#define METATOMIC_TORCH_QUANTITIES_HPP

#include <torch/script.h>

#include <metatensor/torch.hpp>
#include <string>
#include <vector>

#include "metatomic/torch/exports.h"
#include "metatomic/torch/model.hpp"
#include "metatomic/torch/system.hpp"

namespace metatomic_torch {

/// Check that the inputs/outputs of a model conform to the expected structure
/// for the corresponding standard quantities.
///
/// This function checks conformance with the reference documentation in
/// https://docs.metatensor.org/metatomic/latest/quantities/index.html
void METATOMIC_TORCH_EXPORT check_quantities(
    const std::vector<System>& systems,
    const c10::Dict<std::string, ModelOutput>& requested,
    const torch::optional<metatensor_torch::Labels>& selected_atoms,
    const c10::Dict<std::string, metatensor_torch::TensorMap>& values,
    std::string model_dtype,
    std::string inputs_or_outputs
);

/// Get the expected unit dimension of the given quantity used as model input or
/// output.
///
/// This will return one of the following strings:
/// - an empty string for non-standard outputs
/// - "none" for outputs that should be dimensionless (features, …).
/// - "length" for length-like quantities (positions, …);
/// - "momentum" for momentum-like quantities (momenta, …);
/// - "velocity" for velocity-like quantities (velocities, …);
/// - "mass" for mass-like quantities (masses, …);
/// - "energy" for energy-like quantities (energy, energy_ensemble, energy_uncertainty, …);
/// - "force" for force-like quantities (non_conservative_forces, …);
/// - "pressure" for pressure-like quantities (non_conservative_stress, …);
/// - "charge" for charge-like quantities (charges, …);
/// - "heat_flux" for heat flux-like quantities (heat_flux, …);
METATOMIC_TORCH_EXPORT std::string unit_dimension_for_quantity(const std::string& name);

namespace details {
    /// Validate that the given `name` is valid for a model output/input
    ///
    /// The function returns a tuple with:
    /// - a boolean indicating whether this is a known output/input
    /// - the name of the base output/input (empty if custom)
    ///
    /// This is intentionally not exported with `METATOMIC_TORCH_EXPORT`, and is
    /// only intended for internal use.
    std::tuple<bool, std::string> validate_quantity_name(
        const std::string& name, const std::string& context, bool warn_on_deprecated
    );

    /// Same as `unit_dimension_for_quantity`, but without the deprecation
    /// warning for old quantity names.
    ///
    /// This is intentionally not exported with `METATOMIC_TORCH_EXPORT`, and is
    /// only intended for internal use.
    std::string unit_dimension_for_quantity_no_deprecation(const std::string& name);
}

}

#endif
