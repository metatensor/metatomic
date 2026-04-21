#ifndef METATOMIC_TORCH_OUTPUT_HPP
#define METATOMIC_TORCH_OUTPUT_HPP

#include <torch/script.h>

#include <metatensor/torch.hpp>
#include <string>
#include <vector>

#include "metatomic/torch/exports.h"
#include "metatomic/torch/model.hpp"
#include "metatomic/torch/system.hpp"

namespace metatomic_torch {

/// Check that the outputs of a model conform to the expected structure for
/// atomistic models.
///
/// This function checks conformance with the reference documentation in
/// https://docs.metatensor.org/metatomic/latest/outputs/index.html
void METATOMIC_TORCH_EXPORT check_outputs(
    const std::vector<System>& systems,
    const c10::Dict<std::string, ModelOutput>& requested,
    const torch::optional<metatensor_torch::Labels>& selected_atoms,
    const c10::Dict<std::string, metatensor_torch::TensorMap>& outputs,
    std::string model_dtype
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
METATOMIC_TORCH_EXPORT std::string unit_dimension_for_quantity(const std::string& output_name);

namespace details {
    /// Validate that the given `name` is valid for a model output/input
    ///
    /// The function returns a tuple with:
    /// - a boolean indicating whether this is a known output/input
    /// - the name of the base output/input (empty if custom)
    /// - the name of the variant (empty if none)
    ///
    /// This is intentionally not exported with `METATOMIC_TORCH_EXPORT`, and is
    /// only intended for internal use.
    std::tuple<bool, std::string, std::string> validate_name_and_check_variant(
        const std::string& name
    );
}

}

#endif
