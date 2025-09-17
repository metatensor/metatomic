#ifndef METATOMIC_TORCH_MISC_HPP
#define METATOMIC_TORCH_MISC_HPP

#include <string>

#include <torch/types.h>

#include "metatomic/torch/exports.h"

namespace metatomic_torch {

/// Get the runtime version of metatensor-torch as a string
METATOMIC_TORCH_EXPORT std::string version();

/// Select the best device according to the list of `model_devices` from a
/// model, the user-provided `desired_device` and what's available on the 
/// current machine.
METATOMIC_TORCH_EXPORT std::string pick_device(
	std::vector<std::string> model_devices,
	torch::optional<std::string> desired_device = torch::nullopt
);

}

#endif
