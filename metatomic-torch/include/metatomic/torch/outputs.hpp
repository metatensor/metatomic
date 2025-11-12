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
/// Check that the outputs of a model conform to the expected structure for atomistic
/// models.

/// This function checks conformance with the reference documentation in
/// https://docs.metatensor.org/metatomic/latest/outputs/index.html
void _check_outputs(
    const std::vector<System>& systems,
    const c10::Dict<std::string, ModelOutput>& requested,
    const torch::optional<metatensor_torch::Labels>& selected_atoms,
    const c10::Dict<std::string, metatensor_torch::TensorMap>& outputs,
    const int64_t dtype);
}

#endif