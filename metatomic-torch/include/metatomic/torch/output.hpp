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
void _check_outputs(
    const std::vector<System>& systems,
    const std::unordered_map<std::string, ModelOutput>& requested,
    const std::optional<metatensor_torch::Labels>& selected_atoms,
    const std::unordered_map<std::string, metatensor_torch::TensorMap>& outputs,
    const torch::Dtype& expected_dtype);
}

#endif