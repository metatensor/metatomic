#include <torch/script.h>

#include <algorithm>
#include <cassert>
#include <metatensor/torch.hpp>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

#include "./internal/utils.hpp"
#include "metatomic/torch/exports.h"
#include "metatomic/torch/model.hpp"
#include "metatomic/torch/system.hpp"

namespace metatomic_torch {

const std::array energy_bases = {"energy", "energy_ensemble", "energy_uncertainty"};
const std::array energy_gradients = {"strain", "positions"};

torch::Dtype to_torch_dtype(const caffe2::TypeMeta& meta) {
    return c10::typeMetaToScalarType(meta);
}

std::vector<std::string_view> split(const std::string& s, char delimiter) {
    std::vector<std::string_view> result;
    size_t start = 0;
    size_t end = 0;
    while ((end = s.find(delimiter, start)) != std::string::npos) {
        result.push_back(s.substr(start, end - start));
        start = end + 1;
    }
    result.push_back(s.substr(start));
    return result;
}

std::string join_names(const std::vector<std::string>& names) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < names.size(); i++) {
        oss << "'" << names[i] << "'";
        if (i + 1 < names.size()) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

void _validate_single_block(const std::string& name,
                            const metatensor_torch::TensorMap& value) {
    const auto& valid_label = torch::make_intrusive<metatensor_torch::LabelsHolder>(
        "_", torch::tensor({{0}}));
    if (value->keys() != valid_label) {
        C10_THROW_ERROR(ValueError, "invalid keys for \'" + name +
                                        "\' output: expected `Labels(\'_\', [[0]])`");
    }
}

void _validate_atomic_samples(
    const std::string& name,
    const metatensor_torch::TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request,
    const std::optional<metatensor_torch::Labels>& selected_atoms) {
    const torch::Device& device = value->device();
    const metatensor_torch::TensorBlock& block = value->block_by_id(value, 0);

    // Check if the samples names are as expected based on whether the output is
    // per-atom or global
    std::vector<std::string> expected_samples_names;
    if (request->per_atom) {
        expected_samples_names = {"system", "atom"};
    } else {
        expected_samples_names = {"system"};
    }

    if (block->samples()->names() != expected_samples_names) {
        C10_THROW_ERROR(ValueError, "invalid sample names for \'" + name +
                                        "\' output: expected " +
                                        join_names(expected_samples_names) + ", got " +
                                        join_names(block->samples()->names()));
    }

    // Check if the samples match the systems and selected_atoms
    metatensor_torch::Labels expected_samples;
    if (request->per_atom) {
        std::vector<int64_t> flatten_expected_values;
        for (size_t s; s < systems.size(); s++) {
            for (size_t a; a < systems[s]->size(); a++) {
                flatten_expected_values.push_back(static_cast<int64_t>(s));
                flatten_expected_values.push_back(static_cast<int64_t>(a));
            }
        }
        torch::Tensor expected_values = torch::tensor(
            flatten_expected_values, torch::TensorOptions().device(device));
        expected_values = expected_values.reshape({-1, 2});
        expected_samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            torch::IValue(std::vector<std::string>{"system", "atom"}), expected_values,
            metatensor::assume_unique());
        if (selected_atoms) {
            expected_samples = expected_samples->set_intersection(*selected_atoms);
        }
    } else {
        expected_samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            torch::IValue("system"),
            torch::arange(systems.size(), torch::TensorOptions().device(device)),
            metatensor::assume_unique());
        if (selected_atoms) {
            const auto& selected_systems =
                torch::make_intrusive<metatensor_torch::LabelsHolder>(
                    torch::IValue("system"),
                    std::get<0>(torch::_unique(
                        (*selected_atoms)->column("system").reshape({-1, 1}))),
                    metatensor::assume_unique());
            expected_samples = expected_samples->set_intersection(selected_systems);
        }
    }

    if (expected_samples->set_union(block->samples())->size() !=
        expected_samples->size()) {
        C10_THROW_ERROR(ValueError, "invalid samples entries for '" + name +
                                        "\' output, they do not match the `systems` "
                                        "and `selected_atoms`. Expected samples:\n" +
                                        join_names(expected_samples->names()));
    }
}

void _validate_no_components(const std::string& name,
                             const metatensor_torch::TensorBlock& energy_block) {}

void _check_energy_like(const std::string& name,
                        const metatensor_torch::TensorMap& value,
                        const std::vector<System>& systems,
                        const ModelOutput& request,
                        const std::optional<metatensor_torch::Labels>& selected_atoms) {
    assert(name == "energy" || name == "energy_ensemble" ||
           name == "energy_uncertainty");

    // Ensure the output contains a single block with the expected key
    _validate_single_block(name, value);

    // Check samples values from systems & selected_atoms
    _validate_atomic_samples(name, value, systems, request, selected_atoms);
    const auto& energy_block = value->block_by_id(value, 0);
    const auto& device = value->device();

    // Ensure that the block has no components
    _validate_no_components(name, energy_block);

    // The only difference between energy & energy_ensemble is in the properties
    metatensor_torch::Labels expected_properties;
    std::string message;
    if (name == "energy" || name == "energy_uncertainty") {
        expected_properties = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            "energy", torch::tensor({{0}}, torch::TensorOptions().device(device)));
        message = "`Labels(\'energy\', [[0]])`";
    } else {
        assert(name == "energy_ensemble");
        const auto n_ensemble_members = energy_block->values().size(-1);
        expected_properties = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            "energy",
            torch::arange(n_ensemble_members, torch::TensorOptions().device(device)));
        message = "`Labels(\'energy\', [[0], ..., [n]])`";
    }

    if (energy_block->properties() != expected_properties) {
        C10_THROW_ERROR(ValueError, "invalid properties for \'" + name +
                                        " \' output: expected " + message);
    }

    for (const auto& [parameter, gradient] : energy_block->gradients(energy_block)) {
        if (std::find(energy_gradients.begin(), energy_gradients.end(), parameter) ==
            energy_gradients.end()) {
            C10_THROW_ERROR(ValueError,
                            "invalid graident for \'" + name + "output: " + parameter);
        }
        const auto& xyz =
            torch::tensor({{0}, {1}, {2}}, torch::TensorOptions().device(device));
        // strain gradient checks
        if (parameter == "strain") {
            if (gradient->samples()->names() != std::vector<std::string>{"sample"}) {
                C10_THROW_ERROR(ValueError,
                                "invalid samples for \'" + name +
                                    "\' output \'strain\' gradients: expected the "
                                    "names to be [\'sample\'], got " +
                                    join_names(gradient->samples()->names()));
            }

            if (gradient->components().size() != 2) {
                C10_THROW_ERROR(
                    ValueError,
                    "invalid components for \'" + name +
                        "\' output \'strain\' gradients: expected two components");
            }

            if (gradient->components()[0] !=
                torch::make_intrusive<metatensor_torch::LabelsHolder>("xyz_1", xyz)) {
                C10_THROW_ERROR(
                    ValueError,
                    "invalid components for \'" + name +
                        "\' output \'strain\' gradients: expected Labels('xyz_1', "
                        "[[0], [1], [2]]) for the first component");
            }

            if (gradient->components()[1] !=
                torch::make_intrusive<metatensor_torch::LabelsHolder>("xyz_2", xyz)) {
                C10_THROW_ERROR(
                    ValueError,
                    "invalid components for \'" + name +
                        "\' output \'strain\' gradients: expected Labels('xyz_2', "
                        "[[0], [1], [2]]) for the first component");
            }
        }

        // positions gradient checks
        if (parameter == "positions") {
            if (gradient->samples()->names() !=
                std::vector<std::string>{"sample", "system", "atom"}) {
                C10_THROW_ERROR(
                    ValueError,
                    "invalid samples for \'" + name +
                        "\' output \'strain\' gradients: expected the names to be "
                        "[\'sample\', \'system\', \'atom\'], got " +
                        join_names(gradient->samples()->names()));
            }

            if (gradient->components().size() != 1) {
                C10_THROW_ERROR(
                    ValueError,
                    "invalid components for \'" + name +
                        "\' output \'positions\' gradients: expected one component");
            }

            if (gradient->components()[0] !=
                torch::make_intrusive<metatensor_torch::LabelsHolder>("xyz", xyz)) {
                C10_THROW_ERROR(
                    ValueError,
                    "invalid components for \'" + name +
                        "\' output \'strain\' gradients: expected Labels('xyz', [[0], "
                        "[1], [2]]) for the first component");
            }
        }
    }
}

void _check_features(const metatensor_torch::TensorMap& value,
                     const std::vector<System>& systems,
                     const std::unordered_map<std::string, ModelOutput>& requested,
                     const std::optional<metatensor_torch::Labels>& selected_atoms) {}

void _check_non_conservative_forces(
    const metatensor_torch::TensorMap& value,
    const std::vector<System>& systems,
    const std::unordered_map<std::string, ModelOutput>& requested,
    const std::optional<metatensor_torch::Labels>& selected_atoms) {}

void _check_non_conservative_stress(
    const metatensor_torch::TensorMap& value,
    const std::vector<System>& systems,
    const std::unordered_map<std::string, ModelOutput>& requested,
    const std::optional<metatensor_torch::Labels>& selected_atoms) {}

void _check_positions(const metatensor_torch::TensorMap& value,
                      const std::vector<System>& systems,
                      const std::unordered_map<std::string, ModelOutput>& requested,
                      const std::optional<metatensor_torch::Labels>& selected_atoms) {}

void _check_momenta(const metatensor_torch::TensorMap& value,
                    const std::vector<System>& systems,
                    const std::unordered_map<std::string, ModelOutput>& requested,
                    const std::optional<metatensor_torch::Labels>& selected_atoms) {}

void _check_outputs(
    const std::vector<System>& systems,
    const std::unordered_map<std::string, ModelOutput>& requested,
    const std::optional<metatensor_torch::Labels>& selected_atoms,
    const std::unordered_map<std::string, metatensor_torch::TensorMap>& outputs,
    const torch::Dtype& expected_dtype) {
    for (const auto& [name, output] : outputs) {
        if (requested.find(name) == requested.end()) {
            C10_THROW_ERROR(ValueError, "the model produced an output named '" + name +
                                            "', which was not requested");
        }
        if (output->keys()->count() != 0) {
            const torch::Dtype output_dtype =
                to_torch_dtype(output->block_by_id(output, 0)->values().dtype());
            if (output_dtype != expected_dtype) {
                C10_THROW_ERROR(ValueError, "wrong dtype for the " + name +
                                                " output: "
                                                "the model promised " +
                                                scalar_type_name(expected_dtype) +
                                                ", "
                                                "we got " +
                                                scalar_type_name(output_dtype));
            }
        }
        for (const auto& [name, request] : requested) {
            auto it = outputs.find(name);
            if ( it == outputs.end()) {
                C10_THROW_ERROR(ValueError, "the model did not produce the output '" +
                                                name + "' output, which was requested");
            }
            const auto& value = it->second;
            const std::string_view base = split(name, '/')[0];
            if (std::find(energy_bases.begin(), energy_bases.end(), base) !=
                energy_bases.end()) {
                _check_energy_like(std::string(base), value, systems, request, selected_atoms);
            } else if (base == "features") {
            } else if (base == "non_conservative_forces") {
            } else if (base == "non_conservative_stress") {
            } else if (base == "positions") {
            } else if (base == "momenta") {
            } else if (base == "velocities") {
            } else if (name.find("::") != std::string::npos) {
                // this is a non-standard output, there is nothing to check
            } else {
                C10_THROW_ERROR(ValueError,
                                "Invalid output name: '" + name +
                                    "'. Variants should be of the form "
                                    "'<output>/<variant>'. Non-standard output names "
                                    "should have the form '<domain>::<output>'.");
            }
        }
    }
}
}  // namespace metatomic_torch