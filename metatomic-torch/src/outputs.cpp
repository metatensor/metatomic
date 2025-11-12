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

using namespace metatensor_torch;
using namespace metatomic_torch;

static const std::array ENERGY_BASES = {"energy", "energy_ensemble", "energy_uncertainty"};
static const std::array ENERGY_GRADIENTS = {"strain", "positions"};

// torch::Dtype to_torch_dtype(const caffe2::TypeMeta& meta) {
//     return c10::typeMetaToScalarType(meta);
// }

static std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = 0;
    while ((end = s.find(delimiter, start)) != std::string::npos) {
        result.push_back(s.substr(start, end - start));  // safe
        start = end + 1;
    }
    result.push_back(s.substr(start));
    return result;
}

static std::string join_names(const std::vector<std::string>& names) {
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

/// Ensure the TensorMap has a single block with the expected key
static void _validate_single_block(const std::string& name,
                            const TensorMap& value) {

    const auto valid_label = LabelsHolder::create({"_"}, {{0}});
    const auto incoming_label = value->keys();
    if (*valid_label != *incoming_label) {
        C10_THROW_ERROR(ValueError, "invalid keys for \'" + name +
                                        "\' output: expected " + valid_label->str() +
                                        ", get: " + incoming_label->str());
    }
}

/// Validates the sample labels in the output against the expected structure
static void _validate_atomic_samples(
    const std::string& name,
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request,
    const torch::optional<Labels>& selected_atoms) {

    const torch::Device& device = value->device();
    const TensorBlock& block = TensorMapHolder::block_by_id(value, 0);

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
                                        block->samples()->str());
    }

    // Check if the samples match the systems and selected_atoms
    Labels expected_samples;
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
        expected_samples = torch::make_intrusive<LabelsHolder>(
            torch::IValue(std::vector<std::string>{"system", "atom"}), expected_values,
            metatensor::assume_unique());
        if (selected_atoms) {
            expected_samples = expected_samples->set_intersection(*selected_atoms);
        }
    } else {
        expected_samples = torch::make_intrusive<LabelsHolder>(
            torch::IValue("system"),
            torch::arange(static_cast<int64_t>(systems.size()), torch::TensorOptions().device(device))
                .reshape({-1, 1}),
            metatensor::assume_unique());
        if (selected_atoms) {
            const auto& selected_systems =
                torch::make_intrusive<LabelsHolder>(
                    torch::IValue("system"),
                    std::get<0>(torch::_unique(
                        (*selected_atoms)->column("system"))).reshape({-1, 1}),
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

/// Ensure the block has no components
static void _validate_no_components(const std::string& name,
                             const TensorBlock& block) {
    if (block->components().size() != 0) {
        C10_THROW_ERROR(ValueError, "invalid components for " + name +
                                        " output: components should be empty");
    }
}

static void _check_energy_like(const std::string& name,
                        const TensorMap& value,
                        const std::vector<System>& systems,
                        const ModelOutput& request,
                        const torch::optional<Labels>& selected_atoms) {
    // Check the output metadata of energy-related outputs

    assert(name == "energy" || name == "energy_ensemble" ||
           name == "energy_uncertainty");

    // Ensure the output contains a single block with the expected key
    _validate_single_block(name, value);
    // Check samples values from systems & selected_atoms
    _validate_atomic_samples(name, value, systems, request, selected_atoms);
    const auto& energy_block = TensorMapHolder::block_by_id(value, 0);
    const auto& device = value->device();
    // Ensure that the block has no components
    _validate_no_components(name, energy_block);

    // The only difference between energy & energy_ensemble is in the properties
    Labels expected_properties;
    std::string message;
    if (name == "energy" || name == "energy_uncertainty") {
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "energy", torch::tensor({{0}}, torch::TensorOptions().device(device)));
        message = "`Labels(\'energy\', [[0]])`";
    } else {
        assert(name == "energy_ensemble");
        const auto n_ensemble_members = energy_block->values().size(-1);
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "energy",
            torch::arange(n_ensemble_members, torch::TensorOptions().device(device))
                .reshape({-1, 1}));
        message = "`Labels(\'energy\', [[0], ..., [n]])`";
    }

    if (*energy_block->properties() != *expected_properties) {
        C10_THROW_ERROR(ValueError, "invalid properties for \'" + name +
                                        " \' output: expected " + message);
    }

    for (const auto& [parameter, gradient] : energy_block->gradients(energy_block)) {
        if (std::find(ENERGY_GRADIENTS.begin(), ENERGY_GRADIENTS.end(), parameter) ==
            ENERGY_GRADIENTS.end()) {
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

            if (*gradient->components()[0] !=
                *torch::make_intrusive<LabelsHolder>("xyz_1", xyz)) {
                C10_THROW_ERROR(
                    ValueError,
                    "invalid components for \'" + name +
                        "\' output \'strain\' gradients: expected Labels('xyz_1', "
                        "[[0], [1], [2]]) for the first component");
            }

            if (*gradient->components()[1] !=
                *torch::make_intrusive<LabelsHolder>("xyz_2", xyz)) {
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

            if (*gradient->components()[0] !=
                *torch::make_intrusive<LabelsHolder>("xyz", xyz)) {
                C10_THROW_ERROR(
                    ValueError,
                    "invalid components for \'" + name +
                        "\' output \'strain\' gradients: expected Labels('xyz', [[0], "
                        "[1], [2]]) for the first component");
            }
        }
    }
}

/// Check "features" output metadata. It is standardized with Plumed
/// https://www.plumed.org/doc-master/user-doc/html/_m_e_t_a_t_e_n_s_o_r.html
static void _check_features(const TensorMap& value,
                     const std::vector<System>& systems,
                     const ModelOutput& request,
                     const torch::optional<Labels>& selected_atoms) {
    // Ensure the output contains a single block with the expected key
    _validate_single_block("features", value);

    // Check samples values from systems & selected_atoms
    _validate_atomic_samples("features", value, systems, request, selected_atoms);

    const auto& features_block = TensorMapHolder::block_by_id(value, 0);

    // Check that the block has no components
    _validate_no_components("features", features_block);

    // Should not have any explicit gradients
    // all gradient calculations are done using autograd
    if (features_block->gradients_list().size() > 0) {
        C10_THROW_ERROR(ValueError,
                        "invalid gradients for \'features\' output: it should not have "
                        "any explicit gradients. all gradient calculations should be "
                        "done using autograd");
    }
}

/// Check output metadata for non-conservative forces.
static void _check_non_conservative_forces(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request,
    const torch::optional<Labels>& selected_atoms) {

    // Ensure the output contains a single block with the expected key
    _validate_single_block("non_conservative_forces", value);

    // Check samples values from systems & selected_atoms
    _validate_atomic_samples("non_conservative_forces", value, systems, request,
                             selected_atoms);

    const auto& forces_block = TensorMapHolder::block_by_id(value, 0);

    // Check that the block has correct "Cartesian-form" components
    if (forces_block->components().size() != 1) {
        C10_THROW_ERROR(ValueError,
                        "invalid components for \'non_conservative_forces\' output: "
                        "expected one component");
    }
    const auto& expected_component =
        torch::make_intrusive<LabelsHolder>(
            "xyz", torch::tensor({{0}, {1}, {2}},
                                 torch::TensorOptions().device(value->device())));

    if (*forces_block->components()[0] != *expected_component) {
        C10_THROW_ERROR(
            ValueError,
            "invalid components for \'non_conservative_forces\' output: expected " +
                join_names(expected_component->names()) + ", got " +
                join_names(forces_block->components()[0]->names()));
    }

    // Should not have any gradients
    if (forces_block->gradients_list().size() > 0) {
        C10_THROW_ERROR(ValueError,
                        "invalid gradients for \'non_conservative_forces\' output: "
                        "expected no gradients, found " +
                            join_names(forces_block->gradients_list()));
    }
}

/// Check output metadata for the non-conservative stress.
static void _check_non_conservative_stress(const TensorMap& value,
                                    const std::vector<System>& systems,
                                    const ModelOutput& request) {

    // Ensure the output contains a single block with the expected key
    _validate_single_block("non_conservative_stress", value);

    // Check samples values from systems
    _validate_atomic_samples("non_conservative_stress", value, systems, request,
                             torch::nullopt);

    const auto& stress_block = TensorMapHolder::block_by_id(value, 0);
    const auto& xyz =
        torch::tensor({{0}, {1}, {2}}, torch::TensorOptions().device(value->device()));

    // Check that the block has correct "Cartesian-form" components
    if (stress_block->components().size() != 2) {
        C10_THROW_ERROR(ValueError,
                        "invalid components for \'non_conservative_stress\' output: "
                        "expected two components, got " +
                            std::to_string(stress_block->components().size()));
    }

    if (*stress_block->components()[0] !=
        *torch::make_intrusive<LabelsHolder>("xyz_1", xyz)) {
        C10_THROW_ERROR(ValueError,
                        "invalid components for 'non_conservative_stress' output: "
                        "expected Labels(\'xyz_1\', [[0], [1], [2]]), got " +
                            join_names(stress_block->components()[0]->names()));
    }

    if (*stress_block->components()[1] !=
        *torch::make_intrusive<LabelsHolder>("xyz_2", xyz)) {
        C10_THROW_ERROR(ValueError,
                        "invalid components for 'non_conservative_stress' output: "
                        "expected Labels(\'xyz_1\', [[0], [1], [2]]), got " +
                            join_names(stress_block->components()[1]->names()));
    }

    // Should not have any gradients
    if (stress_block->gradients_list().size() > 0) {
        C10_THROW_ERROR(ValueError,
                        "invalid gradients for \'non_conservative_stress\' output: "
                        "expected no gradients, found " +
                            join_names(stress_block->gradients_list()));
    }
}

/// Check output metadata for positions.
static void _check_positions(const TensorMap& value,
                      const std::vector<System>& systems,
                      const ModelOutput& request) {
    // Ensure the output contains a single block with the expected key
    _validate_single_block("positions", value);

    // Check samples values from systems
    _validate_atomic_samples("positions", value, systems, request, torch::nullopt);

    const auto& positions_block = TensorMapHolder::block_by_id(value, 0);

    // Check that the block has correct "Cartesian-form" components
    if (positions_block->components().size() != 1) {
        C10_THROW_ERROR(ValueError,
                        "invalid components for \'positions\' output: expected one "
                        "component, got " +
                            positions_block->components().size());
    }
    const auto expected_component =
        torch::make_intrusive<LabelsHolder>(
            "xyz", torch::tensor({{0}, {1}, {2}},
                                 torch::TensorOptions().device(value->device())));

    if (*positions_block->components()[0] != *expected_component) {
        C10_THROW_ERROR(ValueError,
                        "invalid components for \'positions\' output: expected " +
                            join_names(expected_component->names()) + ", got " +
                            join_names(positions_block->components()[0]->names()));
    }

    const auto expected_properties =
        torch::make_intrusive<LabelsHolder>(
            "positions",
            torch::tensor({{0}}, torch::TensorOptions().device(value->device())));

    if (*positions_block->properties() != *expected_properties) {
        C10_THROW_ERROR(ValueError,
                        "invalid properties for \'positions\' output: expected "
                        "`Labels(\'positions\', [[0]])`, got " +
                            join_names(positions_block->properties()->names()));
    }

    // Should not have any gradients
    if (positions_block->gradients_list().size() > 0) {
        C10_THROW_ERROR(ValueError,
                        "invalid gradients for \'positions\' output: expected no "
                        "gradients, found " +
                            join_names(positions_block->gradients_list()));
    }
}

/// Check output metadata for momenta.
static void _check_momenta(const TensorMap& value,
                    const std::vector<System>& systems,
                    const ModelOutput& request) {

    // Ensure the output contains a single block with the expected key
    _validate_single_block("momenta", value);

    // Check samples values from systems
    _validate_atomic_samples("momenta", value, systems, request, torch::nullopt);

    const auto& momenta_block = TensorMapHolder::block_by_id(value, 0);

    // Check that the block has correct "Cartesian-form" components
    if (momenta_block->components().size() != 1) {
        C10_THROW_ERROR(
            ValueError,
            "invalid components for \'momenta\' output: expected one component, got " +
                momenta_block->components().size());
    }
    const auto expected_component =
        torch::make_intrusive<LabelsHolder>(
            "xyz", torch::tensor({{0}, {1}, {2}},
                                 torch::TensorOptions().device(value->device())));

    if (*momenta_block->components()[0] != *expected_component) {
        C10_THROW_ERROR(ValueError,
                        "invalid components for \'momenta\' output: expected " +
                            join_names(expected_component->names()) + ", got " +
                            join_names(momenta_block->components()[0]->names()));
    }

    const auto expected_properties =
        torch::make_intrusive<LabelsHolder>(
            "momenta",
            torch::tensor({{0}}, torch::TensorOptions().device(value->device())));

    if (*momenta_block->properties() != *expected_properties) {
        C10_THROW_ERROR(ValueError,
                        "invalid properties for \'momenta\' output: expected "
                        "`Labels(\'momenta\', [[0]])`, got " +
                            join_names(momenta_block->properties()->names()));
    }

    // Should not have any gradients
    if (momenta_block->gradients_list().size() > 0) {
        C10_THROW_ERROR(
            ValueError,
            "invalid gradients for \'momenta\' output: expected no gradients, found " +
                join_names(momenta_block->gradients_list()));
    }
}

void _check_outputs(const std::vector<System>& systems,
                    const c10::Dict<std::string, ModelOutput>& requested,
                    const torch::optional<Labels>& selected_atoms,
                    const c10::Dict<std::string, TensorMap>& outputs,
                    const int64_t dtype) {
    const auto expected_dtype = static_cast<torch::ScalarType>(dtype);
    for (const auto& item : outputs) {
        const auto& name = item.key();
        const auto& output = item.value();
        if (requested.find(name) == requested.end()) {
            C10_THROW_ERROR(ValueError, "the model produced an output named '" + name +
                                            "', which was not requested");
        }
        if (output->keys()->count() != 0) {
            const torch::Dtype output_dtype =
                c10::typeMetaToScalarType(TensorMapHolder::block_by_id(output, 0)->values().dtype());
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
    }
    for (const auto& item : requested) {
        const auto& name = item.key();
        const auto& request = item.value();
        auto it = outputs.find(name);
        if (it == outputs.end()) {
            C10_THROW_ERROR(ValueError, "the model did not produce the '" +
                                            name + "' output, which was requested");
        }
        const auto& value = it->value();
        const std::string base = split(name, '/')[0];
        if (std::find(ENERGY_BASES.begin(), ENERGY_BASES.end(), base) !=
            ENERGY_BASES.end()) {
            _check_energy_like(base, value, systems, request, selected_atoms);
        } else if (base == "features") {
            _check_features(value, systems, request, selected_atoms);
        } else if (base == "non_conservative_forces") {
            _check_non_conservative_forces(value, systems, request, selected_atoms);
        } else if (base == "non_conservative_stress") {
            _check_non_conservative_stress(value, systems, request);
        } else if (base == "positions") {
            _check_positions(value, systems, request);
        } else if (base == "momenta") {
            _check_momenta(value, systems, request);
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