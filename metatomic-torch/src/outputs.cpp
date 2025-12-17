#include <algorithm>
#include <cassert>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

#include <torch/script.h>
#include <metatensor/torch.hpp>

#include "metatomic/torch/model.hpp"
#include "metatomic/torch/system.hpp"
#include "metatomic/torch/outputs.hpp"

#include "./internal/utils.hpp"

using namespace metatensor_torch;
using namespace metatomic_torch;


static std::array ENERGY_BASES = {"energy", "energy_ensemble", "energy_uncertainty"};
static std::array ENERGY_GRADIENTS = {"strain", "positions"};

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

static std::string create_list(const int32_t size) {
    std::ostringstream oss;
    oss << "[";
    if (size > 3) {
        oss << "[0], ..., [n]";
    } else {
        for (int32_t i = 0; i < size; i++) {
            oss << "[" << i << "]";
            if (i + 1 < size) {
                oss << ", ";
            }
        }
    }
    oss << "]";
    return oss.str();
}

/// Ensure the TensorMap has a single block with the expected key
static void validate_single_block(const std::string& name, const TensorMap& value) {
    auto expected_label = LabelsHolder::create({"_"}, {{0}});
    if (*value->keys() != *expected_label) {
        C10_THROW_ERROR(ValueError,
            "invalid keys for '" + name + "' output: expected `Labels('_', [[0]])`"
        );
    }
}

/// Validates the sample labels in the output against the expected structure
static void validate_atomic_samples(
    const std::string& name,
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request,
    const torch::optional<Labels>& selected_atoms
) {
    auto tensor_options = torch::TensorOptions().device(value->device());
    TensorBlock block = TensorMapHolder::block_by_id(value, 0);

    // Check if the samples names are as expected based on whether the output is
    // per-atom or global
    std::vector<std::string> expected_samples_names;
    if (request->per_atom) {
        expected_samples_names = {"system", "atom"};
    } else {
        expected_samples_names = {"system"};
    }

    if (block->samples()->names() != expected_samples_names) {
        C10_THROW_ERROR(ValueError,
            "invalid sample names for '" + name + "' output: expected " +
            join_names(expected_samples_names) + ", got " +
            join_names(block->samples()->names())
        );
    }

    // Check if the samples match the systems and selected_atoms
    Labels expected_samples;
    if (request->per_atom) {
        std::vector<int64_t> expected_values_flat;
        for (size_t s; s < systems.size(); s++) {
            for (size_t a; a < systems[s]->size(); a++) {
                expected_values_flat.push_back(static_cast<int64_t>(s));
                expected_values_flat.push_back(static_cast<int64_t>(a));
            }
        }

        auto expected_values = torch::tensor(expected_values_flat, tensor_options);
        expected_samples = torch::make_intrusive<LabelsHolder>(
            std::vector<std::string>{"system", "atom"},
            expected_values.reshape({-1, 2}),
            metatensor::assume_unique{}
        );

        if (selected_atoms) {
            expected_samples = expected_samples->set_intersection(selected_atoms.value());
        }
    } else {
        expected_samples = torch::make_intrusive<LabelsHolder>(
            "system",
            torch::arange(static_cast<int64_t>(systems.size()), tensor_options).reshape({-1, 1}),
            metatensor::assume_unique{}
        );

        if (selected_atoms) {
            auto systems = selected_atoms.value()->column("system");
            auto selected_systems = torch::make_intrusive<LabelsHolder>(
                "system",
                std::get<0>(torch::_unique(systems)).reshape({-1, 1}),
                metatensor::assume_unique{}
            );
            expected_samples = expected_samples->set_intersection(selected_systems);
        }
    }

    if (expected_samples->set_union(block->samples())->size() != expected_samples->size()) {
        C10_THROW_ERROR(ValueError,
            "invalid samples entries for '" + name + "' output, they do not "
            "match the `systems` and `selected_atoms`. Expected samples:\n" +
            expected_samples->print(10, 3)
        );
    }
}

static void validate_components(const std::string& name, const std::vector<metatensor_torch::Labels>& components, const std::vector<Labels>& expected_components) {
    if (components.size() != expected_components.size()) {
        if (expected_components.size() == 0) {
            C10_THROW_ERROR(ValueError,
                "invalid components for " + name + " output: `components` should be empty"
            );
        }
        C10_THROW_ERROR(ValueError,
            "invalid components for '" + name + "' output: "
            "expected" + std::to_string(expected_components.size()) + "component(s)"
        );
    }
    for (size_t i = 0; i < expected_components.size(); i++){
        if (*components[i] != *expected_components[i]) {
            auto label_values = expected_components[i]->values();
            std::string expected_labels = "Labels('" + join_names(expected_components[i]->names()) + "', " + create_list(label_values.size(-1)) + ")`";
            C10_THROW_ERROR(ValueError,
                "invalid components for '" + name + "' output: "
                "expected `" + expected_labels + "`"
            );
        }
    }
}

static void validate_properties(const std::string& name, const TensorBlock& block, const Labels& expected_properties) {
    if (*block->properties() != *expected_properties) {
        auto label_values = expected_properties->values();
        std::string expected_labels = "Labels('" + join_names(expected_properties->names()) + "', " + create_list(label_values.size(-1)) + ")`";
        C10_THROW_ERROR(ValueError,
            "invalid properties for '" + name + "' output: "
            "expected `" + expected_labels + "`"
        );
    }
}

static void validate_gradient(
    const std::string& name,
    const std::string& parameter,
    const TensorBlock& gradient,
    const std::vector<std::string>& expected_samples_names,
    const std::vector<metatensor_torch::Labels>& expected_components
) {

    if (gradient->samples()->names() != expected_samples_names) {
        C10_THROW_ERROR(ValueError,
            "invalid samples for '" + name + "' output '" + parameter + "' gradients: "
            "expected the names to be ['sample'], got " +
            join_names(gradient->samples()->names())
        );
    }

    validate_components(name + " '" + parameter + "' gradients", gradient->components(), expected_components);
}

static void validate_no_gradients(const std::string& name, const TensorBlock& block) {
    if (block->gradients_list().size() > 0) {
        C10_THROW_ERROR(ValueError,
            "invalid gradients for '" + name + "' output: "
            "expected no gradients, found " + join_names(block->gradients_list())
        );
    }
}

static void check_energy_like(
    const std::string& name,
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request,
    const torch::optional<Labels>& selected_atoms
) {
    // Check the output metadata of energy-related outputs
    assert(std::find(ENERGY_BASES.begin(), ENERGY_BASES.end(), name) != ENERGY_BASES.end());

    // Ensure the output contains a single block with the expected key
    validate_single_block(name, value);
    // Check samples values from systems & selected_atoms
    validate_atomic_samples(name, value, systems, request, selected_atoms);
    auto energy_block = TensorMapHolder::block_by_id(value, 0);
    auto tensor_options = torch::TensorOptions().device(value->device());
    // Ensure that the block has no components
    validate_components(name, energy_block->components(), {});

    // The only difference between energy & energy_ensemble is in the properties
    Labels expected_properties;
    if (name == "energy" || name == "energy_uncertainty") {
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "energy",
            torch::tensor({{0}}, tensor_options)
        );
    } else {
        assert(name == "energy_ensemble");
        const auto n_ensemble_members = energy_block->values().size(-1);
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "energy",
            torch::arange(n_ensemble_members, tensor_options).reshape({-1, 1})
        );
    }
    validate_properties(name, energy_block, expected_properties);

    auto gradients = TensorBlockHolder::gradients(energy_block);
    for (const auto& [parameter, gradient]: gradients) {
        if (std::find(ENERGY_GRADIENTS.begin(), ENERGY_GRADIENTS.end(), parameter) == ENERGY_GRADIENTS.end()) {
            C10_THROW_ERROR(ValueError, "invalid gradient for '" + name + "output: " + parameter);
        }
        auto xyz = torch::tensor({{0}, {1}, {2}}, tensor_options);
        // strain gradient checks
        if (parameter == "strain") {
            const std::vector<std::string> expected_samples_names{"sample"};
            std::vector<Labels> expected_components{
                torch::make_intrusive<LabelsHolder>("xyz_1", xyz),
                torch::make_intrusive<LabelsHolder>("xyz_2", xyz)
            };

            validate_gradient(name, parameter, gradient, expected_samples_names, expected_components);
        }

        // positions gradient checks
        if (parameter == "positions") {
            const std::vector<std::string> expected_samples_names{"sample", "system", "atom"};
            std::vector<Labels> expected_components{
                torch::make_intrusive<LabelsHolder>("xyz", xyz)
            };

            validate_gradient(name, parameter, gradient, expected_samples_names, expected_components);
        }
    }
}

/// Check "features" output metadata.
static void check_features(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request,
    const torch::optional<Labels>& selected_atoms
) {
    // Ensure the output contains a single block with the expected key
    validate_single_block("features", value);

    // Check samples values from systems & selected_atoms
    validate_atomic_samples("features", value, systems, request, selected_atoms);

    auto features_block = TensorMapHolder::block_by_id(value, 0);

    // Check that the block has no components
    validate_components("features", features_block->components(), {});

    // Should not have any explicit gradients
    validate_no_gradients("features", features_block);
}

/// Check output metadata for non-conservative forces.
static void check_non_conservative_forces(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request,
    const torch::optional<Labels>& selected_atoms
) {
    // Ensure the output contains a single block with the expected key
    validate_single_block("non_conservative_forces", value);

    // Check samples values from systems & selected_atoms
    validate_atomic_samples("non_conservative_forces", value, systems, request, selected_atoms);
    
    auto forces_block = TensorMapHolder::block_by_id(value, 0);
    auto tensor_options = torch::TensorOptions().device(value->device());
    std::vector<Labels> expected_components{
        torch::make_intrusive<LabelsHolder>(
            "xyz",
            torch::tensor({{0}, {1}, {2}}, tensor_options)
        )
    };

    validate_components("non_conservative_forces", forces_block->components(), expected_components);
    
    // Should not have any gradients
    validate_no_gradients("non_conservative_forces", forces_block);
}

/// Check output metadata for the non-conservative stress.
static void check_non_conservative_stress(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request
) {
    // Ensure the output contains a single block with the expected key
    validate_single_block("non_conservative_stress", value);

    // Check samples values from systems
    validate_atomic_samples("non_conservative_stress", value, systems, request, torch::nullopt);

    auto stress_block = TensorMapHolder::block_by_id(value, 0);
    auto tensor_options = torch::TensorOptions().device(value->device());
    auto xyz = torch::tensor({{0}, {1}, {2}}, tensor_options);
    std::vector<Labels> expected_components{
        torch::make_intrusive<LabelsHolder>("xyz_1", xyz),
        torch::make_intrusive<LabelsHolder>("xyz_2", xyz)
    };
    
    validate_components("non_conservative_stress", stress_block->components(), expected_components);
    
    // Should not have any gradients
    validate_no_gradients("non_conservative_stress", stress_block);
}

/// Check output metadata for positions.
static void check_positions(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request
) {
    // Ensure the output contains a single block with the expected key
    validate_single_block("positions", value);

    // Check samples values from systems
    validate_atomic_samples("positions", value, systems, request, torch::nullopt);
    
    auto tensor_options = torch::TensorOptions().device(value->device());
    auto positions_block = TensorMapHolder::block_by_id(value, 0);
    std::vector<Labels> expected_components{
        torch::make_intrusive<LabelsHolder>(
            "xyz",
            torch::tensor({{0}, {1}, {2}}, tensor_options)
        )
    };
    
    validate_components("positions", positions_block->components(), expected_components);

    auto expected_properties = torch::make_intrusive<LabelsHolder>(
        "positions",
        torch::tensor({{0}}, tensor_options)
    );
    validate_properties("positions", positions_block, expected_properties);

    // Should not have any gradients
    validate_no_gradients("positions", positions_block);
}

/// Check output metadata for momenta.
static void check_momenta(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request
) {
    // Ensure the output contains a single block with the expected key
    validate_single_block("momenta", value);

    // Check samples values from systems
    validate_atomic_samples("momenta", value, systems, request, torch::nullopt);

    
    auto tensor_options = torch::TensorOptions().device(value->device());
    auto momenta_block = TensorMapHolder::block_by_id(value, 0);
    std::vector<Labels> expected_component {
        torch::make_intrusive<LabelsHolder>(
            "xyz",
            torch::tensor({{0}, {1}, {2}}, tensor_options)
        )
    };
    validate_components("momenta", momenta_block->components(), expected_component);

    auto expected_properties = torch::make_intrusive<LabelsHolder>(
        "momenta",
        torch::tensor({{0}}, tensor_options)
    );
    validate_properties("momenta", momenta_block, expected_properties);

    // Should not have any gradients
    validate_no_gradients("momenta", momenta_block);
}

void metatomic_torch::check_outputs(
    const std::vector<System>& systems,
    const c10::Dict<std::string, ModelOutput>& requested,
    const torch::optional<metatensor_torch::Labels>& selected_atoms,
    const c10::Dict<std::string, metatensor_torch::TensorMap>& outputs,
    std::string model_dtype
) {
    torch::ScalarType expected_dtype;
    if (model_dtype == "float32") {
        expected_dtype = torch::kFloat32;
    } else if (model_dtype == "float64") {
        expected_dtype = torch::kFloat64;
    } else {
        C10_THROW_ERROR(ValueError,
            "invalid model dtype: expected 'float32' or 'float64', got '" +
            model_dtype + "'"
        );
    }

    for (const auto& item : outputs) {
        const auto& name = item.key();
        const auto& output = item.value();
        if (!requested.contains(name)) {
            C10_THROW_ERROR(ValueError,
                "the model produced an output named '" + name +"', "
                "which was not requested"
            );
        }

        if (output->keys()->count() != 0) {
            auto output_dtype = output->scalar_type();
            if (output_dtype != expected_dtype) {
                C10_THROW_ERROR(ValueError,
                    "wrong dtype for the " + name + " output: "
                    "the model promised " + scalar_type_name(expected_dtype) + ", "
                    "we got " + scalar_type_name(output_dtype)
                );
            }
        }
    }

    for (const auto& item : requested) {
        const auto& name = item.key();
        const auto& request = item.value();
        auto output = outputs.find(name);
        if (output == outputs.end()) {
            C10_THROW_ERROR(ValueError,
                "the model did not produce the '" + name + "' output, which was requested"
            );
        }
        const auto& value = output->value();
        const std::string base = split(name, '/')[0];
        if (std::find(ENERGY_BASES.begin(), ENERGY_BASES.end(), base) != ENERGY_BASES.end()) {
            check_energy_like(base, value, systems, request, selected_atoms);
        } else if (base == "features") {
            check_features(value, systems, request, selected_atoms);
        } else if (base == "non_conservative_forces") {
            check_non_conservative_forces(value, systems, request, selected_atoms);
        } else if (base == "non_conservative_stress") {
            check_non_conservative_stress(value, systems, request);
        } else if (base == "positions") {
            check_positions(value, systems, request);
        } else if (base == "momenta") {
            check_momenta(value, systems, request);
        } else if (name.find("::") != std::string::npos) {
            // this is a non-standard output, there is nothing to check
        } else {
            C10_THROW_ERROR(ValueError,
                "Invalid output name: '" + name + "'. Variants should be of the form "
                "'<output>/<variant>'. Non-standard output names should have the form "
                "'<domain>::<output>'.");
        }
    }
}
