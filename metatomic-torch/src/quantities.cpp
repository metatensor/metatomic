#include <cassert>

#include <ranges>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <torch/script.h>
#include <metatensor/torch.hpp>

#include "metatomic/torch/model.hpp"
#include "metatomic/torch/system.hpp"
#include "metatomic/torch/quantities.hpp"

#include "./internal/utils.hpp"

using namespace metatensor_torch;
using namespace metatomic_torch;


static std::unordered_set<std::string> ENERGY_BASES = {"energy", "energy_ensemble", "energy_uncertainty"};
static std::unordered_set<std::string> ENERGY_GRADIENTS = {"strain", "positions"};

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

static std::string create_list(int64_t size) {
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
            "invalid keys for '" + name + "': expected `Labels('_', [[0]])`"
        );
    }
}

/// Validates the sample labels against the expected structure
static void validate_atomic_samples(
    const std::string& name,
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request,
    const torch::optional<Labels>& selected_atoms
) {
    auto tensor_options = torch::TensorOptions().device(value->device());
    TensorBlock block = TensorMapHolder::block_by_id(value, 0);

    // Check if the samples names are as expected based on the sample_kind
    std::vector<std::string> expected_samples_names;
    if (request->sample_kind() == "atom") {
        expected_samples_names = {"system", "atom"};
    } else if (request->sample_kind() == "system") {
        expected_samples_names = {"system"};
    } else if (request->sample_kind() == "atom_pair") {
        expected_samples_names = {
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c"
        };
    } else {
        C10_THROW_ERROR(ValueError,
            "Metatomic does not support validating samples for sample_kind"
            "other than 'system', 'atom' or 'atom_pair' at the moment."
            " Received sample_kind '" + request->sample_kind()
        );
    }

    if (block->samples()->names() != expected_samples_names) {
        C10_THROW_ERROR(ValueError,
            "invalid sample names for '" + name + "': expected " +
            join_names(expected_samples_names) + ", got " +
            join_names(block->samples()->names())
        );
    }

    // Check if the samples match the systems and selected_atoms
    Labels expected_samples;
    if (request->sample_kind() == "atom") {
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
    } else if (request->sample_kind() == "system") {
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
    } else if (request->sample_kind() == "atom_pair") {
        // minimal validation, just that indices are in-bounds
        auto values = block->samples()->values().to(torch::kCPU);
        for (int64_t i = 0; i < values.size(0); i++) {
            auto system_idx = values[i][0].item<int64_t>();
            auto first_atom_idx = values[i][1].item<int64_t>();
            auto second_atom_idx = values[i][2].item<int64_t>();

            if (system_idx < 0 || system_idx >= static_cast<int64_t>(systems.size())) {
                C10_THROW_ERROR(ValueError,
                    "invalid system index in samples for '" + name + "': " +
                    std::to_string(system_idx) + " is out of bounds"
                );
            }
            const auto& system = systems[system_idx];
            if (first_atom_idx < 0 || first_atom_idx >= system->size()) {
                C10_THROW_ERROR(ValueError,
                    "invalid first_atom index in samples for '" + name + "': " +
                    std::to_string(first_atom_idx) + " is out of bounds for system " +
                    std::to_string(system_idx)
                );
            }
            if (second_atom_idx < 0 || second_atom_idx >= system->size()) {
                C10_THROW_ERROR(ValueError,
                    "invalid second_atom index in samples for '" + name + "': " +
                    std::to_string(second_atom_idx) + " is out of bounds for system " +
                    std::to_string(system_idx)
                );
            }
        }
    } else {
        C10_THROW_ERROR(ValueError,
            "got invalid sample_kind '" + request->sample_kind() + "' for '" + name + "'"
        );
    }

    if (expected_samples->set_union(block->samples())->size() != expected_samples->size()) {
        C10_THROW_ERROR(ValueError,
            "invalid samples entries for '" + name + "', they do not "
            "match the `systems` and `selected_atoms`. Expected samples:\n" +
            expected_samples->print(10, 3)
        );
    }
}

static void validate_components(const std::string& name, const std::vector<metatensor_torch::Labels>& components, const std::vector<Labels>& expected_components) {
    if (components.size() != expected_components.size()) {
        if (expected_components.size() == 0) {
            C10_THROW_ERROR(ValueError,
                "invalid components for " + name + ": `components` should be empty"
            );
        }
        C10_THROW_ERROR(ValueError,
            "invalid components for '" + name + "': "
            "expected" + std::to_string(expected_components.size()) + "component(s)"
        );
    }
    for (size_t i = 0; i < expected_components.size(); i++){
        if (*components[i] != *expected_components[i]) {
            auto label_values = expected_components[i]->values();
            std::string expected_labels = (
                "Labels('" + join_names(expected_components[i]->names()) + "', " +
                create_list(label_values.size(-1)) + ")`"
            );
            C10_THROW_ERROR(ValueError,
                "invalid components for '" + name + "': "
                "expected `" + expected_labels + "`"
            );
        }
    }
}

static void validate_properties(const std::string& name, const TensorBlock& block, const Labels& expected_properties) {
    if (*block->properties() != *expected_properties) {
        auto label_values = expected_properties->values();
        std::string expected_labels = (
            "Labels('" + join_names(expected_properties->names()) + "', " +
            create_list(label_values.size(-1)) + ")`"
        );
        C10_THROW_ERROR(ValueError,
            "invalid properties for '" + name + "': expected `" + expected_labels + "`"
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
            "invalid samples for '" + name + "' gradients with respect to '" + parameter + "': "
            "expected the names to be " + join_names(expected_samples_names) + ", got " +
            join_names(gradient->samples()->names())
        );
    }

    validate_components(name + " '" + parameter + "' gradients", gradient->components(), expected_components);
}

static void validate_no_gradients(const std::string& name, const TensorBlock& block) {
    if (block->gradients_list().size() > 0) {
        C10_THROW_ERROR(ValueError,
            "invalid gradients for '" + name + "': "
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
    // Check the metadata of energy-related quantities
    assert(ENERGY_BASES.find(name) != ENERGY_BASES.end());

    // Ensure the value contains a single block with the expected key
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
        if (ENERGY_GRADIENTS.find(parameter) == ENERGY_GRADIENTS.end()) {
            C10_THROW_ERROR(ValueError, "invalid gradient for '" + name + ": " + parameter);
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

/// Check metatdata for the "feature" quantity
static void check_feature(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request,
    const torch::optional<Labels>& selected_atoms
) {
    // Ensure the data contains a single block with the expected key
    validate_single_block("feature", value);

    // Check samples values from systems & selected_atoms
    validate_atomic_samples("feature", value, systems, request, selected_atoms);

    auto feature_block = TensorMapHolder::block_by_id(value, 0);

    // Check that the block has no components
    validate_components("feature", feature_block->components(), {});

    // Should not have any explicit gradients
    validate_no_gradients("feature", feature_block);
}

/// Check metatdata for the "non_conservative_force" quantity
static void check_non_conservative_force(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request,
    const torch::optional<Labels>& selected_atoms
) {
    // Ensure the data contains a single block with the expected key
    validate_single_block("non_conservative_force", value);

    // Check samples values from systems & selected_atoms
    validate_atomic_samples("non_conservative_force", value, systems, request, selected_atoms);

    auto forces_block = TensorMapHolder::block_by_id(value, 0);
    auto tensor_options = torch::TensorOptions().device(value->device());
    std::vector<Labels> expected_components{
        torch::make_intrusive<LabelsHolder>(
            "xyz",
            torch::tensor({{0}, {1}, {2}}, tensor_options)
        )
    };
    validate_components("non_conservative_force", forces_block->components(), expected_components);

    Labels expected_properties;
    if (forces_block->properties()->names()[0] == "non_conservative_forces") {
        TORCH_WARN_ONCE(
            "'non_conservative_forces' TensorMap is using a deprecated property name "
            "'non_conservative_forces'. Please use 'non_conservative_force' (singular) instead."
        )
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "non_conservative_forces",
            torch::tensor({{0}}, tensor_options)
        );
    } else {
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "non_conservative_force",
            torch::tensor({{0}}, tensor_options)
        );
    }
    validate_properties("non_conservative_force", forces_block, expected_properties);

    // Should not have any gradients
    validate_no_gradients("non_conservative_force", forces_block);
}

/// Check metatdata for the "non_conservative_stress" quantity
static void check_non_conservative_stress(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request
) {
    // Ensure the data contains a single block with the expected key
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

    auto expected_properties = torch::make_intrusive<LabelsHolder>(
        "non_conservative_stress",
        torch::tensor({{0}}, tensor_options)
    );
    validate_properties("non_conservative_stress", stress_block, expected_properties);

    // Should not have any gradients
    validate_no_gradients("non_conservative_stress", stress_block);
}

/// Check metatdata for the "position" quantity
static void check_position(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request
) {
    // Ensure the data contains a single block with the expected key
    validate_single_block("position", value);

    // Check samples values from systems
    validate_atomic_samples("position", value, systems, request, torch::nullopt);

    auto tensor_options = torch::TensorOptions().device(value->device());
    auto position_block = TensorMapHolder::block_by_id(value, 0);
    std::vector<Labels> expected_components{
        torch::make_intrusive<LabelsHolder>(
            "xyz",
            torch::tensor({{0}, {1}, {2}}, tensor_options)
        )
    };

    validate_components("position", position_block->components(), expected_components);

    Labels expected_properties;
    if (position_block->properties()->names()[0] == "positions") {
        TORCH_WARN_ONCE(
            "The 'position' TensorMap is using a deprecated property name 'positions'. "
            "Please use 'position' (singular) instead."
        )
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "positions",
            torch::tensor({{0}}, tensor_options)
        );
    } else {
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "position",
            torch::tensor({{0}}, tensor_options)
        );
    }
    validate_properties("position", position_block, expected_properties);

    // Should not have any gradients
    validate_no_gradients("position", position_block);
}

/// Check metatdata for the "momentum" quantity
static void check_momentum(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request
) {
    // Ensure the data contains a single block with the expected key
    validate_single_block("momentum", value);

    // Check samples values from systems
    validate_atomic_samples("momentum", value, systems, request, torch::nullopt);

    auto tensor_options = torch::TensorOptions().device(value->device());
    auto momentum_block = TensorMapHolder::block_by_id(value, 0);
    std::vector<Labels> expected_component {
        torch::make_intrusive<LabelsHolder>(
            "xyz",
            torch::tensor({{0}, {1}, {2}}, tensor_options)
        )
    };
    validate_components("momentum", momentum_block->components(), expected_component);

    Labels expected_properties;
    if (momentum_block->properties()->names()[0] == "momenta") {
        TORCH_WARN_ONCE(
            "The 'momentum' TensorMap is using a deprecated property name 'momenta'. "
            "Please use 'momentum' (singular) instead."
        )
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "momenta",
            torch::tensor({{0}}, tensor_options)
        );
    } else {
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "momentum",
            torch::tensor({{0}}, tensor_options)
        );
    }
    validate_properties("momentum", momentum_block, expected_properties);

    // Should not have any gradients
    validate_no_gradients("momentum", momentum_block);
}

/// Check metatdata for the "mass" quantity
static void check_mass(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request
) {
    // Ensure the data contains a single block with the expected key
    validate_single_block("mass", value);

    // Check samples values from systems
    validate_atomic_samples("mass", value, systems, request, torch::nullopt);

    auto tensor_options = torch::TensorOptions().device(value->device());
    auto mass_block = TensorMapHolder::block_by_id(value, 0);

    // Ensure that the block has no components
    validate_components("mass", mass_block->components(), {});

    Labels expected_properties;
    if (mass_block->properties()->names()[0] == "masses") {
        TORCH_WARN_ONCE(
            "The 'mass' TensorMap is using a deprecated property name 'masses'. "
            "Please use 'mass' (singular) instead."
        )
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "masses",
            torch::tensor({{0}}, tensor_options)
        );
    } else {
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "mass",
            torch::tensor({{0}}, tensor_options)
        );
    }
    validate_properties("mass", mass_block, expected_properties);

    // Should not have any gradients
    validate_no_gradients("mass", mass_block);
}

/// Check metatdata for the "velocity" quantity
static void check_velocity(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request
) {
    // Ensure the data contains a single block with the expected key
    validate_single_block("velocity", value);

    // Check samples values from systems
    validate_atomic_samples("velocity", value, systems, request, torch::nullopt);

    auto tensor_options = torch::TensorOptions().device(value->device());
    auto velocity_block = TensorMapHolder::block_by_id(value, 0);
    std::vector<Labels> expected_component {
        torch::make_intrusive<LabelsHolder>(
            "xyz",
            torch::tensor({{0}, {1}, {2}}, tensor_options)
        )
    };
    validate_components("velocity", velocity_block->components(), expected_component);

    Labels expected_properties;
    if (velocity_block->properties()->names()[0] == "velocities") {
        TORCH_WARN_ONCE(
            "The 'velocity' TensorMap is using a deprecated property name 'velocities'. "
            "Please use 'velocity' (singular) instead."
        )
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "velocities",
            torch::tensor({{0}}, tensor_options)
        );
    } else {
        expected_properties = torch::make_intrusive<LabelsHolder>(
            "velocity",
            torch::tensor({{0}}, tensor_options)
        );
    }

    validate_properties("velocity", velocity_block, expected_properties);

    // Should not have any gradients
    validate_no_gradients("velocity", velocity_block);
}

/// Check metatdata for the "charge" quantity
static void check_charge(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request
) {
    // Ensure the data contains a single block with the expected key
    validate_single_block("charge", value);

    // Check samples values from systems
    validate_atomic_samples("charge", value, systems, request, torch::nullopt);

    auto tensor_options = torch::TensorOptions().device(value->device());
    auto charge_block = TensorMapHolder::block_by_id(value, 0);

    // Ensure that the block has no components
    validate_components("charge", charge_block->components(), {});

    auto expected_properties = torch::make_intrusive<LabelsHolder>(
        "charge",
        torch::tensor({{0}}, tensor_options)
    );
    validate_properties("charge", charge_block, expected_properties);

    // Should not have any gradients
    validate_no_gradients("charge", charge_block);
}

/// Check metatdata for the "heat_flux" quantity
static void check_heat_flux(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request
) {
    // Ensure the data contains a single block with the expected key
    validate_single_block("heat_flux", value);

    // Check samples values from systems
    if (request->sample_kind() != "system") {
        C10_THROW_ERROR(ValueError,
            "invalid 'heat_flux': heat_flux is a per-system quantity, "
            "but the request indicates `sample_kind='" + request->sample_kind() + "'`"
        );
    }
    validate_atomic_samples("heat_flux", value, systems, request, torch::nullopt);

    auto tensor_options = torch::TensorOptions().device(value->device());
    auto heat_flux_block = TensorMapHolder::block_by_id(value, 0);
    std::vector<Labels> expected_component {
        torch::make_intrusive<LabelsHolder>(
            "xyz",
            torch::tensor({{0}, {1}, {2}}, tensor_options)
        )
    };
    validate_components("heat_flux", heat_flux_block->components(), expected_component);

    auto expected_properties = torch::make_intrusive<LabelsHolder>(
        "heat_flux",
        torch::tensor({{0}}, tensor_options)
    );
    validate_properties("heat_flux", heat_flux_block, expected_properties);

    // Should not have any gradients
    validate_no_gradients("heat_flux", heat_flux_block);
}

/// Check metadata for spin_multiplicity (per-system scalar).
static void check_spin_multiplicity(
    const TensorMap& value,
    const std::vector<System>& systems,
    const ModelOutput& request
) {
    validate_single_block("spin_multiplicity", value);

    if (request->sample_kind() != "system") {
        C10_THROW_ERROR(ValueError,
            "invalid 'spin_multiplicity': spin_multiplicity is a per-system quantity, "
            "but the request indicates `sample_kind='" + request->sample_kind() + "'`"
        );
    }
    validate_atomic_samples("spin_multiplicity", value, systems, request, torch::nullopt);

    auto tensor_options = torch::TensorOptions().device(value->device());
    auto spin_block = TensorMapHolder::block_by_id(value, 0);

    validate_components("spin_multiplicity", spin_block->components(), {});

    auto expected_properties = torch::make_intrusive<LabelsHolder>(
        "spin_multiplicity",
        torch::tensor({{0}}, tensor_options)
    );
    validate_properties("spin_multiplicity", spin_block, expected_properties);

    validate_no_gradients("spin_multiplicity", spin_block);
}


static std::unordered_map<std::string, std::string> DEPRECATED_NAMES = {
    {"features", "feature"},
    {"non_conservative_forces", "non_conservative_force"},
    {"positions", "position"},
    {"momenta", "momentum"},
    {"masses", "mass"},
    {"velocities", "velocity"},
    {"charges", "charge"},
};

void metatomic_torch::check_quantities(
    const std::vector<System>& systems,
    const c10::Dict<std::string, ModelOutput>& requested,
    const torch::optional<metatensor_torch::Labels>& selected_atoms,
    const c10::Dict<std::string, metatensor_torch::TensorMap>& values,
    std::string model_dtype,
    std::string inputs_or_outputs
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

    bool checking_inputs = false;
    if (inputs_or_outputs == "inputs") {
        checking_inputs = true;
    } else if (inputs_or_outputs == "outputs") {
        checking_inputs = false;
    } else {
        C10_THROW_ERROR(ValueError,
            "internal error: inputs_or_outputs should be 'inputs' or "
            "'outputs', got '" + inputs_or_outputs + "'"
        );
    }

    for (const auto& item : values) {
        const auto& name = item.key();
        if (name.empty()) {
            if (checking_inputs) {
                C10_THROW_ERROR(ValueError,
                    "the model received an input with an empty name, which is not allowed"
                );
            } else {
                C10_THROW_ERROR(ValueError,
                    "the model produced an output with an empty name, which is not allowed"
                );
            }
        }

        if (!requested.contains(name)) {
            if (checking_inputs) {
                C10_THROW_ERROR(ValueError,
                    "the model received an input named '" + name +"', "
                    "which was not requested by the model"
                );
            } else {
                C10_THROW_ERROR(ValueError,
                    "the model produced an output named '" + name +"', "
                    "which was not requested by the engine"
                );
            }
        }

        const auto& value = item.value();
        if (value->keys()->count() != 0) {
            auto value_dtype = value->scalar_type();
            if (value_dtype != expected_dtype) {
                C10_THROW_ERROR(ValueError,
                    "wrong dtype for '" + name + "': "
                    "the model dtype is " + scalar_type_name(expected_dtype) +
                    " but the data uses " + scalar_type_name(value_dtype)
                );
            }
        }
    }

    for (const auto& item : requested) {
        const auto& name = item.key();
        const auto& request = item.value();
        auto it = values.find(name);
        if (it == values.end()) {
            if (checking_inputs) {
                C10_THROW_ERROR(ValueError,
                    "the model did not receive the '" + name + "' requested input from the engine"
                );
            } else {
            C10_THROW_ERROR(ValueError,
                    "the model did not produce the '" + name + "' output requested by the engine"
                );
            }
        }
        const auto& value = it->value();
        std::string base = split(name, '/')[0];

        auto deprecated_it = DEPRECATED_NAMES.find(base);
        if (deprecated_it != DEPRECATED_NAMES.end()) {
            // no warning here, the code in AtomisticModel is handling that
            base = deprecated_it->second;
        }

        if (ENERGY_BASES.find(base) != ENERGY_BASES.end()) {
            check_energy_like(base, value, systems, request, selected_atoms);
        } else if (base == "feature") {
            check_feature(value, systems, request, selected_atoms);
        } else if (base == "non_conservative_force") {
            check_non_conservative_force(value, systems, request, selected_atoms);
        } else if (base == "non_conservative_stress") {
            check_non_conservative_stress(value, systems, request);
        } else if (base == "position") {
            check_position(value, systems, request);
        } else if (base == "momentum") {
            check_momentum(value, systems, request);
        } else if (base == "mass") {
            check_mass(value, systems, request);
        } else if (base == "velocity") {
            check_velocity(value, systems, request);
        } else if (base == "charge") {
            check_charge(value, systems, request);
        } else if (base == "heat_flux") {
            check_heat_flux(value, systems, request);
        } else if (base == "spin_multiplicity") {
            check_spin_multiplicity(value, systems, request);
        } else if (name.find("::") != std::string::npos) {
            // this is a non-standard quantity, there is nothing to check
        } else {
            C10_THROW_ERROR(ValueError,
                "Invalid quantity name: '" + name + "'. Variants should look like "
                "'<quantity>/<variant>'. Non-standard quantity names should look like "
                "'<domain>::<quantity>[/<variant>]'.");
        }
    }
}


/// Known quantities used as input or output, mapped to the corresponding
/// physical dimension (used to check the unit is valid for this quantity).
inline std::unordered_map<std::string, std::string> KNOWN_QUANTITIES = {
    {"energy", "energy"},
    {"energy_ensemble", "energy"},
    {"energy_uncertainty", "energy"},
    {"feature", "none"},
    {"non_conservative_force", "force"},
    {"non_conservative_stress", "pressure"},
    {"position", "length"},
    {"momentum", "momentum"},
    {"velocity", "velocity"},
    {"mass", "mass"},
    {"charge", "charge"},
    {"spin_multiplicity", "none"},
    {"heat_flux", "heat_flux"},
};


std::tuple<bool, std::string> metatomic_torch::details::validate_quantity_name(
    const std::string& name,
    const std::string& context,
    bool warn_on_deprecated
) {
    if (KNOWN_QUANTITIES.find(name) != KNOWN_QUANTITIES.end()) {
        // known quantity, nothing to do
        return {true, name};
    }

    if (DEPRECATED_NAMES.find(name) != DEPRECATED_NAMES.end()) {
        // deprecated quantity, warn and return
        if (warn_on_deprecated) {
            WARN_DEPRECATION_ONCE(
                "the '" + name + "' quantity is deprecated, please update this "
                "code to use '" + DEPRECATED_NAMES.at(name) + "' instead."
            );
        }
        return {true, name};
    }

    auto error_start = "invalid " + context + " name '" + name + "': ";

    auto double_colon = name.rfind("::");
    if (double_colon != std::string::npos) {
        if (double_colon == 0 || double_colon == (name.length() - 2)) {
            C10_THROW_ERROR(ValueError,
                error_start + "non-standard names should look like "
                "'<domain>::<quantity>' with non-empty domain and quantity."
            );
        }

        auto custom_name = name.substr(0, double_colon);
        auto quantity_name = name.substr(double_colon + 2);

        auto slash = custom_name.find('/');
        if (slash != std::string::npos) {
            // "domain/variant::custom" is not allowed
            C10_THROW_ERROR(ValueError,
                error_start + "non-standard name with variant should look like "
                "'<domain>::<quantity>/<variant>'"
            );
        }

        slash = quantity_name.find('/');
        if (slash != std::string::npos) {
            if (slash == 0 || slash == (name.length() - 1)) {
            C10_THROW_ERROR(ValueError,
                    error_start + "non-standard name with variant should look "
                    "like '<domain>::<quantity>/<variant>' with non-empty domain, "
                    "quantity and variant."
                );
            }
        }

        // this is a custom quantity, nothing more to check
        return {false, ""};
    }

    auto slash = name.find('/');
    if (slash != std::string::npos) {
        if (slash == 0 || slash == (name.length() - 1)) {
            C10_THROW_ERROR(ValueError,
                error_start +  "variant names should look like "
                "'<quantity>/<variant>' with non-empty quantity and variant."
            );
        }

        auto base = name.substr(0, slash);
        auto double_colon = base.rfind("::");
        if (double_colon != std::string::npos) {
            // we don't do anything for custom quantities
            return {false, ""};
        }

        auto deprecated_it = DEPRECATED_NAMES.find(base);
        if (KNOWN_QUANTITIES.find(base) == KNOWN_QUANTITIES.end() && deprecated_it == DEPRECATED_NAMES.end()) {
            C10_THROW_ERROR(ValueError,
                error_start + " '" + base + "' is not a known quantity."
            );
        }

        if (deprecated_it != DEPRECATED_NAMES.end()) {
            if (warn_on_deprecated) {
                WARN_DEPRECATION_ONCE(
                    "the '" + base + "' quantity in '" + name + "' is deprecated, "
                    "please update this code to use '" + deprecated_it->second + "' instead."
                );
            }
        }

        return {true, base};
    }

    C10_THROW_ERROR(ValueError,
        error_start + "this is not a known quantity. "
        "Variant names should look like '<quantity>/<variant>'. "
        "Non-standard names should look like '<domain>::<quantity>[/<variant>]'."
    );
}


std::string metatomic_torch::unit_dimension_for_quantity(const std::string& name) {
    auto [is_known, base_name] = details::validate_quantity_name(name, "quantity", false);

    if (!is_known) {
        return "";
    }

    auto it = DEPRECATED_NAMES.find(base_name);
    if (it != DEPRECATED_NAMES.end()) {
        WARN_DEPRECATION_ONCE(
            "the '" + base_name + "' quantity is deprecated, please update this "
            "code to use '" + it->second + "' instead."
        );
        base_name = it->second;
    }

    return KNOWN_QUANTITIES.at(base_name);
}


std::string metatomic_torch::details::unit_dimension_for_quantity_no_deprecation(const std::string& name) {
    auto [is_known, base_name] = details::validate_quantity_name(name, "quantity", false);

    if (!is_known) {
        return "";
    }

    auto it = DEPRECATED_NAMES.find(base_name);
    if (it != DEPRECATED_NAMES.end()) {
        base_name = it->second;
    }

    return KNOWN_QUANTITIES.at(base_name);
}
