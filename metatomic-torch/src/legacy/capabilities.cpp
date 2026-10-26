#include <cstring>

#include <unordered_map>
#include <vector>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <metatensor/torch.hpp>

#include "metatomic/torch/legacy/capabilities.hpp"
#include "metatomic/torch/legacy/quantities.hpp"
#include "metatomic/torch/misc.hpp"

#include "./units.hpp"

using namespace metatomic_torch;

static void read_vector_string_json(
    std::vector<std::string>& output,
    const nlohmann::json& array,
    const std::string& context
) {
    if (!array.is_array()) {
        throw std::runtime_error(context + " must be an array");
    }
    for (const auto& value: array) {
        if (!value.is_string()) {
            throw std::runtime_error(context + " must be an array of string");
        }
        output.emplace_back(value);
    }
}

template<typename T>
static void read_vector_int_json(
    std::vector<T>& output,
    const nlohmann::json& array,
    const std::string& context
) {
    if (!array.is_array()) {
        throw std::runtime_error(context + " must be an array");
    }
    for (const auto& value: array) {
        if (!value.is_number_integer()) {
            throw std::runtime_error(context + " must be an array of integers");
        }
        output.emplace_back(value);
    }
}


void ModelCapabilitiesHolder::set_outputs(torch::Dict<std::string, ModelOutput> outputs) {
    this->set_outputs(outputs, true);
}

void ModelCapabilitiesHolder::set_outputs(torch::Dict<std::string, ModelOutput> outputs, bool warn_on_deprecated) {
    std::unordered_map<std::string, std::vector<std::string>> variants;
    for (const auto& it: outputs) {
        auto [is_standard, base] = details::validate_quantity_name(it.key(), "model output", warn_on_deprecated);
        if (is_standard) {
            variants[base].emplace_back(it.key());
        };
    }

    for (const auto& kv: variants) {
        const auto& base = kv.first;
        const auto& all_names = kv.second;

        // check descriptions for each variant group
        if (all_names.size() > 1) {
            for (const auto& name : all_names) {
                if (outputs.at(name)->description.empty()) {
                    TORCH_WARN(
                        "'", base, "' defines ", all_names.size(), " output "
                        "variants and '", name, "' has an empty description. ",
                        "Consider adding meaningful descriptions helping users "
                        "to distinguish between them."
                    );
                }
            }
        }

        for (const auto& name : all_names) {
            auto output = outputs.at(name);
            auto unit = output->unit();
            auto dimension = details::unit_dimension_for_quantity_no_deprecation(base);

            if (dimension.empty()) {
                // quantity with unknown dimension, just check if the unit is valid
                details::validate_unit("", unit);
            } else {
                if (dimension != "none" && unit.empty()) {
                    TORCH_WARN(
                        "output '", name, "' has an empty unit. ",
                        "Consider adding a unit to ensure correct unit conversion."
                    );
                } else if (dimension == "none" && !unit.empty()) {
                    TORCH_WARN(
                        "output '", name, "' is dimensionless but has a "
                        "non-empty unit '", unit, "'."
                    );
                } else {
                    details::validate_unit(dimension, unit);
                }
            }
        }
    }

    outputs_ = outputs;
}

void ModelCapabilitiesHolder::set_length_unit(std::string unit) {
    details::validate_unit("length", unit);
    this->length_unit_ = std::move(unit);
}

void ModelCapabilitiesHolder::set_dtype(std::string dtype) {
    if (dtype == "float32" || dtype == "float64") {
        dtype_ = std::move(dtype);
    } else {
        C10_THROW_ERROR(ValueError,
            "`dtype` can be one of ['float32', 'float64'], got '" + dtype + "'"
        );
    }
}

double ModelCapabilitiesHolder::engine_interaction_range(const std::string& engine_length_unit) const {
    return interaction_range * unit_conversion_factor(length_unit_, engine_length_unit);
}

std::string ModelCapabilitiesHolder::to_json() const {
    nlohmann::json result;

    result["class"] = "ModelCapabilities";

    auto outputs = nlohmann::json::object();
    for (const auto& it: this->outputs()) {
        outputs[it.key()] = details::model_output_to_json(*it.value());
    }
    result["outputs"] = outputs;
    result["atomic_types"] = this->atomic_types;

    // Store interaction_range using its binary representation to ensure
    // perfect round-tripping of the data
    static_assert(sizeof(double) == sizeof(int64_t));
    int64_t int_interaction_range = 0;
    std::memcpy(&int_interaction_range, &this->interaction_range, sizeof(double));
    result["interaction_range"] = int_interaction_range;

    result["length_unit"] = this->length_unit();
    result["supported_devices"] = this->supported_devices;
    result["dtype"] = this->dtype();

    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

ModelCapabilities ModelCapabilitiesHolder::from_json(std::string_view json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelCapabilities, expected an object");
    }

    if (!data.contains("class") || !data["class"].is_string()) {
        throw std::runtime_error("expected 'class' in JSON for ModelCapabilities, did not find it");
    }

    if (data["class"] != "ModelCapabilities") {
        throw std::runtime_error("'class' in JSON for ModelCapabilities must be 'ModelCapabilities'");
    }

    auto result = torch::make_intrusive<ModelCapabilitiesHolder>();
    if (data.contains("outputs")) {
        auto outputs = torch::Dict<std::string, ModelOutput>();
        if (!data["outputs"].is_object()) {
            throw std::runtime_error("'outputs' in JSON for ModelCapabilities must be an object");
        }

        for (const auto& output: data["outputs"].items()) {
            outputs.insert(output.key(), ModelOutputHolder::from_json(output.value().dump()));
        }

        // no deprecation warnings when loading from JSON, since (a) the model might
        // have been saved with an older version of the code and (b) AtomisticModel
        // adds deprecated outputs to the model automatically for backward compatibility
        result->set_outputs(outputs, false);
    }

    if (data.contains("atomic_types")) {
        read_vector_int_json(
            result->atomic_types,
            data["atomic_types"],
            "'atomic_types' in JSON for ModelCapabilities"
        );
    }

    if (data.contains("interaction_range")) {
        if (!data["interaction_range"].is_number_integer()) {
            throw std::runtime_error("'interaction_range' in JSON for ModelCapabilities must be a number");
        }

        auto int_interaction_range = data["interaction_range"].get<int64_t>();
        double interaction_range = 0;
        std::memcpy(&interaction_range, &int_interaction_range, sizeof(double));

        result->interaction_range = interaction_range;
    }

    if (data.contains("length_unit")) {
        if (!data["length_unit"].is_string()) {
            throw std::runtime_error("'length_unit' in JSON for ModelCapabilities must be a string");
        }
        result->set_length_unit(data["length_unit"]);
    }

    if (data.contains("supported_devices")) {
        read_vector_string_json(
            result->supported_devices,
            data["supported_devices"],
            "'supported_devices' in JSON for ModelCapabilities"
        );
    }

    if (data.contains("dtype")) {
        if (!data["dtype"].is_string()) {
            throw std::runtime_error("'dtype' in JSON for ModelCapabilities must be a string");
        }
        result->set_dtype(data["dtype"]);
    }

    return result;
}
