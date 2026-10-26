#include <cstring>
#include <iostream>

#include <sstream>
#include <filesystem>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <metatensor/torch.hpp>

#include "metatomic/torch/model.hpp"
#include "metatomic/torch/misc.hpp"
#include "metatomic/torch/quantities.hpp"
#include "metatomic/torch/units.hpp"

#include "./internal/utils.hpp"
#include "./internal/shared_libraries.hpp"

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

/******************************************************************************/

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

ModelOutputHolder::ModelOutputHolder() = default;

ModelOutputHolder::ModelOutputHolder(
    std::string quantity,
    std::string unit,
    std::string sample_kind,
    std::vector<std::string> explicit_gradients_,
    std::string description_
):
    description(std::move(description_)),
    explicit_gradients(std::move(explicit_gradients_))
{
    this->set_quantity(std::move(quantity));
    this->set_unit(std::move(unit));
    this->set_sample_kind(std::move(sample_kind));
}

ModelOutputHolder::ModelOutputHolder(
    std::string quantity,
    std::string unit,
    bool per_atom_,
    std::vector<std::string> explicit_gradients_,
    std::string description_
):
    description(std::move(description_)),
    explicit_gradients(std::move(explicit_gradients_))
{
    this->set_quantity(std::move(quantity));
    this->set_unit(std::move(unit));
    this->set_per_atom(per_atom_);
}

ModelOutputHolder::ModelOutputHolder(
    std::string quantity,
    std::string unit,
    torch::IValue per_atom_or_sample_kind,
    std::vector<std::string> explicit_gradients_,
    std::string description_,
    torch::optional<bool> per_atom,
    torch::optional<std::string> sample_kind
):
    description(std::move(description_)),
    explicit_gradients(std::move(explicit_gradients_))
{
    this->set_quantity(std::move(quantity));
    this->set_unit(std::move(unit));

    if (per_atom_or_sample_kind.isNone()) {
        // check the kwargs for backward compatibility
        if (sample_kind.has_value() && per_atom.has_value()) {
            C10_THROW_ERROR(ValueError, "cannot specify both `per_atom` and `sample_kind`");
        } else if (sample_kind.has_value()) {
            this->set_sample_kind(sample_kind.value());
        } else if (per_atom.has_value()) {
            this->set_per_atom(per_atom.value());
        }
    } else if (per_atom_or_sample_kind.isBool()) {
        if (per_atom.has_value()) {
            C10_THROW_ERROR(ValueError,
                "cannot specify `per_atom` both as a positional and keyword argument"
            );
        }
        this->set_per_atom(per_atom_or_sample_kind.toBool());
    } else if (per_atom_or_sample_kind.isString()) {
        if (sample_kind.has_value()) {
            C10_THROW_ERROR(ValueError,
                "cannot specify `sample_kind` both as a positional and keyword argument"
            );
        }
        this->set_sample_kind(per_atom_or_sample_kind.toStringRef());
    } else {
        C10_THROW_ERROR(ValueError,
            "positional argument for `per_atom`/`sample_kind` must be either a boolean or a string"
        );
    }
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

void ModelOutputHolder::set_quantity(std::string quantity) {
    if (!quantity.empty()) {
        WARN_DEPRECATION_ONCE(
            "ModelOutput.quantity is deprecated and will be removed in a future version"
        );
    }

    if (details::valid_dimension(quantity)) {
        details::validate_unit(quantity, unit_);
    }

    this->quantity_ = std::move(quantity);
}

void ModelOutputHolder::set_unit(std::string unit) {
    // just check that we can parse the unit
    details::validate_unit("", unit);

    this->unit_ = std::move(unit);
}

static nlohmann::json model_output_to_json(const ModelOutputHolder& self) {
    nlohmann::json result;

    result["class"] = "ModelOutput";
    result["unit"] = self.unit();
    result["sample_kind"] = self.sample_kind();
    result["explicit_gradients"] = self.explicit_gradients;
    result["description"] = self.description;

    return result;
}

std::string ModelOutputHolder::to_json() const {
    return model_output_to_json(*this).dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

ModelOutput ModelOutputHolder::from_json(std::string_view json) {
    auto data = nlohmann::json::parse(json);
    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelOutput, expected an object");
    }

    if (!data.contains("class") || !data["class"].is_string()) {
        throw std::runtime_error("expected 'class' in JSON for ModelOutput, did not find it");
    }

    if (data["class"] != "ModelOutput") {
        throw std::runtime_error("'class' in JSON for ModelOutput must be 'ModelOutput'");
    }

    auto result = torch::make_intrusive<ModelOutputHolder>();

    if (data.contains("unit")) {
        if (!data["unit"].is_string()) {
            throw std::runtime_error("'unit' in JSON for ModelOutput must be a string");
        }
        result->set_unit(data["unit"]);
    }

    if (data.contains("sample_kind")) {
        if (!data["sample_kind"].is_string()) {
            throw std::runtime_error("'sample_kind' in JSON for ModelOutput must be a string");
        }
        result->set_sample_kind(data["sample_kind"]);
    } else if (data.contains("per_atom")) {
        if (!data["per_atom"].is_boolean()) {
            throw std::runtime_error("'per_atom' in JSON for ModelOutput must be a boolean");
        }
        result->set_per_atom_no_deprecation(data["per_atom"]);
    } else {
        result->set_sample_kind("system");
    }

    if (data.contains("explicit_gradients")) {
        read_vector_string_json(
            result->explicit_gradients,
            data["explicit_gradients"],
            "'explicit_gradients' in JSON for ModelOutput"
        );
    }

    if (data.contains("description")) {
        if (!data["description"].is_string()) {
            throw std::runtime_error("'description' in JSON for ModelOutput must be a string");
        }
        result->description = data["description"];
    } else {
        // backward compatibility
        result->description = "";
    }

    return result;
}

static std::set<std::string> SUPPORTED_SAMPLE_KINDS = {
    "system",
    "atom",
    "atom_pair",
};

void ModelOutputHolder::set_sample_kind(std::string sample_kind) {
    if (sample_kind == "atom") {
        this->set_per_atom_no_deprecation(true);
    } else if (sample_kind == "system") {
        this->set_per_atom_no_deprecation(false);
    } else {
        if (SUPPORTED_SAMPLE_KINDS.find(sample_kind) == SUPPORTED_SAMPLE_KINDS.end()) {
            C10_THROW_ERROR(ValueError,
                "invalid sample_kind '" + sample_kind + "': supported values are [" +
                torch::str(SUPPORTED_SAMPLE_KINDS) + "]"
            );
        }

        this->sample_kind_ = std::move(sample_kind);
    }
}

std::string ModelOutputHolder::sample_kind() const {
    if (sample_kind_.has_value()) {
        return sample_kind_.value();
    } else if (this->get_per_atom_no_deprecation()) {
        return "atom";
    } else {
        return "system";
    }
}

void ModelOutputHolder::set_per_atom(bool per_atom_) {
    WARN_DEPRECATION_ONCE(
        "`per_atom` is deprecated, please use `sample_kind` instead"
    );

    this->set_per_atom_no_deprecation(per_atom_);
}

bool ModelOutputHolder::get_per_atom() const {
    WARN_DEPRECATION_ONCE(
        "`per_atom` is deprecated, please use `sample_kind` instead"
    );

    return this->get_per_atom_no_deprecation();
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

void ModelOutputHolder::set_per_atom_no_deprecation(bool per_atom) {
    this->per_atom = per_atom;

    this->sample_kind_ = torch::nullopt;
}

bool ModelOutputHolder::get_per_atom_no_deprecation() const {
    if (sample_kind_.has_value()) {
        if (sample_kind_.value() == "atom") {
            return true;
        } else if (sample_kind_.value() == "system") {
            return false;
        } else {
            C10_THROW_ERROR(
                ValueError,
                "Can't infer `per_atom` from `sample_kind` '" + this->sample_kind() + "'. "
                "`per_atom` only makes sense for `sample_kind` 'atom' and 'system'."
            );
        }
    }
    return per_atom;
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

/******************************************************************************/


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
        outputs[it.key()] = model_output_to_json(*it.value());
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

/******************************************************************************/

static void check_selected_atoms(const torch::optional<metatensor_torch::Labels>& selected_atoms) {
    if (selected_atoms) {
        if (selected_atoms.value()->names() != std::vector<std::string>{"system", "atom"}) {
            std::ostringstream oss;
            oss << '[';
            for (const auto& name: selected_atoms.value()->names()) {
                oss << '\'' << name << "', ";
            }
            oss << ']';

            C10_THROW_ERROR(ValueError,
                "invalid `selected_atoms` names: expected ['system', 'atom'], "
                "got " + oss.str()
            );
        }
    }
}

void ModelEvaluationOptionsHolder::set_length_unit(std::string unit) {
    details::validate_unit("length", unit);
    this->length_unit_ = std::move(unit);
}

ModelEvaluationOptionsHolder::ModelEvaluationOptionsHolder(
    std::string length_unit_,
    torch::Dict<std::string, ModelOutput> outputs_,
    torch::optional<metatensor_torch::Labels> selected_atoms
):
    outputs(outputs_),
    selected_atoms_(std::move(selected_atoms))
{
    this->set_length_unit(std::move(length_unit_));
    check_selected_atoms(selected_atoms_);
}


void ModelEvaluationOptionsHolder::set_selected_atoms(torch::optional<metatensor_torch::Labels> selected_atoms) {
    check_selected_atoms(selected_atoms);
    selected_atoms_ = std::move(selected_atoms);
}


std::string ModelEvaluationOptionsHolder::to_json() const {
    nlohmann::json result;

    result["class"] = "ModelEvaluationOptions";
    result["length_unit"] = this->length_unit();

    if (this->selected_atoms_) {
        const auto& selected_atoms = this->selected_atoms_.value();

        auto selected_json = nlohmann::json::object();
        selected_json["names"] = selected_atoms->names();
        auto values = selected_atoms->values().to(torch::kCPU).contiguous();
        auto size = static_cast<size_t>(selected_atoms->size() * selected_atoms->count());
        selected_json["values"] = std::vector<int32_t>(
            values.data_ptr<int32_t>(),
            values.data_ptr<int32_t>() + size
        );

        result["selected_atoms"] = std::move(selected_json);
    } else {
        result["selected_atoms"] = nlohmann::json();
    }

    auto outputs = nlohmann::json::object();
    for (const auto& it: this->outputs) {
        outputs[it.key()] = model_output_to_json(*it.value());
    }
    result["outputs"] = outputs;

    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

ModelEvaluationOptions ModelEvaluationOptionsHolder::from_json(std::string_view json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelEvaluationOptions, expected an object");
    }

    if (!data.contains("class") || !data["class"].is_string()) {
        throw std::runtime_error("expected 'class' in JSON for ModelEvaluationOptions, did not find it");
    }

    if (data["class"] != "ModelEvaluationOptions") {
        throw std::runtime_error("'class' in JSON for ModelEvaluationOptions must be 'ModelEvaluationOptions'");
    }

    auto result = torch::make_intrusive<ModelEvaluationOptionsHolder>();
    if (data.contains("length_unit")) {
        if (!data["length_unit"].is_string()) {
            throw std::runtime_error("'length_unit' in JSON for ModelEvaluationOptions must be a string");
        }
        result->set_length_unit(data["length_unit"]);
    }

    if (data.contains("selected_atoms")) {
        if (data["selected_atoms"].is_null()) {
            // nothing to do
        } else {
            if (!data["selected_atoms"].is_object()) {
                throw std::runtime_error("'selected_atoms' in JSON for ModelEvaluationOptions must be an object");
            }

            if (!data["selected_atoms"].contains("names")) {
                throw std::runtime_error("'selected_atoms.names' in JSON for ModelEvaluationOptions must be an array");
            }

            auto names = std::vector<std::string>();
            read_vector_string_json(
                names,
                data["selected_atoms"]["names"],
                "'selected_atoms.names' in JSON for ModelEvaluationOptions"
            );

            if (!data["selected_atoms"].contains("values")) {
                throw std::runtime_error("'selected_atoms.values' in JSON for ModelEvaluationOptions must be an array");
            }

            auto values = std::vector<int32_t>();
            read_vector_int_json(
                values,
                data["selected_atoms"]["values"],
                "'selected_atoms.values' in JSON for ModelEvaluationOptions"
            );
            assert(values.size() % 2 == 0);

            result->set_selected_atoms(torch::make_intrusive<metatensor_torch::LabelsHolder>(
                std::move(names),
                torch::tensor(values).reshape({-1, 2})
            ));
        }
    }

    if (data.contains("outputs")) {
        if (!data["outputs"].is_object()) {
            throw std::runtime_error("'outputs' in JSON for ModelEvaluationOptions must be an object");
        }

        for (const auto& output: data["outputs"].items()) {
            result->outputs.insert(output.key(), ModelOutputHolder::from_json(output.value().dump()));
        }
    }

    return result;
}

/******************************************************************************/

/// Get a `metatomic::ModelMetadata::Builder` initialized with all the data
/// from `metadata`, to create a modified copy of `metadata`.
static metatomic::ModelMetadata::Builder metadata_builder(const metatomic::ModelMetadata& metadata) {
    auto builder = metatomic::ModelMetadata::builder();
    builder.name(metadata.name())
        .description(metadata.description())
        .authors(metadata.authors())
        .references(metadata.references())
        .extra(metadata.extra());
    return builder;
}

static void validate_authors(const std::vector<std::string>& authors) {
    for (const auto& author: authors) {
        if (author.empty()) {
            C10_THROW_ERROR(ValueError, "author can not be empty string in ModelMetadata");
        }
    }
}

static void validate_references(const std::vector<std::string>& references, const std::string& section) {
    for (const auto& reference: references) {
        if (reference.empty()) {
            C10_THROW_ERROR(ValueError,
                "reference can not be empty string (in '" + section + "' section)"
            );
        }
    }
}

static metatomic::ModelMetadata::References references_from_torch(
    const torch::Dict<std::string, std::vector<std::string>>& references
) {
    auto builder = metatomic::ModelMetadata::References::builder();
    for (const auto& item: references) {
        const auto& section = item.key();
        auto values = item.value();
        validate_references(values, section);

        if (section == "model") {
            builder.model(std::move(values));
        } else if (section == "architecture") {
            builder.architecture(std::move(values));
        } else if (section == "implementation") {
            builder.implementation(std::move(values));
        } else {
            C10_THROW_ERROR(ValueError, "unknown key in references: " + section);
        }
    }
    return builder.build();
}

static std::map<std::string, std::string> extra_from_torch(const torch::Dict<std::string, std::string>& extra) {
    auto result = std::map<std::string, std::string>();
    for (const auto& item: extra) {
        result.emplace(item.key(), item.value());
    }
    return result;
}

ModelMetadataHolder::ModelMetadataHolder():
    metadata_(metatomic::ModelMetadata::builder().build())
{}

ModelMetadataHolder::ModelMetadataHolder(
    std::string name,
    std::string description,
    std::vector<std::string> authors,
    torch::Dict<std::string, std::vector<std::string>> references,
    torch::Dict<std::string, std::string> extra
): ModelMetadataHolder(metatomic::ModelMetadata::builder()
    .name(std::move(name))
    .description(std::move(description))
    .authors(std::move(authors))
    .references(references_from_torch(references))
    .extra(extra_from_torch(extra))
    .build()
) {}

ModelMetadataHolder::ModelMetadataHolder(metatomic::ModelMetadata metadata):
    metadata_(std::move(metadata))
{
    validate_authors(metadata_.authors());
    validate_references(metadata_.references().model(), "model");
    validate_references(metadata_.references().architecture(), "architecture");
    validate_references(metadata_.references().implementation(), "implementation");
}

void ModelMetadataHolder::set_name(std::string name) {
    metadata_ = metadata_builder(metadata_).name(std::move(name)).build();
}

void ModelMetadataHolder::set_description(std::string description) {
    metadata_ = metadata_builder(metadata_).description(std::move(description)).build();
}

void ModelMetadataHolder::set_authors(std::vector<std::string> authors) {
    validate_authors(authors);
    metadata_ = metadata_builder(metadata_).authors(std::move(authors)).build();
}

torch::Dict<std::string, std::vector<std::string>> ModelMetadataHolder::references() const {
    const auto& references = metadata_.references();

    auto result = torch::Dict<std::string, std::vector<std::string>>();
    result.insert("model", references.model());
    result.insert("architecture", references.architecture());
    result.insert("implementation", references.implementation());
    return result;
}

void ModelMetadataHolder::set_references(torch::Dict<std::string, std::vector<std::string>> references) {
    metadata_ = metadata_builder(metadata_)
        .references(references_from_torch(references))
        .build();
}

torch::Dict<std::string, std::string> ModelMetadataHolder::extra() const {
    auto result = torch::Dict<std::string, std::string>();
    for (const auto& [key, value]: metadata_.extra()) {
        result.insert(key, value);
    }
    return result;
}

void ModelMetadataHolder::set_extra(torch::Dict<std::string, std::string> extra) {
    metadata_ = metadata_builder(metadata_).extra(extra_from_torch(extra)).build();
}

std::string ModelMetadataHolder::print() const {
    try {
        return metadata_.print();
    } catch (const metatomic::Error& e) {
        C10_THROW_ERROR(ValueError, e.what());
    }
}

std::string ModelMetadataHolder::to_json() const {
    nlohmann::json result = metadata_;
    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

ModelMetadata ModelMetadataHolder::from_json(std::string_view json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelMetadata, expected an object");
    }

    if (data.contains("class")) {
        // Legacy JSON format, used in metatomic-torch<0.11. All the fields
        // were optional in this format.
        if (!data["class"].is_string() || data["class"] != "ModelMetadata") {
            throw std::runtime_error("'class' in legacy JSON for ModelMetadata must be 'ModelMetadata'");
        }

        data.erase("class");
        data["type"] = "metatomic_model_metadata";

        if (!data.contains("name")) {
            data["name"] = "";
        }

        if (!data.contains("description")) {
            data["description"] = "";
        }

        if (!data.contains("authors")) {
            data["authors"] = nlohmann::json::array();
        }

        if (!data.contains("references")) {
            data["references"] = nlohmann::json::object();
        }

        if (data["references"].is_object()) {
            for (const auto* section: {"model", "architecture", "implementation"}) {
                if (!data["references"].contains(section)) {
                    data["references"][section] = nlohmann::json::array();
                }
            }
        }

        if (!data.contains("extra")) {
            data["extra"] = nlohmann::json::object();
        }
    }

    return torch::make_intrusive<ModelMetadataHolder>(data.get<metatomic::ModelMetadata>());
}

/******************************************************************************/

struct Version {
    Version(std::string version): string(std::move(version)) {
        size_t pos = 0;
        try {
            this->major = std::stoi(this->string, &pos);
        } catch (const std::invalid_argument&) {
            C10_THROW_ERROR(ValueError, "invalid version number: " + this->string);
        }

        if (this->string[pos] != '.' || this->string.size() == pos) {
            C10_THROW_ERROR(ValueError, "invalid version number: " + this->string);
        }

        auto minor_version = this->string.substr(pos + 1);
        try {
            this->minor = std::stoi(minor_version, &pos);
        } catch (const std::invalid_argument&) {
            C10_THROW_ERROR(ValueError, "invalid version number: " + this->string);
        }
    }

    /// Check if two version are compatible. `same_minor` indicates whether two
    /// versions should have the same major AND minor number to be considered
    /// compatible.
    bool is_compatible(const Version& other, bool same_minor = false) const {
        if (this->major != other.major) {
            return false;
        }

        if (this->major == 0) {
            same_minor = true;
        }

        if (same_minor && this->minor != other.minor) {
            return false;
        }

        return true;
    }

    std::string string;
    int major = 0;
    int minor = 0;
};

struct Library {
    std::string name;
    std::string path;
};

void from_json(const nlohmann::json& json, Library& extension) {
    json.at("name").get_to(extension.name);
    json.at("path").get_to(extension.path);
}


/// Convert (ptr, len) tuple to a string
static std::string record_to_string(std::tuple<at::DataPtr, size_t> data) {
    return std::string(
        static_cast<char*>(std::get<0>(data).get()),
        std::get<1>(data)
    );
}


/// Check if a library is already loaded. To handle multiple platforms, this
/// does fuzzy matching on the file name; assuming that the name of the library
/// is the same across platforms.
static bool library_already_loaded(
    const std::vector<std::string>& loaded_libraries,
    const std::string& name
) {
    for (const auto& library: loaded_libraries) {
        auto filename = std::filesystem::path(library).filename().string();
        if (filename.find(name) != std::string::npos) {
            return true;
        }
    }
    return false;
}


/// Load a shared library (either TorchScript extension or dependency of
/// extension) in the process
static void load_library(
    const Library& library,
    c10::optional<std::string> extensions_directory,
    bool is_dependency
) {
    auto candidates = std::vector<std::string>();
    if (library.path[0] == '/') {
        candidates.push_back(library.path);
    }

    if (extensions_directory) {
        candidates.push_back(extensions_directory.value() + "/" + library.path);
    }

    auto loaded = details::load_library(library.name, candidates);

    if (!loaded) {
        std::ostringstream oss;
        oss << "failed to load ";
        if (is_dependency) {
            oss << "extension dependency ";
        } else {
            oss << "TorchScript extension ";
        }
        oss << library.name << ". We tried the following:\n";
        for (const auto& candidate: candidates) {
            oss << " - " << candidate << "\n";
        }
        oss << " - loading " << library.name << " directly by name\n";

        if (getenv("METATOMIC_DEBUG_EXTENSIONS_LOADING") == nullptr) {
            oss << "You can set `METATOMIC_DEBUG_EXTENSIONS_LOADING=1` ";
            oss << "in your environemnt for more information\n";
        }

        TORCH_WARN(oss.str());
    }
}

void metatomic_torch::load_model_extensions(
    std::string path,
    c10::optional<std::string> extensions_directory
) {
    auto reader = caffe2::serialize::PyTorchStreamReader(path);

    if (!reader.hasRecord("extra/metatomic-version")) {
        C10_THROW_ERROR(ValueError,
            "file at '" + path + "' does not contain a metatomic model"
        );
    }

    auto debug = getenv("METATOMIC_DEBUG_EXTENSIONS_LOADING") != nullptr;
    auto loaded_libraries = metatomic_torch::details::get_loaded_libraries();

    std::vector<Library> dependencies = nlohmann::json::parse(record_to_string(
        reader.getRecord("extra/extensions-deps")
    ));
    for (const auto& dep: dependencies) {
        if (!library_already_loaded(loaded_libraries, dep.name)) {
            load_library(dep, extensions_directory, /*is_dependency=*/true);
        } else if (debug) {
            std::cerr << dep.name << " dependency was already loaded" << std::endl;
        }
    }

    std::vector<Library> extensions = nlohmann::json::parse(record_to_string(
        reader.getRecord("extra/extensions")
    ));
    for (const auto& ext: extensions) {
        if (ext.name == "metatensor_torch") {
            continue;
        }

        if (!library_already_loaded(loaded_libraries, ext.name)) {
            load_library(ext, extensions_directory, /*is_dependency=*/false);
        } else if (debug) {
            std::cerr << ext.name << " extension was already loaded" << std::endl;
        }
    }
}

ModelMetadata metatomic_torch::read_model_metadata(std::string path) {
    auto reader = caffe2::serialize::PyTorchStreamReader(path);
    if (!reader.hasRecord("extra/model-metadata")) {
        C10_THROW_ERROR(ValueError,
            "could not find model metadata in file at '" + path +
            "', did you export your model with metatensor-torch >=0.5.4?"
        );
    }

    return ModelMetadataHolder::from_json(
        record_to_string(reader.getRecord("extra/model-metadata"))
    );
}

void metatomic_torch::check_atomistic_model(std::string path) {
    auto reader = caffe2::serialize::PyTorchStreamReader(path);

    if (!reader.hasRecord("extra/metatomic-version")) {
        C10_THROW_ERROR(ValueError,
            "file at '" + path + "' does not contain a metatomic model"
        );
    }

    auto recorded_mta_version = Version(record_to_string(
        reader.getRecord("extra/metatomic-version")
    ));
    auto current_mta_version = Version(metatomic_torch::version());

    if (!current_mta_version.is_compatible(recorded_mta_version)) {
        TORCH_WARN(
            "Current metatomic version (", current_mta_version.string, ") ",
            "is not compatible with the version (", recorded_mta_version.string,
            ") used to export the model at '", path, "'; proceed at your own risk."
        );
    }

    // Check that the extensions loaded while the model was exported are also
    // loaded now. Since the model can be exported from a different machine, or
    // the extensions might change how they organize code, we only try to do
    // fuzzy matching on the file name, and warn if we can not find a match.
    std::vector<Library> extensions = nlohmann::json::parse(record_to_string(
        reader.getRecord("extra/extensions")
    ));

    auto loaded_libraries = metatomic_torch::details::get_loaded_libraries();

    for (const auto& extension: extensions) {
        if (!library_already_loaded(loaded_libraries, extension.name)) {
            TORCH_WARN(
                "The model at '", path, "' was exported with extension '",
                extension.name, "' loaded (from '", extension.path, "'), ",
                "but it does not seem to be currently loaded; proceed at your own risk."
            );
        }
    }
}

metatensor_torch::Module metatomic_torch::load_atomistic_model(
    std::string path,
    c10::optional<std::string> extensions_directory
) {
    load_model_extensions(path, extensions_directory);
    check_atomistic_model(path);

    torch::jit::Module model;
    try {
        model = torch::jit::load(path);
    } catch (const std::exception& e) {
        auto error_str = std::string(e.what());
        if (
            (error_str.find("Unknown type name '__torch__.torch.classes.") != std::string::npos)
            || (error_str.find("Unknown builtin op") != std::string::npos)
        ) {
            std::string extra;
            if (!extensions_directory) {
                extra = "\nMake sure to provide the path to the extensions directory "
                        "if your extensions are not installed system-wide.";
            } else {
                extra = "\nMake sure that all extensions are available in the "
                        "extensions directory you provided.";
            }

            throw std::runtime_error(
                "failed to load the model at '" + path + "': " + error_str + "\n"
                "This is likely due to missing TorchScript extensions." + extra
            );
        } else {
            throw;
        }
    }

    return metatensor_torch::Module(model);
}
