#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <metatensor/torch.hpp>

#include "metatomic/torch/metadata.hpp"


using namespace metatomic_torch;

static metatomic::SampleKind sample_kind_from_string(const std::string& sample_kind) {
    if (sample_kind == "system") {
        return metatomic::SampleKind::System;
    } else if (sample_kind == "atom") {
        return metatomic::SampleKind::Atom;
    } else if (sample_kind == "atom_pair") {
        return metatomic::SampleKind::AtomPair;
    } else {
        C10_THROW_ERROR(ValueError,
            "sample_kind must be one of ['system', 'atom', 'atom_pair'], got " + sample_kind
        );
    }
}

static std::vector<metatomic::Gradients> gradients_from_strings(const std::vector<std::string>& gradients) {
    auto result = std::vector<metatomic::Gradients>();
    result.reserve(gradients.size());
    for (const auto& gradient: gradients) {
        if (gradient == "positions") {
            result.push_back(metatomic::Gradients::Positions);
        } else if (gradient == "strain") {
            result.push_back(metatomic::Gradients::Strain);
        } else {
            C10_THROW_ERROR(ValueError,
                "gradient must be one of ['positions', 'strain'], got " + gradient
            );
        }
    }
    return result;
}

/// Get a `metatomic::Quantity::Builder` initialized with all the data from
/// `quantity`, to create a modified copy of `quantity`.
static metatomic::Quantity::Builder quantity_builder(const metatomic::Quantity& quantity) {
    auto builder = metatomic::Quantity::builder();
    builder.name(quantity.name())
        .unit(quantity.unit())
        .sample_kind(quantity.sample_kind())
        .gradients(quantity.gradients())
        .description(quantity.description());
    return builder;
}

QuantityHolder::QuantityHolder(
    std::string name,
    std::string unit,
    const std::string& sample_kind,
    const torch::optional<std::vector<std::string>>& gradients,
    torch::optional<std::string> description
): QuantityHolder(metatomic::Quantity::builder()
    .name(std::move(name))
    .unit(std::move(unit))
    .sample_kind(sample_kind_from_string(sample_kind))
    .gradients(gradients_from_strings(gradients.value_or(std::vector<std::string>())))
    .description(std::move(description).value_or(""))
    .build()
) {}

QuantityHolder::QuantityHolder(metatomic::Quantity quantity):
    quantity_(std::move(quantity))
{}

void QuantityHolder::set_name(std::string name) {
    quantity_ = quantity_builder(quantity_).name(std::move(name)).build();
}

void QuantityHolder::set_unit(std::string unit) {
    quantity_ = quantity_builder(quantity_).unit(std::move(unit)).build();
}

std::string QuantityHolder::sample_kind() const {
    return nlohmann::json(quantity_.sample_kind()).get<std::string>();
}

void QuantityHolder::set_sample_kind(const std::string& sample_kind) {
    quantity_ = quantity_builder(quantity_)
        .sample_kind(sample_kind_from_string(sample_kind))
        .build();
}

std::vector<std::string> QuantityHolder::gradients() const {
    auto result = std::vector<std::string>();
    for (const auto& gradient: quantity_.gradients()) {
        result.push_back(nlohmann::json(gradient).get<std::string>());
    }
    return result;
}

void QuantityHolder::set_gradients(const std::vector<std::string>& gradients) {
    quantity_ = quantity_builder(quantity_)
        .gradients(gradients_from_strings(gradients))
        .build();
}

torch::optional<std::string> QuantityHolder::description() const {
    if (quantity_.description().empty()) {
        return torch::nullopt;
    }
    return quantity_.description();
}

void QuantityHolder::set_description(torch::optional<std::string> description) {
    quantity_ = quantity_builder(quantity_)
        .description(std::move(description).value_or(""))
        .build();
}

std::string QuantityHolder::repr() const {
    auto result = "Quantity(name='" + this->name() + "', unit='" + this->unit() + "', ";
    result += "sample_kind='" + this->sample_kind() + "', gradients=[";

    const auto gradients = this->gradients();
    for (size_t i = 0; i < gradients.size(); i++) {
        if (i != 0) {
            result += ", ";
        }
        result += "'" + gradients[i] + "'";
    }

    result += "], description=";
    if (quantity_.description().empty()) {
        result += "None";
    } else {
        result += "'" + quantity_.description() + "'";
    }
    result += ")";

    return result;
}

std::string QuantityHolder::to_json() const {
    nlohmann::json result = quantity_;
    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

Quantity QuantityHolder::from_json(std::string_view json) {
    auto data = nlohmann::json::parse(json);
    return torch::make_intrusive<QuantityHolder>(data.get<metatomic::Quantity>());
}

/******************************************************************************/

static std::vector<metatomic::Quantity> outputs_from_torch(const std::vector<Quantity>& outputs) {
    auto result = std::vector<metatomic::Quantity>();
    result.reserve(outputs.size());
    for (const auto& output: outputs) {
        result.push_back(output->as_metatomic());
    }
    return result;
}

static std::vector<int32_t> atomic_types_from_torch(const std::vector<int64_t>& atomic_types) {
    auto result = std::vector<int32_t>();
    result.reserve(atomic_types.size());
    for (auto atomic_type: atomic_types) {
        if (atomic_type < std::numeric_limits<int32_t>::min() || atomic_type > std::numeric_limits<int32_t>::max()) {
            C10_THROW_ERROR(ValueError,
                "atomic types must fit in a 32-bit integer, got " + std::to_string(atomic_type)
            );
        }
        result.push_back(static_cast<int32_t>(atomic_type));
    }
    return result;
}

static std::vector<metatomic::ModelCapabilities::Device> devices_from_strings(const std::vector<std::string>& devices) {
    auto result = std::vector<metatomic::ModelCapabilities::Device>();
    result.reserve(devices.size());
    for (const auto& device: devices) {
        if (device == "cpu") {
            result.push_back(metatomic::ModelCapabilities::Device::CPU);
        } else if (device == "cuda") {
            result.push_back(metatomic::ModelCapabilities::Device::CUDA);
        } else if (device == "rocm" || device == "hip") {
            // "hip" is the name of this device in torch
            result.push_back(metatomic::ModelCapabilities::Device::ROCM);
        } else if (device == "metal" || device == "mps") {
            // "mps" is the name of this device in torch
            result.push_back(metatomic::ModelCapabilities::Device::Metal);
        } else {
            C10_THROW_ERROR(ValueError,
                "device must be one of ['cpu', 'cuda', 'rocm', 'hip', 'metal', 'mps'], got " + device
            );
        }
    }
    return result;
}

static metatomic::ModelCapabilities::DType dtype_from_string(const std::string& dtype) {
    if (dtype == "float32") {
        return metatomic::ModelCapabilities::DType::Float32;
    } else if (dtype == "float64") {
        return metatomic::ModelCapabilities::DType::Float64;
    } else {
        C10_THROW_ERROR(ValueError,
            "dtype must be one of ['float32', 'float64'], got " + dtype
        );
    }
}

/// Get a `metatomic::ModelCapabilities::Builder` initialized with all the data
/// from `capabilities`, to create a modified copy of `capabilities`.
static metatomic::ModelCapabilities::Builder capabilities_builder(const metatomic::ModelCapabilities& capabilities) {
    auto builder = metatomic::ModelCapabilities::builder();
    builder.atomic_types(capabilities.atomic_types())
        .interaction_range(capabilities.interaction_range())
        .length_unit(capabilities.length_unit())
        .supported_devices(capabilities.supported_devices())
        .dtype(capabilities.dtype())
        .outputs(capabilities.outputs());
    return builder;
}

ModelCapabilitiesHolder::ModelCapabilitiesHolder(
    const std::vector<int64_t>& atomic_types,
    double interaction_range,
    std::string length_unit,
    const std::vector<std::string>& supported_devices,
    const std::string& dtype,
    const torch::optional<std::vector<Quantity>>& outputs
): ModelCapabilitiesHolder(metatomic::ModelCapabilities::builder()
    .atomic_types(atomic_types_from_torch(atomic_types))
    .interaction_range(interaction_range)
    .length_unit(std::move(length_unit))
    .supported_devices(devices_from_strings(supported_devices))
    .dtype(dtype_from_string(dtype))
    .outputs(outputs_from_torch(outputs.value_or(std::vector<Quantity>())))
    .build()
) {}

ModelCapabilitiesHolder::ModelCapabilitiesHolder(metatomic::ModelCapabilities capabilities):
    capabilities_(std::move(capabilities))
{}

std::vector<Quantity> ModelCapabilitiesHolder::outputs() const {
    auto result = std::vector<Quantity>();
    result.reserve(capabilities_.outputs().size());
    for (const auto& output: capabilities_.outputs()) {
        result.push_back(torch::make_intrusive<QuantityHolder>(output));
    }
    return result;
}

void ModelCapabilitiesHolder::set_outputs(const std::vector<Quantity>& outputs) {
    capabilities_ = capabilities_builder(capabilities_)
        .outputs(outputs_from_torch(outputs))
        .build();
}

std::vector<int64_t> ModelCapabilitiesHolder::atomic_types() const {
    const auto& atomic_types = capabilities_.atomic_types();
    return std::vector<int64_t>(atomic_types.begin(), atomic_types.end());
}

void ModelCapabilitiesHolder::set_atomic_types(const std::vector<int64_t>& atomic_types) {
    capabilities_ = capabilities_builder(capabilities_)
        .atomic_types(atomic_types_from_torch(atomic_types))
        .build();
}

void ModelCapabilitiesHolder::set_interaction_range(double interaction_range) {
    capabilities_ = capabilities_builder(capabilities_)
        .interaction_range(interaction_range)
        .build();
}

void ModelCapabilitiesHolder::set_length_unit(std::string length_unit) {
    capabilities_ = capabilities_builder(capabilities_)
        .length_unit(std::move(length_unit))
        .build();
}

std::vector<std::string> ModelCapabilitiesHolder::supported_devices() const {
    auto result = std::vector<std::string>();
    for (const auto& device: capabilities_.supported_devices()) {
        result.push_back(nlohmann::json(device).get<std::string>());
    }
    return result;
}

void ModelCapabilitiesHolder::set_supported_devices(const std::vector<std::string>& supported_devices) {
    capabilities_ = capabilities_builder(capabilities_)
        .supported_devices(devices_from_strings(supported_devices))
        .build();
}

std::string ModelCapabilitiesHolder::dtype() const {
    return nlohmann::json(capabilities_.dtype()).get<std::string>();
}

void ModelCapabilitiesHolder::set_dtype(const std::string& dtype) {
    capabilities_ = capabilities_builder(capabilities_)
        .dtype(dtype_from_string(dtype))
        .build();
}

std::string ModelCapabilitiesHolder::repr() const {
    auto result = std::string("ModelCapabilities(outputs=[");

    const auto outputs = this->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        if (i != 0) {
            result += ", ";
        }
        result += outputs[i]->repr();
    }

    result += "], atomic_types=[";
    const auto& atomic_types = capabilities_.atomic_types();
    for (size_t i = 0; i < atomic_types.size(); i++) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(atomic_types[i]);
    }

    // use the shortest representation that round-trips, like Python does
    auto interaction_range = this->interaction_range();
    auto interaction_range_str = std::string();
    if (std::isinf(interaction_range)) {
        interaction_range_str = "inf";
    } else {
        auto buffer = std::array<char, 32>();
        for (int precision = 1; precision <= 17; precision++) {
            std::snprintf(buffer.data(), buffer.size(), "%.*g", precision, interaction_range);
            if (std::strtod(buffer.data(), nullptr) == interaction_range) {
                break;
            }
        }
        interaction_range_str = buffer.data();
        if (interaction_range_str.find_first_of(".e") == std::string::npos) {
            interaction_range_str += ".0";
        }
    }

    result += "], interaction_range=" + interaction_range_str;
    result += ", length_unit='" + this->length_unit() + "', supported_devices=[";

    const auto supported_devices = this->supported_devices();
    for (size_t i = 0; i < supported_devices.size(); i++) {
        if (i != 0) {
            result += ", ";
        }
        result += "'" + supported_devices[i] + "'";
    }

    result += "], dtype='" + this->dtype() + "')";

    return result;
}

std::string ModelCapabilitiesHolder::to_json() const {
    nlohmann::json result = capabilities_;
    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

ModelCapabilities ModelCapabilitiesHolder::from_json(std::string_view json) {
    auto data = nlohmann::json::parse(json);
    return torch::make_intrusive<ModelCapabilitiesHolder>(data.get<metatomic::ModelCapabilities>());
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
