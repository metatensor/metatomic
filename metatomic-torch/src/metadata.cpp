#include <cstring>
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
