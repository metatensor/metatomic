#include <cstring>
#include <vector>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <metatensor/torch.hpp>

#include "metatomic/torch/legacy/output.hpp"

#include "../internal/utils.hpp"

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

    this->quantity_ = std::move(quantity);
}

void ModelOutputHolder::set_unit(std::string unit) {
    this->unit_ = std::move(unit);
}

nlohmann::json metatomic_torch::details::model_output_to_json(const ModelOutputHolder& self) {
    nlohmann::json result;

    result["class"] = "ModelOutput";
    result["unit"] = self.unit();
    result["sample_kind"] = self.sample_kind();
    result["explicit_gradients"] = self.explicit_gradients;
    result["description"] = self.description;

    return result;
}

std::string ModelOutputHolder::to_json() const {
    return details::model_output_to_json(*this).dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
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


std::string metatomic_torch::pick_output(
    std::string requested_output,
    torch::Dict<std::string, ModelOutput> outputs,
    torch::optional<std::string> desired_variant
) {
    std::vector<std::string> matching_keys;
    bool has_exact = false;

    for (const auto& output: outputs) {
        const auto& key = output.key();

        // match either exact `requested_output` or `requested_output/<variant>`
        if (key == requested_output
            || (key.size() > requested_output.size()
                && key.compare(0, requested_output.size(), requested_output) == 0
                && key[requested_output.size()] == '/')) {
            matching_keys.emplace_back(key);

            if (key == requested_output) {
                has_exact = true;
            }
        }
    }

    if (matching_keys.empty()) {
        C10_THROW_ERROR(ValueError,
            "output '" + requested_output + "' not found in outputs"
        );
    }

    if (desired_variant != torch::nullopt) {
        const auto& output = requested_output + "/" + desired_variant.value();
        auto it = std::find(matching_keys.begin(), matching_keys.end(), output);
        if (it != matching_keys.end()) {
            return *it;
        }
        C10_THROW_ERROR(ValueError,
            "variant '" + desired_variant.value() + "' for output '" + requested_output +
            "' not found in outputs"
        );
    } else if (has_exact) {
        return requested_output;
    } else {
        std::ostringstream oss;
        oss << "output '" << requested_output << "' has no default variant and no `desired_variant` was given. Available variants are:";

        size_t maxlen = 0;
        for (const auto& key: matching_keys) {
            maxlen = std::max(key.size(), maxlen);
        }

        for (const auto& key: matching_keys) {
            const auto& description = outputs.at(key)->description;
            std::string padding(maxlen - key.size(), ' ');
            oss << "\n - '" << key << "'" << padding << ": " << description;
        }
        C10_THROW_ERROR(ValueError, oss.str());
    }
}
