#pragma once

#include <cstdint>
#include <cstring>

#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace metatomic {

class PairListOptions final {
public:
    PairListOptions() = default;

    PairListOptions(
        double cutoff_value,
        bool full_list_value,
        bool strict_value,
        std::vector<std::string> requestors_list = {}
    ):
        cutoff(cutoff_value),
        full_list(full_list_value),
        strict(strict_value),
        requestors(std::move(requestors_list))
    {}

    double cutoff = 0.0;
    bool full_list = false;
    bool strict = false;
    std::vector<std::string> requestors;

    std::string to_json() const;
    static PairListOptions from_json(const std::string& json);
};

class ModelMetadata final {
public:
    ModelMetadata() = default;

    ModelMetadata(
        std::string model_name,
        std::string model_description,
        std::vector<std::string> model_authors,
        std::map<std::string, std::vector<std::string>> model_references = {},
        std::map<std::string, std::string> extra_metadata = {}
    ):
        name(std::move(model_name)),
        description(std::move(model_description)),
        authors(std::move(model_authors)),
        references(std::move(model_references)),
        extra(std::move(extra_metadata))
    {}

    std::string name;
    std::string description;
    std::vector<std::string> authors;
    std::map<std::string, std::vector<std::string>> references;
    std::map<std::string, std::string> extra;

    std::string to_json() const;
    static ModelMetadata from_json(const std::string& json);
};

class Quantity final {
public:
    Quantity() = default;

    Quantity(
        std::string quantity_name,
        std::string quantity_unit,
        std::vector<std::string> quantity_gradients,
        std::string quantity_sample_kind
    ):
        name(std::move(quantity_name)),
        unit(std::move(quantity_unit)),
        gradients(std::move(quantity_gradients)),
        sample_kind(std::move(quantity_sample_kind))
    {}

    std::string name;
    std::string unit;
    std::vector<std::string> gradients;
    std::string sample_kind;

    std::string to_json() const;
    static Quantity from_json(const std::string& json);
};

namespace details {
    inline std::string double_to_hex(double value) {
        uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "unexpected double size");
        std::memcpy(&bits, &value, sizeof(bits));

        auto stream = std::ostringstream();
        stream << "0x" << std::hex << bits;
        return stream.str();
    }

    inline double hex_to_double(const std::string& value) {
        auto bits = uint64_t(0);
        auto stream = std::istringstream(value);
        if (value.rfind("0x", 0) == 0 || value.rfind("0X", 0) == 0) {
            stream.seekg(2);
        }
        stream >> std::hex >> bits;

        auto result = 0.0;
        static_assert(sizeof(bits) == sizeof(result), "unexpected double size");
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
} // namespace details

inline std::string PairListOptions::to_json() const {
    return nlohmann::json{
        {"type", "metatomic_pair_options"},
        {"cutoff", details::double_to_hex(cutoff)},
        {"full_list", full_list},
        {"strict", strict},
        {"requestors", requestors},
    }.dump();
}

inline PairListOptions PairListOptions::from_json(const std::string& string) {
    auto json = nlohmann::json::parse(string);
    auto options = PairListOptions();

    options.cutoff = details::hex_to_double(json.at("cutoff").get<std::string>());
    options.full_list = json.value("full_list", false);
    options.strict = json.value("strict", false);
    options.requestors = json.value("requestors", std::vector<std::string>{});

    return options;
}

inline std::string ModelMetadata::to_json() const {
    return nlohmann::json{
        {"type", "metatomic_model_metadata"},
        {"name", name},
        {"description", description},
        {"authors", authors},
        {"references", references},
        {"extra", extra},
    }.dump();
}

inline ModelMetadata ModelMetadata::from_json(const std::string& string) {
    auto json = nlohmann::json::parse(string);
    auto metadata = ModelMetadata();

    metadata.name = json.value("name", "");
    metadata.description = json.value("description", "");
    metadata.authors = json.value("authors", std::vector<std::string>{});
    metadata.references = json.value("references", std::map<std::string, std::vector<std::string>>{});
    metadata.extra = json.value("extra", std::map<std::string, std::string>{});

    return metadata;
}

inline std::string Quantity::to_json() const {
    return nlohmann::json{
        {"type", "metatomic_quantity"},
        {"name", name},
        {"unit", unit},
        {"gradients", gradients},
        {"sample_kind", sample_kind},
    }.dump();
}

inline Quantity Quantity::from_json(const std::string& string) {
    auto json = nlohmann::json::parse(string);
    auto quantity = Quantity();

    quantity.name = json.value("name", "");
    quantity.unit = json.value("unit", "");
    quantity.gradients = json.value("gradients", std::vector<std::string>{});
    quantity.sample_kind = json.value("sample_kind", "");

    return quantity;
}

} // namespace metatomic
