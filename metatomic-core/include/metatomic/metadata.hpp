#pragma once

#include <vector>

#include <nlohmann/json.hpp>

namespace metatomic{
    struct PairListOptions{
        double cutoff = 0.0;
        bool full_list = false;
        bool strict = false;
        std::vector<std::string> requestors;
    };

    void to_json(nlohmann::json& j, const PairListOptions& p){
        j = nlohmann::json{
            {"cutoff", p.cutoff},
            {"full_list", p.full_list},
            {"strict", p.strict},
            {"requestors", p.requestors}
        };
    }

    void from_json(const nlohmann::json& j, PairListOptions& p){
        j.at("cutoff").get_to(p.cutoff);
        j.at("full_list").get_to(p.full_list);
        j.at("strict").get_to(p.strict);
        j.at("requestors").get_to(p.requestors);
    }

    struct ModelMetadata{
        std::string type;
        std::string name;
        std::string description;
        std::vector<std::string> authors;
        struct References{
            std::vector<std::string> architecture;
            std::vector<std::string> model;
            std::vector<std::string> implementation;
        } references;
        nlohmann::json extra;
    };

    void to_json(nlohmann::json& j, const ModelMetadata& m){
        j = nlohmann::json{
            {"type", m.type},
            {"name", m.name},
            {"description", m.description},
            {"authors", m.authors},
            {"references", {
                {"architecture", m.references.architecture},
                {"model", m.references.model},
                {"implementation", m.references.implementation}
            }},
            {"extra", m.extra}
        };
    }

    void from_json(const nlohmann::json& j, ModelMetadata& m){
        j.at("type").get_to(m.type);
        j.at("name").get_to(m.name);
        j.at("description").get_to(m.description);
        j.at("authors").get_to(m.authors);
        j.at("references").at("architecture").get_to(m.references.architecture);
        j.at("references").at("model").get_to(m.references.model);
        j.at("references").at("implementation").get_to(m.references.implementation);
        j.at("extra").get_to(m.extra);
    }

    struct Quantity{
        std::string quantity;
        std::string unit;
        bool per_atom = false;
    };

    void to_json(nlohmann::json& j, const Quantity& q){
        j = nlohmann::json{
            {"quantity", q.quantity},
            {"unit", q.unit},
            {"per_atom", q.per_atom}
        };
    }

    void from_json(const nlohmann::json& j, Quantity& q){
        j.at("quantity").get_to(q.quantity);
        j.at("unit").get_to(q.unit);
        j.at("per_atom").get_to(q.per_atom);
    }

} // namespace metatomic
