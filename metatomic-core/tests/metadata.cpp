#include <catch.hpp>

#include "metatomic.hpp"

TEST_CASE("JSON serialization C++ API") {
    SECTION("PairListOptions"){
        metatomic::PairListOptions p1;
        p1.cutoff = 3.0;
        p1.full_list = true;
        p1.strict = false;
        p1.requestors = {"model1", "model2"};

        nlohmann::json j = p1;

        CHECK(j["cutoff"] == 3.0);
        CHECK(j["full_list"] == true);
        CHECK(j["strict"] == false);
        CHECK(j["requestors"].is_array());
        CHECK(j["requestors"].size() == 2);
        CHECK(j["requestors"][0] == "model1");
        CHECK(j["requestors"][1] == "model2");

        metatomic::PairListOptions p2 = j.get<metatomic::PairListOptions>();
        CHECK(p2.cutoff == 3.0);
        CHECK(p2.full_list == true);
        CHECK(p2.strict == false);
        CHECK(p2.requestors.size() == 2);
        CHECK(p2.requestors[0] == "model1");
        CHECK(p2.requestors[1] == "model2");
    }

    SECTION("ModelMetadata"){
        metatomic::ModelMetadata m1;
        m1.type = "metatomic_model_metadata";
        m1.name = "test model";
        m1.description = "A test model for unit testing";
        m1.authors = {"Author 1", "Author 2"};
        m1.references.architecture = {"arch ref 1", "arch ref 2"};
        m1.references.model = {"model ref 1"};
        m1.references.implementation = {"impl ref 1", "impl ref 2"};
        m1.extra = {{"key", "value"}};

        nlohmann::json j = m1;

        CHECK(j["type"] == "metatomic_model_metadata");
        CHECK(j["name"] == "test model");
        CHECK(j["description"] == "A test model for unit testing");
        CHECK(j["authors"].is_array());
        CHECK(j["authors"].size() == 2);
        CHECK(j["authors"][0] == "Author 1");
        CHECK(j["authors"][1] == "Author 2");
        CHECK(j["references"]["architecture"].is_array());
        CHECK(j["references"]["architecture"].size() == 2);
        CHECK(j["references"]["architecture"][0] == "arch ref 1");
        CHECK(j["references"]["architecture"][1] == "arch ref 2");
        CHECK(j["references"]["model"].is_array());
        CHECK(j["references"]["model"].size() == 1);
        CHECK(j["references"]["model"][0] == "model ref 1");
        CHECK(j["references"]["implementation"].is_array());
        CHECK(j["references"]["implementation"].size() == 2);
        CHECK(j["references"]["implementation"][0] == "impl ref 1");
        CHECK(j["references"]["implementation"][1] == "impl ref 2");
        CHECK(j["extra"]["key"] == "value");
    }
}
