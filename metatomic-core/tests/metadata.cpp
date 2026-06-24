#include <catch.hpp>

#include "metatomic.hpp"

TEST_CASE("JSON serialization C++ API") {
    SECTION("PairListOptions"){
        double cutoff = 3.0;
        std::string cutoff_hex = "0x4008000000000000";

        metatomic::PairListOptions p1(cutoff, true, false, {"model1", "model2"});

        nlohmann::json j = p1;

        CHECK(j["cutoff"] == cutoff_hex);
        CHECK(j["full_list"] == true);
        CHECK(j["strict"] == false);
        CHECK(j["requestors"].is_array());
        CHECK(j["requestors"].size() == 2);
        CHECK(j["requestors"][0] == "model1");
        CHECK(j["requestors"][1] == "model2");

        auto p2 = j.get<metatomic::PairListOptions>();
        CHECK(p2.cutoff == Approx(cutoff));
        CHECK(p2.full_list == true);
        CHECK(p2.strict == false);
        CHECK(p2.requestors.size() == 2);
        CHECK(p2.requestors[0] == "model1");
        CHECK(p2.requestors[1] == "model2");
    }

    SECTION("References") {
        metatomic::References r1(
            {"model ref 1", "model ref 2"},
            {"architecture ref 1"},
            {"implementation ref 1", "implementation ref 2"}
        );

        nlohmann::json j = r1;

        CHECK(j["model"].is_array());
        CHECK(j["model"].size() == 2);
        CHECK(j["model"][0] == "model ref 1");
        CHECK(j["model"][1] == "model ref 2");

        CHECK(j["architecture"].is_array());
        CHECK(j["architecture"].size() == 1);
        CHECK(j["architecture"][0] == "architecture ref 1");

        CHECK(j["implementation"].is_array());
        CHECK(j["implementation"].size() == 2);
        CHECK(j["implementation"][0] == "implementation ref 1");
        CHECK(j["implementation"][1] == "implementation ref 2");

        auto r2 = j.get<metatomic::References>();
        CHECK(r2.model[0] == "model ref 1");
        CHECK(r2.model[1] == "model ref 2");
        CHECK(r2.architecture.size() == 1);
        CHECK(r2.architecture[0] == "architecture ref 1");
        CHECK(r2.implementation.size() == 2);
        CHECK(r2.implementation[0] == "implementation ref 1");
        CHECK(r2.implementation[1] == "implementation ref 2");
    }
}
