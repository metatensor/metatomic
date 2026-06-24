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
}
