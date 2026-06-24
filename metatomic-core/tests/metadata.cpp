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

    SECTION("ModelMetadata") {
        auto create_example = []() {
            metatomic::ModelMetadata m;
            m.name = "test-model";
            m.authors = {"Alice", "Bob <bob@test.com>"};
            m.description = "A test model";
            m.references = metatomic::References(
                {"doi:10.1234/test"},
                {"doi:10.1234/arch"},
                {"https://github.com/test"}
            );
            m.extra = {{"key1", "value1"}, {"key2", "value2"}};
            return m;
        };

        SECTION("roundtrip") {
            auto m1 = create_example();
            nlohmann::json j = m1;

            CHECK(j["type"] == "metatomic_model_metadata");
            CHECK(j["name"] == "test-model");
            CHECK(j["authors"].is_array());
            CHECK(j["authors"].size() == 2);
            CHECK(j["authors"][0] == "Alice");
            CHECK(j["authors"][1] == "Bob <bob@test.com>");
            CHECK(j["description"] == "A test model");
            CHECK(j["references"]["model"][0] == "doi:10.1234/test");
            CHECK(j["references"]["architecture"][0] == "doi:10.1234/arch");
            CHECK(j["references"]["implementation"][0] == "https://github.com/test");
            CHECK(j["extra"]["key1"] == "value1");
            CHECK(j["extra"]["key2"] == "value2");

            auto m2 = j.get<metatomic::ModelMetadata>();
            CHECK(m2.name == m1.name);
            CHECK(m2.authors == m1.authors);
            CHECK(m2.description == m1.description);
            CHECK(m2.references.model == m1.references.model);
            CHECK(m2.references.architecture == m1.references.architecture);
            CHECK(m2.references.implementation == m1.references.implementation);
            CHECK(m2.extra == m1.extra);
        }

        SECTION("rejects invalid json") {
            auto m1 = create_example();
            nlohmann::json j = m1;

            CHECK_THROWS_WITH(
                nlohmann::json("not an object").get<metatomic::ModelMetadata>(),
                Catch::Matchers::StartsWith("invalid JSON data for ModelMetadata, expected an object")
            );

            {
                auto j_copy = j;
                j_copy["type"] = "something-else";
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelMetadata>(),
                    Catch::Matchers::StartsWith("'type' in JSON for ModelMetadata must be 'metatomic_model_metadata'")
                );
            }

            {
                auto j_copy = j;
                j_copy.erase("name");
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelMetadata>(),
                    Catch::Matchers::StartsWith("'name' in JSON for ModelMetadata must be a string")
                );
            }

            {
                auto j_copy = j;
                j_copy["name"] = 42;
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelMetadata>(),
                    Catch::Matchers::StartsWith("'name' in JSON for ModelMetadata must be a string")
                );
            }

            {
                auto j_copy = j;
                j_copy["authors"] = "Alice";
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelMetadata>(),
                    Catch::Matchers::StartsWith("'authors' in JSON for ModelMetadata must be an array")
                );
            }

            {
                auto j_copy = j;
                j_copy["authors"] = {"Alice", 42};
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelMetadata>(),
                    Catch::Matchers::StartsWith("'authors' in JSON for ModelMetadata must be an array of strings")
                );
            }

            {
                auto j_copy = j;
                j_copy.erase("description");
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelMetadata>(),
                    Catch::Matchers::StartsWith("'description' in JSON for ModelMetadata must be a string")
                );
            }

            {
                auto j_copy = j;
                j_copy["extra"] = "not-an-object";
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelMetadata>(),
                    Catch::Matchers::StartsWith("'extra' in JSON for ModelMetadata must be an object")
                );
            }

            {
                auto j_copy = j;
                j_copy["extra"] = {{"key", 42}};
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelMetadata>(),
                    Catch::Matchers::StartsWith("'extra' in JSON for ModelMetadata must be an object with string values")
                );
            }

            {
                auto j_copy = j;
                j_copy["references"] = "not-an-object";
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelMetadata>(),
                    Catch::Matchers::StartsWith("invalid JSON data for references in ModelMetadata, expected an object")
                );
            }
        }

        SECTION("printing") {
            auto m1 = create_example();
            std::string output = m1.print();
            std::string expected =
                "This is the test-model model\n"
                "============================\n"
                "\n"
                "A test model\n"
                "\n"
                "Model authors\n"
                "-------------\n"
                "\n"
                "- Alice\n"
                "- Bob <bob@test.com>\n"
                "\n"
                "Model references\n"
                "----------------\n"
                "\n"
                "Please cite the following references when using this model:\n"
                "- about this specific model:\n"
                "  * doi:10.1234/test\n"
                "- about the architecture of this model:\n"
                "  * doi:10.1234/arch\n"
                "- about the implementation of this model:\n"
                "  * https://github.com/test\n";

            CHECK(output == expected);
        }
    }
}
