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
        metatomic::ModelMetadata::References r1(
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

        auto r2 = j.get<metatomic::ModelMetadata::References>();
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
            return metatomic::ModelMetadata(
                "test-model",
                {"Alice", "Bob"},
                "A test model",
                metatomic::ModelMetadata::References(
                    {"doi:10.1234/test"},
                    {"doi:10.1234/arch"},
                    {"https://github.com/test"}
                ),
                std::map<std::string, std::string>{
                    {"key1", "value1"},
                    {"key2", "value2"}
                }
            );
        };

        SECTION("JSON roundtrip conversion") {
            auto m1 = create_example();
            nlohmann::json j = m1;

            CHECK(j["type"] == "metatomic_model_metadata");
            CHECK(j["name"] == "test-model");
            CHECK(j["authors"].is_array());
            CHECK(j["authors"].size() == 2);
            CHECK(j["authors"][0] == "Alice");
            CHECK(j["authors"][1] == "Bob");
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

        SECTION("Invalid JSON data") {
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

        SECTION("Model metadata formatting") {
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
                "- Bob\n"
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

    SECTION("DType") {
        SECTION("JSON roundtrip conversion") {
            auto dtype1 = metatomic::ModelCapabilities::DType::Float32;
            nlohmann::json j = dtype1;
            CHECK(j == "float32");
            auto dtype2 = j.get<metatomic::ModelCapabilities::DType>();
            CHECK(dtype2 == metatomic::ModelCapabilities::DType::Float32);

            auto dtype3 = metatomic::ModelCapabilities::DType::Float64;
            nlohmann::json j2 = dtype3;
            CHECK(j2 == "float64");
            auto dtype4 = j2.get<metatomic::ModelCapabilities::DType>();
            CHECK(dtype4 == metatomic::ModelCapabilities::DType::Float64);
        }

        SECTION("Invalid JSON data") {
            CHECK_THROWS_WITH(
                nlohmann::json(42).get<metatomic::ModelCapabilities::DType>(),
                Catch::Matchers::StartsWith("dtype in JSON for ModelCapabilities must be a string")
            );

            CHECK_THROWS_WITH(
                nlohmann::json("float16").get<metatomic::ModelCapabilities::DType>(),
                Catch::Matchers::StartsWith("invalid string for dtype in JSON for ModelCapabilities, expected 'float32' or 'float64'")
            );
        }
    }

    SECTION("Quantity") {
        SECTION("JSON roundtrip conversion with description") {
            metatomic::ModelCapabilities::Quantity q1(
                "energy",
                "eV",
                "total energy of the system",
                {metatomic::ModelCapabilities::Gradients::Positions},
                metatomic::ModelCapabilities::SampleKind::System
            );
            nlohmann::json j = q1;

            CHECK(j["type"] == "metatomic_quantity");
            CHECK(j["name"] == "energy");
            CHECK(j["unit"] == "eV");
            CHECK(j["description"] == "total energy of the system");
            CHECK(j["gradients"].is_array());
            CHECK(j["gradients"].size() == 1);
            CHECK(j["gradients"][0] == "positions");
            CHECK(j["sample_kind"] == "system");

            auto q2 = j.get<metatomic::ModelCapabilities::Quantity>();
            CHECK(q2.name == q1.name);
            CHECK(q2.unit == q1.unit);
            CHECK(q2.description == q1.description);
            CHECK(q2.gradients == q1.gradients);
            CHECK(q2.sample_kind == q1.sample_kind);
        }

        SECTION("JSON roundtrip conversion without description") {
            metatomic::ModelCapabilities::Quantity q1(
                "charge",
                "e",
                std::nullopt,
                {},
                metatomic::ModelCapabilities::SampleKind::Atom
            );
            nlohmann::json j = q1;

            CHECK(j["type"] == "metatomic_quantity");
            CHECK(j["name"] == "charge");
            CHECK(j["unit"] == "e");
            CHECK(!j.contains("description"));
            CHECK(j["gradients"].is_array());
            CHECK(j["gradients"].size() == 0);
            CHECK(j["sample_kind"] == "atom");

            auto q2 = j.get<metatomic::ModelCapabilities::Quantity>();
            CHECK(q2.name == q1.name);
            CHECK(q2.unit == q1.unit);
            CHECK(!q2.description.has_value());
            CHECK(q2.gradients.empty());
            CHECK(q2.sample_kind == q1.sample_kind);
        }

        SECTION("Empty description is treated as no description") {
            nlohmann::json j = {
                {"type", "metatomic_quantity"},
                {"name", "charge"},
                {"unit", "e"},
                {"description", ""},
                {"gradients", nlohmann::json::array()},
                {"sample_kind", "atom"}
            };

            auto q = j.get<metatomic::ModelCapabilities::Quantity>();
            CHECK(q.name == "charge");
            CHECK(q.unit == "e");
            CHECK(!q.description.has_value());
            CHECK(q.gradients.empty());
            CHECK(q.sample_kind == metatomic::ModelCapabilities::SampleKind::Atom);
        }

        SECTION("Invalid JSON data") {
            CHECK_THROWS_WITH(
                nlohmann::json("not an object").get<metatomic::ModelCapabilities::Quantity>(),
                Catch::Matchers::StartsWith("invalid JSON data for Quantity, expected an object")
            );

            {
                nlohmann::json j = {{"type", "wrong-type"}};
                CHECK_THROWS_WITH(
                    j.get<metatomic::ModelCapabilities::Quantity>(),
                    Catch::Matchers::StartsWith("'type' in JSON for Quantity must be 'metatomic_quantity'")
                );
            }

            {
                nlohmann::json j = {
                    {"type", "metatomic_quantity"},
                    {"name", 42}
                };
                CHECK_THROWS_WITH(
                    j.get<metatomic::ModelCapabilities::Quantity>(),
                    Catch::Matchers::StartsWith("'name' in JSON for Quantity must be a string")
                );
            }

            {
                nlohmann::json j = {
                    {"type", "metatomic_quantity"},
                    {"name", "energy"},
                    {"unit", "eV"},
                    {"gradients", "positions"}
                };
                CHECK_THROWS_WITH(
                    j.get<metatomic::ModelCapabilities::Quantity>(),
                    Catch::Matchers::StartsWith("'gradients' in JSON for Quantity must be an array")
                );
            }

            {
                nlohmann::json j = {
                    {"type", "metatomic_quantity"},
                    {"name", "energy"},
                    {"unit", "eV"},
                    {"gradients", {"positions"}},
                    {"sample_kind", "unknown"}
                };
                CHECK_THROWS_WITH(
                    j.get<metatomic::ModelCapabilities::Quantity>(),
                    Catch::Matchers::StartsWith("'sample_kind' in JSON for Quantity must be 'atom', 'system' or 'atom_pair', got 'unknown'")
                );
            }
        }
    }

    SECTION("ModelCapabilities") {
        auto create_example = []() {
            std::vector<metatomic::ModelCapabilities::Quantity> outputs = {
                metatomic::ModelCapabilities::Quantity(
                    "energy",
                    "eV",
                    "total energy",
                    {metatomic::ModelCapabilities::Gradients::Positions},
                    metatomic::ModelCapabilities::SampleKind::System
                ),
                metatomic::ModelCapabilities::Quantity(
                    "charge",
                    "e",
                    std::nullopt,
                    {},
                    metatomic::ModelCapabilities::SampleKind::Atom
                )
            };

            return metatomic::ModelCapabilities(
                outputs,
                {1, 6, 8},
                5.0,
                "Angstrom",
                {metatomic::ModelCapabilities::Device::CPU, metatomic::ModelCapabilities::Device::CUDA},
                metatomic::ModelCapabilities::DType::Float32
            );
        };

        SECTION("JSON roundtrip conversion") {
            auto c1 = create_example();
            nlohmann::json j = c1;

            CHECK(j["type"] == "metatomic_model_capabilities");
            CHECK(j["outputs"].is_array());
            CHECK(j["outputs"].size() == 2);
            CHECK(j["outputs"][0]["name"] == "energy");
            CHECK(j["outputs"][1]["name"] == "charge");
            CHECK(j["atomic_types"].is_array());
            CHECK(j["atomic_types"].size() == 3);
            CHECK(j["atomic_types"][0] == 1);
            CHECK(j["atomic_types"][1] == 6);
            CHECK(j["atomic_types"][2] == 8);
            CHECK(j["interaction_range"] == Approx(5.0));
            CHECK(j["length_unit"] == "Angstrom");
            CHECK(j["supported_devices"].is_array());
            CHECK(j["supported_devices"].size() == 2);
            CHECK(j["supported_devices"][0] == "cpu");
            CHECK(j["supported_devices"][1] == "cuda");
            CHECK(j["dtype"] == "float32");

            auto c2 = j.get<metatomic::ModelCapabilities>();
            CHECK(c2.outputs.size() == c1.outputs.size());
            CHECK(c2.outputs[0].name == c1.outputs[0].name);
            CHECK(c2.outputs[1].name == c1.outputs[1].name);
            CHECK(c2.atomic_types == c1.atomic_types);
            CHECK(c2.interaction_range == Approx(c1.interaction_range));
            CHECK(c2.length_unit == c1.length_unit);
            CHECK(c2.supported_devices == c1.supported_devices);
            CHECK(c2.dtype == c1.dtype);
        }

        SECTION("Invalid JSON data") {
            auto c1 = create_example();
            nlohmann::json j = c1;

            CHECK_THROWS_WITH(
                nlohmann::json("not an object").get<metatomic::ModelCapabilities>(),
                Catch::Matchers::StartsWith("invalid JSON data for ModelCapabilities, expected an object")
            );

            {
                auto j_copy = j;
                j_copy["type"] = "something-else";
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelCapabilities>(),
                    Catch::Matchers::StartsWith("'type' in JSON for ModelCapabilities must be 'metatomic_model_capabilities'")
                );
            }

            {
                auto j_copy = j;
                j_copy["outputs"] = "energy";
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelCapabilities>(),
                    Catch::Matchers::StartsWith("'outputs' in JSON for ModelCapabilities must be an array")
                );
            }

            {
                auto j_copy = j;
                j_copy["atomic_types"] = "1";
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelCapabilities>(),
                    Catch::Matchers::StartsWith("'atomic_types' in JSON for ModelCapabilities must be an array")
                );
            }

            {
                auto j_copy = j;
                j_copy["atomic_types"] = {1, "x"};
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelCapabilities>(),
                    Catch::Matchers::StartsWith("'atomic_types' in JSON for ModelCapabilities must be an array of integers")
                );
            }

            {
                auto j_copy = j;
                j_copy.erase("interaction_range");
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelCapabilities>(),
                    Catch::Matchers::StartsWith("'interaction_range' in JSON for ModelCapabilities must be a number")
                );
            }

            {
                auto j_copy = j;
                j_copy["interaction_range"] = -1.0;
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelCapabilities>(),
                    Catch::Matchers::StartsWith("'interaction_range' in JSON for ModelCapabilities must be non-negative")
                );
            }

            {
                auto j_copy = j;
                j_copy["length_unit"] = "eV";
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelCapabilities>(),
                    Catch::Matchers::StartsWith("invalid parameter: dimension mismatch")
                );
            }

            {
                auto j_copy = j;
                j_copy["supported_devices"] = "cpu";
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelCapabilities>(),
                    Catch::Matchers::StartsWith("'supported_devices' in JSON for ModelCapabilities must be an array")
                );
            }

            {
                auto j_copy = j;
                j_copy["supported_devices"] = {"cpu", "wat"};
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelCapabilities>(),
                    Catch::Matchers::StartsWith("invalid string for device in JSON for ModelCapabilities, expected 'cpu', 'cuda', 'rocm', or 'metal'")
                );
            }

            {
                auto j_copy = j;
                j_copy["dtype"] = "float16";
                CHECK_THROWS_WITH(
                    j_copy.get<metatomic::ModelCapabilities>(),
                    Catch::Matchers::StartsWith("invalid string for dtype in JSON for ModelCapabilities, expected 'float32' or 'float64'")
                );
            }
        }
    }
}
