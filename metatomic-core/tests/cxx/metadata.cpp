#include <catch.hpp>

#include "metatomic.hpp"

TEST_CASE("JSON serialization C++ API") {
    SECTION("PairListOptions"){
        double cutoff = 3.0;
        std::string cutoff_hex = "0x4008000000000000";

        SECTION("Builder construction") {
            auto p1 = metatomic::PairListOptions::builder()
                .cutoff(cutoff)
                .full_list(true)
                .strict(false)
                .add_requestor("model1")
                .add_requestor("model2")
                .build();

            nlohmann::json j = p1;

            CHECK(j["cutoff"] == cutoff_hex);
            CHECK(j["full_list"] == true);
            CHECK(j["strict"] == false);
            CHECK(j["requestors"].is_array());
            CHECK(j["requestors"].size() == 2);
            CHECK(j["requestors"][0] == "model1");
            CHECK(j["requestors"][1] == "model2");

            auto p2 = j.get<metatomic::PairListOptions>();
            CHECK(p2.cutoff() == Approx(cutoff));
            CHECK(p2.full_list() == true);
            CHECK(p2.strict() == false);
            CHECK(p2.requestors().size() == 2);
            CHECK(p2.requestors()[0] == "model1");
            CHECK(p2.requestors()[1] == "model2");
        }

        SECTION("Builder with requestors set as a list") {
            std::vector<std::string> requestors = {"model1", "model2"};
            auto p1 = metatomic::PairListOptions::builder()
                .cutoff(cutoff)
                .full_list(true)
                .strict(false)
                .requestors(requestors)
                .build();

            nlohmann::json j = p1;

            CHECK(j["cutoff"] == cutoff_hex);
            CHECK(j["full_list"] == true);
            CHECK(j["strict"] == false);
            CHECK(j["requestors"].is_array());
            CHECK(j["requestors"].size() == 2);
            CHECK(j["requestors"][0] == "model1");
            CHECK(j["requestors"][1] == "model2");

            auto p2 = j.get<metatomic::PairListOptions>();
            CHECK(p2.cutoff() == Approx(cutoff));
            CHECK(p2.full_list() == true);
            CHECK(p2.strict() == false);
            CHECK(p2.requestors().size() == 2);
            CHECK(p2.requestors()[0] == "model1");
            CHECK(p2.requestors()[1] == "model2");
        }

        SECTION("build() validates completeness") {
            CHECK_THROWS_WITH(
                metatomic::PairListOptions::builder().build(),
                Catch::Matchers::StartsWith("cutoff must be set before building PairListOptions")
            );

            CHECK_THROWS_WITH(
                metatomic::PairListOptions::builder().full_list(true).build(),
                Catch::Matchers::StartsWith("cutoff must be set before building PairListOptions")
            );

            CHECK_THROWS_WITH(
                metatomic::PairListOptions::builder().cutoff(cutoff).build(),
                Catch::Matchers::StartsWith("full_list must be set before building PairListOptions")
            );

            CHECK_THROWS_WITH(
                metatomic::PairListOptions::builder().cutoff(-1.0),
                Catch::Matchers::StartsWith("cutoff must be a finite positive number")
            );
        }

        SECTION("setters mutate existing instances") {
            auto p1 = metatomic::PairListOptions::builder()
                .cutoff(cutoff)
                .full_list(true)
                .build();

            p1.cutoff(2.0).strict(false).add_requestor("model1").add_requestor("model2");

            CHECK(p1.cutoff() == Approx(2.0));
            CHECK(p1.full_list() == true);
            CHECK(p1.strict() == false);
            CHECK(p1.requestors().size() == 2);
            CHECK(p1.requestors()[0] == "model1");
            CHECK(p1.requestors()[1] == "model2");

            p1.clear_requestors();
            CHECK(p1.requestors().empty());
        }

        SECTION("add_requestor ignores empty strings and duplicates") {
            auto p1 = metatomic::PairListOptions::builder()
                .cutoff(cutoff)
                .full_list(true)
                .add_requestor("model1")
                .add_requestor("")
                .add_requestor("model2")
                .add_requestor("model1")
                .build();

            auto requestors = p1.requestors();
            CHECK(requestors.size() == 2);
            CHECK(requestors[0] == "model1");
            CHECK(requestors[1] == "model2");

            nlohmann::json j = p1;
            CHECK(j["requestors"].size() == 2);
            CHECK(j["requestors"][0] == "model1");
            CHECK(j["requestors"][1] == "model2");
        }

        SECTION("clear_requestors empties the list") {
            auto p1 = metatomic::PairListOptions::builder()
                .cutoff(cutoff)
                .full_list(true)
                .requestors({"model1", "model2"})
                .build();
            p1.clear_requestors();

            CHECK(p1.requestors().empty());

            nlohmann::json j = p1;
            CHECK(j["requestors"].is_array());
            CHECK(j["requestors"].size() == 0);
        }
    }

    SECTION("References") {
        SECTION("Builder construction") {
            auto r1 = metatomic::ModelMetadata::References::builder()
                .model({"model ref 1", "model ref 2"})
                .architecture({"architecture ref 1"})
                .implementation({"implementation ref 1", "implementation ref 2"})
                .build();

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
            CHECK(r2.model()[0] == "model ref 1");
            CHECK(r2.model()[1] == "model ref 2");
            CHECK(r2.architecture().size() == 1);
            CHECK(r2.architecture()[0] == "architecture ref 1");
            CHECK(r2.implementation().size() == 2);
            CHECK(r2.implementation()[0] == "implementation ref 1");
            CHECK(r2.implementation()[1] == "implementation ref 2");
        }

        SECTION("Builder with no setters succeeds") {
            auto r1 = metatomic::ModelMetadata::References::builder().build();

            CHECK(r1.model().empty());
            CHECK(r1.architecture().empty());
            CHECK(r1.implementation().empty());
        }

        SECTION("Setters mutate existing instances") {
            auto r1 = metatomic::ModelMetadata::References::builder().build();
            r1.model({"model ref 1", "model ref 2"});
            r1.architecture({"architecture ref 1"});
            r1.implementation({"implementation ref 1", "implementation ref 2"});

            nlohmann::json j = r1;

            CHECK(j["model"].size() == 2);
            CHECK(j["model"][0] == "model ref 1");
            CHECK(j["architecture"].size() == 1);
            CHECK(j["implementation"].size() == 2);

            auto r2 = j.get<metatomic::ModelMetadata::References>();
            CHECK(r2.model()[0] == "model ref 1");
            CHECK(r2.architecture().size() == 1);
            CHECK(r2.implementation().size() == 2);
        }

        SECTION("add and clear reference sections") {
            auto r1 = metatomic::ModelMetadata::References::builder().build();
            r1.add_model("model ref 1");
            r1.add_model("model ref 2");
            r1.add_architecture("architecture ref 1");
            r1.add_implementation("implementation ref 1");
            r1.add_implementation("implementation ref 2");

            CHECK(r1.model().size() == 2);
            CHECK(r1.model()[0] == "model ref 1");
            CHECK(r1.model()[1] == "model ref 2");
            CHECK(r1.architecture().size() == 1);
            CHECK(r1.architecture()[0] == "architecture ref 1");
            CHECK(r1.implementation().size() == 2);
            CHECK(r1.implementation()[1] == "implementation ref 2");

            r1.clear_model();
            CHECK(r1.model().empty());
            CHECK(r1.architecture().size() == 1);

            r1.clear_architecture();
            r1.clear_implementation();
            CHECK(r1.architecture().empty());
            CHECK(r1.implementation().empty());
        }
    }

    SECTION("ModelMetadata") {
        auto create_example = []() {
            return metatomic::ModelMetadata::builder()
                .name("test-model")
                .authors({"Alice", "Bob"})
                .description("A test model")
                .references(metatomic::ModelMetadata::References::builder()
                    .model({"doi:10.1234/test"})
                    .architecture({"doi:10.1234/arch"})
                    .implementation({"https://github.com/test"})
                    .build())
                .extra(std::map<std::string, std::string>{
                    {"key1", "value1"},
                    {"key2", "value2"}
                })
                .build();
        };

        auto create_example_with_setters = []() {
            auto metadata = metatomic::ModelMetadata::builder().build();
            metadata.name("test-model");
            metadata.authors({"Alice", "Bob"});
            metadata.description("A test model");
            metadata.references(metatomic::ModelMetadata::References::builder()
                .model({"doi:10.1234/test"})
                .architecture({"doi:10.1234/arch"})
                .implementation({"https://github.com/test"})
                .build());
            metadata.extra(std::map<std::string, std::string>{
                {"key1", "value1"},
                {"key2", "value2"}
            });
            return metadata;
        };

        SECTION("JSON roundtrip conversion with builder") {
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
            CHECK(m2.name() == m1.name());
            CHECK(m2.authors() == m1.authors());
            CHECK(m2.description() == m1.description());
            CHECK(m2.references().model() == m1.references().model());
            CHECK(m2.references().architecture() == m1.references().architecture());
            CHECK(m2.references().implementation() == m1.references().implementation());
            CHECK(m2.extra() == m1.extra());
        }

        SECTION("JSON roundtrip conversion with builder (accumulated fields)") {
            auto m1 = create_example_with_setters();
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
            CHECK(m2.name() == m1.name());
            CHECK(m2.authors() == m1.authors());
            CHECK(m2.description() == m1.description());
            CHECK(m2.references().model() == m1.references().model());
            CHECK(m2.references().architecture() == m1.references().architecture());
            CHECK(m2.references().implementation() == m1.references().implementation());
            CHECK(m2.extra() == m1.extra());
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

        SECTION("Builder with no setters succeeds") {
            auto m1 = metatomic::ModelMetadata::builder().build();

            CHECK(m1.name().empty());
            CHECK(m1.authors().empty());
            CHECK(m1.description().empty());
            CHECK(m1.references().model().empty());
            CHECK(m1.references().architecture().empty());
            CHECK(m1.references().implementation().empty());
            CHECK(m1.extra().empty());
        }

        SECTION("add and clear authors, references, and extra") {
            auto m1 = metatomic::ModelMetadata::builder().build();
            m1.name("test-model")
              .add_author("Alice")
              .add_author("Bob")
              .add_reference("model", "doi:10.1234/test")
              .add_reference("architecture", "doi:10.1234/arch")
              .add_reference("implementation", "https://github.com/test")
              .add_extra("key1", "value1")
              .add_extra("key2", "value2");

            CHECK(m1.authors().size() == 2);
            CHECK(m1.authors()[0] == "Alice");
            CHECK(m1.authors()[1] == "Bob");
            CHECK(m1.references().model().size() == 1);
            CHECK(m1.references().architecture().size() == 1);
            CHECK(m1.references().implementation().size() == 1);
            CHECK(m1.extra().size() == 2);
            CHECK(m1.extra().at("key1") == "value1");
            CHECK(m1.extra().at("key2") == "value2");

            nlohmann::json j = m1;
            CHECK(j["authors"].size() == 2);
            CHECK(j["references"]["model"].size() == 1);
            CHECK(j["extra"].size() == 2);

            m1.clear_reference("model");
            CHECK(m1.references().model().empty());
            CHECK(m1.references().architecture().size() == 1);
            CHECK(m1.references().implementation().size() == 1);

            m1.clear_authors();
            m1.clear_references();
            m1.clear_extra();
            CHECK(m1.authors().empty());
            CHECK(m1.references().model().empty());
            CHECK(m1.references().architecture().empty());
            CHECK(m1.references().implementation().empty());
            CHECK(m1.extra().empty());

            CHECK_THROWS_WITH(
                m1.add_reference("invalid", "ref"),
                Catch::Matchers::StartsWith("reference section must be 'model', 'architecture', or 'implementation', got 'invalid'")
            );

            CHECK_THROWS_WITH(
                m1.clear_reference("invalid"),
                Catch::Matchers::StartsWith("reference section must be 'model', 'architecture', or 'implementation', got 'invalid'")
            );
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
            auto q1 = metatomic::Quantity::builder()
                .name("energy")
                .unit("eV")
                .sample_kind(metatomic::SampleKind::System)
                .description("total energy of the system")
                .gradients({metatomic::Gradients::Positions})
                .build();

            nlohmann::json j = q1;

            CHECK(j["type"] == "metatomic_quantity");
            CHECK(j["name"] == "energy");
            CHECK(j["unit"] == "eV");
            CHECK(j["description"] == "total energy of the system");
            CHECK(j["gradients"].is_array());
            CHECK(j["gradients"].size() == 1);
            CHECK(j["gradients"][0] == "positions");
            CHECK(j["sample_kind"] == "system");

            auto q2 = j.get<metatomic::Quantity>();
            CHECK(q2.name() == q1.name());
            CHECK(q2.unit() == q1.unit());
            CHECK(q2.description() == q1.description());
            CHECK(q2.gradients() == q1.gradients());
            CHECK(q2.sample_kind() == q1.sample_kind());
        }

        SECTION("JSON roundtrip conversion without description") {
            auto q1 = metatomic::Quantity::builder()
                .name("charge")
                .unit("e")
                .sample_kind(metatomic::SampleKind::Atom)
                .build();

            nlohmann::json j = q1;

            CHECK(j["type"] == "metatomic_quantity");
            CHECK(j["name"] == "charge");
            CHECK(j["unit"] == "e");
            CHECK(!j.contains("description"));
            CHECK(j["gradients"].is_array());
            CHECK(j["gradients"].size() == 0);
            CHECK(j["sample_kind"] == "atom");

            auto q2 = j.get<metatomic::Quantity>();
            CHECK(q2.name() == q1.name());
            CHECK(q2.unit() == q1.unit());
            CHECK(q2.description().empty());
            CHECK(q2.gradients().empty());
            CHECK(q2.sample_kind() == q1.sample_kind());
        }

        SECTION("build() validates completeness") {
            CHECK_THROWS_WITH(
                metatomic::Quantity::builder().build(),
                Catch::Matchers::StartsWith("name must be set before building Quantity")
            );

            CHECK_THROWS_WITH(
                metatomic::Quantity::builder().name("energy").build(),
                Catch::Matchers::StartsWith("unit must be set before building Quantity")
            );

            CHECK_THROWS_WITH(
                metatomic::Quantity::builder().name("energy").unit("eV").build(),
                Catch::Matchers::StartsWith("sample_kind must be set before building Quantity")
            );
        }

        SECTION("setters mutate existing instances") {
            auto q1 = metatomic::Quantity::builder()
                .name("energy")
                .unit("eV")
                .sample_kind(metatomic::SampleKind::System)
                .build();

            q1.unit("kJ/mol").description("updated").add_gradient(metatomic::Gradients::Positions);

            CHECK(q1.unit() == "kJ/mol");
            CHECK(q1.description() == "updated");
            CHECK(q1.gradients().size() == 1);
            CHECK(q1.gradients()[0] == metatomic::Gradients::Positions);

            q1.clear_gradients();
            CHECK(q1.gradients().empty());
        }

        SECTION("add and clear gradients") {
            auto q1 = metatomic::Quantity::builder()
                .name("energy")
                .unit("eV")
                .sample_kind(metatomic::SampleKind::System)
                .add_gradient(metatomic::Gradients::Positions)
                .add_gradient(metatomic::Gradients::Strain)
                .build();

            CHECK(q1.gradients().size() == 2);
            CHECK(q1.gradients()[0] == metatomic::Gradients::Positions);
            CHECK(q1.gradients()[1] == metatomic::Gradients::Strain);

            nlohmann::json j = q1;
            CHECK(j["gradients"].size() == 2);
            CHECK(j["gradients"][0] == "positions");
            CHECK(j["gradients"][1] == "strain");

            q1.clear_gradients();
            CHECK(q1.gradients().empty());
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

            auto q = j.get<metatomic::Quantity>();
            CHECK(q.name() == "charge");
            CHECK(q.unit() == "e");
            CHECK(q.description().empty());
            CHECK(q.gradients().empty());
            CHECK(q.sample_kind() == metatomic::SampleKind::Atom);
        }

        SECTION("Invalid JSON data") {
            CHECK_THROWS_WITH(
                nlohmann::json("not an object").get<metatomic::Quantity>(),
                Catch::Matchers::StartsWith("invalid JSON data for Quantity, expected an object")
            );

            {
                nlohmann::json j = {{"type", "wrong-type"}};
                CHECK_THROWS_WITH(
                    j.get<metatomic::Quantity>(),
                    Catch::Matchers::StartsWith("'type' in JSON for Quantity must be 'metatomic_quantity'")
                );
            }

            {
                nlohmann::json j = {
                    {"type", "metatomic_quantity"},
                    {"name", 42}
                };
                CHECK_THROWS_WITH(
                    j.get<metatomic::Quantity>(),
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
                    j.get<metatomic::Quantity>(),
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
                    j.get<metatomic::Quantity>(),
                    Catch::Matchers::StartsWith("'sample_kind' in JSON for Quantity must be 'atom', 'system' or 'atom_pair', got 'unknown'")
                );
            }
        }
    }

    SECTION("ModelCapabilities") {
        auto make_quantity = [](const std::string& name, const std::string& unit,
                                metatomic::SampleKind sample_kind,
                                const std::string& description = "",
                                std::vector<metatomic::Gradients> gradients = {}) {
            return metatomic::Quantity::builder()
                .name(name)
                .unit(unit)
                .sample_kind(sample_kind)
                .description(description)
                .gradients(std::move(gradients))
                .build();
        };

        auto create_example = [&]() {
            std::vector<metatomic::Quantity> outputs = {
                make_quantity("energy", "eV", metatomic::SampleKind::System,
                              "total energy", {metatomic::Gradients::Positions}),
                make_quantity("charge", "e", metatomic::SampleKind::Atom),
            };

            return metatomic::ModelCapabilities::builder()
                .atomic_types({1, 6, 8})
                .interaction_range(5.0)
                .length_unit("Angstrom")
                .supported_devices({metatomic::ModelCapabilities::Device::CPU,
                                    metatomic::ModelCapabilities::Device::CUDA})
                .dtype(metatomic::ModelCapabilities::DType::Float32)
                .outputs(std::move(outputs))
                .build();
        };

        auto create_example_with_setters = [&]() {
            std::vector<metatomic::Quantity> outputs = {
                make_quantity("energy", "eV", metatomic::SampleKind::System,
                              "total energy", {metatomic::Gradients::Positions}),
                make_quantity("charge", "e", metatomic::SampleKind::Atom),
            };

            return metatomic::ModelCapabilities::builder()
                .atomic_types({1, 6, 8})
                .interaction_range(5.0)
                .length_unit("Angstrom")
                .supported_devices({metatomic::ModelCapabilities::Device::CPU,
                                    metatomic::ModelCapabilities::Device::CUDA})
                .dtype(metatomic::ModelCapabilities::DType::Float32)
                .outputs(std::move(outputs))
                .build();
        };

        SECTION("JSON roundtrip conversion with builder") {
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
            CHECK(c2.outputs().size() == c1.outputs().size());
            CHECK(c2.outputs()[0].name() == c1.outputs()[0].name());
            CHECK(c2.outputs()[1].name() == c1.outputs()[1].name());
            CHECK(c2.atomic_types() == c1.atomic_types());
            CHECK(c2.interaction_range() == Approx(c1.interaction_range()));
            CHECK(c2.length_unit() == c1.length_unit());
            CHECK(c2.supported_devices() == c1.supported_devices());
            CHECK(c2.dtype() == c1.dtype());
        }

        SECTION("JSON roundtrip conversion with builder (accumulated fields)") {
            auto c1 = create_example_with_setters();
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
            CHECK(c2.outputs().size() == c1.outputs().size());
            CHECK(c2.outputs()[0].name() == c1.outputs()[0].name());
            CHECK(c2.outputs()[1].name() == c1.outputs()[1].name());
            CHECK(c2.atomic_types() == c1.atomic_types());
            CHECK(c2.interaction_range() == Approx(c1.interaction_range()));
            CHECK(c2.length_unit() == c1.length_unit());
            CHECK(c2.supported_devices() == c1.supported_devices());
            CHECK(c2.dtype() == c1.dtype());
        }

        SECTION("build() validates completeness") {
            CHECK_THROWS_WITH(
                metatomic::ModelCapabilities::builder().build(),
                Catch::Matchers::StartsWith("atomic_types must be set before building ModelCapabilities")
            );

            CHECK_THROWS_WITH(
                metatomic::ModelCapabilities::builder().atomic_types({1}).build(),
                Catch::Matchers::StartsWith("interaction_range must be set before building ModelCapabilities")
            );

            CHECK_THROWS_WITH(
                metatomic::ModelCapabilities::builder()
                    .atomic_types({1}).interaction_range(5.0).build(),
                Catch::Matchers::StartsWith("length_unit must be set before building ModelCapabilities")
            );

            CHECK_THROWS_WITH(
                metatomic::ModelCapabilities::builder()
                    .atomic_types({1}).interaction_range(5.0).length_unit("Angstrom").build(),
                Catch::Matchers::StartsWith("supported_devices must be set before building ModelCapabilities")
            );

            CHECK_THROWS_WITH(
                metatomic::ModelCapabilities::builder()
                    .atomic_types({1}).interaction_range(5.0).length_unit("Angstrom")
                    .supported_devices({metatomic::ModelCapabilities::Device::CPU}).build(),
                Catch::Matchers::StartsWith("dtype must be set before building ModelCapabilities")
            );

            CHECK_THROWS_WITH(
                metatomic::ModelCapabilities::builder().interaction_range(-1.0),
                Catch::Matchers::StartsWith("interaction_range must be non-negative")
            );
        }

        SECTION("setters mutate existing instances") {
            auto c1 = create_example();

            c1.interaction_range(10.0).length_unit("nanometer");

            CHECK(c1.interaction_range() == Approx(10.0));
            CHECK(c1.length_unit() == "nanometer");

            c1.add_atomic_type(12);
            CHECK(c1.atomic_types().size() == 4);
            CHECK(c1.atomic_types()[3] == 12);

            c1.clear_atomic_types();
            CHECK(c1.atomic_types().empty());
        }

        SECTION("add and clear outputs, atomic types, and supported devices") {
            auto c1 = metatomic::ModelCapabilities::builder()
                .atomic_types({})
                .interaction_range(5.0)
                .length_unit("Angstrom")
                .supported_devices({})
                .dtype(metatomic::ModelCapabilities::DType::Float32)
                .build();

            c1.add_output(make_quantity("energy", "eV", metatomic::SampleKind::System,
                                        "total energy", {metatomic::Gradients::Positions}));
            c1.add_output(make_quantity("charge", "e", metatomic::SampleKind::Atom));

            c1.add_atomic_type(1);
            c1.add_atomic_type(6);
            c1.add_atomic_type(8);

            c1.add_supported_device(metatomic::ModelCapabilities::Device::CPU);
            c1.add_supported_device(metatomic::ModelCapabilities::Device::CUDA);

            CHECK(c1.outputs().size() == 2);
            CHECK(c1.outputs()[0].name() == "energy");
            CHECK(c1.outputs()[1].name() == "charge");
            CHECK(c1.atomic_types().size() == 3);
            CHECK(c1.atomic_types()[0] == 1);
            CHECK(c1.atomic_types()[1] == 6);
            CHECK(c1.atomic_types()[2] == 8);
            CHECK(c1.supported_devices().size() == 2);
            CHECK(c1.supported_devices()[0] == metatomic::ModelCapabilities::Device::CPU);
            CHECK(c1.supported_devices()[1] == metatomic::ModelCapabilities::Device::CUDA);

            nlohmann::json j = c1;
            CHECK(j["outputs"].size() == 2);
            CHECK(j["atomic_types"].size() == 3);
            CHECK(j["supported_devices"].size() == 2);

            c1.clear_outputs();
            c1.clear_atomic_types();
            c1.clear_supported_devices();
            CHECK(c1.outputs().empty());
            CHECK(c1.atomic_types().empty());
            CHECK(c1.supported_devices().empty());
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
