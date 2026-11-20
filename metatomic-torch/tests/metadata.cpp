#include <torch/torch.h>

#include <metatomic/torch.hpp>
using namespace metatomic_torch;

#include <catch.hpp>
using namespace Catch::Matchers;


TEST_CASE("ModelMetadata") {
    // save to JSON
    auto metadata = torch::make_intrusive<ModelMetadataHolder>();
    metadata->set_name("some name");
    metadata->set_description("describing it");
    metadata->set_authors({"John Doe", "Napoleon"});
    auto references = torch::Dict<std::string, std::vector<std::string>>();
    references.insert("model", std::vector<std::string>{"some-ref"});
    references.insert("architecture", std::vector<std::string>{"ref-2", "ref-3"});
    metadata->set_references(references);

    const auto* expected = R"({
    "authors": [
        "John Doe",
        "Napoleon"
    ],
    "description": "describing it",
    "extra": {},
    "name": "some name",
    "references": {
        "architecture": [
            "ref-2",
            "ref-3"
        ],
        "implementation": [],
        "model": [
            "some-ref"
        ]
    },
    "type": "metatomic_model_metadata"
})";
    CHECK(metadata->to_json() == expected);

    // load from JSON
    metadata = ModelMetadataHolder::from_json(expected);
    CHECK(metadata->name() == "some name");
    CHECK(metadata->description() == "describing it");
    CHECK(metadata->authors() == std::vector<std::string>{"John Doe", "Napoleon"});
    CHECK(metadata->references().at("model") == std::vector<std::string>{"some-ref"});
    CHECK(metadata->references().at("architecture") == std::vector<std::string>{"ref-2", "ref-3"});
    CHECK(metadata->references().at("implementation").empty());
    CHECK(metadata->extra().empty());

    // load from legacy JSON
    std::string json =R"({
    "class": "ModelMetadata",
    "name": "foo",
    "description": "test",
    "authors": ["me", "myself"],
    "references": {
        "implementation": ["torch-power!"],
        "model": ["took a while to train"]
    },
    "extra": {
        "key": "value"
    }
})";

    metadata = ModelMetadataHolder::from_json(json);
    CHECK(metadata->name() == "foo");
    CHECK(metadata->description() == "test");
    CHECK(metadata->authors() == std::vector<std::string>{"me", "myself"});
    CHECK(metadata->references().at("implementation") == std::vector<std::string>{"torch-power!"});
    CHECK(metadata->references().at("model") == std::vector<std::string>{"took a while to train"});
    CHECK(metadata->references().at("architecture").empty());
    CHECK(metadata->extra().at("key") == "value");

    metadata = ModelMetadataHolder::from_json("{\"class\": \"ModelMetadata\"}");
    CHECK(metadata->name().empty());
    CHECK(metadata->authors().empty());

    CHECK_THROWS_WITH(
        ModelMetadataHolder::from_json("{}"),
        StartsWith("'type' in JSON for ModelMetadata must be 'metatomic_model_metadata'")
    );
    CHECK_THROWS_WITH(
        ModelMetadataHolder::from_json("{\"class\": \"nope\"}"),
        StartsWith("'class' in legacy JSON for ModelMetadata must be 'ModelMetadata'")
    );

    CHECK_THROWS_WITH(
        metadata->set_authors({"me", ""}),
        StartsWith("author can not be empty string in ModelMetadata")
    );

    references = torch::Dict<std::string, std::vector<std::string>>();
    references.insert("unknown", std::vector<std::string>{"ref"});
    CHECK_THROWS_WITH(
        metadata->set_references(references),
        StartsWith("unknown key in references: unknown")
    );

    references = torch::Dict<std::string, std::vector<std::string>>();
    references.insert("model", std::vector<std::string>{""});
    CHECK_THROWS_WITH(
        metadata->set_references(references),
        StartsWith("reference can not be empty string (in 'model' section)")
    );

    // printing
    metadata = torch::make_intrusive<ModelMetadataHolder>();
    metadata->set_name("name");
    metadata->set_description(R"(Lorem ipsum dolor sit amet, consectetur
adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna
aliqua. Ut enim ad minim veniam, quis nostrud exercitation.)");
    metadata->set_authors({"Short author", "Some extremely long author that will take more than one line in the printed output"});
    references = torch::Dict<std::string, std::vector<std::string>>();
    references.insert("model", std::vector<std::string>{
        "a very long reference that will take more than one line in the printed output"
    });
    references.insert("architecture", std::vector<std::string>{"ref-2", "ref-3"});
    metadata->set_references(references);

    expected = R"(This is the name model
======================

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation.

Model authors
-------------

- Short author
- Some extremely long author that will take more than one line in the printed
  output

Model references
----------------

Please cite the following references when using this model:
- about this specific model:
  * a very long reference that will take more than one line in the printed
    output
- about the architecture of this model:
  * ref-2
  * ref-3
)";

    CHECK(metadata->print() == expected);
}
