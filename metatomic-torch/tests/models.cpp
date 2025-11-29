#include <torch/torch.h>

#include <metatomic/torch.hpp>
using namespace metatomic_torch;

#include <catch.hpp>
using namespace Catch::Matchers;


TEST_CASE("Models metadata") {
    SECTION("NeighborListOptions") {
        // save to JSON
        auto options = torch::make_intrusive<NeighborListOptionsHolder>(
            /*cutoff=*/ 3.5426,
            /*full_list=*/ true,
            /*strict=*/ true,
            /*requestor=*/ "request"
        );
        options->add_requestor("another request");

        const auto* expected = R"({
    "class": "NeighborListOptions",
    "cutoff": 4615159644819978768,
    "full_list": true,
    "length_unit": "",
    "requestors": [
        "request",
        "another request"
    ],
    "strict": true
})";
        CHECK(options->to_json() == expected);

        // load from JSON
        std::string json = R"({
    "cutoff": 4615159644819978768,
    "full_list": false,
    "strict": false,
    "class": "NeighborListOptions",
    "requestors": ["some request", "hello.world"]
})";
        options = NeighborListOptionsHolder::from_json(json);
        CHECK(options->cutoff() == 3.5426);
        CHECK(options->full_list() == false);
        CHECK(options->strict() == false);
        CHECK(options->requestors() == std::vector<std::string>{"some request", "hello.world"});

        CHECK_THROWS_WITH(
            NeighborListOptionsHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for NeighborListOptions, did not find it")
        );
        CHECK_THROWS_WITH(
            NeighborListOptionsHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for NeighborListOptions must be 'NeighborListOptions'")
        );

        CHECK_THROWS_WITH(options->set_length_unit("unknown"),
            StartsWith("unknown unit 'unknown' for length")
        );
    }

    SECTION("ModelOutput") {
        // save to JSON
        auto output = torch::make_intrusive<ModelOutputHolder>();
        output->description = "my awesome energy";
        output->set_quantity("energy");
        output->set_unit("kJ / mol");
        output->per_atom = false;
        output->explicit_gradients = {"baz", "not.this-one_"};

        const auto* expected = R"({
    "class": "ModelOutput",
    "description": "my awesome energy",
    "explicit_gradients": [
        "baz",
        "not.this-one_"
    ],
    "per_atom": false,
    "quantity": "energy",
    "unit": "kJ / mol"
})";
        CHECK(output->to_json() == expected);

        // load from JSON
        std::string json = R"({
    "class": "ModelOutput",
    "quantity": "length",
    "explicit_gradients": []
})";
        output = ModelOutputHolder::from_json(json);
        CHECK(output->quantity() == "length");
        CHECK(output->unit().empty());
        CHECK(output->per_atom == false);
        CHECK(output->explicit_gradients.empty());

        CHECK_THROWS_WITH(
            ModelOutputHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for ModelOutput, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelOutputHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for ModelOutput must be 'ModelOutput'")
        );

        CHECK_THROWS_WITH(output->set_unit("unknown"),
            StartsWith("unknown unit 'unknown' for length")
        );

        struct WarningHandler: public torch::WarningHandler {
            virtual ~WarningHandler() override = default;
            void process(const torch::Warning& warning) override {
                CHECK(warning.msg() == "unknown quantity 'unknown', only [energy force length momentum pressure] are supported");
            }
        };

        auto* old_handler = torch::WarningUtils::get_warning_handler();
        auto check_expected_warning = WarningHandler();
        torch::WarningUtils::set_warning_handler(&check_expected_warning);

        output->set_quantity("unknown"),

        torch::WarningUtils::set_warning_handler(old_handler);
    }

    SECTION("ModelEvaluationOptions") {
        // save to JSON
        auto options = torch::make_intrusive<ModelEvaluationOptionsHolder>();
        options->set_length_unit("nanometer");

        options->outputs.insert("output_1", torch::make_intrusive<ModelOutputHolder>());

        auto output = torch::make_intrusive<ModelOutputHolder>();
        output->per_atom = true;
        output->set_quantity("something");
        output->set_unit("something");
        options->outputs.insert("output_2", output);

        const auto* expected = R"({
    "class": "ModelEvaluationOptions",
    "length_unit": "nanometer",
    "outputs": {
        "output_1": {
            "class": "ModelOutput",
            "description": "",
            "explicit_gradients": [],
            "per_atom": false,
            "quantity": "",
            "unit": ""
        },
        "output_2": {
            "class": "ModelOutput",
            "description": "",
            "explicit_gradients": [],
            "per_atom": true,
            "quantity": "something",
            "unit": "something"
        }
    },
    "selected_atoms": null
})";
        CHECK(options->to_json() == expected);


        // load from JSON
        std::string json =R"({
    "length_unit": "Angstrom",
    "outputs": {
        "foo": {
            "explicit_gradients": ["test"],
            "class": "ModelOutput"
        }
    },
    "selected_atoms": {
        "names": ["system", "atom"],
        "values": [0, 1, 4, 5]
    },
    "class": "ModelEvaluationOptions"
})";

        options = ModelEvaluationOptionsHolder::from_json(json);
        CHECK(options->length_unit() == "Angstrom");
        auto expected_selection = metatensor_torch::LabelsHolder::create(
            {"system", "atom"},
            {{0, 1}, {4, 5}}
        );
        CHECK(*options->get_selected_atoms().value() == *expected_selection);

        output = options->outputs.at("foo");
        CHECK(output->quantity().empty());
        CHECK(output->unit().empty());
        CHECK(output->per_atom == false);
        CHECK(output->explicit_gradients == std::vector<std::string>{"test"});

        CHECK_THROWS_WITH(
            ModelEvaluationOptionsHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for ModelEvaluationOptions, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelEvaluationOptionsHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for ModelEvaluationOptions must be 'ModelEvaluationOptions'")
        );

        CHECK_THROWS_WITH(options->set_length_unit("unknown"),
            StartsWith("unknown unit 'unknown' for length")
        );
    }

    SECTION("ModelCapabilities") {
        // save to JSON
        auto capabilities = torch::make_intrusive<ModelCapabilitiesHolder>();
        capabilities->set_length_unit("nanometer");
        capabilities->interaction_range = 1.4;
        capabilities->atomic_types = {1, 2, -43};
        capabilities->set_dtype("float32");
        capabilities->supported_devices = {"cuda", "xla", "cpu"};

        auto output = torch::make_intrusive<ModelOutputHolder>();
        output->per_atom = true;
        output->set_quantity("length");
        output->explicit_gradients.emplace_back("µ-λ");

        auto outputs = torch::Dict<std::string, ModelOutput>();
        outputs.insert("tests::bar", output);
        capabilities->set_outputs(outputs);

        const auto* expected = R"({
    "atomic_types": [
        1,
        2,
        -43
    ],
    "class": "ModelCapabilities",
    "dtype": "float32",
    "interaction_range": 4608983858650965606,
    "length_unit": "nanometer",
    "outputs": {
        "tests::bar": {
            "class": "ModelOutput",
            "description": "",
            "explicit_gradients": [
                "\u00b5-\u03bb"
            ],
            "per_atom": true,
            "quantity": "length",
            "unit": ""
        }
    },
    "supported_devices": [
        "cuda",
        "xla",
        "cpu"
    ]
})";
        CHECK(capabilities->to_json() == expected);


        // load from JSON
        std::string json =R"({
    "length_unit": "µm",
    "outputs": {
        "tests::foo": {
            "explicit_gradients": ["\u00b5-test"],
            "class": "ModelOutput"
        }
    },
    "atomic_types": [
        1,
        -2
    ],
    "class": "ModelCapabilities"
})";

        capabilities = ModelCapabilitiesHolder::from_json(json);
        CHECK(capabilities->length_unit() == "µm");
        CHECK(capabilities->dtype().empty());
        CHECK(capabilities->atomic_types == std::vector<int64_t>{1, -2});

        output = capabilities->outputs().at("tests::foo");
        CHECK(output->quantity().empty());
        CHECK(output->unit().empty());
        CHECK(output->per_atom == false);
        CHECK(output->explicit_gradients == std::vector<std::string>{"µ-test"});

        CHECK_THROWS_WITH(
            ModelCapabilitiesHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for ModelCapabilities, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelCapabilitiesHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for ModelCapabilities must be 'ModelCapabilities'")
        );

        CHECK_THROWS_WITH(capabilities->set_length_unit("unknown"),
            StartsWith("unknown unit 'unknown' for length")
        );

        auto capabilities_variants = torch::make_intrusive<ModelCapabilitiesHolder>();
        auto output_variant = torch::make_intrusive<ModelOutputHolder>();
        output_variant->per_atom = true;
        output_variant->description = "variant output";

        auto outputs_variant = torch::Dict<std::string, ModelOutput>();
        outputs_variant.insert("energy", output_variant);
        outputs_variant.insert("energy/PBE0", output_variant);

        // should not throw
        capabilities_variants->set_outputs(outputs_variant);

        // both keys must be present in the stored outputs
        auto stored = capabilities_variants->outputs();
        CHECK(stored.find("energy") != stored.end());
        CHECK(stored.find("energy/PBE0") != stored.end());

        auto capabilities_non_standard = torch::make_intrusive<ModelCapabilitiesHolder>();
        auto output_non_standard = torch::make_intrusive<ModelOutputHolder>();
        auto outputs_non_standard = torch::Dict<std::string, ModelOutput>();

        // missing domain in custom output
        outputs_non_standard.insert("::not-a-standard", output_non_standard);
        CHECK_THROWS_WITH(
            capabilities_non_standard->set_outputs(outputs_non_standard),
            Contains(
                "Invalid name for model output: '::not-a-standard'. "
                "Non-standard names should look like '<domain>::<output>' "
                "with non-empty domain and output."
            )
        );
        outputs_non_standard.clear();

        // missing output in custom output
        outputs_non_standard.insert("not-a-standard::", output_non_standard);
        CHECK_THROWS_WITH(
            capabilities_non_standard->set_outputs(outputs_non_standard),
            Contains(
                "Invalid name for model output: 'not-a-standard::'. "
                "Non-standard names should look like '<domain>::<output>' "
                "with non-empty domain and output."
            )
        );
        outputs_non_standard.clear();

        // missing output in custom output
        outputs_non_standard.insert("not-a-standard::something::", output_non_standard);
        CHECK_THROWS_WITH(
            capabilities_non_standard->set_outputs(outputs_non_standard),
            Contains(
                "Invalid name for model output: 'not-a-standard::something::'. "
                "Non-standard names should look like '<domain>::<output>' "
                "with non-empty domain and output"
            )
        );
        outputs_non_standard.clear();

        // missing output in variant
        outputs_non_standard.insert("/not-a-standard", output_non_standard);
        CHECK_THROWS_WITH(
            capabilities_non_standard->set_outputs(outputs_non_standard),
            Contains(
                "Invalid name for model output: '/not-a-standard'. Variant names "
                "should look like '<output>/<variant>' with non-empty output and variant."
            )
        );
        outputs_non_standard.clear();

        // missing variant
        outputs_non_standard.insert("energy/", output_non_standard);
        CHECK_THROWS_WITH(
            capabilities_non_standard->set_outputs(outputs_non_standard),
            Contains(
                "Invalid name for model output: 'energy/'. Variant names should "
                "look like '<output>/<variant>' with non-empty output and variant."
            )
        );
        outputs_non_standard.clear();

        // empty output in custom name with variant
        outputs_non_standard.insert("not-a-standard::/not-a-standard", output_non_standard);
        CHECK_THROWS_WITH(
            capabilities_non_standard->set_outputs(outputs_non_standard),
            Contains(
                "Invalid name for model output: 'not-a-standard::/not-a-standard'. "
                "Non-standard name with variant should look like "
                "'<domain>::<output>/<variant>' with non-empty domain, output and variant."
            )
        );
        outputs_non_standard.clear();

        // test for intended naming
        outputs_non_standard.insert("energy", output_non_standard);
        outputs_non_standard.insert("custom::custom-output/variant", output_non_standard);
        CHECK_NOTHROW(capabilities_non_standard->set_outputs(outputs_non_standard));
        outputs_non_standard.clear();

        // "foo" (not known, no ::, no /)
        outputs_non_standard.insert("foo", output_non_standard);
        CHECK_THROWS_WITH(
            capabilities_non_standard->set_outputs(outputs_non_standard),
            Contains(
                "Invalid name for model output: 'foo' is not a known output. "
                "Variant names should be of the form '<output>/<variant>'. "
                "Non-standard names should have the form '<domain>::<output>'."
            )
        );

        // check for variant description warning
        struct WarningHandler: public torch::WarningHandler {
            virtual ~WarningHandler() override = default;
            void process(const torch::Warning& warning) override {
                CHECK(warning.msg() == "'energy' defines 3 output variants and 'energy/foo' has an empty description. "
                "Consider adding meaningful descriptions helping users to distinguish between them.");
            }
        };

        auto* old_handler = torch::WarningUtils::get_warning_handler();
        auto check_expected_warning = WarningHandler();
        torch::WarningUtils::set_warning_handler(&check_expected_warning);

        auto output_variant_no_desc = torch::make_intrusive<ModelOutputHolder>();
        outputs_variant.insert("energy/foo", output_variant_no_desc);
        capabilities_variants->set_outputs(outputs_variant);

        torch::WarningUtils::set_warning_handler(old_handler);
    }

    SECTION("ModelMetadata") {
        // save to JSON
        auto metadata = torch::make_intrusive<ModelMetadataHolder>();
        metadata->name = "some name";
        metadata->description = "describing it";
        metadata->authors = {"John Doe", "Napoleon"};
        metadata->references.insert("model", std::vector<std::string>{"some-ref"});
        metadata->references.insert("architecture", std::vector<std::string>{"ref-2", "ref-3"});

        const auto* expected = R"({
    "authors": [
        "John Doe",
        "Napoleon"
    ],
    "class": "ModelMetadata",
    "description": "describing it",
    "extra": {},
    "name": "some name",
    "references": {
        "architecture": [
            "ref-2",
            "ref-3"
        ],
        "model": [
            "some-ref"
        ]
    }
})";
        CHECK(metadata->to_json() == expected);


        // load from JSON
        std::string json =R"({
    "class": "ModelMetadata",
    "name": "foo",
    "description": "test",
    "authors": ["me", "myself"],
    "references": {
        "implementation": ["torch-power!"],
        "model": ["took a while to train"]
    }
})";

        metadata = ModelMetadataHolder::from_json(json);
        CHECK(metadata->name == "foo");
        CHECK(metadata->description == "test");
        CHECK(metadata->authors == std::vector<std::string>{"me", "myself"});
        CHECK(metadata->references.at("implementation") == std::vector<std::string>{"torch-power!"});
        CHECK(metadata->references.at("model") == std::vector<std::string>{"took a while to train"});

        CHECK_THROWS_WITH(
            ModelMetadataHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for ModelMetadata, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelMetadataHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for ModelMetadata must be 'ModelMetadata'")
        );

        // printing
        metadata = torch::make_intrusive<ModelMetadataHolder>();
        metadata->name = "name";
        metadata->description = R"(Lorem ipsum dolor sit amet, consectetur
adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna
aliqua. Ut enim ad minim veniam, quis nostrud exercitation.)";
        metadata->authors = {"Short author", "Some extremely long author that will take more than one line in the printed output"};
        metadata->references.insert("model", std::vector<std::string>{
            "a very long reference that will take more than one line in the printed output"
        });
        metadata->references.insert("architecture", std::vector<std::string>{"ref-2", "ref-3"});

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
}
