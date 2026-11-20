#include <torch/torch.h>

#include "metatomic/torch/legacy/output.hpp"
#include "metatomic/torch/legacy/model.hpp"
#include "metatomic/torch/legacy/capabilities.hpp"

#include <catch.hpp>
using namespace Catch::Matchers;

using namespace metatomic_torch;


TEST_CASE("Pick variant") {
    auto output_base = torch::make_intrusive<ModelOutputHolder>();
    output_base->description = "my awesome energy";

    auto variantA = torch::make_intrusive<ModelOutputHolder>();
    variantA->description = "Variant A of the output";

    auto variantfoo = torch::make_intrusive<ModelOutputHolder>();
    variantfoo->description = "Variant foo of the output";

    auto outputs = torch::Dict<std::string, ModelOutput>();
    outputs.insert("energy", output_base);
    outputs.insert("energy/A", variantA);
    outputs.insert("energy/foo", variantfoo);

    CHECK(metatomic_torch::pick_output("energy", outputs) == "energy");
    CHECK(metatomic_torch::pick_output("energy", outputs, "A") == "energy/A");
    CHECK_THROWS_WITH(metatomic_torch::pick_output("foo", outputs), StartsWith("output 'foo' not found in outputs"));
    CHECK_THROWS_WITH(metatomic_torch::pick_output("energy", outputs, "C"), StartsWith("variant 'C' for output 'energy' not found in outputs"));

    (void)outputs.erase("energy");
    const auto *err = "output 'energy' has no default variant and no `desired_variant` was given. "
        "Available variants are:\n"
        " - 'energy/A'  : Variant A of the output\n"
        " - 'energy/foo': Variant foo of the output";
    CHECK_THROWS_WITH(metatomic_torch::pick_output("energy", outputs), StartsWith(err));
}

TEST_CASE("ModelOutput") {
    // save to JSON
    auto output = torch::make_intrusive<ModelOutputHolder>();
    output->description = "my awesome energy";
    output->set_unit("kJ / mol");
    output->set_sample_kind("system");
    output->explicit_gradients = {"baz", "not.this-one_"};

    const auto* expected = R"({
    "class": "ModelOutput",
    "description": "my awesome energy",
    "explicit_gradients": [
        "baz",
        "not.this-one_"
    ],
    "sample_kind": "system",
    "unit": "kJ / mol"
})";
    CHECK(output->to_json() == expected);

    // load from JSON
    std::string json = R"({
    "class": "ModelOutput",
    "explicit_gradients": []
})";
    output = ModelOutputHolder::from_json(json);
    CHECK(output->unit().empty());
    CHECK(output->sample_kind() == "system");
    CHECK(output->explicit_gradients.empty());

    CHECK_THROWS_WITH(
        ModelOutputHolder::from_json("{}"),
        StartsWith("expected 'class' in JSON for ModelOutput, did not find it")
    );
    CHECK_THROWS_WITH(
        ModelOutputHolder::from_json("{\"class\": \"nope\"}"),
        StartsWith("'class' in JSON for ModelOutput must be 'ModelOutput'")
    );
}

TEST_CASE("ModelEvaluationOptions") {
    // save to JSON
    auto options = torch::make_intrusive<ModelEvaluationOptionsHolder>();
    options->set_length_unit("nanometer");

    options->outputs.insert("output_1", torch::make_intrusive<ModelOutputHolder>());

    auto output = torch::make_intrusive<ModelOutputHolder>();
    output->set_sample_kind("atom");
    output->set_unit("eV");
    options->outputs.insert("output_2", output);

    const auto* expected = R"({
    "class": "ModelEvaluationOptions",
    "length_unit": "nanometer",
    "outputs": {
        "output_1": {
            "class": "ModelOutput",
            "description": "",
            "explicit_gradients": [],
            "sample_kind": "system",
            "unit": ""
        },
        "output_2": {
            "class": "ModelOutput",
            "description": "",
            "explicit_gradients": [],
            "sample_kind": "atom",
            "unit": "eV"
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
    CHECK(output->unit().empty());
    CHECK(output->sample_kind() == "system");
    CHECK(output->explicit_gradients == std::vector<std::string>{"test"});

    CHECK_THROWS_WITH(
        ModelEvaluationOptionsHolder::from_json("{}"),
        StartsWith("expected 'class' in JSON for ModelEvaluationOptions, did not find it")
    );
    CHECK_THROWS_WITH(
        ModelEvaluationOptionsHolder::from_json("{\"class\": \"nope\"}"),
        StartsWith("'class' in JSON for ModelEvaluationOptions must be 'ModelEvaluationOptions'")
    );
}

TEST_CASE("ModelCapabilities") {
    // save to JSON
    auto capabilities = torch::make_intrusive<ModelCapabilitiesHolder>();
    capabilities->set_length_unit("nanometer");
    capabilities->interaction_range = 1.4;
    capabilities->atomic_types = {1, 2, -43};
    capabilities->set_dtype("float32");
    capabilities->supported_devices = {"cuda", "xla", "cpu"};

    auto output = torch::make_intrusive<ModelOutputHolder>();
    output->set_sample_kind("atom");
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
            "sample_kind": "atom",
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
            "per_atom": true,
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
    CHECK(output->unit().empty());
    // check that we can load JSON with `per_atom` and without `sample_kind`
    CHECK(output->sample_kind() == "atom");
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
        Contains("unknown unit 'unknown'")
    );

    auto capabilities_variants = torch::make_intrusive<ModelCapabilitiesHolder>();
    auto output_variant = torch::make_intrusive<ModelOutputHolder>();
    output_variant->set_sample_kind("atom");
    output_variant->set_unit("kJ/mol");
    output_variant->description = "variant output";

    auto outputs_variant = torch::Dict<std::string, ModelOutput>();
    outputs_variant.insert("energy", output_variant);
    outputs_variant.insert("energy/PBE0", output_variant);

    // should not throw
    capabilities_variants->set_outputs(outputs_variant);

    // both keys must be present in the stored outputs
    auto stored = capabilities_variants->outputs();
    CHECK(stored.contains("energy"));
    CHECK(stored.contains("energy/PBE0"));

    auto capabilities_non_standard = torch::make_intrusive<ModelCapabilitiesHolder>();
    auto output_non_standard = torch::make_intrusive<ModelOutputHolder>();
    auto outputs_non_standard = torch::Dict<std::string, ModelOutput>();

    // missing domain in custom output
    outputs_non_standard.insert("::not-a-standard", output_non_standard);
    CHECK_THROWS_WITH(
        capabilities_non_standard->set_outputs(outputs_non_standard),
        Contains(
            "invalid model output name '::not-a-standard': "
            "non-standard names should look like '<domain>::<quantity>' "
            "with non-empty domain and quantity."
        )
    );
    outputs_non_standard.clear();

    // missing output in custom output
    outputs_non_standard.insert("not-a-standard::", output_non_standard);
    CHECK_THROWS_WITH(
        capabilities_non_standard->set_outputs(outputs_non_standard),
        Contains(
            "invalid model output name 'not-a-standard::': "
            "non-standard names should look like '<domain>::<quantity>' "
            "with non-empty domain and quantity."
        )
    );
    outputs_non_standard.clear();

    // missing output in custom output
    outputs_non_standard.insert("not-a-standard::something::", output_non_standard);
    CHECK_THROWS_WITH(
        capabilities_non_standard->set_outputs(outputs_non_standard),
        Contains(
            "invalid model output name 'not-a-standard::something::': "
            "non-standard names should look like '<domain>::<quantity>' "
            "with non-empty domain and quantity"
        )
    );
    outputs_non_standard.clear();

    // missing output in variant
    outputs_non_standard.insert("/not-a-standard", output_non_standard);
    CHECK_THROWS_WITH(
        capabilities_non_standard->set_outputs(outputs_non_standard),
        Contains(
            "invalid model output name '/not-a-standard': "
            "variant names should look like '<quantity>/<variant>' "
            "with non-empty quantity and variant."
        )
    );
    outputs_non_standard.clear();

    // missing variant
    outputs_non_standard.insert("energy/", output_non_standard);
    CHECK_THROWS_WITH(
        capabilities_non_standard->set_outputs(outputs_non_standard),
        Contains(
            "invalid model output name 'energy/': "
            "variant names should look like '<quantity>/<variant>' "
            "with non-empty quantity and variant."
        )
    );
    outputs_non_standard.clear();

    // empty output in custom name with variant
    outputs_non_standard.insert("not-a-standard::/not-a-standard", output_non_standard);
    CHECK_THROWS_WITH(
        capabilities_non_standard->set_outputs(outputs_non_standard),
        Contains(
            "invalid model output name 'not-a-standard::/not-a-standard': "
            "non-standard name with variant should look like "
            "'<domain>::<quantity>/<variant>' with non-empty domain, quantity and variant."
        )
    );
    outputs_non_standard.clear();

    // test for intended naming
    output_non_standard->set_unit("Ry");
    outputs_non_standard.insert("energy", output_non_standard);
    outputs_non_standard.insert("custom::custom-output/variant", output_non_standard);
    CHECK_NOTHROW(capabilities_non_standard->set_outputs(outputs_non_standard));
    outputs_non_standard.clear();

    // "foo" (not known, no ::, no /)
    outputs_non_standard.insert("foo", output_non_standard);
    CHECK_THROWS_WITH(
        capabilities_non_standard->set_outputs(outputs_non_standard),
        Contains(
            "invalid model output name 'foo': this is not a known quantity. "
            "Variant names should look like '<quantity>/<variant>'. "
            "Non-standard names should look like '<domain>::<quantity>[/<variant>]'"
        )
    );

    // check for variant description warning
    struct WarningHandler: public torch::WarningHandler {
        virtual ~WarningHandler() override = default;
        void process(const torch::Warning& warning) override {
            auto expected = std::string(
                "'energy' defines 3 output variants and 'energy/foo' has an empty description. "
                "Consider adding meaningful descriptions helping users to distinguish between them."
            );
            CHECK(warning.msg() == expected);
        }
    };

    auto* old_handler = torch::WarningUtils::get_warning_handler();
    auto check_expected_warning = WarningHandler();
    torch::WarningUtils::set_warning_handler(&check_expected_warning);

    auto output_variant_no_desc = torch::make_intrusive<ModelOutputHolder>();
    output_variant_no_desc->set_unit("eV");
    outputs_variant.insert("energy/foo", output_variant_no_desc);
    capabilities_variants->set_outputs(outputs_variant);

    torch::WarningUtils::set_warning_handler(old_handler);
}
