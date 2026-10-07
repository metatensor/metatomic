#include <filesystem>

#include <catch.hpp>

#include <metatensor.hpp>
#include <metatomic.hpp>

#include "./helpers.hpp"

TEST_CASE("Python plugin") {
    const auto* plugin_path = std::getenv("METATOMIC_TESTS_PYTHON_PLUGIN");
    REQUIRE(plugin_path != nullptr);

    metatomic::load_plugin(plugin_path);

    auto path = std::filesystem::path(__FILE__).parent_path() / "python-model.py";
    auto model = metatomic::load_model(path.string(), std::nullopt, "python");

    auto capabilities = model.capabilities();
    CHECK(capabilities.length_unit() == "nm");

    const auto& outputs = capabilities.outputs();
    REQUIRE(outputs.size() == 1);
    CHECK(outputs[0].name() == "test::sum_of_positions");

    CHECK(model.metadata().name() == "sum of positions");
    CHECK(model.requested_pair_lists().empty());
    CHECK(model.requested_inputs().empty());

    auto systems = std::vector<metatomic::System>();
    systems.push_back(test_system(4));
    systems.push_back(test_system(2));

    auto results = metatomic::execute_model(
        model, systems, std::nullopt, {outputs[0]}, /*check_consistency=*/ true
    );
    REQUIRE(results.size() == 1);

    auto block = results[0].block_by_id(0);
    CHECK(block.samples() == metatensor::Labels({"system"}, {{0}, {1}}));

    auto values = block.values<float>();
    CHECK(values(0, 0) == Approx(78.0));
    CHECK(values(1, 0) == Approx(21.0));

    auto selected_atoms = metatensor::Labels({"system", "atom"}, {{0, 1}});
    CHECK_THROWS_WITH(
        metatomic::execute_model(model, systems, selected_atoms, {outputs[0]}, true),
        "AssertionError: selected_atoms is not supported by this model"
    );

    CHECK_THROWS_WITH(
        metatomic::execute_model(model, systems, std::nullopt, {outputs[0], outputs[0]}, true),
        "ValueError: expected exactly one requested output, got 2"
    );
}
