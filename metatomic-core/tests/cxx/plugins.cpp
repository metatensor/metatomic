#include <catch.hpp>

#include "metatomic.hpp"


TEST_CASE("Load plugins") {
    metatomic::load_plugin(PLUGIN_DIR "/test-c-plugin.so");

    REQUIRE_THROWS_WITH(
        metatomic::load_model("some_model", "{}", "test-c-plugin"),
        "invalid parameter: failed to load model from 'some_model': plugin 'test-c-plugin' could not load the model"
    );

    REQUIRE_THROWS_WITH(
        metatomic::load_model("some_model"),
        "invalid parameter: failed to load model from 'some_model': tried the "
        "following plugins, but none could load the model: test-c-plugin"
    );

    REQUIRE_THROWS_WITH(
        metatomic::load_plugin(PLUGIN_DIR "/bad-abi-plugin.so"),
        "invalid parameter: can not register plugin 'bad-abi-plugin': "
        "plugin ABI version is 2, but metatomic expects 1"
    );
}


TEST_CASE("Load C++ plugins") {
    metatomic::load_plugin(PLUGIN_DIR "/test-cxx-plugin.so");

    auto raw_model = metatomic::load_model("test-cxx-model", "{}", "test-cxx-plugin");
    auto model = metatomic::ExternalModel(raw_model);

    auto metadata = model.metadata();
    CHECK(metadata.name() == "simple C++ plugin model");

    auto capabilities = model.capabilities();
    CHECK(capabilities.length_unit() == "nm");
    REQUIRE(capabilities.outputs().size() == 1);
    CHECK(capabilities.outputs()[0].name() == "energy");

    REQUIRE_THROWS_WITH(
        metatomic::load_model("unknown", "{}", "test-cxx-plugin"),
        "invalid parameter: failed to load model from 'unknown': plugin "
        "'test-cxx-plugin' could not load the model"
    );

    REQUIRE_THROWS_WITH(
        metatomic::load_model("throws", "{}", "test-cxx-plugin"),
        "load_model_cxx: intentional failure for 'throws'"
    );
}
