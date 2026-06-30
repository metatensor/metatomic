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
