#include <catch.hpp>

#include "metatomic.h"


TEST_CASE("Load plugins") {
    auto status = mta_load_plugin(PLUGIN_DIR "/test-c-plugin.so");
    CHECK(status == MTA_SUCCESS);

    // try to load the model with an explicit plugin name
    struct mta_model_t model;
    status = mta_load_model("test-c-plugin", "some_model", "{}", &model);
    CHECK(status == MTA_MODEL_NOT_SUPPORTED_ERROR);

    // load the plugin without specifying the plugin name
    status = mta_load_model(nullptr, "some_model", "{}", &model);
    CHECK(status == MTA_INVALID_PARAMETER_ERROR);

    const char* error_message;
    const char* error_origin;

    status = mta_last_error(&error_message, &error_origin, nullptr);
    REQUIRE(status == MTA_SUCCESS);

    CHECK(std::string(error_origin) == "metatomic-core");
    const char* expected_message = (
        "invalid parameter: failed to load model from 'some_model': tried the "
        "following plugins, but none could load the model: test-c-plugin"
    );
    CHECK(std::string(error_message) == expected_message);


    status = mta_load_plugin(PLUGIN_DIR "/bad-abi-plugin.so");
    CHECK(status == MTA_INVALID_PARAMETER_ERROR);

    status = mta_last_error(&error_message, &error_origin, nullptr);
    REQUIRE(status == MTA_SUCCESS);

    CHECK(std::string(error_origin) == "metatomic-core");
    expected_message = (
        "invalid parameter: can not register plugin 'bad-abi-plugin': "
        "plugin ABI version is 2, but metatomic expects 1"
    );
    CHECK(std::string(error_message) == expected_message);
}
