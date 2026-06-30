#include <catch.hpp>

#include "metatomic.h"


TEST_CASE("Load plugins") {
    auto status = mta_load_plugin(PLUGIN_DIR "/test-c-plugin.so");
    CHECK(status == MTA_SUCCESS);

    const char* error_message;
    const char* error_origin;

    struct mta_model_t model;
    status = mta_load_model("some_model", "{}", "test-c-plugin", &model);
    CHECK(status == MTA_INVALID_PARAMETER_ERROR);

    status = mta_last_error(&error_message, &error_origin, nullptr);
    REQUIRE(status == MTA_SUCCESS);

    CHECK(std::string(error_origin) == "metatomic-core");
    CHECK(std::string(error_message) == (
        "invalid parameter: failed to load model from 'some_model': plugin 'test-c-plugin' could not load the model"
    ));

    status = mta_load_model("some_model", "{}", nullptr, &model);
    CHECK(status == MTA_INVALID_PARAMETER_ERROR);

    status = mta_last_error(&error_message, &error_origin, nullptr);
    REQUIRE(status == MTA_SUCCESS);

    CHECK(std::string(error_origin) == "metatomic-core");
    CHECK(std::string(error_message) == (
        "invalid parameter: failed to load model from 'some_model': tried the "
        "following plugins, but none could load the model: test-c-plugin"
    ));


    status = mta_load_plugin(PLUGIN_DIR "/bad-abi-plugin.so");
    CHECK(status == MTA_INVALID_PARAMETER_ERROR);

    status = mta_last_error(&error_message, &error_origin, nullptr);
    REQUIRE(status == MTA_SUCCESS);

    CHECK(std::string(error_origin) == "metatomic-core");
    CHECK(std::string(error_message) == (
        "invalid parameter: can not register plugin 'bad-abi-plugin': "
        "plugin ABI version is 2, but metatomic expects 1"
    ));
}
