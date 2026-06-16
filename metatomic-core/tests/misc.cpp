#include <cstring>

#include <catch.hpp>

#include "metatomic.h"
#include "metatomic.hpp"


// TEST_CASE("Version macros") {
//     CHECK(std::string(METATOMIC_VERSION) == mta_version());

//     auto version = std::to_string(METATOMIC_VERSION_MAJOR) + "."
//         + std::to_string(METATOMIC_VERSION_MINOR) + "."
//         + std::to_string(METATOMIC_VERSION_PATCH);

//     // METATOMIC_VERSION should start with `x.y.z`
//     CHECK(std::string(METATOMIC_VERSION).find(version) == 0);
// }

TEST_CASE("mta_string_t") {
    auto* str = mta_string_create("hello");
    REQUIRE(str != nullptr);

    const char* view = mta_string_view(str);
    CHECK(std::strlen(view) == 5);
    CHECK(std::string(view) == "hello");
    mta_string_free(str);

    // empty string
    str = mta_string_create("");
    REQUIRE(str != nullptr);
    CHECK(std::string(mta_string_view(str)) == "");
    mta_string_free(str);

    // special characters
    str = mta_string_create("a\nb\tc\xFFºµ");
    REQUIRE(str != nullptr);
    CHECK(std::string(mta_string_view(str)) == std::string("a\nb\tc\xFFºµ"));
    mta_string_free(str);

    // long string
    std::string long_str(10000, 'x');
    str = mta_string_create(long_str.c_str());
    REQUIRE(str != nullptr);
    CHECK(std::string(mta_string_view(str)) == long_str);
    mta_string_free(str);

    // free on a null pointer should work
    mta_string_free(nullptr);
}

TEST_CASE("unit conversion factor") {
    SECTION("C API") {
        double factor = 0.0;

        // same unit -> factor = 1.0
        auto status = mta_unit_conversion_factor("m", "m", &factor);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(factor == 1.0);

        // kJ/mol -> eV
        CHECK(mta_unit_conversion_factor("kJ/mol", "eV", &factor) == MTA_SUCCESS);
        CHECK(factor == Approx(0.010364269656262174).epsilon(1e-15));

        // dimension mismatch -> error
        status = mta_unit_conversion_factor("m", "kg", &factor);
        REQUIRE(status != MTA_SUCCESS);

        const char* error_msg = nullptr;
        mta_last_error(&error_msg, nullptr, nullptr);
        CHECK(std::string(error_msg) ==
            "invalid parameter: dimension mismatch in unit conversion: "
            "'m' has dimension [L] but 'kg' has dimension [M]"
        );
    }

    SECTION("C++ API") {
        // same unit -> factor = 1.0
        auto factor = metatomic::unit_conversion_factor("m", "m");
        CHECK(factor == 1.0);

        // kJ/mol -> eV
        factor = metatomic::unit_conversion_factor("kJ/mol", "eV");
        CHECK(factor == Approx(0.010364269656262174).epsilon(1e-15));

        // dimension mismatch -> error
        try{
            factor = metatomic::unit_conversion_factor("m", "kg");
        }
        catch(metatomic::Error& e){
            CHECK(std::string(e.what()) == "invalid parameter: dimension mismatch in unit conversion: 'm' has dimension [L] but 'kg' has dimension [M]");
        }
    }
}


TEST_CASE("model metatdata formatting") {
    std::string json =R"({
    "type": "metatomic_model_metadata",
    "name": "name",
    "description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation.",
    "authors": ["Short author", "Some extremely long author that will take more than one line in the printed output"],
    "references": {
        "architecture": ["ref-2", "ref-3"],
        "model": ["a very long reference that will take more than one line in the printed output"],
        "implementation": []
    },
    "extra": {}
})";
    auto* mta_string = mta_string_create("");
    REQUIRE(mta_string != nullptr);
    auto status = mta_format_metadata(json.c_str(), &mta_string);
    REQUIRE(status == MTA_SUCCESS);
    const auto* expected = R"(This is the name model
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
    CHECK(std::string(mta_string_view(mta_string)) == expected);
    mta_string_free(mta_string);
}

TEST_CASE("metadata", "C API"){
    SECTION("JSON serialization "){
        mta_pair_list_options_t *metadata = nullptr;
        auto status = mta_pair_list_options_create(0.42, true, true, &metadata);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(metadata != nullptr);
        char* json = nullptr;
        status = mta_pair_list_options_to_json(metadata, &json);
        REQUIRE(status == MTA_SUCCESS);
        // cutoff is 0.42 in double precision converted to hex
        CHECK(std::string(json) == R"({"type":"metatomic_pair_options","cutoff":"0x3fdae147ae147ae1","full_list":true,"strict":true,"requestors":[]})");
        mta_pair_list_options_free(metadata);
    }
    SECTION("JSON deserialization"){
        const char* json = "{\"type\":\"metatomic_pair_options\",\"cutoff\":\"0x3fdae147ae147ae1\",\"full_list\":true,\"strict\":true,\"requestors\":[]}";
        mta_pair_list_options_t *options = nullptr;
        auto status = mta_pair_list_options_from_json(json, &options);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(options != nullptr);
        // char* type = nullptr;
        // status = mta_pair_list_options_get_type(options, &type);
        // REQUIRE(status == MTA_SUCCESS);
        // CHECK(std::string(type) == "metatomic_pair_options");
        double cutoff = -1.0;
        status = mta_pair_list_options_get_cutoff(options, &cutoff);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(cutoff == Approx(0.42).epsilon(1e-15));
        bool full_list = false;
        status = mta_pair_list_options_get_full_list(options, &full_list);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(full_list == true);
        bool strict = false;
        status = mta_pair_list_options_get_strict(options, &strict);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(strict == true);
        char* requestor = nullptr;
        size_t num_requestors = 0;
        status = mta_pair_list_options_requestors_count(options, &num_requestors);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(num_requestors == 0);
        mta_pair_list_options_free(options);
    }
    SECTION("JSON deserialization with wrong JSON"){
        // boolean values are erroneously stored as strings
        const char* json = "{\"type\":\"metatomic_pair_options\",\"cutoff\":\"0x3fdae147ae147ae1\",\"full_list\":\"true\",\"strict\":\"true\",\"requestors\":[]}";
        mta_pair_list_options_t *options = nullptr;
        auto status = mta_pair_list_options_from_json(json, &options);
        CHECK(status == MTA_SERIALIZATION_ERROR);
    }
    SECTION("requestors"){
        mta_pair_list_options_t *options = nullptr;
        auto status = mta_pair_list_options_create(0.42, true, true, &options);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(options != nullptr);
        const char* requestor1 = "requestor1";
        const char* requestor2 = "requestor2";
        status = mta_pair_list_options_add_requestor(options, requestor1);
        REQUIRE(status == MTA_SUCCESS);
        status = mta_pair_list_options_add_requestor(options, requestor2);
        REQUIRE(status == MTA_SUCCESS);
        char* requestor = nullptr;
        size_t num_requestors = 0;
        status = mta_pair_list_options_requestors_count(options, &num_requestors);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(num_requestors == 2);
        status = mta_pair_list_options_get_requestor(options, 0, &requestor);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(requestor) == requestor1);
        status = mta_pair_list_options_get_requestor(options, 1, &requestor);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(requestor) == requestor2);
        mta_pair_list_options_free(options);
    }
}
