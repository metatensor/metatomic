#include <cstring>

#include <catch.hpp>

#include "metatomic.h"
#include "metatomic.hpp"


TEST_CASE("Version macros") {
    CHECK(std::string(METATOMIC_VERSION) == mta_version());

    auto version = std::to_string(METATOMIC_VERSION_MAJOR) + "."
        + std::to_string(METATOMIC_VERSION_MINOR) + "."
        + std::to_string(METATOMIC_VERSION_PATCH);

    // METATOMIC_VERSION should start with `x.y.z`
    CHECK(std::string(METATOMIC_VERSION).find(version) == 0);
}

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
