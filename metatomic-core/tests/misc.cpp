#include <cstring>

#include <catch.hpp>

#include "metatomic.h"


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
