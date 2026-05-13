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
