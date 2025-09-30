#include "metatomic/torch.hpp"

#include "../src/internal/shared_libraries.cpp"

#include <catch.hpp>

TEST_CASE("List shared libraries") {
    // force linking to metatomic_torch
    CHECK(!metatomic_torch::version().empty());

    auto libraries = metatomic_torch::details::get_loaded_libraries();

    bool found_metatensor = false;
    bool found_metatensor_torch = false;
    bool found_metatomic_torch = false;
    for (const auto& path : libraries) {
        if (path.find("metatensor_torch") != std::string::npos) {
            found_metatensor_torch = true;
            continue;
        }

        if (path.find("metatomic_torch") != std::string::npos) {
            found_metatomic_torch = true;
            continue;
        }

        if (path.find("metatensor") != std::string::npos) {
            found_metatensor = true;
            continue;
        }
    }

    CHECK(found_metatensor);
    CHECK(found_metatensor_torch);
    CHECK(found_metatomic_torch);
}
