#include <catch.hpp>

#include "metatomic.hpp"


TEST_CASE("unit conversion factor") {
    // same unit -> factor = 1.0
    auto factor = metatomic::unit_conversion_factor("m", "m");
    CHECK(factor == 1.0);

    // kJ/mol -> eV
    factor = metatomic::unit_conversion_factor("kJ/mol", "eV");
    CHECK(factor == Approx(0.010364269656262174).epsilon(1e-15));

    REQUIRE_THROWS_WITH(
        metatomic::unit_conversion_factor("m", "kg"),
        "invalid parameter: dimension mismatch in unit conversion: "
        "'m' has dimension [L] but 'kg' has dimension [M]"
    );
}


TEST_CASE("metatdata formatting") {
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

    CHECK(metatomic::format_metadata(json) == expected);
}
