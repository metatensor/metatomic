#include <cmath>
#include <string>

#include <torch/torch.h>

#include "metatomic/torch.hpp"

#include <catch.hpp>
using Catch::Detail::Approx;
using Catch::Matchers::Contains;
using Catch::Matchers::StartsWith;

class DeprecationWarningHandler : public torch::WarningHandler {
public:
    std::vector<std::string> messages;

    void process(const torch::Warning& warning) override {
        messages.push_back(warning.msg());
    }
};

// check that two doubles are approximately equal within a certain number of ULPs
static bool check_equal_ulp(double a, double b, size_t ulp = 4) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b);
    } else if (std::isinf(a) || std::isinf(b)) {
        return std::isinf(a) && std::isinf(b);
    } else {
        double next_a = a;
        double prev_a = a;
        for (size_t i = 0; i < ulp; ++i) {
            next_a = std::nextafter(next_a, std::numeric_limits<double>::infinity());
            prev_a = std::nextafter(prev_a, -std::numeric_limits<double>::infinity());
        }

        double next_b = b;
        double prev_b = b;
        for (size_t i = 0; i < ulp; ++i) {
            next_b = std::nextafter(next_b, std::numeric_limits<double>::infinity());
            prev_b = std::nextafter(prev_b, -std::numeric_limits<double>::infinity());
        }

        return (a >= prev_b && a <= next_b) && (b >= prev_a && b <= next_a);
    }
}

#define CHECK_APPROX_ULP(a, b) do { if (!check_equal_ulp(a, b)) { CHECK(a == b); } } while(0)


TEST_CASE("Simple conversions") {
    // Angstrom <-> Bohr
    double a_to_bohr = metatomic_torch::unit_conversion_factor("Angstrom", "Bohr");
    CHECK(a_to_bohr == Approx(1.88972612590485));

    double bohr_to_a = metatomic_torch::unit_conversion_factor("Bohr", "Angstrom");
    CHECK_APPROX_ULP(bohr_to_a, 0.52917721054482);

    // Angstrom <-> nm
    double a_to_nm = metatomic_torch::unit_conversion_factor("Angstrom", "nm");
    CHECK_APPROX_ULP(a_to_nm, 0.1);

    // Angstrom <-> meter
    double a_to_m = metatomic_torch::unit_conversion_factor("Angstrom", "meter");
    CHECK_APPROX_ULP(a_to_m, 1e-10);

    // eV <-> meV
    double ev_to_mev = metatomic_torch::unit_conversion_factor("eV", "meV");
    CHECK_APPROX_ULP(ev_to_mev, 1000.0);

    // eV <-> Hartree
    double ev_to_hartree = metatomic_torch::unit_conversion_factor("eV", "Hartree");
    CHECK(ev_to_hartree == Approx(0.0367493));

    // eV <-> Rydberg
    double ev_to_ry = metatomic_torch::unit_conversion_factor("eV", "Ry");
    CHECK(ev_to_ry == Approx(0.0734987));
}


TEST_CASE("Compound unit expressions") {
    // eV/Angstrom <-> Hartree/Bohr (force)
    double f_conv = metatomic_torch::unit_conversion_factor("eV/Angstrom", "Hartree/Bohr");
    CHECK(f_conv == Approx(0.0194469));

    // kJ/mol <-> kcal/mol (energy)
    double e_conv = metatomic_torch::unit_conversion_factor("kJ/mol", "kcal/mol");
    CHECK_APPROX_ULP(e_conv, 1000.0 / 4184.0);

    // eV/Angstrom^3 (pressure) -- identity
    double p_id = metatomic_torch::unit_conversion_factor("eV/Angstrom^3", "eV/A^3");
    CHECK_APPROX_ULP(p_id, 1.0);
}


TEST_CASE("Fractional power expressions") {
    // (eV*u)^(1/2) should have dimension of momentum: M*L/T
    // Compare to u*A/fs
    double conv = metatomic_torch::unit_conversion_factor("(eV*u)^(1/2)", "u*A/fs");
    // Cross-check: sqrt(eV_SI * u_SI) / (u_SI * A_SI / fs_SI)
    double ev_si = 1.602176634e-19;
    double u_si = 1.66053906660e-27;
    double a_si = 1e-10;
    double fs_si = 1e-15;
    double expected = std::sqrt(ev_si * u_si) / (u_si * a_si / fs_si);
    CHECK(conv == Approx(expected));

    // (eV/u)^(1/2) has dimension of velocity: L/T
    double v_conv = metatomic_torch::unit_conversion_factor("(eV/u)^(1/2)", "A/fs");
    double v_expected = std::sqrt(ev_si / u_si) / (a_si / fs_si);
    CHECK(v_conv == Approx(v_expected));
}


TEST_CASE("Case insensitive unit lookup") {
    double c1 = metatomic_torch::unit_conversion_factor("eV", "hartree");
    double c2 = metatomic_torch::unit_conversion_factor("EV", "HARTREE");
    double c3 = metatomic_torch::unit_conversion_factor("Ev", "Hartree");
    CHECK_APPROX_ULP(c1, c2);
    CHECK_APPROX_ULP(c1, c3);
}


TEST_CASE("Whitespace in unit expressions") {
    double c1 = metatomic_torch::unit_conversion_factor("eV / Angstrom", "Hartree/Bohr");
    double c2 = metatomic_torch::unit_conversion_factor("eV/Angstrom", "Hartree/Bohr");
    CHECK_APPROX_ULP(c1, c2);

    double c3 = metatomic_torch::unit_conversion_factor("( eV * u ) ^ ( 1 / 2 )", "u*A/fs");
    double c4 = metatomic_torch::unit_conversion_factor("(eV*u)^(1/2)", "u*A/fs");
    CHECK_APPROX_ULP(c3, c4);
}


TEST_CASE("Errors") {
    CHECK_THROWS_WITH(
        metatomic_torch::unit_conversion_factor("eV", "Angstrom"),
        Contains(
            "dimension mismatch in unit conversion: 'eV' has dimension "
            "L^2 T^-2 M but 'Angstrom' has dimension L"
        )
    );

    CHECK_THROWS_WITH(
        metatomic_torch::unit_conversion_factor("foobar", "eV"),
        Contains("unknown unit 'foobar'")
    );

    CHECK_THROWS_WITH(
        metatomic_torch::unit_conversion_factor("eV*(", "eV"),
        Contains("unit expression has unbalanced parentheses")
    );

    CHECK_THROWS_WITH(
        metatomic_torch::unit_conversion_factor("ev^eV", "eV"),
        Contains(
            "exponent in unit expression must be dimensionless, "
            "got dimension L^2 T^-2 M for exponent 'eV'"
        )
    );

    SECTION("Overflow") {
        // Test overflow with extreme exponent on unit with small SI factor
        // u (atomic mass unit) has factor 1.66e-27, so u^(-50) overflows
        CHECK_THROWS_WITH(
            metatomic_torch::unit_conversion_factor("u^-50", "u^-50"),
            Contains(
                "unit conversion factor overflows: exponentiation result is "
                "infinite or NaN for 'u ^ -50'"
            )
        );

        // eV has factor 1.6e-19, so eV^(-100) overflows
        CHECK_THROWS_WITH(
            metatomic_torch::unit_conversion_factor("eV^(-100)", "eV^(-100)"),
            Contains(
                "unit conversion factor overflows: exponentiation result is "
                "infinite or NaN for 'eV ^ -100'"
            )
        );

        // Test overflow with multiplication of units with extreme factors
        CHECK_THROWS_WITH(
            metatomic_torch::unit_conversion_factor("u^(-10) * u^(-10)", "u^(-20)"),
            Contains(
                "unit conversion factor overflows: multiplication result is "
                "infinite or NaN for '(u ^ -10) * (u ^ -10)'"
            )
        );

        // Test overflow with division creating extreme factor
        CHECK_THROWS_WITH(
            metatomic_torch::unit_conversion_factor("mol^10 / mol^(-10)", "mol"),
            Contains(
                "unit conversion factor overflows: division result is "
                "infinite or NaN for '(mol ^ 10) / (mol ^ -10)'"
            )
        );
    }
}



TEST_CASE("mol as dimensionless scaling factor") {
    // kJ/mol to eV: kJ_SI / mol_SI / eV_SI
    double conv = metatomic_torch::unit_conversion_factor("kJ/mol", "eV");
    double kj_si = 1000.0;
    double mol_si = 6.02214076e23;
    double ev_si = 1.602176634e-19;
    double expected = (kj_si / mol_si) / ev_si;
    CHECK_APPROX_ULP(conv, expected);
}


TEST_CASE("hbar/Bohr momentum conversion") {
    double conv = metatomic_torch::unit_conversion_factor("hbar/Bohr", "u*A/fs");
    // hbar_SI / Bohr_SI / (u_SI * A_SI / fs_SI)
    double hbar_si = 1.0545718176462e-34;
    double bohr_si = 5.2917721054482e-11;
    double u_si = 1.6605390689252e-27;
    double a_si = 1e-10;
    double fs_si = 1e-15;
    double expected = (hbar_si / bohr_si) / (u_si * a_si / fs_si);
    CHECK_APPROX_ULP(conv, expected);
}


TEST_CASE("Identity conversions") {
    CHECK_APPROX_ULP(metatomic_torch::unit_conversion_factor("eV", "eV"), 1.0);
    CHECK_APPROX_ULP(metatomic_torch::unit_conversion_factor("Angstrom", "Angstrom"), 1.0);
    CHECK_APPROX_ULP(metatomic_torch::unit_conversion_factor("u*A/fs", "u*A/fs"), 1.0);

    CHECK(metatomic_torch::unit_conversion_factor("", "eV") == 1.0);
    CHECK(metatomic_torch::unit_conversion_factor("eV", "") == 1.0);
    CHECK(metatomic_torch::unit_conversion_factor("", "") == 1.0);
}


TEST_CASE("Fractional power accumulated error") {
    // Test that (eV*u)^(1/3) then ^3 equals eV*u within tolerance
    // This verifies the 1e-10 dimension comparison tolerance handles
    // floating-point accumulation from fractional exponents
    CHECK(metatomic_torch::unit_conversion_factor("((eV*u)^(1/3))^3", "eV*u") == Approx(1.0));

    // Test with 1/2 then ^2
    CHECK_APPROX_ULP(metatomic_torch::unit_conversion_factor("((eV*u)^(1/2))^2", "eV*u"), 1.0);

    // Test nested: ((eV^(1/2))^2) should equal eV
    CHECK_APPROX_ULP(metatomic_torch::unit_conversion_factor("(eV^(1/2))^2", "eV"), 1.0);
}


TEST_CASE("Time unit conversions") {
    // second -> femtosecond
    double s_to_fs = metatomic_torch::unit_conversion_factor("s", "fs");
    CHECK_APPROX_ULP(s_to_fs, 1e15);

    // second -> picosecond
    double s_to_ps = metatomic_torch::unit_conversion_factor("second", "ps");
    CHECK_APPROX_ULP(s_to_ps, 1e12);

    // nanosecond -> femtosecond
    double ns_to_fs = metatomic_torch::unit_conversion_factor("ns", "fs");
    CHECK_APPROX_ULP(ns_to_fs, 1e6);

    // microsecond -> nanosecond
    double us_to_ns = metatomic_torch::unit_conversion_factor("us", "ns");
    CHECK_APPROX_ULP(us_to_ns, 1e3);
}


TEST_CASE("Micro sign (U+00B5) handling") {
    double c1 = metatomic_torch::unit_conversion_factor("um", "Angstrom");
    double c2 = metatomic_torch::unit_conversion_factor("µm", "Angstrom");
    CHECK_APPROX_ULP(c1, c2);

    // µs -> ns (microsecond via micro sign)
    double c3 = metatomic_torch::unit_conversion_factor("us", "ns");
    double c4 = metatomic_torch::unit_conversion_factor("µs", "ns");
    CHECK_APPROX_ULP(c3, c4);
}


TEST_CASE("ModelOutput rejects mismatched quantity and unit") {
    // energy quantity with a force unit
    CHECK_THROWS_WITH(
        torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
            "energy", "eV/A", "system", std::vector<std::string>{}, ""
        ),
        Contains(
            "unit 'eV/A' has dimension L T^-2 M which is incompatible "
            "with quantity 'energy' (expected L^2 T^-2 M)"
        )
    );

    // force quantity with an energy unit
    CHECK_THROWS_WITH(
        torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
            "force", "eV", "system", std::vector<std::string>{}, ""
        ),
        Contains(
            "unit 'eV' has dimension L^2 T^-2 M which is incompatible "
            "with quantity 'force' (expected L T^-2 M)"
        )
    );

    // length quantity with a pressure unit
    CHECK_THROWS_WITH(
        torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
            "length", "eV/A^3", "system", std::vector<std::string>{}, ""
        ),
        Contains(
            "unit 'eV/A^3' has dimension L^-1 T^-2 M which is incompatible with "
            "quantity 'length' (expected L)"
        )
    );
}


TEST_CASE("ModelOutput accepts matching quantity and unit") {
    // These should not throw
    torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        "energy", "eV", "system", std::vector<std::string>{}, ""
    );
    torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        "force", "eV/A", "system", std::vector<std::string>{}, ""
    );
    torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        "pressure", "eV/A^3", "system", std::vector<std::string>{}, ""
    );
    torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        "length", "Angstrom", "system", std::vector<std::string>{}, ""
    );
    torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        "momentum", "u*A/fs", "system", std::vector<std::string>{}, ""
    );
    torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        "velocity", "A/fs", "system", std::vector<std::string>{}, ""
    );
}


// Call the deprecated 3-arg overload without triggering -Wdeprecated-declarations
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
static double unit_conversion_factor_previous(const std::string& q, const std::string& f, const std::string& t) {
    return metatomic_torch::unit_conversion_factor(q, f, t);
}
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

TEST_CASE("3-arg API backward compatibility") {
    DeprecationWarningHandler handler;
    torch::WarningUtils::WarningHandlerGuard guard(&handler);
    torch::WarningUtils::set_warnAlways(true);

    double conv = unit_conversion_factor_previous("energy", "eV", "meV");
    CHECK_APPROX_ULP(conv, 1000.0);
}
