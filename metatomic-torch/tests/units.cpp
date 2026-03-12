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

// ---- Simple conversions ----

TEST_CASE("Simple length conversions") {
    // Angstrom <-> Bohr
    double a_to_bohr = metatomic_torch::unit_conversion_factor("Angstrom", "Bohr");
    CHECK(a_to_bohr == Approx(1.8897259886).epsilon(1e-6));

    double bohr_to_a = metatomic_torch::unit_conversion_factor("Bohr", "Angstrom");
    CHECK(bohr_to_a == Approx(0.529177210903).epsilon(1e-6));

    // Angstrom <-> nm
    double a_to_nm = metatomic_torch::unit_conversion_factor("Angstrom", "nm");
    CHECK(a_to_nm == Approx(0.1).epsilon(1e-12));

    // Angstrom <-> meter
    double a_to_m = metatomic_torch::unit_conversion_factor("Angstrom", "meter");
    CHECK(a_to_m == Approx(1e-10).epsilon(1e-20));
}

TEST_CASE("Simple energy conversions") {
    // eV <-> meV
    double ev_to_mev = metatomic_torch::unit_conversion_factor("eV", "meV");
    CHECK(ev_to_mev == Approx(1000.0).epsilon(1e-10));

    // eV <-> Hartree
    double ev_to_hartree = metatomic_torch::unit_conversion_factor("eV", "Hartree");
    CHECK(ev_to_hartree == Approx(0.0367493).epsilon(1e-4));

    // eV <-> Rydberg
    double ev_to_ry = metatomic_torch::unit_conversion_factor("eV", "Ry");
    CHECK(ev_to_ry == Approx(0.0734987).epsilon(1e-4));
}

// ---- Compound expressions ----

TEST_CASE("Compound unit expressions") {
    // eV/Angstrom <-> Hartree/Bohr (force)
    double f_conv = metatomic_torch::unit_conversion_factor("eV/Angstrom", "Hartree/Bohr");
    CHECK(f_conv == Approx(0.0194469).epsilon(1e-3));

    // kJ/mol <-> kcal/mol (energy)
    double e_conv = metatomic_torch::unit_conversion_factor("kJ/mol", "kcal/mol");
    CHECK(e_conv == Approx(1000.0 / 4184.0).epsilon(1e-6));

    // eV/Angstrom^3 (pressure) -- identity
    double p_id = metatomic_torch::unit_conversion_factor("eV/Angstrom^3", "eV/A^3");
    CHECK(p_id == Approx(1.0).epsilon(1e-10));
}

// ---- Fractional powers ----

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
    CHECK(conv == Approx(expected).epsilon(1e-4));

    // (eV/u)^(1/2) has dimension of velocity: L/T
    double v_conv = metatomic_torch::unit_conversion_factor("(eV/u)^(1/2)", "A/fs");
    double v_expected = std::sqrt(ev_si / u_si) / (a_si / fs_si);
    CHECK(v_conv == Approx(v_expected).epsilon(1e-4));
}

// ---- Case insensitivity ----

TEST_CASE("Case insensitive unit lookup") {
    double c1 = metatomic_torch::unit_conversion_factor("eV", "hartree");
    double c2 = metatomic_torch::unit_conversion_factor("EV", "HARTREE");
    double c3 = metatomic_torch::unit_conversion_factor("Ev", "Hartree");
    CHECK(c1 == Approx(c2).epsilon(1e-12));
    CHECK(c1 == Approx(c3).epsilon(1e-12));
}

// ---- Whitespace handling ----

TEST_CASE("Whitespace in unit expressions") {
    double c1 = metatomic_torch::unit_conversion_factor("eV / Angstrom", "Hartree/Bohr");
    double c2 = metatomic_torch::unit_conversion_factor("eV/Angstrom", "Hartree/Bohr");
    CHECK(c1 == Approx(c2).epsilon(1e-12));

    double c3 = metatomic_torch::unit_conversion_factor("( eV * u ) ^ ( 1 / 2 )", "u*A/fs");
    double c4 = metatomic_torch::unit_conversion_factor("(eV*u)^(1/2)", "u*A/fs");
    CHECK(c3 == Approx(c4).epsilon(1e-12));
}

// ---- Dimension mismatch errors ----

TEST_CASE("Dimension mismatch error") {
    CHECK_THROWS_WITH(
        metatomic_torch::unit_conversion_factor("eV", "Angstrom"),
        Contains("dimension mismatch")
    );
}

// ---- Unknown token errors ----

TEST_CASE("Unknown unit token error") {
    CHECK_THROWS_WITH(
        metatomic_torch::unit_conversion_factor("foobar", "eV"),
        Contains("unknown unit")
    );
}

// ---- Malformed expression errors ----

TEST_CASE("Malformed expression errors") {
    CHECK_THROWS_WITH(
        metatomic_torch::unit_conversion_factor("eV*(", "eV"),
        Contains("parentheses")
    );
}

// ---- Empty string handling ----

TEST_CASE("Empty unit string returns 1.0") {
    CHECK(metatomic_torch::unit_conversion_factor("", "eV") == 1.0);
    CHECK(metatomic_torch::unit_conversion_factor("eV", "") == 1.0);
    CHECK(metatomic_torch::unit_conversion_factor("", "") == 1.0);
}

// ---- Overflow/underflow handling ----

TEST_CASE("Overflow detection in exponentiation") {
    // Test overflow with extreme exponent on unit with small SI factor
    // u (atomic mass unit) has factor 1.66e-27, so u^(-50) overflows
    CHECK_THROWS_WITH(
        metatomic_torch::unit_conversion_factor("u^(-50)", "u^(-50)"),
        Contains("overflows")
    );
    
    // eV has factor 1.6e-19, so eV^(-100) overflows
    CHECK_THROWS_WITH(
        metatomic_torch::unit_conversion_factor("eV^(-100)", "eV^(-100)"),
        Contains("overflows")
    );
}

TEST_CASE("Overflow detection in multiplication") {
    // Test overflow with multiplication of units with extreme factors
    // u^(-25) * u^(-25) = u^(-50), which overflows
    CHECK_THROWS_WITH(
        metatomic_torch::unit_conversion_factor("u^(-25) * u^(-25)", "u^(-50)"),
        Contains("overflows")
    );
}

TEST_CASE("Overflow detection in division") {
    // Test overflow with division creating extreme factor
    // u / u^50 = u^(-49), which overflows (u has factor 1.66e-27)
    CHECK_THROWS_WITH(
        metatomic_torch::unit_conversion_factor("u", "u^50"),
        Contains("overflows")
    );
}

// ---- Backward compatibility: 3-arg API ----

// Call the deprecated 3-arg overload without triggering -Wdeprecated-declarations
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
static double call_3arg(const std::string& q, const std::string& f, const std::string& t) {
    return metatomic_torch::unit_conversion_factor(q, f, t);
}
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

TEST_CASE("3-arg API backward compatibility") {
    DeprecationWarningHandler handler;
    torch::WarningUtils::WarningHandlerGuard guard(&handler);
    torch::WarningUtils::set_warnAlways(true);

    double conv = call_3arg("energy", "eV", "meV");
    CHECK(conv == Approx(1000.0).epsilon(1e-10));
}

// ---- mol handling (dimensionless scaling) ----

TEST_CASE("mol as dimensionless scaling factor") {
    // kJ/mol to eV: kJ_SI / mol_SI / eV_SI
    double conv = metatomic_torch::unit_conversion_factor("kJ/mol", "eV");
    double kj_si = 1000.0;
    double mol_si = 6.02214076e23;
    double ev_si = 1.602176634e-19;
    double expected = (kj_si / mol_si) / ev_si;
    CHECK(conv == Approx(expected).epsilon(1e-6));
}

// ---- Hbar derived unit ----

TEST_CASE("hbar/Bohr momentum conversion") {
    double conv = metatomic_torch::unit_conversion_factor("hbar/Bohr", "u*A/fs");
    // hbar_SI / Bohr_SI / (u_SI * A_SI / fs_SI)
    double hbar_si = 1.054571817e-34;
    double bohr_si = 5.29177210903e-11;
    double u_si = 1.66053906660e-27;
    double a_si = 1e-10;
    double fs_si = 1e-15;
    double expected = (hbar_si / bohr_si) / (u_si * a_si / fs_si);
    CHECK(conv == Approx(expected).epsilon(1e-3));
}

// ---- Identity conversions ----

TEST_CASE("Identity conversions") {
    CHECK(metatomic_torch::unit_conversion_factor("eV", "eV") == Approx(1.0));
    CHECK(metatomic_torch::unit_conversion_factor("Angstrom", "Angstrom") == Approx(1.0));
    CHECK(metatomic_torch::unit_conversion_factor("u*A/fs", "u*A/fs") == Approx(1.0));
}

// ---- Accumulated floating-point error with fractional powers ----

TEST_CASE("Fractional power accumulated error") {
    // Test that (eV*u)^(1/3) then ^3 equals eV*u within tolerance
    // This verifies the 1e-10 dimension comparison tolerance handles
    // floating-point accumulation from fractional exponents
    double conv = metatomic_torch::unit_conversion_factor(
        "((eV*u)^(1/3))^3", "eV*u"
    );
    CHECK(conv == Approx(1.0).epsilon(1e-6));
    
    // Test with 1/2 then ^2
    double conv2 = metatomic_torch::unit_conversion_factor(
        "((eV*u)^(1/2))^2", "eV*u"
    );
    CHECK(conv2 == Approx(1.0).epsilon(1e-6));
    
    // Test nested: ((eV^(1/2))^2) should equal eV
    double conv3 = metatomic_torch::unit_conversion_factor(
        "(eV^(1/2))^2", "eV"
    );
    CHECK(conv3 == Approx(1.0).epsilon(1e-6));
}

// ---- Time unit conversions ----

TEST_CASE("Time unit conversions") {
    // second -> femtosecond
    double s_to_fs = metatomic_torch::unit_conversion_factor("s", "fs");
    CHECK(s_to_fs == Approx(1e15).epsilon(1e-6));

    // second -> picosecond
    double s_to_ps = metatomic_torch::unit_conversion_factor("second", "ps");
    CHECK(s_to_ps == Approx(1e12).epsilon(1e-6));

    // nanosecond -> femtosecond
    double ns_to_fs = metatomic_torch::unit_conversion_factor("ns", "fs");
    CHECK(ns_to_fs == Approx(1e6).epsilon(1e-6));

    // microsecond -> nanosecond
    double us_to_ns = metatomic_torch::unit_conversion_factor("us", "ns");
    CHECK(us_to_ns == Approx(1e3).epsilon(1e-6));
}

// ---- Micro sign handling ----

TEST_CASE("Micro sign (U+00B5) handling") {
    double c1 = metatomic_torch::unit_conversion_factor("um", "Angstrom");
    double c2 = metatomic_torch::unit_conversion_factor("µm", "Angstrom");
    CHECK(c1 == Approx(c2).epsilon(1e-12));

    // µs -> ns (microsecond via micro sign)
    double c3 = metatomic_torch::unit_conversion_factor("us", "ns");
    double c4 = metatomic_torch::unit_conversion_factor("µs", "ns");
    CHECK(c3 == Approx(c4).epsilon(1e-12));
}

// ---- Quantity-unit dimension validation ----

TEST_CASE("ModelOutput rejects mismatched quantity and unit") {
    // energy quantity with a force unit
    CHECK_THROWS_WITH(
        torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
            "energy", "eV/A", false, std::vector<std::string>{}, ""
        ),
        Contains("incompatible with quantity")
    );

    // force quantity with an energy unit
    CHECK_THROWS_WITH(
        torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
            "force", "eV", false, std::vector<std::string>{}, ""
        ),
        Contains("incompatible with quantity")
    );

    // length quantity with a pressure unit
    CHECK_THROWS_WITH(
        torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
            "length", "eV/A^3", false, std::vector<std::string>{}, ""
        ),
        Contains("incompatible with quantity")
    );
}

TEST_CASE("ModelOutput accepts matching quantity and unit") {
    // These should not throw
    torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        "energy", "eV", false, std::vector<std::string>{}, ""
    );
    torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        "force", "eV/A", false, std::vector<std::string>{}, ""
    );
    torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        "pressure", "eV/A^3", false, std::vector<std::string>{}, ""
    );
    torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        "length", "Angstrom", false, std::vector<std::string>{}, ""
    );
    torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        "momentum", "u*A/fs", false, std::vector<std::string>{}, ""
    );
    torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        "velocity", "A/fs", false, std::vector<std::string>{}, ""
    );
}

// ---- Multithreading tests ----

#include <thread>

TEST_CASE("Thread-local cache works correctly") {
    // Test that the thread-local cache works correctly across multiple threads
    // Each thread should have its own cache, and results should be consistent
    
    const size_t num_threads = 10;
    const size_t iterations_per_thread = 100;
    std::vector<double> results(num_threads * iterations_per_thread);
    
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([t, &results, iterations_per_thread]() {
            for (size_t i = 0; i < iterations_per_thread; ++i) {
                // All threads call the same conversion repeatedly
                double factor = metatomic_torch::unit_conversion_factor("eV", "meV");
                results[t * iterations_per_thread + i] = factor;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // All results should be consistent (1000.0 for eV -> meV)
    for (size_t i = 0; i < results.size(); ++i) {
        CHECK(results[i] == Approx(1000.0).epsilon(1e-10));
    }
}

TEST_CASE("Concurrent different unit conversions") {
    // Test that different threads can safely convert different units
    const size_t num_threads = 5;
    std::vector<double> results(num_threads);
    std::vector<std::thread> threads;
    
    const std::vector<std::pair<std::string, std::string>> conversions = {
        {"eV", "meV"},           // energy
        {"angstrom", "bohr"},    // length
        {"fs", "ps"},            // time
        {"u", "kg"},             // mass
        {"eV/A", "Hartree/Bohr"} // force
    };
    
    const std::vector<double> expected = {
        1000.0,      // eV -> meV
        1.8897259886, // angstrom -> bohr
        0.001,       // fs -> ps
        1.66053906660e-27, // u -> kg
        0.0194469    // eV/A -> Hartree/Bohr
    };
    
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([t, &results, &conversions]() {
            for (size_t i = 0; i < 50; ++i) {
                results[t] = metatomic_torch::unit_conversion_factor(
                    conversions[t].first, conversions[t].second
                );
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Check each thread got the correct result
    for (size_t t = 0; t < num_threads; ++t) {
        CHECK(results[t] == Approx(expected[t]).epsilon(1e-4));
    }
}
