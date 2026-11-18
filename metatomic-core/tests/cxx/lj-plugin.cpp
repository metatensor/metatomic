#include <catch.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <metatomic.hpp>
#include <metatensor.hpp>

using namespace metatensor;
using namespace metatomic;

/// Build a test system for the LJ plugin from `positions` (`n_atoms x 3`) and
/// `cell` (rows are the cell vectors, a zero row is a non-periodic direction),
/// with a brute-force half pair list for the plugin cutoff of 3.0
inline System test_system(const std::vector<double>& positions, const std::vector<double>& cell) {
    const double cutoff = 3.0;
    auto n_atoms = positions.size() / 3;

    auto pbc_data = std::vector<uint8_t>(3, 0);
    for (size_t abc = 0; abc < 3; abc++) {
        pbc_data[abc] = cell[3 * abc] != 0.0 || cell[3 * abc + 1] != 0.0 || cell[3 * abc + 2] != 0.0;
    }

    auto pair_samples = std::vector<int32_t>();
    auto pair_values = std::vector<double>();
    auto max_shift = [&](size_t abc) { return pbc_data[abc] ? 2 : 0; };
    for (int32_t i = 0; i < static_cast<int32_t>(n_atoms); i++) {
        for (int32_t j = i; j < static_cast<int32_t>(n_atoms); j++) {
            for (int32_t a = -max_shift(0); a <= max_shift(0); a++) {
                for (int32_t b = -max_shift(1); b <= max_shift(1); b++) {
                    for (int32_t c = -max_shift(2); c <= max_shift(2); c++) {
                        // only keep one of the two images of an atom with itself
                        if (i == j && std::make_tuple(a, b, c) <= std::make_tuple(0, 0, 0)) {
                            continue;
                        }

                        auto shift = std::array<double, 3>{static_cast<double>(a), static_cast<double>(b), static_cast<double>(c)};
                        auto d = std::array<double, 3>{};
                        for (size_t xyz = 0; xyz < 3; xyz++) {
                            d[xyz] = positions[3 * static_cast<size_t>(j) + xyz] - positions[3 * static_cast<size_t>(i) + xyz];
                            for (size_t abc = 0; abc < 3; abc++) {
                                d[xyz] += shift[abc] * cell[3 * abc + xyz];
                            }
                        }

                        if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < cutoff * cutoff) {
                            pair_samples.insert(pair_samples.end(), {i, j, a, b, c});
                            pair_values.insert(pair_values.end(), d.begin(), d.end());
                        }
                    }
                }
            }
        }
    }

    auto type_array = DataArrayBase::to_mts_array(
        std::make_unique<SimpleDataArray<int32_t>>(
            std::vector<uintptr_t>{n_atoms}, std::vector<int32_t>(n_atoms, 1)
        )
    );

    auto positions_array = DataArrayBase::to_mts_array(
        std::make_unique<SimpleDataArray<double>>(
            std::vector<uintptr_t>{n_atoms, 3}, positions
        )
    );

    auto cell_array = DataArrayBase::to_mts_array(
        std::make_unique<SimpleDataArray<double>>(
            std::vector<uintptr_t>{3, 3}, cell
        )
    );

    auto pbc_array = DataArrayBase::to_mts_array(
        std::make_unique<SimpleDataArray<bool>>(
            std::vector<uintptr_t>{3}, std::move(pbc_data)
        )
    );

    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    auto system = System(
        "Angstrom",
        DLPackTensor(type_array.as_dlpack(cpu, nullptr, version)),
        DLPackTensor(positions_array.as_dlpack(cpu, nullptr, version)),
        DLPackTensor(cell_array.as_dlpack(cpu, nullptr, version)),
        DLPackTensor(pbc_array.as_dlpack(cpu, nullptr, version))
    );

    auto n_pairs = pair_values.size() / 3;
    auto pairs_samples_array = DataArrayBase::to_mts_array(
        std::make_unique<SimpleDataArray<int32_t>>(
            std::vector<uintptr_t>{n_pairs, 5}, std::move(pair_samples)
        )
    );

    auto pairs_block = TensorBlock(
        std::make_unique<SimpleDataArray<double>>(
            std::vector<uintptr_t>{n_pairs, 3, 1}, std::move(pair_values)
        ),
        Labels(
            {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"},
            std::move(pairs_samples_array),
            assume_unique{}
        ),
        {Labels({"xyz"}, {{0}, {1}, {2}})},
        Labels({"distance"}, {{0}})
    );

    auto pairs_options = PairListOptions::builder()
        .cutoff(cutoff)
        .full_list(false)
        .strict(false)
        .build();

    system.add_pairs(pairs_options, std::move(pairs_block));

    return system;
}

/// Build a test system for LJ plugin, with 2 atoms separated by `distance` and
/// no PBC
inline System test_system(double distance) {
    return test_system({0.0, 0.0, 0.0, distance, 0.0, 0.0}, std::vector<double>(9, 0.0));
}

/// Apply the strain to each row of `rows` (`n x 3`), as `r -> r (1 + strain)`
static std::vector<double> deform(std::vector<double> rows, const std::array<double, 9>& strain) {
    for (size_t i = 0; i < rows.size(); i += 3) {
        auto row = std::array<double, 3>{rows[i], rows[i + 1], rows[i + 2]};
        for (size_t xyz_2 = 0; xyz_2 < 3; xyz_2++) {
            for (size_t xyz_1 = 0; xyz_1 < 3; xyz_1++) {
                rows[i + xyz_2] += row[xyz_1] * strain[3 * xyz_1 + xyz_2];
            }
        }
    }
    return rows;
}

static double lj_energy(double distance, double sigma, double epsilon, double cutoff) {
    auto sigma_r = sigma / distance;
    auto sigma_r_6 = std::pow(sigma_r, 6);
    auto sigma_cutoff_6 = std::pow(sigma / cutoff, 6);
    return 4.0 * epsilon * (
        sigma_r_6 * sigma_r_6 - sigma_r_6
        - sigma_cutoff_6 * sigma_cutoff_6 + sigma_cutoff_6
    );
}

/// dE/dr for the Lennard-Jones potential (the shift does not depend on r)
static double lj_derivative(double distance, double sigma, double epsilon) {
    auto sigma_6 = std::pow(sigma, 6);
    return 4.0 * epsilon * (
        6.0 * sigma_6 / std::pow(distance, 7)
        - 12.0 * sigma_6 * sigma_6 / std::pow(distance, 13)
    );
}

TEST_CASE("Lennard-Jones plugin") {
    metatomic::load_plugin(LJ_PLUGIN_PATH);

    SECTION("model loading errors") {
        const char* plugin = "metatomic-lj-plugin";
        CHECK_THROWS_WITH(
            load_model("not-lj", std::nullopt, plugin),
            "invalid parameter: failed to load model from 'not-lj': plugin 'metatomic-lj-plugin' could not load the model"
        );

        const char* model = "metatomic-lj-model";
        const auto* options = R"({"unknown":"1"})";
        CHECK_THROWS_WITH(
            load_model(model, options, plugin),
            "unknown Lennard-Jones option: 'unknown'"
        );

        options = R"({"sigma": "1", "epsilon": "1", "atomic_type": "1", "cutoff": "0", "length_unit": "A", "energy_unit": "kJ/mol"})";
        CHECK_THROWS_WITH(
            load_model(model, options, plugin),
            "Lennard-Jones option 'cutoff' must be finite and positive"
        );

        options = R"({"sigma": 1.0, "epsilon": "1", "atomic_type": "1", "cutoff": "1", "length_unit": "A", "energy_unit": "kJ/mol"})";
        CHECK_THROWS_WITH(
            load_model(model, options, plugin),
            "invalid parameter: JSON option 'sigma' has a non-string value in `mta_load_model`"
        );

        options = R"({"sigma": "1", "epsilon": "1", "atomic_type": "1.6", "cutoff": "1", "length_unit": "A", "energy_unit": "kJ/mol"})";
        CHECK_THROWS_WITH(
            load_model(model, options, plugin),
            "Lennard-Jones option 'atomic_type' must be an integer"
        );
    }

    const char* options = R"({"sigma": "1", "epsilon": "1", "atomic_type": "1", "cutoff": "3.0", "length_unit": "A", "energy_unit": "kJ/mol"})";

    SECTION("Global energy and forces") {
        auto model = load_model("metatomic-lj-model", options);

        auto systems = std::vector<System>();
        systems.emplace_back(test_system(2.0));
        systems.emplace_back(test_system(1.2));

        auto global = Quantity::builder()
            .name("energy")
            .unit("kJ/mol")
            .sample_kind(SampleKind::System)
            .add_gradient(Gradients::Positions);

        auto per_atom = Quantity::builder()
            .name("energy")
            .unit("kJ/mol")
            .sample_kind(SampleKind::Atom);

        auto results = metatomic::execute_model(
            model, systems, std::nullopt, {global.build(), per_atom.build()}, true
        );

        REQUIRE(results.size() == 2);

        // global energy
        auto block = results[0].block_by_id(0);
        CHECK(block.samples() == Labels({"system"}, {{0}, {1}}));

        auto values = block.values<double>();

        CHECK(values(0, 0) == Approx(lj_energy(2.0, 1.0, 1.0, 3.0)).epsilon(1e-12));
        CHECK(values(1, 0) == Approx(lj_energy(1.2, 1.0, 1.0, 3.0)).epsilon(1e-12));

        // forces from the global energy
        auto gradient_block = block.gradient("positions");
        CHECK(gradient_block.samples() == Labels(
            {"sample", "system", "atom"},
            {{0, 0, 0}, {0, 0, 1}, {1, 1, 0}, {1, 1, 1}}
        ));

        auto gradient = gradient_block.values<double>();
        auto step = 1e-6;
        auto distance = 2.0;
        auto finite_difference = (
            lj_energy(distance + step, 1.0, 1.0, 3.0)
            - lj_energy(distance - step, 1.0, 1.0, 3.0)
        ) / (2.0 * step);
        CHECK(gradient(0, 0, 0) == Approx(-finite_difference).epsilon(1e-8));
        CHECK(gradient(1, 0, 0) == Approx(finite_difference).epsilon(1e-8));

        distance = 1.2;
        finite_difference = (
            lj_energy(distance + step, 1.0, 1.0, 3.0)
            - lj_energy(distance - step, 1.0, 1.0, 3.0)
        ) / (2.0 * step);
        CHECK(gradient(2, 0, 0) == Approx(-finite_difference).epsilon(1e-8));
        CHECK(gradient(3, 0, 0) == Approx(finite_difference).epsilon(1e-8));

        // per-atom energy
        block = results[1].block_by_id(0);
        CHECK(block.samples() == Labels({"system", "atom"}, {{0, 0}, {0, 1}, {1, 0}, {1, 1}}));

        values = block.values<double>();
        CHECK(values(0, 0) == Approx(0.5 * lj_energy(2.0, 1.0, 1.0, 3.0)).epsilon(1e-12));
        CHECK(values(1, 0) == Approx(0.5 * lj_energy(2.0, 1.0, 1.0, 3.0)).epsilon(1e-12));
        CHECK(values(2, 0) == Approx(0.5 * lj_energy(1.2, 1.0, 1.0, 3.0)).epsilon(1e-12));
        CHECK(values(3, 0) == Approx(0.5 * lj_energy(1.2, 1.0, 1.0, 3.0)).epsilon(1e-12));
    }

    SECTION("Selected atoms") {
        auto model = load_model("metatomic-lj-model", options);

        auto systems = std::vector<System>();
        systems.emplace_back(test_system(2.0));
        systems.emplace_back(test_system(1.2));

        // select only one atom in each system
        auto selected_atoms = Labels({"system", "atom"}, {{0, 0}, {1, 1}});

        auto global = Quantity::builder()
            .name("energy")
            .unit("kJ/mol")
            .sample_kind(SampleKind::System)
            .add_gradient(Gradients::Positions);

        auto per_atom = Quantity::builder()
            .name("energy")
            .unit("kJ/mol")
            .sample_kind(SampleKind::Atom);

        auto results = metatomic::execute_model(
            model, systems, selected_atoms, {global.build(), per_atom.build()}, true
        );

        REQUIRE(results.size() == 2);

        // global energy, only containing half of the pair energy
        auto block = results[0].block_by_id(0);
        CHECK(block.samples() == Labels({"system"}, {{0}, {1}}));

        auto values = block.values<double>();

        CHECK(values(0, 0) == Approx(0.5 * lj_energy(2.0, 1.0, 1.0, 3.0)).epsilon(1e-12));
        CHECK(values(1, 0) == Approx(0.5 * lj_energy(1.2, 1.0, 1.0, 3.0)).epsilon(1e-12));

        // forces from the global energy, including on the unselected atoms
        auto gradient_block = block.gradient("positions");
        CHECK(gradient_block.samples() == Labels(
            {"sample", "system", "atom"},
            {{0, 0, 0}, {0, 0, 1}, {1, 1, 0}, {1, 1, 1}}
        ));

        auto gradient = gradient_block.values<double>();
        auto step = 1e-6;
        auto distance = 2.0;
        auto finite_difference = (
            lj_energy(distance + step, 1.0, 1.0, 3.0)
            - lj_energy(distance - step, 1.0, 1.0, 3.0)
        ) / (2.0 * step);
        CHECK(gradient(0, 0, 0) == Approx(-0.5 * finite_difference).epsilon(1e-8));
        CHECK(gradient(1, 0, 0) == Approx(0.5 * finite_difference).epsilon(1e-8));

        distance = 1.2;
        finite_difference = (
            lj_energy(distance + step, 1.0, 1.0, 3.0)
            - lj_energy(distance - step, 1.0, 1.0, 3.0)
        ) / (2.0 * step);
        CHECK(gradient(2, 0, 0) == Approx(-0.5 * finite_difference).epsilon(1e-8));
        CHECK(gradient(3, 0, 0) == Approx(0.5 * finite_difference).epsilon(1e-8));

        // per-atom energy, only for the selected atoms
        block = results[1].block_by_id(0);
        CHECK(block.samples() == selected_atoms);

        values = block.values<double>();
        CHECK(values(0, 0) == Approx(0.5 * lj_energy(2.0, 1.0, 1.0, 3.0)).epsilon(1e-12));
        CHECK(values(1, 0) == Approx(0.5 * lj_energy(1.2, 1.0, 1.0, 3.0)).epsilon(1e-12));
    }

    SECTION("Strain gradient for two atoms") {
        auto model = load_model("metatomic-lj-model", options);

        auto displacements = std::vector<std::array<double, 3>>{{1.1, 0.4, -0.7}, {-0.5, 1.3, 0.9}};
        auto systems = std::vector<System>();
        for (const auto& d: displacements) {
            systems.emplace_back(test_system({0.0, 0.0, 0.0, d[0], d[1], d[2]}, std::vector<double>(9, 0.0)));
        }

        auto energy = Quantity::builder()
            .name("energy")
            .unit("kJ/mol")
            .sample_kind(SampleKind::System)
            .add_gradient(Gradients::Positions)
            .add_gradient(Gradients::Strain)
            .build();

        // all atoms, then one atom per system (getting half of the pair)
        auto selections = std::vector<std::pair<std::optional<Labels>, double>>{
            {std::nullopt, 1.0},
            {Labels({"system", "atom"}, {{0, 0}, {1, 1}}), 0.5},
        };
        for (const auto& [selected_atoms, scale]: selections) {
            auto results = metatomic::execute_model(model, systems, selected_atoms, {energy}, true);
            REQUIRE(results.size() == 1);

            auto block = results[0].block_by_id(0);
            auto strain_block = block.gradient("strain");
            CHECK(strain_block.samples() == Labels({"sample"}, {{0}, {1}}));
            REQUIRE(strain_block.components().size() == 2);
            CHECK(strain_block.components()[0] == Labels({"xyz_1"}, {{0}, {1}, {2}}));
            CHECK(strain_block.components()[1] == Labels({"xyz_2"}, {{0}, {1}, {2}}));

            auto strain = strain_block.values<double>();
            auto positions = block.gradient("positions").values<double>();
            for (size_t system_i = 0; system_i < displacements.size(); system_i++) {
                const auto& d = displacements[system_i];
                auto distance = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                auto derivative = scale * lj_derivative(distance, 1.0, 1.0) / distance;

                for (size_t xyz_1 = 0; xyz_1 < 3; xyz_1++) {
                    CHECK(positions(2 * system_i, xyz_1, 0) == Approx(-derivative * d[xyz_1]).epsilon(1e-12));
                    CHECK(positions(2 * system_i + 1, xyz_1, 0) == Approx(derivative * d[xyz_1]).epsilon(1e-12));

                    for (size_t xyz_2 = 0; xyz_2 < 3; xyz_2++) {
                        auto expected = d[xyz_1] * derivative * d[xyz_2];
                        CHECK(strain(system_i, xyz_1, xyz_2, 0) == Approx(expected).epsilon(1e-12));
                    }
                }
            }
        }
    }

    SECTION("Strain gradient against finite differences") {
        auto model = load_model("metatomic-lj-model", options);

        // a triclinic periodic cell, so that all strain components are non-zero
        // and pairs with periodic images contribute, including atoms with their
        // own image (the third cell vector is shorter than the cutoff)
        auto positions = std::vector<double>{
            0.1, 0.2, 0.3,
            1.3, 0.9, 0.1,
            0.4, 1.8, 1.5,
            2.5, 2.2, 2.8,
        };
        auto cell = std::vector<double>{
            4.0, 0.0, 0.0,
            0.5, 4.2, 0.0,
            -0.3, 0.4, 2.8,
        };

        // system 0 is the reference, systems 2k+1 and 2k+2 are deformed by
        // +step and -step along strain component k
        auto step = 1e-6;
        auto systems = std::vector<System>();
        systems.emplace_back(test_system(positions, cell));
        for (size_t k = 0; k < 9; k++) {
            for (auto sign: {1.0, -1.0}) {
                auto strain = std::array<double, 9>{};
                strain[k] = sign * step;
                systems.emplace_back(test_system(deform(positions, strain), deform(cell, strain)));
            }
        }

        // select atoms 0 and 2 in all systems
        auto selected_values = std::vector<int32_t>();
        for (int32_t system_i = 0; system_i < static_cast<int32_t>(systems.size()); system_i++) {
            selected_values.insert(selected_values.end(), {system_i, 0, system_i, 2});
        }
        auto n_selected = selected_values.size() / 2;
        auto selected_array = DataArrayBase::to_mts_array(
            std::make_unique<SimpleDataArray<int32_t>>(
                std::vector<uintptr_t>{n_selected, 2}, std::move(selected_values)
            )
        );

        auto energy = Quantity::builder()
            .name("energy")
            .unit("kJ/mol")
            .sample_kind(SampleKind::System)
            .add_gradient(Gradients::Strain)
            .build();

        auto selections = std::vector<std::optional<Labels>>{
            std::nullopt,
            Labels({"system", "atom"}, std::move(selected_array), assume_unique{}),
        };
        for (const auto& selected_atoms: selections) {
            auto results = metatomic::execute_model(model, systems, selected_atoms, {energy}, true);
            REQUIRE(results.size() == 1);

            auto block = results[0].block_by_id(0);
            auto values = block.values<double>();
            auto strain = block.gradient("strain").values<double>();
            for (size_t k = 0; k < 9; k++) {
                auto finite_difference = (values(2 * k + 1, 0) - values(2 * k + 2, 0)) / (2.0 * step);
                CHECK(strain(0, k / 3, k % 3, 0) == Approx(finite_difference).epsilon(1e-6));
            }
        }
    }
}
