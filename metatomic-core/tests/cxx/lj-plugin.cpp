#include <catch.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <metatomic.hpp>
#include <metatensor.hpp>

using namespace metatensor;
using namespace metatomic;

/// Build a test system for LJ plugin, with 2 atoms separated by `distance` and
/// no PBC
inline System test_system(double distance) {
    auto type_data = std::vector<int32_t>{1, 1};
    auto type_array = DataArrayBase::to_mts_array(
        std::make_unique<SimpleDataArray<int32_t>>(
            std::vector<uintptr_t>{2}, std::move(type_data)
        )
    );

    auto positions_data = std::vector<double>{0.0, 0.0, 0.0, distance, 0.0, 0.0};
    auto positions_array = DataArrayBase::to_mts_array(
        std::make_unique<SimpleDataArray<double>>(
            std::vector<uintptr_t>{2, 3}, std::move(positions_data)
        )
    );

    auto cell_data = std::vector<double>(9, 0.0);
    auto cell_array = DataArrayBase::to_mts_array(
        std::make_unique<SimpleDataArray<double>>(
            std::vector<uintptr_t>{3, 3}, std::move(cell_data)
        )
    );

    auto pbc_data = std::vector<uint8_t>{1, 1, 1};
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

    // create the pairs
    auto pairs_data = std::vector<double>{distance, 0.0, 0.0};
    auto pairs_array = std::make_unique<SimpleDataArray<double>>(
        std::vector<uintptr_t>{1, 3, 1}, std::move(pairs_data)
    );

    auto pairs_block = TensorBlock(
        std::move(pairs_array),
        Labels(
            {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"},
            {{0, 1, 0, 0, 0}}
        ),
        {Labels({"xyz"}, {{0}, {1}, {2}})},
        Labels({"distance"}, {{0}})
    );

    auto pairs_options = PairListOptions::builder()
        .cutoff(3.0)
        .full_list(false)
        .strict(false)
        .build();

    system.add_pairs(pairs_options, std::move(pairs_block));

    return system;
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

TEST_CASE("Lennard-Jones plugin") {
    static bool loaded = false;
    if (!loaded) {
        metatomic::load_plugin(LJ_PLUGIN_PATH);
        loaded = true;
    }

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
}
