#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <catch.hpp>
#include <metatomic.hpp>
#include <metatensor.hpp>
#include <nlohmann/json.hpp>


template <typename T>
static metatomic::DLPackTensor dlpack_tensor(
    std::vector<uintptr_t> shape,
    std::vector<typename metatensor::SimpleDataArray<T>::storage_t> data
) {
    auto array = std::make_unique<metatensor::SimpleDataArray<T>>(
        std::move(shape), std::move(data)
    );
    auto mts_array = metatensor::DataArrayBase::to_mts_array(std::move(array));
    return metatomic::DLPackTensor(mts_array.as_dlpack(
        {kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}
    ));
}

static metatomic::System two_atom_system(
    double distance,
    int32_t first_type = 1,
    int32_t second_type = 1
) {
    return metatomic::System(
        "Angstrom",
        dlpack_tensor<int32_t>({2}, {first_type, second_type}),
        dlpack_tensor<double>({2, 3}, {0.0, 0.0, 0.0, 0.0, 0.0, distance}),
        dlpack_tensor<double>({3, 3}, std::vector<double>(9, 0.0)),
        dlpack_tensor<bool>({3}, {0, 0, 0})
    );
}

static metatomic::System periodic_two_atom_system(double cell, double x2) {
    return metatomic::System(
        "Angstrom",
        dlpack_tensor<int32_t>({2}, {1, 1}),
        dlpack_tensor<double>({2, 3}, {0.0, 0.0, 0.0, x2, 0.0, 0.0}),
        dlpack_tensor<double>({3, 3}, {
            cell, 0.0, 0.0,
            0.0, cell, 0.0,
            0.0, 0.0, cell
        }),
        dlpack_tensor<bool>({3}, {1, 1, 1})
    );
}

static metatomic::System three_atom_system() {
    return metatomic::System(
        "Angstrom",
        dlpack_tensor<int32_t>({3}, {1, 1, 1}),
        dlpack_tensor<double>({3, 3}, {
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.5,
            0.0, 0.0, 5.5
        }),
        dlpack_tensor<double>({3, 3}, std::vector<double>(9, 0.0)),
        dlpack_tensor<bool>({3}, {0, 0, 0})
    );
}

static metatensor::TensorBlock pair_block(
    const std::vector<double>& displacements,
    const std::vector<int32_t>& samples
) {
    const auto pair_count = samples.size() / 5;
    auto values = std::make_unique<metatensor::SimpleDataArray<double>>(
        std::vector<uintptr_t>{pair_count, 3, 1},
        displacements
    );
    return metatensor::TensorBlock(
        std::move(values),
        metatensor::Labels(
            {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"},
            samples.data(),
            pair_count
        ),
        {metatensor::Labels({"xyz"}, {{0}, {1}, {2}})},
        metatensor::Labels({"distance"}, {{0}})
    );
}

static metatensor::TensorBlock pair_block(double distance) {
    return pair_block(
        {0.0, 0.0, distance},
        {0, 1, 0, 0, 0}
    );
}

static metatomic::ExternalModel load_lj(const std::string& options = "{}") {
    return metatomic::ExternalModel(
        metatomic::load_model("lennard-jones", options, "lj-plugin")
    );
}

static void add_lj_pairs(
    metatomic::ExternalModel& model,
    metatomic::System& system,
    metatensor::TensorBlock pairs
) {
    system.add_pairs(model.requested_pair_lists()[0], std::move(pairs));
}

static void add_lj_pairs(metatomic::ExternalModel& model, metatomic::System& system, double distance) {
    add_lj_pairs(model, system, pair_block(distance));
}

static std::vector<metatensor::TensorMap> run(
    metatomic::ExternalModel& model,
    const std::vector<metatomic::System>& systems,
    const std::optional<metatensor::Labels>& selected,
    metatomic::SampleKind kind,
    bool positions_gradient
) {
    auto output = metatomic::Quantity::builder()
        .name("energy")
        .unit("eV")
        .sample_kind(kind);
    if (positions_gradient) {
        output.add_gradient(metatomic::Gradients::Positions);
    }
    return metatomic::execute_model(
        model, systems, selected, {output.build()}, true
    );
}

static double lj_energy(double distance, double sigma, double epsilon, double cutoff) {
    auto ratio = sigma / distance;
    auto ratio6 = std::pow(ratio, 6);
    auto cutoff_ratio6 = std::pow(sigma / cutoff, 6);
    return 4.0 * epsilon * (
        ratio6 * ratio6 - ratio6
        - cutoff_ratio6 * cutoff_ratio6 + cutoff_ratio6
    );
}

TEST_CASE("Lennard-Jones plugin") {
    static bool loaded = false;
    if (!loaded) {
        metatomic::load_plugin(LJ_PLUGIN_PATH);
        loaded = true;
    }

    SECTION("rejects unsupported models and invalid options") {
        CHECK_THROWS(metatomic::load_model("not-lj", "{}", "lj-plugin"));
        CHECK_THROWS(load_lj(R"({"sigma":1.0})"));
        CHECK_THROWS(load_lj(R"({"unknown":"1"})"));
        CHECK_THROWS(load_lj(R"({"cutoff":"0"})"));
    }

    SECTION("reports model information") {
        auto model = load_lj();
        auto capabilities = model.capabilities();
        CHECK(capabilities.atomic_types() == std::vector<int64_t>{1});
        CHECK(capabilities.interaction_range() == 3.0);
        CHECK(capabilities.length_unit() == "Angstrom");
        CHECK(model.metadata().name() == "Lennard-Jones test model");

        auto pairs = model.requested_pair_lists();
        REQUIRE(pairs.size() == 1);
        CHECK(pairs[0].cutoff() == 3.0);
        CHECK_FALSE(pairs[0].full_list());
        CHECK_FALSE(pairs[0].strict());

        REQUIRE(capabilities.outputs().size() == 2);
        CHECK(capabilities.outputs()[0].sample_kind() == metatomic::SampleKind::System);
        CHECK(capabilities.outputs()[1].sample_kind() == metatomic::SampleKind::Atom);
    }

    SECTION("parses a list of atomic types") {
        auto model = load_lj(R"({"atomic_type":"1, 6"})");
        CHECK(model.capabilities().atomic_types() == std::vector<int64_t>{1, 6});
    }

    SECTION("computes energy and positions gradient") {
        const double distance = 1.5;
        auto model = load_lj(
            R"({"sigma":"1.0","epsilon":"1.0","cutoff":"3.0"})"
        );
        std::vector<metatomic::System> systems;
        systems.push_back(two_atom_system(distance));
        add_lj_pairs(model, systems[0], distance);

        auto result = run(
            model, systems, std::nullopt, metatomic::SampleKind::System, true
        );
        REQUIRE(result.size() == 1);
        auto block = result[0].block_by_id(0);
        auto values = block.values<double>();
        const auto expected = lj_energy(distance, 1.0, 1.0, 3.0);
        CHECK(values(0, 0) == Approx(expected).epsilon(1e-12));

        auto gradient = block.gradient("positions").values<double>();
        const auto step = 1e-6;
        const auto finite_difference = (
            lj_energy(distance + step, 1.0, 1.0, 3.0)
            - lj_energy(distance - step, 1.0, 1.0, 3.0)
        ) / (2.0 * step);
        CHECK(gradient(0, 2, 0) == Approx(-finite_difference).epsilon(1e-8));
        CHECK(gradient(1, 2, 0) == Approx(finite_difference).epsilon(1e-8));
    }

    SECTION("omits an unrequested gradient") {
        auto model = load_lj();
        std::vector<metatomic::System> systems;
        systems.push_back(two_atom_system(1.5));
        add_lj_pairs(model, systems[0], 1.5);

        auto result = run(
            model, systems, std::nullopt, metatomic::SampleKind::System, false
        );
        CHECK(result[0].block_by_id(0).gradients_list().empty());
    }

    SECTION("computes energy for multiple systems") {
        auto model = load_lj();
        std::vector<metatomic::System> systems;
        systems.push_back(two_atom_system(1.5));
        systems.push_back(two_atom_system(2.0));
        add_lj_pairs(model, systems[0], 1.5);
        add_lj_pairs(model, systems[1], 2.0);

        auto result = run(
            model, systems, std::nullopt, metatomic::SampleKind::System, true
        );
        auto block = result[0].block_by_id(0);
        auto values = block.values<double>();
        CHECK(values(0, 0) == Approx(lj_energy(1.5, 1.0, 1.0, 3.0)).epsilon(1e-12));
        CHECK(values(1, 0) == Approx(lj_energy(2.0, 1.0, 1.0, 3.0)).epsilon(1e-12));

        auto gradient_samples = block.gradient("positions").samples();
        CHECK(gradient_samples.count() == 4);
        const auto sample_values = gradient_samples.values_cpu();
        CHECK(sample_values(0, 0) == 0);
        CHECK(sample_values(0, 1) == 0);
        CHECK(sample_values(2, 0) == 1);
        CHECK(sample_values(2, 1) == 1);
    }

    SECTION("applies selected atoms with a half-pair split") {
        const double distance = 1.5;
        auto model = load_lj();
        std::vector<metatomic::System> systems;
        systems.push_back(two_atom_system(distance));
        add_lj_pairs(model, systems[0], distance);

        auto selected = metatensor::Labels({"system", "atom"}, {{0, 0}});
        auto result = run(
            model, systems, selected, metatomic::SampleKind::System, true
        );
        auto block = result[0].block_by_id(0);
        auto values = block.values<double>();
        const auto expected = 0.5 * lj_energy(distance, 1.0, 1.0, 3.0);
        CHECK(values(0, 0) == Approx(expected).epsilon(1e-12));

        auto gradient = block.gradient("positions").values<double>();
        const auto step = 1e-6;
        const auto finite_difference = 0.5 * (
            lj_energy(distance + step, 1.0, 1.0, 3.0)
            - lj_energy(distance - step, 1.0, 1.0, 3.0)
        ) / (2.0 * step);
        CHECK(gradient(0, 2, 0) == Approx(-finite_difference).epsilon(1e-8));
        CHECK(gradient(1, 2, 0) == Approx(finite_difference).epsilon(1e-8));
    }

    SECTION("computes per-atom energies that sum to the system energy") {
        const double distance = 1.5;
        auto model = load_lj();
        std::vector<metatomic::System> systems;
        systems.push_back(two_atom_system(distance));
        add_lj_pairs(model, systems[0], distance);

        auto result = run(
            model, systems, std::nullopt, metatomic::SampleKind::Atom, false
        );
        auto block = result[0].block_by_id(0);
        auto values = block.values<double>();
        const auto expected = 0.5 * lj_energy(distance, 1.0, 1.0, 3.0);
        CHECK(values(0, 0) == Approx(expected).epsilon(1e-12));
        CHECK(values(1, 0) == Approx(expected).epsilon(1e-12));
        CHECK(values(0, 0) + values(1, 0) == Approx(
            lj_energy(distance, 1.0, 1.0, 3.0)
        ).epsilon(1e-12));
        CHECK(block.gradients_list().empty());

        auto selected = metatensor::Labels({"system", "atom"}, {{0, 1}});
        auto selected_result = run(
            model, systems, selected, metatomic::SampleKind::Atom, false
        );
        auto selected_values = selected_result[0].block_by_id(0).values<double>();
        CHECK(selected_result[0].block_by_id(0).samples().count() == 1);
        CHECK(selected_values(0, 0) == Approx(expected).epsilon(1e-12));
    }

    SECTION("uses the same pair potential for mixed atomic types") {
        const double distance = 1.5;
        auto model = load_lj(R"({"atomic_type":"1,6"})");
        CHECK(model.capabilities().atomic_types() == std::vector<int64_t>{1, 6});

        std::vector<metatomic::System> systems;
        systems.push_back(two_atom_system(distance, 1, 6));
        add_lj_pairs(model, systems[0], distance);

        auto result = run(
            model, systems, std::nullopt, metatomic::SampleKind::System, false
        );
        auto values = result[0].block_by_id(0).values<double>();
        CHECK(values(0, 0) == Approx(lj_energy(distance, 1.0, 1.0, 3.0)).epsilon(1e-12));
    }

    SECTION("skips pairs past the cutoff") {
        auto model = load_lj();
        std::vector<metatomic::System> systems;
        systems.push_back(three_atom_system());
        add_lj_pairs(model, systems[0], pair_block(
            {0.0, 0.0, 1.5, 0.0, 0.0, 4.0},
            {0, 1, 0, 0, 0, 1, 2, 0, 0, 0}
        ));

        auto result = run(
            model, systems, std::nullopt, metatomic::SampleKind::Atom, false
        );
        auto values = result[0].block_by_id(0).values<double>();
        const auto expected = 0.5 * lj_energy(1.5, 1.0, 1.0, 3.0);
        CHECK(values(0, 0) == Approx(expected).epsilon(1e-12));
        CHECK(values(1, 0) == Approx(expected).epsilon(1e-12));
        CHECK(values(2, 0) == Approx(0.0).margin(1e-12));
    }

    SECTION("applies selected atoms across multiple systems") {
        auto model = load_lj();
        std::vector<metatomic::System> systems;
        systems.push_back(two_atom_system(1.5));
        systems.push_back(two_atom_system(2.0));
        add_lj_pairs(model, systems[0], 1.5);
        add_lj_pairs(model, systems[1], 2.0);

        auto selected = metatensor::Labels({"system", "atom"}, {{0, 0}, {1, 1}});
        auto result = run(
            model, systems, selected, metatomic::SampleKind::System, true
        );
        auto block = result[0].block_by_id(0);
        auto values = block.values<double>();
        const auto samples = block.samples().values_cpu();
        REQUIRE(block.samples().count() == 2);
        CHECK(samples(0, 0) == 0);
        CHECK(samples(1, 0) == 1);
        CHECK(values(0, 0) == Approx(0.5 * lj_energy(1.5, 1.0, 1.0, 3.0)).epsilon(1e-12));
        CHECK(values(1, 0) == Approx(0.5 * lj_energy(2.0, 1.0, 1.0, 3.0)).epsilon(1e-12));

        auto only_second = metatensor::Labels({"system", "atom"}, {{1, 0}, {1, 1}});
        auto second = run(
            model, systems, only_second, metatomic::SampleKind::System, false
        );
        auto second_block = second[0].block_by_id(0);
        const auto second_samples = second_block.samples().values_cpu();
        auto second_values = second_block.values<double>();
        REQUIRE(second_block.samples().count() == 1);
        CHECK(second_samples(0, 0) == 1);
        CHECK(second_values(0, 0) == Approx(lj_energy(2.0, 1.0, 1.0, 3.0)).epsilon(1e-12));
    }

    SECTION("returns system and per-atom energy in one execute") {
        const double distance = 1.5;
        auto model = load_lj();
        std::vector<metatomic::System> systems;
        systems.push_back(two_atom_system(distance));
        add_lj_pairs(model, systems[0], distance);

        auto system_energy = metatomic::Quantity::builder()
            .name("energy")
            .unit("eV")
            .sample_kind(metatomic::SampleKind::System)
            .add_gradient(metatomic::Gradients::Positions)
            .build();
        auto atom_energy = metatomic::Quantity::builder()
            .name("energy")
            .unit("eV")
            .sample_kind(metatomic::SampleKind::Atom)
            .build();
        auto result = metatomic::execute_model(
            model, systems, std::nullopt, {system_energy, atom_energy}, true
        );
        REQUIRE(result.size() == 2);

        const auto expected = lj_energy(distance, 1.0, 1.0, 3.0);
        auto system_values = result[0].block_by_id(0).values<double>();
        auto atom_values = result[1].block_by_id(0).values<double>();
        CHECK(system_values(0, 0) == Approx(expected).epsilon(1e-12));
        CHECK(atom_values(0, 0) + atom_values(1, 0) == Approx(expected).epsilon(1e-12));
        CHECK_FALSE(result[0].block_by_id(0).gradients_list().empty());
        CHECK(result[1].block_by_id(0).gradients_list().empty());
    }

    SECTION("uses pair displacements for periodic cell-shift images") {
        const double cell = 10.0;
        const double wrapped = 1.0;
        auto model = load_lj();
        std::vector<metatomic::System> systems;
        systems.push_back(periodic_two_atom_system(cell, cell - wrapped));
        add_lj_pairs(model, systems[0], pair_block(
            {-wrapped, 0.0, 0.0},
            {0, 1, -1, 0, 0}
        ));

        auto result = run(
            model, systems, std::nullopt, metatomic::SampleKind::System, true
        );
        auto block = result[0].block_by_id(0);
        auto values = block.values<double>();
        CHECK(values(0, 0) == Approx(lj_energy(wrapped, 1.0, 1.0, 3.0)).epsilon(1e-12));

        auto gradient = block.gradient("positions").values<double>();
        const auto step = 1e-6;
        const auto finite_difference = (
            lj_energy(wrapped + step, 1.0, 1.0, 3.0)
            - lj_energy(wrapped - step, 1.0, 1.0, 3.0)
        ) / (2.0 * step);
        CHECK(gradient(0, 0, 0) == Approx(finite_difference).epsilon(1e-8));
        CHECK(gradient(1, 0, 0) == Approx(-finite_difference).epsilon(1e-8));
    }
}
