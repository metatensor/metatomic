#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <catch.hpp>

#include <metatensor.hpp>
#include "metatomic.hpp"
#include "helpers.hpp"


// Local helper for the construction-error test: `types` with the wrong dtype.
static metatomic::DLPackTensor types_float_tensor(size_t n_atoms) {
    auto type_data = std::vector<float>();
    type_data.reserve(n_atoms);
    for (size_t i=0; i<n_atoms; i++) {
        type_data.push_back(static_cast<float>(i * 3 + 1));
    }

    auto array = std::make_unique<metatensor::SimpleDataArray<float>>(
        std::vector<uintptr_t>{n_atoms}, std::move(type_data)
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));

    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return metatomic::DLPackTensor(mts.as_dlpack(cpu, nullptr, version));
}

static metatensor::TensorBlock pair_block() {
    auto samples = metatensor::Labels(
        {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"},
        {{0, 1, 0, 0, 0}}
    );
    auto components = std::vector<metatensor::Labels>{
        metatensor::Labels({"xyz"}, {{0}, {1}, {2}})
    };
    auto properties = metatensor::Labels({"distance"}, {{0}});

    auto values = std::make_unique<metatensor::SimpleDataArray<float>>(
        std::vector<uintptr_t>{1, 3, 1}, std::vector<float>{1.5F, 2.5F, 3.5F}
    );

    return metatensor::TensorBlock(std::move(values), samples, components, properties);
}

static metatensor::TensorMap custom_data() {
    auto keys = metatensor::Labels({"key"}, {{0}});

    auto samples = metatensor::Labels({"sample"}, {{0}});
    auto properties = metatensor::Labels({"property"}, {{0}});
    auto values = std::make_unique<metatensor::SimpleDataArray<float>>(
        std::vector<uintptr_t>{1, 1}, std::vector<float>{42.0F}
    );
    auto block = metatensor::TensorBlock(std::move(values), samples, {}, properties);

    auto blocks = std::vector<metatensor::TensorBlock>();
    blocks.push_back(std::move(block));
    return metatensor::TensorMap(keys, std::move(blocks));
}


TEST_CASE("System basics") {
    auto system = test_system(4);

    CHECK(system.size() == 4);
    CHECK(system.length_unit() == "nm");
}

TEST_CASE("System construction errors") {
    // wrong dtype for `types` (float instead of int32)
    REQUIRE_THROWS_WITH(
        metatomic::System(
            "Angstrom",
            types_float_tensor(3),
            positions_tensor(3),
            cell_tensor(),
            pbc_tensor()
        ),
        "invalid parameter: `types` must be a tensor of 32-bit integers"
    );
}

TEST_CASE("System data") {
    auto system = test_system(4);

    SECTION("types") {
        auto types = system.types();
        REQUIRE(static_cast<bool>(types));
        CHECK(types->dl_tensor.ndim == 1);
        CHECK(types->dl_tensor.shape[0] == 4);
        CHECK(types->dl_tensor.dtype.code == kDLInt);
        CHECK(types->dl_tensor.dtype.bits == 32);

        auto* data = reinterpret_cast<int32_t*>(
            static_cast<char*>(types->dl_tensor.data) + types->dl_tensor.byte_offset
        );
        CHECK(data[0] == 1);
        CHECK(data[3] == 10);
    }

    SECTION("positions") {
        auto positions = system.positions();
        REQUIRE(static_cast<bool>(positions));
        CHECK(positions->dl_tensor.ndim == 2);
        CHECK(positions->dl_tensor.shape[0] == 4);
        CHECK(positions->dl_tensor.shape[1] == 3);
        CHECK(positions->dl_tensor.dtype.code == kDLFloat);

        auto* data = reinterpret_cast<float*>(
            static_cast<char*>(positions->dl_tensor.data) + positions->dl_tensor.byte_offset
        );
        CHECK(data[0] == 1.0F);
        CHECK(data[9] == 10.0F);
    }

    SECTION("cell") {
        auto cell = system.cell();
        REQUIRE(static_cast<bool>(cell));
        CHECK(cell->dl_tensor.ndim == 2);
        CHECK(cell->dl_tensor.shape[0] == 3);
        CHECK(cell->dl_tensor.shape[1] == 3);
    }

    SECTION("pbc") {
        auto pbc = system.pbc();
        REQUIRE(static_cast<bool>(pbc));
        CHECK(pbc->dl_tensor.ndim == 1);
        CHECK(pbc->dl_tensor.shape[0] == 3);
        CHECK(pbc->dl_tensor.dtype.code == kDLBool);

        auto* data = reinterpret_cast<bool*>(
            static_cast<char*>(pbc->dl_tensor.data) + pbc->dl_tensor.byte_offset
        );
        CHECK(data[0] == true);
        CHECK(data[1] == false);
        CHECK(data[2] == true);
    }
}

TEST_CASE("System pairs") {
    auto system = test_system(4);

    auto options = metatomic::PairListOptions();
    options.cutoff(1.0);
    options.full_list(true);
    options.strict(false);
    options.add_requestor("test");

    system.add_pairs(options, pair_block());

    const auto* options_json = R"({
        "type": "metatomic_pair_options",
        "cutoff": "0x40364ccccccccccd",
        "full_list": false,
        "strict": true,
        "requestors": [""]
    })";

    system.add_pairs(options_json, pair_block());

    auto pairs = system.pairs(options);
    CHECK(pairs.samples().count() == 1);
    CHECK(pairs.properties().size() == 1);

    auto known = system.known_pairs();
    CHECK(known.size() == 2);
    CHECK(known[0].cutoff() == 1.0);
    CHECK(known[0].full_list() == true);
    CHECK(known[0].strict() == false);
    CHECK(known[0].requestors().size() == 1);
    CHECK(known[0].requestors()[0] == "test");

    CHECK(known[1].cutoff() == 22.3);
    CHECK(known[1].full_list() == false);
    CHECK(known[1].strict() == true);
    CHECK(known[1].requestors().size() == 0);
}

TEST_CASE("System custom data") {
    auto system = test_system(4);

    system.add_custom_data("test::my_data", custom_data());

    auto data = system.custom_data("test::my_data");
    CHECK(data.keys().count() == 1);

    // retrieving unknown data throws
    REQUIRE_THROWS(system.custom_data("test::no_such_data"));

    system.add_custom_data("test::other_data", custom_data());
    auto names = system.known_custom_data();
    std::sort(names.begin(), names.end());
    CHECK(names.size() == 2);
    CHECK(names[0] == "test::my_data");
    CHECK(names[1] == "test::other_data");
}

TEST_CASE("System ownership") {
    SECTION("move") {
        auto system = test_system(4);
        auto* ptr = system.as_mta_system_t();

        auto moved = std::move(system);
        CHECK(moved.as_mta_system_t() == ptr);
        CHECK(moved.size() == 4);
    }

    SECTION("release / unsafe_from_ptr round-trip") {
        auto system = test_system(4);
        auto* raw = system.release();
        REQUIRE(raw != nullptr);

        auto owned = metatomic::System::unsafe_from_ptr(raw);
        CHECK(owned.size() == 4);
    }

    SECTION("unsafe_view_from_ptr does not free") {
        auto system = test_system(4);

        {
            auto view = metatomic::System::unsafe_view_from_ptr(system.as_mta_system_t());
            CHECK(view.size() == 4);
        }

        // the original system is still usable after the view is destroyed
        CHECK(system.size() == 4);
    }
}
