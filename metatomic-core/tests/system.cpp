#include <cstring>
#include <string>

#include <catch.hpp>

#include <metatensor.hpp>
#include "metatomic.h"


template <typename T> static DLManagedTensorVersioned* types_tensor(size_t n_atoms) {
    std::vector<T> type_data;
    type_data.reserve(n_atoms);
    for (size_t i = 0; i < n_atoms; i++) {
        type_data.push_back(static_cast<T>(i * 3 + 1));
    }
    auto array = std::make_unique<metatensor::SimpleDataArray<T>>(
        std::vector<uintptr_t>{n_atoms},
        std::move(type_data)
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return mts.as_dlpack(cpu, nullptr, version);
}

template <typename T> static DLManagedTensorVersioned* cell_tensor() {
    auto array = std::make_unique<metatensor::SimpleDataArray<T>>(
        std::vector<uintptr_t>{3, 3},
        std::vector<T>{
            T(10.0), T(0.0), T(0.0),
            T(0.0), T(0.0), T(0.0),
            T(0.0), T(0.0), T(10.0),
        }
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return mts.as_dlpack(cpu, nullptr, version);
}

template <typename T> static DLManagedTensorVersioned* positions_tensor(size_t n_atoms) {
    std::vector<T> position_data;
    position_data.reserve(n_atoms * 3);
    for (size_t i = 0; i < n_atoms; i++) {
        position_data.push_back(static_cast<T>(i * 3 + 1));
        position_data.push_back(static_cast<T>(i * 3 + 2));
        position_data.push_back(static_cast<T>(i * 3 + 3));
    }
    auto array = std::make_unique<metatensor::SimpleDataArray<T>>(
        std::vector<uintptr_t>{n_atoms, 3},
        std::move(position_data)
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return mts.as_dlpack(cpu, nullptr, version);
}

template <typename T> static DLManagedTensorVersioned* pbc_tensor() {
    std::vector<T> pbc_data = {1, 0, 1};
    auto array = std::make_unique<metatensor::SimpleDataArray<T>>(
        std::vector<uintptr_t>{3},
        std::move(pbc_data)
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return mts.as_dlpack(cpu, nullptr, version);
}


/// SimpleDataArray<bool> doesn't compile (std::vector<bool> has no data()
/// method). We use SimpleDataArray<uint8_t> and patch the dtype code
/// from kDLUInt to kDLBool.
template <> DLManagedTensorVersioned* pbc_tensor<bool>() {
    std::vector<uint8_t> pbc_data = {1, 0, 1};
    auto array = std::make_unique<metatensor::SimpleDataArray<uint8_t>>(
        std::vector<uintptr_t>{3},
        std::move(pbc_data)
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    auto* tensor = mts.as_dlpack(cpu, nullptr, version);

    tensor->dl_tensor.dtype.code = DLDataTypeCode::kDLBool;

    return tensor;
}

static mts_block_t* pair_block() {
    auto samples = metatensor::Labels(
        {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"},
        {{0, 1, 0, 0, 0}}
    );

    auto components = metatensor::Labels({"xyz"}, {{0}, {1}, {2}});
    std::vector<const mts_labels_t*> components_list = {
        components.as_mts_labels_t()
    };

    auto properties = metatensor::Labels({"distance"}, {{0}});

    auto values = std::make_unique<metatensor::SimpleDataArray<float>>(
        std::vector<uintptr_t>{1, 3, 1},
        std::vector<float>{1.5F, 2.5F, 3.5F}
    );
    auto values_mts = metatensor::DataArrayBase::to_mts_array(std::move(values));

    auto* block = mts_block(
        std::move(values_mts).release(),
        samples.as_mts_labels_t(),
        components_list.data(),
        components_list.size(),
        properties.as_mts_labels_t()
    );
    REQUIRE(block != nullptr);
    return block;
}

static mts_tensormap_t* custom_data() {
    auto keys = metatensor::Labels({"key"}, {{0}});
    auto samples = metatensor::Labels({"sample"}, {{0}});
    auto properties = metatensor::Labels({"property"}, {{0}});

    auto values = std::make_unique<metatensor::SimpleDataArray<float>>(
        std::vector<uintptr_t>{1, 1},
        std::vector<float>{42.0F}
    );
    auto values_mts = metatensor::DataArrayBase::to_mts_array(std::move(values));

    auto* block = mts_block(
        std::move(values_mts).release(),
        samples.as_mts_labels_t(),
        nullptr,
        0,
        properties.as_mts_labels_t()
    );
    REQUIRE(block != nullptr);

    std::vector<mts_block_t*> blocks = {block};
    auto* tensormap = mts_tensormap(
        keys.as_mts_labels_t(),
        blocks.data(),
        blocks.size()
    );
    REQUIRE(tensormap != nullptr);

    return tensormap;
}

TEST_CASE("system") {
    SECTION("create and free") {
        mta_system_t* system_f32 = nullptr;
        auto status = mta_system_create(
            "nm",
            types_tensor<int32_t>(4),
            positions_tensor<float>(4),
            cell_tensor<float>(),
            pbc_tensor<bool>(),
            &system_f32
        );
        CHECK(status == MTA_SUCCESS);
        REQUIRE(system_f32 != nullptr);

        status = mta_system_free(system_f32);
        CHECK(status == MTA_SUCCESS);

        mta_system_t* system_f64 = nullptr;
        status = mta_system_create(
            "nm",
            types_tensor<int32_t>(4),
            positions_tensor<double>(4),
            cell_tensor<double>(),
            pbc_tensor<bool>(),
            &system_f64
        );
        CHECK(status == MTA_SUCCESS);
        REQUIRE(system_f64 != nullptr);

        status = mta_system_free(system_f64);
        CHECK(status == MTA_SUCCESS);

        // free on null pointer is fine
        status = mta_system_free(nullptr);
        REQUIRE(status == MTA_SUCCESS);
    }

    SECTION("errors") {
        mta_system_t* system = nullptr;

        // wrong dtype for types (float instead of int32)
        auto status = mta_system_create(
            "Angstrom",
            types_tensor<float>(3),
            positions_tensor<float>(3),
            cell_tensor<float>(),
            pbc_tensor<bool>(),
            &system
        );
        CHECK(status != MTA_SUCCESS);
        CHECK(system == nullptr);

        const char* message = nullptr;
        mta_last_error(&message, nullptr, nullptr);
        CHECK(std::string(message) == "invalid parameter: `types` must be a tensor of 32-bit integers");

        // wrong dtype for positions (int32 instead of float)
        status = mta_system_create(
            "Angstrom",
            types_tensor<int32_t>(3),
            positions_tensor<int32_t>(3),
            cell_tensor<float>(),
            pbc_tensor<bool>(),
            &system
        );
        CHECK(status != MTA_SUCCESS);
        CHECK(system == nullptr);

        mta_last_error(&message, nullptr, nullptr);
        CHECK(std::string(message) == "invalid parameter: `positions` must be a tensor of 32 or 64-bit floating point data");

        // wrong dtype for cell (int32 instead of float)
        status = mta_system_create(
            "Angstrom",
            types_tensor<int32_t>(3),
            positions_tensor<float>(3),
            cell_tensor<int32_t>(),
            pbc_tensor<bool>(),
            &system
        );
        CHECK(status != MTA_SUCCESS);
        CHECK(system == nullptr);

        mta_last_error(&message, nullptr, nullptr);
        CHECK(std::string(message) == "invalid parameter: `cell` must have the same dtype as `positions`, got i32 and f32");

        // wrong dtype for pbc (float instead of bool)
        status = mta_system_create(
            "Angstrom",
            types_tensor<int32_t>(3),
            positions_tensor<float>(3),
            cell_tensor<float>(),
            pbc_tensor<float>(),
            &system
        );
        CHECK(status != MTA_SUCCESS);
        CHECK(system == nullptr);

        mta_last_error(&message, nullptr, nullptr);
        CHECK(std::string(message) == "invalid parameter: `pbc` must be a tensor of booleans");

        // mismatched positions/type shapes
        status = mta_system_create(
            "Angstrom",
            types_tensor<int32_t>(3),
            positions_tensor<float>(5),
            cell_tensor<float>(),
            pbc_tensor<float>(),
            &system
        );
        CHECK(status != MTA_SUCCESS);
        CHECK(system == nullptr);

        mta_last_error(&message, nullptr, nullptr);
        CHECK(std::string(message) == "invalid parameter: `positions` must be a (n_atoms x 3) tensor, got a tensor with shape [5, 3]");


        // wrong cell shape
        auto* cell = cell_tensor<float>();
        cell->dl_tensor.shape[0] = 9;
        cell->dl_tensor.shape[1] = 1;
        status = mta_system_create(
            "Angstrom",
            types_tensor<int32_t>(3),
            positions_tensor<float>(3),
            cell,
            pbc_tensor<bool>(),
            &system
        );
        CHECK(status != MTA_SUCCESS);
        CHECK(system == nullptr);

        mta_last_error(&message, nullptr, nullptr);
        CHECK(std::string(message) == "invalid parameter: `cell` must be a (3 x 3) tensor, got a tensor with shape [9, 1]");


        // wrong pbc shape
        auto* pbc = pbc_tensor<bool>();
        pbc->dl_tensor.shape[0] = 2;
        status = mta_system_create(
            "Angstrom",
            types_tensor<int32_t>(3),
            positions_tensor<float>(3),
            cell_tensor<float>(),
            pbc,
            &system
        );
        CHECK(status != MTA_SUCCESS);
        CHECK(system == nullptr);

        mta_last_error(&message, nullptr, nullptr);
        CHECK(std::string(message) == "invalid parameter: `pbc` must contain 3 entries, got a tensor with shape [2]");
    }

    SECTION("size") {
        mta_system_t* system = nullptr;
        auto status = mta_system_create(
            "nm",
            types_tensor<int32_t>(4),
            positions_tensor<float>(4),
            cell_tensor<float>(),
            pbc_tensor<bool>(),
            &system
        );
        CHECK(status == MTA_SUCCESS);
        REQUIRE(system != nullptr);

        uintptr_t size = 0;
        status = mta_system_size(system, &size);
        CHECK(status == MTA_SUCCESS);
        CHECK(size == 4);

        status = mta_system_free(system);
        CHECK(status == MTA_SUCCESS);
    }

    SECTION("length unit") {
        mta_system_t* system = nullptr;
        auto status = mta_system_create(
            "nm",
            types_tensor<int32_t>(4),
            positions_tensor<float>(4),
            cell_tensor<float>(),
            pbc_tensor<bool>(),
            &system
        );
        CHECK(status == MTA_SUCCESS);
        REQUIRE(system != nullptr);

        mta_string_t unit = nullptr;
        status = mta_system_get_length_unit(system, &unit);
        CHECK(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(unit)) == "nm");
        mta_string_free(unit);

        status = mta_system_free(system);
        CHECK(status == MTA_SUCCESS);
    }
}

TEST_CASE("system data") {
    mta_system_t* system = nullptr;
    auto status = mta_system_create(
        "nm",
        types_tensor<int32_t>(4),
        positions_tensor<float>(4),
        cell_tensor<float>(),
        pbc_tensor<bool>(),
        &system
    );
    CHECK(status == MTA_SUCCESS);
    REQUIRE(system != nullptr);

    DLManagedTensorVersioned* data = nullptr;

    SECTION("types") {
        status = mta_system_get_data(
            system, MTA_SYSTEM_DATA_TYPES, &data
        );
        CHECK(status == MTA_SUCCESS);
        REQUIRE(data != nullptr);

        CHECK(data->dl_tensor.ndim == 1);
        CHECK(data->dl_tensor.shape[0] == 4);
        CHECK(data->dl_tensor.dtype.code == kDLInt);
        CHECK(data->dl_tensor.dtype.bits == 32);

        auto* types = reinterpret_cast<int32_t*>(static_cast<char*>(data->dl_tensor.data) + data->dl_tensor.byte_offset);
        CHECK(types[0] == 1);
        CHECK(types[1] == 4);
        CHECK(types[2] == 7);
        CHECK(types[3] == 10);
    }

    SECTION("positions") {
        status = mta_system_get_data(
            system, MTA_SYSTEM_DATA_POSITIONS, &data
        );
        CHECK(status == MTA_SUCCESS);
        REQUIRE(data != nullptr);

        CHECK(data->dl_tensor.ndim == 2);
        CHECK(data->dl_tensor.shape[0] == 4);
        CHECK(data->dl_tensor.shape[1] == 3);
        CHECK(data->dl_tensor.dtype.code == kDLFloat);
        CHECK(data->dl_tensor.dtype.bits == 32);

        auto* positions = reinterpret_cast<float*>(static_cast<char*>(data->dl_tensor.data) + data->dl_tensor.byte_offset);
        CHECK(positions[0] == 1.0F);
        CHECK(positions[3] == 4.0F);
        CHECK(positions[6] == 7.0F);
        CHECK(positions[9] == 10.0F);
    }

    SECTION("cell") {
        status = mta_system_get_data(
            system, MTA_SYSTEM_DATA_CELL, &data
        );
        CHECK(status == MTA_SUCCESS);
        REQUIRE(data != nullptr);

        CHECK(data->dl_tensor.ndim == 2);
        CHECK(data->dl_tensor.shape[0] == 3);
        CHECK(data->dl_tensor.shape[1] == 3);
        CHECK(data->dl_tensor.dtype.code == kDLFloat);
        CHECK(data->dl_tensor.dtype.bits == 32);

        auto* cell = reinterpret_cast<float*>(static_cast<char*>(data->dl_tensor.data) + data->dl_tensor.byte_offset);
        CHECK(cell[0] == 10.0F);
        CHECK(cell[4] == 0.0F);
        CHECK(cell[8] == 10.0F);
    }

    SECTION("pbc") {
        status = mta_system_get_data(
            system, MTA_SYSTEM_DATA_PBC, &data
        );
        CHECK(status == MTA_SUCCESS);
        REQUIRE(data != nullptr);

        CHECK(data->dl_tensor.ndim == 1);
        CHECK(data->dl_tensor.shape[0] == 3);
        CHECK(data->dl_tensor.dtype.code == kDLBool);
        CHECK(data->dl_tensor.dtype.bits == 8);

        auto* pbc = reinterpret_cast<bool*>(static_cast<char*>(data->dl_tensor.data) + data->dl_tensor.byte_offset);
        CHECK(pbc[0] == true);
        CHECK(pbc[1] == false);
        CHECK(pbc[2] == true);
    }

    data->deleter(data);

    status = mta_system_free(system);
    CHECK(status == MTA_SUCCESS);
}


TEST_CASE("system pairs") {
    mta_system_t* system = nullptr;
    auto status = mta_system_create(
        "nm",
        types_tensor<int32_t>(4),
        positions_tensor<float>(4),
        cell_tensor<float>(),
        pbc_tensor<bool>(),
        &system
    );
    CHECK(status == MTA_SUCCESS);
    REQUIRE(system != nullptr);

    const auto* options_json = R"({
        "type": "metatomic_pair_options",
        "cutoff": "0x00001000",
        "full_list": true,
        "strict": false,
        "requestors": ["test"]
    })";

    auto* pairs = pair_block();
    status = mta_system_add_pairs(system, options_json, pairs);
    CHECK(status == MTA_SUCCESS);

    const mts_block_t* recovered_pairs = nullptr;
    status = mta_system_get_pairs(system, options_json, &recovered_pairs);
    CHECK(status == MTA_SUCCESS);
    // we get the same pointer back
    CHECK(static_cast<const void*>(recovered_pairs) == static_cast<const void*>(pairs));

    // Add a second block with different options
    const auto* other_json = R"({
        "type": "metatomic_pair_options",
        "cutoff": "0x00001000",
        "full_list": true,
        "strict": true,
        "requestors": []
    })";

    pairs = pair_block();
    status = mta_system_add_pairs(system, other_json, pairs);
    CHECK(status == MTA_SUCCESS);

    // Check known pairs contains both
    mta_string_t known = nullptr;
    status = mta_system_known_pairs(system, &known);
    CHECK(status == MTA_SUCCESS);
    REQUIRE(known != nullptr);

    auto known_str = std::string(mta_string_view(known));
    mta_string_free(known);

    auto first = known_str.find("metatomic_pair_options");
    CHECK(first != std::string::npos);
    known_str = known_str.substr(first + 1);
    auto second = known_str.find("metatomic_pair_options");
    CHECK(second != std::string::npos);

    mta_system_free(system);
}

TEST_CASE("system custom data") {
    mta_system_t* system = nullptr;
    auto status = mta_system_create(
        "Angstrom",
        types_tensor<int32_t>(4),
        positions_tensor<float>(4),
        cell_tensor<float>(),
        pbc_tensor<bool>(),
        &system
    );
    CHECK(status == MTA_SUCCESS);
    REQUIRE(system != nullptr);

    auto* data = custom_data();
    status = mta_system_add_custom_data(system, "test::my_data", data);
    CHECK(status == MTA_SUCCESS);

    const mts_tensormap_t* retrieved = nullptr;
    status = mta_system_get_custom_data(
        system, "test::my_data", &retrieved
    );
    CHECK(status == MTA_SUCCESS);
    CHECK(retrieved != nullptr);
    CHECK(static_cast<const void*>(retrieved) == static_cast<const void*>(data));

    retrieved = nullptr;
    status = mta_system_get_custom_data(
        system, "test::no_such_data", &retrieved
    );
    CHECK(status != MTA_SUCCESS);
    CHECK(retrieved == nullptr);

    data = custom_data();
    status = mta_system_add_custom_data(system, "test::other_data", data);
    CHECK(status == MTA_SUCCESS);

    mta_string_t names = nullptr;
    status = mta_system_known_custom_data(system, &names);
    CHECK(status == MTA_SUCCESS);
    CHECK(names != nullptr);

    auto names_str = std::string(mta_string_view(names));
    CHECK(names_str.find("test::my_data") != std::string::npos);
    CHECK(names_str.find("test::other_data") != std::string::npos);

    mta_system_free(system);
}
