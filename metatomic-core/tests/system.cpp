#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

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
        CHECK(std::string(message) == "invalid parameter: `cell` must have the same dtype as `positions`");

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

/// Build a system containing all kinds of data (basic data, pairs, and custom
/// data) for use in serialization round-trip tests.
static mta_system_t* full_test_system() {
    mta_system_t* system = nullptr;
    auto status = mta_system_create(
        "Angstrom",
        types_tensor<int32_t>(4),
        positions_tensor<float>(4),
        cell_tensor<float>(),
        pbc_tensor<bool>(),
        &system
    );
    REQUIRE(status == MTA_SUCCESS);
    REQUIRE(system != nullptr);

    const auto* pairs_options_json = R"({
        "type": "metatomic_pair_options",
        "cutoff": "0x00001000",
        "full_list": true,
        "strict": false,
        "requestors": ["test"]
    })";
    status = mta_system_add_pairs(system, pairs_options_json, pair_block());
    CHECK(status == MTA_SUCCESS);

    status = mta_system_add_custom_data(system, "test::my_data", custom_data());
    CHECK(status == MTA_SUCCESS);

    return system;
}

/// Check that the given system contains the data expected from
/// `full_test_system`, independently of how it was loaded back.
static void check_full_system_data(const mta_system_t* system) {
    uintptr_t size = 0;
    CHECK(mta_system_size(system, &size) == MTA_SUCCESS);
    CHECK(size == 4);

    mta_string_t unit = nullptr;
    CHECK(mta_system_get_length_unit(system, &unit) == MTA_SUCCESS);
    CHECK(std::string(mta_string_view(unit)) == "Angstrom");
    mta_string_free(unit);

    DLManagedTensorVersioned* data = nullptr;

    // types
    CHECK(mta_system_get_data(system, MTA_SYSTEM_DATA_TYPES, &data) == MTA_SUCCESS);
    REQUIRE(data != nullptr);
    CHECK(data->dl_tensor.ndim == 1);
    CHECK(data->dl_tensor.shape[0] == 4);
    CHECK(data->dl_tensor.dtype.code == kDLInt);
    CHECK(data->dl_tensor.dtype.bits == 32);
    {
        auto* types = reinterpret_cast<int32_t*>(
            static_cast<char*>(data->dl_tensor.data) + data->dl_tensor.byte_offset
        );
        CHECK(types[0] == 1);
        CHECK(types[1] == 4);
        CHECK(types[2] == 7);
        CHECK(types[3] == 10);
    }
    data->deleter(data);

    // positions
    CHECK(mta_system_get_data(system, MTA_SYSTEM_DATA_POSITIONS, &data) == MTA_SUCCESS);
    REQUIRE(data != nullptr);
    CHECK(data->dl_tensor.ndim == 2);
    CHECK(data->dl_tensor.shape[0] == 4);
    CHECK(data->dl_tensor.shape[1] == 3);
    CHECK(data->dl_tensor.dtype.code == kDLFloat);
    CHECK(data->dl_tensor.dtype.bits == 32);
    {
        auto* positions = reinterpret_cast<float*>(
            static_cast<char*>(data->dl_tensor.data) + data->dl_tensor.byte_offset
        );
        CHECK(positions[0] == 1.0F);
        CHECK(positions[3] == 4.0F);
        CHECK(positions[6] == 7.0F);
        CHECK(positions[9] == 10.0F);
    }
    data->deleter(data);

    // cell
    CHECK(mta_system_get_data(system, MTA_SYSTEM_DATA_CELL, &data) == MTA_SUCCESS);
    REQUIRE(data != nullptr);
    CHECK(data->dl_tensor.ndim == 2);
    CHECK(data->dl_tensor.shape[0] == 3);
    CHECK(data->dl_tensor.shape[1] == 3);
    CHECK(data->dl_tensor.dtype.code == kDLFloat);
    CHECK(data->dl_tensor.dtype.bits == 32);
    {
        auto* cell = reinterpret_cast<float*>(
            static_cast<char*>(data->dl_tensor.data) + data->dl_tensor.byte_offset
        );
        CHECK(cell[0] == 10.0F);
        CHECK(cell[4] == 0.0F);
        CHECK(cell[8] == 10.0F);
    }
    data->deleter(data);

    // pbc
    CHECK(mta_system_get_data(system, MTA_SYSTEM_DATA_PBC, &data) == MTA_SUCCESS);
    REQUIRE(data != nullptr);
    CHECK(data->dl_tensor.ndim == 1);
    CHECK(data->dl_tensor.shape[0] == 3);
    CHECK(data->dl_tensor.dtype.code == kDLBool);
    CHECK(data->dl_tensor.dtype.bits == 8);
    {
        auto* pbc = reinterpret_cast<uint8_t*>(
            static_cast<char*>(data->dl_tensor.data) + data->dl_tensor.byte_offset
        );
        CHECK(pbc[0] == 1);
        CHECK(pbc[1] == 0);
        CHECK(pbc[2] == 1);
    }
    data->deleter(data);

    // known pairs survive the round-trip
    mta_string_t known = nullptr;
    CHECK(mta_system_known_pairs(system, &known) == MTA_SUCCESS);
    REQUIRE(known != nullptr);
    {
        auto known_str = std::string(mta_string_view(known));
        CHECK(known_str.find("metatomic_pair_options") != std::string::npos);
        CHECK(known_str.find("\"full_list\":true") != std::string::npos);
    }
    mta_string_free(known);

    // the pairs block can be retrieved
    const auto* pairs_options_json = R"({
        "type": "metatomic_pair_options",
        "cutoff": "0x00001000",
        "full_list": true,
        "strict": false,
        "requestors": []
    })";
    const mts_block_t* pairs = nullptr;
    CHECK(mta_system_get_pairs(system, pairs_options_json, &pairs) == MTA_SUCCESS);
    CHECK(pairs != nullptr);

    // custom data survives the round-trip
    mta_string_t names = nullptr;
    CHECK(mta_system_known_custom_data(system, &names) == MTA_SUCCESS);
    REQUIRE(names != nullptr);
    {
        auto names_str = std::string(mta_string_view(names));
        CHECK(names_str.find("test::my_data") != std::string::npos);
    }
    mta_string_free(names);

    const mts_tensormap_t* retrieved = nullptr;
    CHECK(mta_system_get_custom_data(system, "test::my_data", &retrieved) == MTA_SUCCESS);
    CHECK(retrieved != nullptr);
}

/// `DataArrayBase` storing boolean data as `uint8_t` (since
/// `std::vector<bool>` has no `data()` method, `SimpleDataArray<bool>` can not
/// be used). This class reports its dtype as `kDLBool` so that the metatensor
/// serialization code correctly handles it.
class BoolDataArray: public metatensor::SimpleDataArray<uint8_t> {
public:
    using SimpleDataArray::SimpleDataArray;

    DLDataType dtype() const override {
        DLDataType dtype;
        dtype.code = DLDataTypeCode::kDLBool;
        dtype.bits = 8;
        dtype.lanes = 1;
        return dtype;
    }

    DLManagedTensorVersioned* as_dlpack(
        DLDevice device,
        const int64_t* stream,
        DLPackVersion max_version
    ) override {
        auto* managed = SimpleDataArray::as_dlpack(device, stream, max_version);
        managed->dl_tensor.dtype.code = DLDataTypeCode::kDLBool;
        return managed;
    }

    std::unique_ptr<DataArrayBase> copy(DLDevice device) const override {
        if (device.device_type != kDLCPU) {
            throw metatensor::Error("BoolDataArray only supports copying to CPU");
        }
        return std::unique_ptr<DataArrayBase>(new BoolDataArray(*this));
    }

    std::unique_ptr<DataArrayBase> create(
        std::vector<uintptr_t> shape,
        metatensor::MtsArray fill_value
    ) const override {
        DLDevice cpu_device = {kDLCPU, 0};
        DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        auto fill_dlpack = fill_value.as_dlpack_array<uint8_t>(cpu_device, nullptr, version);

        if (!fill_dlpack.shape().empty()) {
            throw metatensor::Error("`fill_value` must be a single scalar");
        }

        auto scalar = fill_dlpack.data()[0];
        return std::unique_ptr<DataArrayBase>(new BoolDataArray(std::move(shape), scalar));
    }
};

/// `mts_realloc_buffer_t` callback backed by a `std::vector<uint8_t>`.
static uint8_t* vector_realloc(void* user_data, uint8_t* /*ptr*/, uintptr_t new_size) {
    auto* buffer = static_cast<std::vector<uint8_t>*>(user_data);
    buffer->resize(new_size, 0);
    return buffer->data();
}

/// `mts_create_array_callback_t` that delegates to
/// `metatensor::details::default_create_array`, but handles `kDLBool` by
/// creating a `BoolDataArray` (`SimpleDataArray<bool>` does not compile since
/// `std::vector<bool>` has no `data()` method). Can be removed once
/// https://github.com/metatensor/metatensor/pull/1164 is released.
static mts_status_t create_array_with_bool(
    const uintptr_t* shape_ptr,
    uintptr_t shape_count,
    DLDataType dtype,
    mts_array_t* array
) {
    if (dtype.code == kDLBool && dtype.bits == 8 && dtype.lanes == 1) {
        auto shape = std::vector<uintptr_t>();
        for (uintptr_t i = 0; i < shape_count; i++) {
            shape.push_back(shape_ptr[i]);
        }
        auto cxx_array = std::make_unique<BoolDataArray>(shape);
        *array = metatensor::DataArrayBase::to_mts_array(std::move(cxx_array)).release();
        return MTS_SUCCESS;
    }

    return metatensor::details::default_create_array(shape_ptr, shape_count, dtype, array);
}

TEST_CASE("system serialization") {
    SECTION("save and load to a file") {
        auto* system = full_test_system();

        auto path = (std::filesystem::temp_directory_path() / "metatomic-test-system.mta").string();

        CHECK(mta_save(path.c_str(), system) == MTA_SUCCESS);

        mta_system_t* loaded = nullptr;
        auto status = mta_load(
            path.c_str(),
            create_array_with_bool,
            &loaded
        );
        CHECK(status == MTA_SUCCESS);
        REQUIRE(loaded != nullptr);

        check_full_system_data(loaded);

        CHECK(mta_system_free(loaded) == MTA_SUCCESS);
        CHECK(mta_system_free(system) == MTA_SUCCESS);
        std::remove(path.c_str());
    }

    SECTION("save and load to an in-memory buffer") {
        auto* system = full_test_system();

        std::vector<uint8_t> buffer;
        uint8_t* ptr = buffer.data();
        uintptr_t size = buffer.size();

        auto status = mta_save_buffer(
            &ptr, &size, &buffer, vector_realloc, system
        );
        CHECK(status == MTA_SUCCESS);
        buffer.resize(size);

        mta_system_t* loaded = nullptr;
        status = mta_load_buffer(
            buffer.data(), buffer.size(),
            create_array_with_bool,
            &loaded
        );
        CHECK(status == MTA_SUCCESS);
        REQUIRE(loaded != nullptr);

        check_full_system_data(loaded);

        CHECK(mta_system_free(loaded) == MTA_SUCCESS);
        CHECK(mta_system_free(system) == MTA_SUCCESS);
    }
}
