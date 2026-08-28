#include <catch.hpp>

#include <metatensor.hpp>

#include "metatomic.h"
#include "tensor_utils.hpp"

#include <cstring>


struct SimpleModelData {
    double scale;
};

static mta_status_t unload_impl(void* model_data) {
    delete static_cast<SimpleModelData*>(model_data);
    return MTA_SUCCESS;
}

static mta_status_t metadata_impl(const void* model_data, mta_string_t* metadata_json) {
    (void) model_data;

    *metadata_json = mta_string_create(R"({
        "name": "test C model",
        "description": "small model used as a C API example",
        "authors": [],
        "references": {
            "model": [],
            "implementation": [],
            "architecture": []
        }
    })");
    return MTA_SUCCESS;
}

static mta_status_t capabilities_impl(const void* model_data, mta_string_t* capabilities_json) {
    (void) model_data;

    *capabilities_json = mta_string_create(R"({
        "type": "metatomic_model_capabilities",
        "outputs": [{
            "type": "metatomic_quantity",
            "name": "energy",
            "unit": "eV",
            "gradients": [],
            "sample_kind": "system"
        }],
        "atomic_types": [1, 6, 8],
        "interaction_range": 4.5,
        "length_unit": "nm",
        "supported_devices": ["cpu"],
        "dtype": "float32"
    })");
    return MTA_SUCCESS;
}

static mta_status_t supported_outputs_impl(
    const void* model_data,
    mta_string_t* outputs_json
) {
    (void) model_data;
    *outputs_json = mta_string_create(R"([{
        "type": "metatomic_quantity",
        "name": "energy",
        "unit": "eV",
        "gradients": [],
        "sample_kind": "system"
    }])");
    return MTA_SUCCESS;
}

static mta_status_t requested_pair_lists_impl(
    const void* model_data,
    mta_string_t* pair_options_json
) {
    (void) model_data;
    *pair_options_json = mta_string_create("[]");
    return MTA_SUCCESS;
}

static mta_status_t requested_inputs_impl(
    const void* model_data,
    mta_string_t* requested_inputs_json
) {
    (void) model_data;
    *requested_inputs_json = mta_string_create("[]");
    return MTA_SUCCESS;
}


mts_tensormap_t* scalar_tensormap(double value) {
    auto values = std::make_unique<metatensor::SimpleDataArray<float>>(
        std::vector<uintptr_t>{1, 1},
        std::vector<float>{static_cast<float>(value)}
    );

    auto array = metatensor::DataArrayBase::to_mts_array(std::move(values));

    auto samples = metatensor::Labels({"system"}, {{0}});
    auto properties = metatensor::Labels({"energy"}, {{0}});

    auto* block = mts_block(
        std::move(array).release(),
        samples.as_mts_labels_t(),
        nullptr,
        0,
        properties.as_mts_labels_t()
    );
    if (block == nullptr) {
        return nullptr;
    }

    auto keys = metatensor::Labels({"_"}, {{0}});
    auto blocks = std::vector<mts_block_t*>{block};
    return mts_tensormap(keys.as_mts_labels_t(), blocks.data(), blocks.size());
}

static mta_status_t execute_inner_impl(
    void* model_data,
    const mta_system_t* const* systems,
    uintptr_t systems_count,
    const mts_labels_t* selected_atoms,
    const char* requested_outputs_json,
    mts_tensormap_t** outputs,
    uintptr_t outputs_count
) {
    (void)systems;
    (void)systems_count;
    (void)selected_atoms;
    (void)requested_outputs_json;

    auto* data = static_cast<SimpleModelData*>(model_data);

    for (uintptr_t i = 0; i < outputs_count; i++) {
        outputs[i] = scalar_tensormap(data->scale);
    }

    return MTA_SUCCESS;
}

static mta_status_t load_model_impl(
    const char* load_from,
    const char* options_json,
    mta_model_t* model
) {
    (void)options_json;
    assert(model != nullptr);

    if (std::strcmp(load_from, "test-c-model") != 0) {
        return MTA_MODEL_NOT_SUPPORTED_ERROR;
    }

    model->data = new SimpleModelData{2.0};
    model->unload = unload_impl;
    model->metadata = metadata_impl;
    model->capabilities = capabilities_impl;
    model->supported_outputs = supported_outputs_impl;
    model->requested_pair_lists = requested_pair_lists_impl;
    model->requested_inputs = requested_inputs_impl;
    model->execute_inner = execute_inner_impl;

    return MTA_SUCCESS;
}

TEST_CASE("simple C model can be registered and loaded through the C API") {
    static auto PLUGIN = mta_plugin_t {
        MTA_ABI_VERSION,
        "test-c-plugin",
        load_model_impl,
    };
    mta_register_plugin(PLUGIN);

    auto model = mta_model_t{};
    auto status = mta_load_model("test-c-model", "{}", "test-c-plugin", &model);
    REQUIRE(status == MTA_SUCCESS);

    CHECK(model.data != nullptr);
    CHECK(model.unload != nullptr);
    CHECK(model.metadata != nullptr);
    CHECK(model.capabilities != nullptr);
    CHECK(model.supported_outputs != nullptr);
    CHECK(model.requested_pair_lists != nullptr);
    CHECK(model.requested_inputs != nullptr);
    CHECK(model.execute_inner != nullptr);

    mta_string_t metadata = nullptr;
    status = model.metadata(model.data, &metadata);
    REQUIRE(status == MTA_SUCCESS);

    CHECK(metadata != nullptr);
    auto metadata_str = std::string(mta_string_view(metadata));
    mta_string_free(metadata);

    CHECK(metadata_str.find("\"name\": \"test C model\"") != std::string::npos);


    mta_string_t pair_lists = nullptr;
    status = model.requested_pair_lists(model.data, &pair_lists);
    REQUIRE(status == MTA_SUCCESS);

    CHECK(pair_lists != nullptr);
    CHECK(std::strcmp(mta_string_view(pair_lists), "[]") == 0);
    mta_string_free(pair_lists);

    REQUIRE(model.unload(model.data) == MTA_SUCCESS);
}

TEST_CASE("execute a model through the C API") {
    static auto PLUGIN = mta_plugin_t {
        MTA_ABI_VERSION,
        "test-c-plugin",
        load_model_impl,
    };
    mta_register_plugin(PLUGIN);

    auto model = mta_model_t{};
    auto status = mta_load_model("test-c-model", "{}", "test-c-plugin", &model);
    REQUIRE(status == MTA_SUCCESS);

    // create a water molecule system (O, H, H) with types matching capabilities [1, 6, 8]
    auto types_array = std::make_unique<metatensor::SimpleDataArray<int32_t>>(
        std::vector<uintptr_t>{3}, std::vector<int32_t>{8, 1, 1}
    );
    auto types_mts = metatensor::DataArrayBase::to_mts_array(std::move(types_array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    auto* types_dlpack = types_mts.as_dlpack(cpu, nullptr, version);

    mta_system_t* system = nullptr;
    status = mta_system_create(
        "nm",
        types_dlpack,
        positions_tensor<float>(3),
        cell_tensor<float>(),
        pbc_tensor(),
        &system
    );
    REQUIRE(status == MTA_SUCCESS);
    REQUIRE(system != nullptr);

    // run the model
    const char* requested_outputs = R"([{
        "type": "metatomic_quantity",
        "name": "energy",
        "unit": "eV",
        "gradients": [],
        "sample_kind": "system"
    }])";

    mts_tensormap_t* output = nullptr;
    status = mta_execute_model(
        model,
        &system,
        1,
        nullptr,
        requested_outputs,
        true,
        &output,
        1
    );
    CHECK(status == MTA_SUCCESS);

    if (output != nullptr) {
        mts_tensormap_free(output);
    }

    mta_system_free(system);
    REQUIRE(model.unload(model.data) == MTA_SUCCESS);
}
