#include <catch.hpp>

#include "metatomic.h"
#include "metatomic.hpp"

TEST_CASE("model metadata formatting") {
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
    auto* mta_string = mta_string_create("");
    REQUIRE(mta_string != nullptr);
    auto status = mta_format_metadata(json.c_str(), &mta_string);
    REQUIRE(status == MTA_SUCCESS);
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
    CHECK(std::string(mta_string_view(mta_string)) == expected);
    mta_string_free(mta_string);
}

TEST_CASE("pair list metadata", "C API"){
    SECTION("JSON serialization"){
        mta_pair_list_options_t *metadata = nullptr;
        auto status = mta_pair_list_options_create(0.42, true, true, &metadata);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(metadata != nullptr);

        mta_string_t json = nullptr;
        status = mta_pair_list_options_to_json(metadata, &json);
        REQUIRE(status == MTA_SUCCESS);

        // cutoff is 0.42 in double precision converted to hex
        CHECK(std::string(mta_string_view(json)) == R"({"type":"metatomic_pair_options","cutoff":"0x3fdae147ae147ae1","full_list":true,"strict":true,"requestors":[]})");
        mta_string_free(json);
        mta_pair_list_options_free(metadata);
    }
    SECTION("JSON deserialization"){
        const char* json = "{\"type\":\"metatomic_pair_options\",\"cutoff\":\"0x3fdae147ae147ae1\",\"full_list\":true,\"strict\":true,\"requestors\":[]}";
        mta_pair_list_options_t *options = nullptr;
        auto status = mta_pair_list_options_from_json(json, &options);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(options != nullptr);

        mta_string_t type = nullptr;
        status = mta_pair_list_options_get_type(options, &type);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(type)) == "metatomic_pair_options");
        mta_string_free(type);

        double cutoff = -1.0;
        status = mta_pair_list_options_get_cutoff(options, &cutoff);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(cutoff == Approx(0.42).epsilon(1e-15));

        bool full_list = false;
        status = mta_pair_list_options_get_full_list(options, &full_list);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(full_list == true);

        bool strict = false;
        status = mta_pair_list_options_get_strict(options, &strict);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(strict == true);

        size_t num_requestors = 0;
        status = mta_pair_list_options_requestors_count(options, &num_requestors);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(num_requestors == 0);

        mta_pair_list_options_free(options);
    }
    SECTION("JSON deserialization with wrong JSON"){
        // boolean values are erroneously stored as strings
        const char* json = "{\"type\":\"metatomic_pair_options\",\"cutoff\":\"0x3fdae147ae147ae1\",\"full_list\":\"true\",\"strict\":\"true\",\"requestors\":[]}";

        mta_pair_list_options_t *options = nullptr;
        auto status = mta_pair_list_options_from_json(json, &options);
        CHECK(status == MTA_SERIALIZATION_ERROR);
    }
    SECTION("requestors"){
        mta_pair_list_options_t *options = nullptr;
        auto status = mta_pair_list_options_create(0.42, true, true, &options);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(options != nullptr);

        const char* requestor1 = "requestor1";
        status = mta_pair_list_options_add_requestor(options, requestor1);
        REQUIRE(status == MTA_SUCCESS);

        const char* requestor2 = "requestor2";
        status = mta_pair_list_options_add_requestor(options, requestor2);
        REQUIRE(status == MTA_SUCCESS);

        mta_string_t requestor = nullptr;
        size_t num_requestors = 0;
        status = mta_pair_list_options_requestors_count(options, &num_requestors);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(num_requestors == 2);

        status = mta_pair_list_options_get_requestor(options, 0, &requestor);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(requestor)) == requestor1);
        mta_string_free(requestor);

        status = mta_pair_list_options_get_requestor(options, 1, &requestor);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(requestor)) == requestor2);
        mta_string_free(requestor);

        mta_pair_list_options_free(options);
    }
}

TEST_CASE("model metadata", "C API"){
    const char* json = R"({
        "type": "metatomic_model_metadata",
        "name": "model name",
        "description": "model name is awesome",
        "authors": ["Author One", "Author Two"],
        "references": {
            "architecture": ["reference one", "reference two", "refrerence three"],
            "model": ["model reference"],
            "implementation": []
        },
        "extra": {
            "foo": "bar"
        }
    })";

    SECTION("JSON deserialization"){
        mta_model_metadata_t *metadata = nullptr;
        auto status = mta_model_metadata_from_json(json, &metadata);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(metadata != nullptr);

        mta_string_t name = nullptr;
        status = mta_model_metadata_get_name(metadata, &name);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(name)) == "model name");
        mta_string_free(name);

        mta_string_t description = nullptr;
        status = mta_model_metadata_get_description(metadata, &description);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(description)) == "model name is awesome");
        mta_string_free(description);

        size_t num_authors = 0;
        status = mta_model_metadata_authors_count(metadata, &num_authors);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(num_authors == 2);

        mta_string_t author0 = nullptr;
        status = mta_model_metadata_get_author(metadata, 0, &author0);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(author0)) == "Author One");
        mta_string_free(author0);

        mta_string_t author1 = nullptr;
        status = mta_model_metadata_get_author(metadata, 1, &author1);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(author1)) == "Author Two");
        mta_string_free(author1);

        size_t num_arch_refs = 0;
        status = mta_model_metadata_references_count(metadata, MTA_REFERENCES_ARCHITECTURE, &num_arch_refs);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(num_arch_refs == 3);

        size_t num_model_refs = 0;
        status = mta_model_metadata_references_count(metadata, MTA_REFERENCES_MODEL, &num_model_refs);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(num_model_refs == 1);

        size_t num_impl_refs = 0;
        status = mta_model_metadata_references_count(metadata, MTA_REFERENCES_IMPLEMENTATION, &num_impl_refs);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(num_impl_refs == 0);

        mta_string_t arch_ref0 = nullptr;
        status = mta_model_metadata_get_reference(metadata, MTA_REFERENCES_ARCHITECTURE, 0, &arch_ref0);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(arch_ref0)) == "reference one");
        mta_string_free(arch_ref0);

        mta_string_t arch_ref2 = nullptr;
        status = mta_model_metadata_get_reference(metadata, MTA_REFERENCES_ARCHITECTURE, 2, &arch_ref2);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(arch_ref2)) == "refrerence three");
        mta_string_free(arch_ref2);

        mta_string_t model_ref0 = nullptr;
        status = mta_model_metadata_get_reference(metadata, MTA_REFERENCES_MODEL, 0, &model_ref0);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(model_ref0)) == "model reference");
        mta_string_free(model_ref0);

        size_t num_extra = 0;
        status = mta_model_metadata_extra_count(metadata, &num_extra);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(num_extra == 1);

        mta_string_t extra_key0 = nullptr;
        status = mta_model_metadata_get_extra_key(metadata, 0, &extra_key0);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(extra_key0)) == "foo");
        mta_string_free(extra_key0);

        mta_string_t extra_value0 = nullptr;
        status = mta_model_metadata_get_extra_value(metadata, "foo", &extra_value0);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(extra_value0)) == "bar");
        mta_string_free(extra_value0);

        mta_model_metadata_free(metadata);
    }

    SECTION("JSON serialization"){
        mta_model_metadata_t *metadata = nullptr;
        auto status = mta_model_metadata_from_json(json, &metadata);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(metadata != nullptr);

        mta_string_t serialized = nullptr;
        status = mta_model_metadata_to_json(metadata, &serialized);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(serialized != nullptr);

        CHECK(std::string(mta_string_view(serialized)) ==
              R"({"type":"metatomic_model_metadata","name":"model name","authors":["Author One","Author Two"],"description":"model name is awesome","references":{"model":["model reference"],"architecture":["reference one","reference two","refrerence three"],"implementation":[]},"extra":{"foo":"bar"}})");

        mta_string_free(serialized);
        mta_model_metadata_free(metadata);
    }

    SECTION("Check out of bound requests"){
        mta_model_metadata_t *metadata = nullptr;
        auto status = mta_model_metadata_from_json(json, &metadata);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(metadata != nullptr);

        mta_string_t author = nullptr;
        status = mta_model_metadata_get_author(metadata, 2, &author);
        CHECK(status == MTA_INVALID_PARAMETER_ERROR);

        mta_string_t ref = nullptr;
        status = mta_model_metadata_get_reference(metadata, MTA_REFERENCES_MODEL, 1, &ref);
        CHECK(status == MTA_INVALID_PARAMETER_ERROR);

        status = mta_model_metadata_get_reference(metadata, MTA_REFERENCES_IMPLEMENTATION, 0, &ref);
        CHECK(status == MTA_INVALID_PARAMETER_ERROR);

        mta_string_t extra_key = nullptr;
        status = mta_model_metadata_get_extra_key(metadata, 1, &extra_key);
        CHECK(status == MTA_INVALID_PARAMETER_ERROR);

        mta_model_metadata_free(metadata);
    }
}

TEST_CASE("model capabilities", "C API"){
    const char* json = R"({
        "type": "metatomic_model_capabilities",
        "outputs": [{
            "type": "metatomic_quantity",
            "name": "energy",
            "unit": "eV",
            "description": "total energy",
            "gradients": ["positions"],
            "sample_kind": "system"
        }, {
            "type": "metatomic_quantity",
            "name": "charge",
            "unit": "e",
            "gradients": [],
            "sample_kind": "atom"
        }],
        "atomic_types": [1, 6, 8],
        "interaction_range": 5.5,
        "length_unit": "Angstrom",
        "supported_devices": ["cpu", "cuda"],
        "dtype": "float32"
    })";

    SECTION("JSON deserialization"){
        mta_model_capabilities_t *capabilities = nullptr;
        auto status = mta_model_capabilities_from_json(json, &capabilities);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(capabilities != nullptr);

        double interaction_range = -1.0;
        status = mta_model_capabilities_get_interaction_range(capabilities, &interaction_range);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(interaction_range == Approx(5.5).epsilon(1e-15));

        mta_string_t length_unit = nullptr;
        status = mta_model_capabilities_get_length_unit(capabilities, &length_unit);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(length_unit)) == "Angstrom");
        mta_string_free(length_unit);

        mta_dtype_t dtype = MTA_DTYPE_FLOAT64;
        status = mta_model_capabilities_get_dtype(capabilities, &dtype);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(dtype == MTA_DTYPE_FLOAT32);

        size_t num_outputs = 0;
        status = mta_model_capabilities_outputs_count(capabilities, &num_outputs);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(num_outputs == 2);

        mta_string_t output0 = nullptr;
        status = mta_model_capabilities_get_output_json(capabilities, 0, &output0);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(output0)) ==
              R"({"type":"metatomic_quantity","name":"energy","unit":"eV","description":"total energy","gradients":["positions"],"sample_kind":"system"})");
        mta_string_free(output0);

        mta_string_t output1 = nullptr;
        status = mta_model_capabilities_get_output_json(capabilities, 1, &output1);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(std::string(mta_string_view(output1)) ==
              R"({"type":"metatomic_quantity","name":"charge","unit":"e","gradients":[],"sample_kind":"atom"})");
        mta_string_free(output1);

        size_t num_atomic_types = 0;
        status = mta_model_capabilities_atomic_types_count(capabilities, &num_atomic_types);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(num_atomic_types == 3);

        int64_t atomic_type0 = -1;
        status = mta_model_capabilities_get_atomic_type(capabilities, 0, &atomic_type0);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(atomic_type0 == 1);

        int64_t atomic_type1 = -1;
        status = mta_model_capabilities_get_atomic_type(capabilities, 1, &atomic_type1);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(atomic_type1 == 6);

        int64_t atomic_type2 = -1;
        status = mta_model_capabilities_get_atomic_type(capabilities, 2, &atomic_type2);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(atomic_type2 == 8);

        size_t num_devices = 0;
        status = mta_model_capabilities_supported_devices_count(capabilities, &num_devices);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(num_devices == 2);

        mta_device_t device0 = MTA_DEVICE_CUDA;
        status = mta_model_capabilities_get_supported_device(capabilities, 0, &device0);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(device0 == MTA_DEVICE_CPU);

        mta_device_t device1 = MTA_DEVICE_CPU;
        status = mta_model_capabilities_get_supported_device(capabilities, 1, &device1);
        REQUIRE(status == MTA_SUCCESS);
        CHECK(device1 == MTA_DEVICE_CUDA);

        mta_model_capabilities_free(capabilities);
    }

    SECTION("JSON serialization"){
        mta_model_capabilities_t *capabilities = nullptr;
        auto status = mta_model_capabilities_from_json(json, &capabilities);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(capabilities != nullptr);

        mta_string_t serialized = nullptr;
        status = mta_model_capabilities_to_json(capabilities, &serialized);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(serialized != nullptr);

        CHECK(std::string(mta_string_view(serialized)) ==
              R"({"type":"metatomic_model_capabilities","outputs":[{"type":"metatomic_quantity","name":"energy","unit":"eV","description":"total energy","gradients":["positions"],"sample_kind":"system"},{"type":"metatomic_quantity","name":"charge","unit":"e","gradients":[],"sample_kind":"atom"}],"atomic_types":[1,6,8],"interaction_range":5.5,"length_unit":"Angstrom","supported_devices":["cpu","cuda"],"dtype":"float32"})");

        mta_string_free(serialized);
        mta_model_capabilities_free(capabilities);
    }

    SECTION("Check out of bound requests"){
        mta_model_capabilities_t *capabilities = nullptr;
        auto status = mta_model_capabilities_from_json(json, &capabilities);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(capabilities != nullptr);

        mta_string_t output = nullptr;
        status = mta_model_capabilities_get_output_json(capabilities, 2, &output);
        CHECK(status == MTA_INVALID_PARAMETER_ERROR);

        int64_t atomic_type = -1;
        status = mta_model_capabilities_get_atomic_type(capabilities, 3, &atomic_type);
        CHECK(status == MTA_INVALID_PARAMETER_ERROR);

        mta_device_t device = MTA_DEVICE_CPU;
        status = mta_model_capabilities_get_supported_device(capabilities, 2, &device);
        CHECK(status == MTA_INVALID_PARAMETER_ERROR);

        mta_model_capabilities_free(capabilities);
    }

    SECTION("JSON deserialization with wrong type"){
        const char* wrong_json = R"({
            "type": "something-else",
            "outputs": [],
            "atomic_types": [],
            "interaction_range": 0.0,
            "length_unit": "Angstrom",
            "supported_devices": ["cpu"],
            "dtype": "float32"
        })";

        mta_model_capabilities_t *capabilities = nullptr;
        auto status = mta_model_capabilities_from_json(wrong_json, &capabilities);
        CHECK(status == MTA_SERIALIZATION_ERROR);
    }
}
