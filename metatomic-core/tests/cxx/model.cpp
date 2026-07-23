#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <catch.hpp>
#include <metatensor.hpp>

#include "metatomic.hpp"
#include "helpers.hpp"


class SimpleCppModel: public metatomic::ModelBase {
public:
    explicit SimpleCppModel(double scale): scale_(scale) {}

    metatomic::ModelCapabilities capabilities() const override {
        metatomic::ModelCapabilities caps;
        caps.atomic_types({1, 6, 8});
        caps.interaction_range(4.5);
        caps.length_unit("nm");
        caps.supported_devices({metatomic::ModelCapabilities::Device::CPU});
        caps.dtype(metatomic::ModelCapabilities::DType::Float32);

        caps.add_output(metatomic::Quantity(
            "energy", "eV", metatomic::SampleKind::System
        ));

        return caps;
    }

    metatomic::ModelMetadata metadata() const override {
        metatomic::ModelMetadata meta;
        meta.name("simple C++ model");
        meta.description("test model for ModelBase");
        return meta;
    }

    std::vector<metatomic::PairListOptions> requested_pair_lists() const override {
        return {};
    }

    std::vector<metatomic::Quantity> requested_inputs() const override {
        return {};
    }

    std::vector<metatensor::TensorMap> execute(
        const std::vector<metatomic::System>& systems,
        const metatensor::Labels* selected_atoms,
        const std::vector<metatomic::Quantity>& requested_outputs
    ) override {
        (void) selected_atoms;

        std::vector<metatensor::TensorMap> outputs;
        outputs.reserve(requested_outputs.size());

        for (const auto& output: requested_outputs) {
            if (output.name() != "energy") {
                throw metatomic::Error("unknown output: " + output.name());
            }

            double energy = 0.0;
            for (const auto& system: systems) {
                energy += scale_ * static_cast<double>(system.size());
            }

            outputs.push_back(scalar_tensor(energy, output.name()));
        }

        return outputs;
    }

private:
    double scale_;
};


static mta_status_t load_cpp_model(
    const char* load_from,
    const char* options_json,
    mta_model_t* model
) {
    (void) options_json;

    if (std::strcmp(load_from, "test-cpp-model") != 0) {
        return MTA_MODEL_NOT_SUPPORTED_ERROR;
    }

    auto cpp_model = std::make_unique<SimpleCppModel>(2.5);
    *model = metatomic::ModelBase::to_mta_model(std::move(cpp_model));
    return MTA_SUCCESS;
}


namespace {
    /// Register the C++ model plugin once when the test executable loads.
    struct CppPluginRegistrar {
        CppPluginRegistrar() {
            static mta_plugin_t PLUGIN = {
                MTA_ABI_VERSION,
                "test-cpp-plugin",
                load_cpp_model,
            };
            auto status = mta_register_plugin(PLUGIN);
            if (status != MTA_SUCCESS) {
                throw metatomic::Error("failed to register test-cpp-plugin");
            }
        }
    } CPP_PLUGIN_REGISTRAR;
}


class ThrowingModel: public metatomic::ModelBase {
public:
    metatomic::ModelCapabilities capabilities() const override {
        throw metatomic::Error("capabilities failed");
    }

    metatomic::ModelMetadata metadata() const override {
        return metatomic::ModelMetadata();
    }

    std::vector<metatomic::PairListOptions> requested_pair_lists() const override {
        return {};
    }

    std::vector<metatomic::Quantity> requested_inputs() const override {
        return {};
    }

    std::vector<metatensor::TensorMap> execute(
        const std::vector<metatomic::System>&,
        const metatensor::Labels*,
        const std::vector<metatomic::Quantity>&
    ) override {
        return {};
    }
};


TEST_CASE("ModelBase can be used directly from C++") {
    auto model = std::make_unique<SimpleCppModel>(2.5);

    auto caps = model->capabilities();
    CHECK(caps.atomic_types().size() == 3);
    CHECK(caps.outputs().size() == 1);
    CHECK(caps.outputs()[0].name() == "energy");

    // supported_outputs defaults to capabilities().outputs()
    auto supported = model->supported_outputs();
    CHECK(supported.size() == 1);
    CHECK(supported[0].name() == "energy");

    auto system = test_system(4);
    auto systems = std::vector<metatomic::System>();
    systems.push_back(std::move(system));

    auto outputs = model->execute(systems, nullptr, caps.outputs());

    REQUIRE(outputs.size() == 1);
    CHECK(outputs[0].keys().count() == 1);

    auto block = outputs[0].block_by_id(0);
    auto values = block.values<double>();
    REQUIRE(values.data() != nullptr);
    CHECK(values.data()[0] == Approx(10.0));
}


TEST_CASE("ModelBase model can be loaded and used through the C API") {
    auto raw_model = metatomic::load_model("test-cpp-model", "{}", "test-cpp-plugin");

    CHECK(raw_model.capabilities != nullptr);
    CHECK(raw_model.metadata != nullptr);

    mta_string_t metadata = nullptr;
    auto status = raw_model.metadata(raw_model.data, &metadata);
    REQUIRE(status == MTA_SUCCESS);
    auto metadata_str = std::string(mta_string_view(metadata));
    CHECK(metadata_str.find("simple C++ model") != std::string::npos);
    mta_string_free(metadata);

    mta_string_t capabilities = nullptr;
    status = raw_model.capabilities(raw_model.data, &capabilities);
    REQUIRE(status == MTA_SUCCESS);
    auto capabilities_str = std::string(mta_string_view(capabilities));
    CHECK(capabilities_str.find("\"length_unit\":\"nm\"") != std::string::npos);
    mta_string_free(capabilities);

    CHECK(raw_model.unload(raw_model.data) == MTA_SUCCESS);
}


TEST_CASE("ModelWrapper can wrap a C++ model created with to_mta_model") {
    auto raw_model = metatomic::ModelBase::to_mta_model(
        std::make_unique<SimpleCppModel>(3.0)
    );
    auto model = metatomic::ModelWrapper(std::move(raw_model));

    auto system = test_system(4);
    auto systems = std::vector<metatomic::System>();
    systems.push_back(std::move(system));

    auto requested = model.capabilities().outputs();

    auto outputs = model.execute(systems, nullptr, requested);
    REQUIRE(outputs.size() == 1);

    // 4 atoms * scale 3.0 = 12.0
    auto block = outputs[0].block_by_id(0);
    auto values = block.values<double>();
    REQUIRE(values.data() != nullptr);
    CHECK(values.data()[0] == Approx(12.0));
}


TEST_CASE("ModelWrapper can wrap a model loaded from a plugin") {
    auto raw_model = metatomic::load_model("test-cpp-model", "{}", "test-cpp-plugin");
    auto model = metatomic::ModelWrapper(std::move(raw_model));

    auto caps = model.capabilities();
    CHECK(caps.length_unit() == "nm");
    CHECK(caps.outputs().size() == 1);
    CHECK(caps.outputs()[0].name() == "energy");

    auto metadata = model.metadata();
    CHECK(metadata.name() == "simple C++ model");

    CHECK(model.requested_pair_lists().empty());
    CHECK(model.requested_inputs().empty());

    auto system = test_system(4);
    auto systems = std::vector<metatomic::System>();
    systems.push_back(std::move(system));

    auto outputs = model.execute(systems, nullptr, caps.outputs());
    REQUIRE(outputs.size() == 1);

    auto block = outputs[0].block_by_id(0);
    auto values = block.values<double>();
    REQUIRE(values.data() != nullptr);
    CHECK(values.data()[0] == Approx(10.0));
}


TEST_CASE("ModelBase exceptions are propagated through mta_model_t callbacks") {
    auto raw_model = metatomic::ModelBase::to_mta_model(
        std::make_unique<ThrowingModel>()
    );

    mta_string_t caps = nullptr;
    auto status = raw_model.capabilities(raw_model.data, &caps);
    CHECK(status != MTA_SUCCESS);

    raw_model.unload(raw_model.data);
}


TEST_CASE("ModelWrapper move and view semantics") {
    SECTION("move") {
        auto raw = metatomic::ModelBase::to_mta_model(
            std::make_unique<SimpleCppModel>(1.0)
        );
        auto model = metatomic::ModelWrapper(std::move(raw));
        CHECK(model.as_mta_model_t() != nullptr);

        auto moved = std::move(model);
        CHECK(moved.as_mta_model_t() != nullptr);
        CHECK(moved.capabilities().outputs().size() == 1);
    }

    SECTION("view does not own") {
        auto raw = metatomic::ModelBase::to_mta_model(
            std::make_unique<SimpleCppModel>(1.0)
        );
        {
            auto view = metatomic::ModelWrapper::unsafe_view_from_ptr(raw);
            CHECK(view.capabilities().outputs().size() == 1);
        }
        // raw is still valid and must be unloaded manually
        CHECK(raw.unload(raw.data) == MTA_SUCCESS);
    }
}
