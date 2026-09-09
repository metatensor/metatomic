#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <catch.hpp>
#include <metatensor.hpp>

#include "metatomic.hpp"
#include "helpers.hpp"


class SimpleCppModel: public metatomic::BaseModel {
public:
    explicit SimpleCppModel(double scale): scale_(scale) {}

    metatomic::ModelCapabilities capabilities() const final {
        auto capabilities = metatomic::ModelCapabilities();
        capabilities.atomic_types({1, 6, 8});
        capabilities.interaction_range(4.5);
        capabilities.length_unit("nm");
        capabilities.supported_devices({metatomic::ModelCapabilities::Device::CPU});
        capabilities.dtype(metatomic::ModelCapabilities::DType::Float32);

        capabilities.add_output(metatomic::Quantity(
            "energy", "eV", metatomic::SampleKind::System
        ));

        capabilities.add_output(metatomic::Quantity(
            "custom::output", "eV", metatomic::SampleKind::Atom
        ));

        return capabilities;
    }

    metatomic::ModelMetadata metadata() const final {
        auto metadata = metatomic::ModelMetadata();
        metadata.name("simple C++ model");
        metadata.description("test model for BaseModel");
        return metadata;
    }

    std::vector<metatomic::PairListOptions> requested_pair_lists() const final {
        return {};
    }

    std::vector<metatomic::Quantity> requested_inputs() const final {
        return {};
    }

    std::vector<metatensor::TensorMap> execute_inner(
        const std::vector<metatomic::System>& systems,
        const metatensor::Labels* selected_atoms,
        const std::vector<metatomic::Quantity>& requested_outputs
    ) final {
        std::vector<metatensor::TensorMap> outputs;
        outputs.reserve(requested_outputs.size());

        size_t atom_count = 0;
        if (selected_atoms != nullptr) {
            atom_count = selected_atoms->count();
        } else {
            for (const auto& system: systems) {
                atom_count += system.size();
            }
        }

        for (const auto& output: requested_outputs) {
            if (output.name() != "energy") {
                throw metatomic::Error("unknown output: " + output.name());
            }

            double energy = scale_ * static_cast<double>(atom_count);
            outputs.push_back(scalar_tensor(energy, output.name()));
        }

        return outputs;
    }

private:
    double scale_;
};

TEST_CASE("BaseModel") {
    auto model = std::make_unique<SimpleCppModel>(2.5);

    auto capabilities = model->capabilities();
    CHECK(capabilities.atomic_types().size() == 3);

    const auto& outputs = capabilities.outputs();
    CHECK(outputs.size() == 2);
    CHECK(outputs[0].name() == "energy");
    CHECK(outputs[1].name() == "custom::output");
    CHECK(outputs[1].sample_kind() == metatomic::SampleKind::Atom);

    auto system = test_system(4);
    auto systems = std::vector<metatomic::System>();
    systems.push_back(std::move(system));

    // NOTE: we call execute_inner directly only for testing
    // in practice, the model should be executed through the `mta_execute_model` function
    auto requested_outputs = std::vector<metatomic::Quantity>{outputs[0]};
    auto results = model->execute_inner(systems, nullptr, requested_outputs);

    REQUIRE(results.size() == 1);
    CHECK(results[0].keys().count() == 1);

    auto block = results[0].block_by_id(0);
    auto values = block.values<double>();
    REQUIRE(values.data() != nullptr);
    CHECK(values.data()[0] == Approx(10.0));
}


TEST_CASE("Wrap mta_model_t with ExternalModel") {
    auto raw_model = metatomic::BaseModel::to_mta_model(
        std::make_unique<SimpleCppModel>(3.0)
    );
    auto model = metatomic::ExternalModel(raw_model);

    auto outputs = model.capabilities().outputs();
    CHECK(outputs.size() == 2);
    CHECK(outputs[0].name() == "energy");
    CHECK(outputs[1].name() == "custom::output");

    // TODO: uncomment once `mta_execute_model` is implemented on the Rust side.
    // auto system = test_system(4);
    // std::vector<metatomic::System> systems;
    // systems.push_back(std::move(system));
    //
    // auto outputs = metatomic::execute_model(
    //     model, systems, nullptr, outputs, false
    // );
    // REQUIRE(outputs.size() == 1);
    //
    // // 4 atoms * scale 3.0 = 12.0
    // auto block = outputs[0].block_by_id(0);
    // auto values = block.values<double>();
    // REQUIRE(values.data() != nullptr);
    // CHECK(values.data()[0] == Approx(12.0));
}


TEST_CASE("ExternalModel move semantics") {
    auto raw = metatomic::BaseModel::to_mta_model(
        std::make_unique<SimpleCppModel>(1.0)
    );
    auto model = metatomic::ExternalModel(raw);
    CHECK(model.as_mta_model_t() != nullptr);

    auto moved = std::move(model);
    CHECK(moved.as_mta_model_t() != nullptr);
    CHECK(moved.capabilities().outputs().size() == 2);
}


TEST_CASE("ExternalModel release transfers ownership") {
    auto raw = metatomic::BaseModel::to_mta_model(
        std::make_unique<SimpleCppModel>(2.0)
    );
    auto model = metatomic::ExternalModel(raw);

    // release the raw model back to the caller; the ExternalModel is empty
    // and will not call unload on destruction
    auto released = model.release();
    CHECK(released.unload != nullptr);

    // re-wrap the released model to verify it is still valid
    auto wrapped = metatomic::ExternalModel(released);

    auto outputs = wrapped.capabilities().outputs();
    CHECK(outputs.size() == 2);
    CHECK(outputs[0].name() == "energy");
    CHECK(outputs[1].name() == "custom::output");

    // TODO: uncomment once `mta_execute_model` is implemented on the Rust side.
    // auto system = test_system(4);
    // std::vector<metatomic::System> systems;
    // systems.push_back(std::move(system));
    //
    // auto outputs = metatomic::execute_model(
    //     wrapped, systems, nullptr, outputs, false
    // );
    // REQUIRE(outputs.size() == 1);
    //
    // // 4 atoms * scale 2.0 = 8.0
    // auto block = outputs[0].block_by_id(0);
    // auto values = block.values<double>();
    // REQUIRE(values.data() != nullptr);
    // CHECK(values.data()[0] == Approx(8.0));
}


TEST_CASE("to_mta_model for ExternalModel") {
    auto inner_raw = metatomic::BaseModel::to_mta_model(
        std::make_unique<SimpleCppModel>(3.0)
    );
    void* inner_data = inner_raw.data;
    auto external = std::make_unique<metatomic::ExternalModel>(inner_raw);
    auto outer_raw = metatomic::BaseModel::to_mta_model(std::move(external));

    // `to_mta_model` short-circuits for `ExternalModel`
    // The raw model's data pointer should be the same as the inner model's data pointer.
    CHECK(outer_raw.data == inner_data);

    // The raw model's callbacks must all be set by `to_mta_model`.
    CHECK(outer_raw.capabilities != nullptr);
    CHECK(outer_raw.metadata != nullptr);
    CHECK(outer_raw.requested_pair_lists != nullptr);
    CHECK(outer_raw.requested_inputs != nullptr);
    CHECK(outer_raw.execute_inner != nullptr);
    CHECK(outer_raw.unload != nullptr);

    // Wrap the raw model back in an ExternalModel to test through the C++ interface.
    auto model = metatomic::ExternalModel(outer_raw);

    auto capabilities = model.capabilities();
    CHECK(capabilities.length_unit() == "nm");

    auto metadata = model.metadata();
    CHECK(metadata.name() == "simple C++ model");

    const auto& outputs = capabilities.outputs();
    REQUIRE(outputs.size() == 2);
    CHECK(outputs[0].name() == "energy");
    CHECK(outputs[1].name() == "custom::output");
    CHECK(outputs[1].sample_kind() == metatomic::SampleKind::Atom);

    CHECK(model.requested_pair_lists().empty());
    CHECK(model.requested_inputs().empty());
}
