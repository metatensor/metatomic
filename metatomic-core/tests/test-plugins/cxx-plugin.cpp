#include <map>
#include <memory>
#include <string>
#include <vector>

#include <metatensor.hpp>
#include "metatomic.hpp"


class SimpleModel: public metatomic::BaseModel {
public:
    metatomic::ModelCapabilities capabilities() const override final {
        return metatomic::ModelCapabilities::builder()
            .atomic_types({1, 6, 8})
            .interaction_range(4.5)
            .length_unit("nm")
            .supported_devices({metatomic::ModelCapabilities::Device::CPU})
            .dtype(metatomic::ModelCapabilities::DType::Float32)
            .add_output(metatomic::Quantity::builder()
                .name("energy")
                .unit("eV")
                .sample_kind(metatomic::SampleKind::System)
                .build())
            .build();
    }

    metatomic::ModelMetadata metadata() const override final {
        return metatomic::ModelMetadata::builder()
            .name("simple C++ plugin model")
            .description("test model for MTA_REGISTER_CXX_PLUGIN")
            .build();
    }

    std::vector<metatomic::PairListOptions> requested_pair_lists() const final {
        return {};
    }

    std::vector<metatomic::Quantity> requested_inputs() const final {
        return {};
    }

    std::vector<metatensor::TensorMap> execute_inner(
        const std::vector<metatomic::System>&,
        const metatensor::Labels*,
        const std::vector<metatomic::Quantity>&
    ) final {
        return {};
    }
};


std::unique_ptr<metatomic::BaseModel> load_model_cxx(
    std::string load_from,
    std::map<std::string, std::string> options
) {
    if (load_from == "throws") {
        throw metatomic::Error("load_model_cxx: intentional failure for '" + load_from + "'");
    }

    if (load_from != "test-cxx-model") {
        // this plugin can not load this model
        return nullptr;
    }

    return std::make_unique<SimpleModel>();
}


MTA_REGISTER_CXX_PLUGIN("test-cxx-plugin", load_model_cxx);
