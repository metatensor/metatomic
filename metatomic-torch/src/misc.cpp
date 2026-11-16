#include <string>
#include <vector>
#include <cstring>

#include <torch/torch.h>
#include <metatensor/torch.hpp>
#include <metatomic/utils.hpp>

#include "metatomic/torch/misc.hpp"
#include "metatomic/torch/system.hpp"
#include "metatomic/torch/version.h"

#include "./internal/utils.hpp"


using namespace metatomic_torch;

std::string metatomic_torch::version() {
    return METATOMIC_TORCH_VERSION;
}


double metatomic_torch::unit_conversion_factor(
    const std::string& from_unit,
    const std::string& to_unit
) {
    try {
        return metatomic::unit_conversion_factor(from_unit, to_unit);
    } catch (const metatomic::Error& e) {
        C10_THROW_ERROR(ValueError, e.what());
    }
}


void metatomic_torch::save(const std::string& path, const System& system) {
    metatomic::io::save(path, system->as_metatomic());
}

System metatomic_torch::load(const std::string& path) {
    auto system = metatomic::io::load(path, metatensor_torch::details::create_torch_array);
    return SystemHolder::from_metatomic(std::move(system));
}

torch::Tensor metatomic_torch::save_buffer(const System& system) {
    auto buffer = metatomic::io::save_buffer(system->as_metatomic());
    // move the buffer to the heap so it can escape this function
    // `torch::from_blob` does not take ownership of the data,
    // so we need to register a custom deleter to clean up when
    // the tensor is no longer used
    auto* buffer_data = new std::vector<uint8_t>(std::move(buffer));

    auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCPU);
    auto deleter = [=](void* data) {
        delete buffer_data;
    };

    // use a tensor of bytes to store the data
    return torch::from_blob(
        buffer_data->data(),
        {static_cast<int64_t>(buffer_data->size())},
        deleter,
        options
    );
}


System metatomic_torch::load_buffer(torch::Tensor buffer) {
    if (buffer.scalar_type() != torch::kUInt8) {
        C10_THROW_ERROR(ValueError,
            "`buffer` must be a tensor of uint8, not " +
            scalar_type_name(buffer.scalar_type())
        );
    }

    if (buffer.device().type() != torch::kCPU) {
        C10_THROW_ERROR(ValueError,
            "`buffer` must be a tensor on CPU, not on " + buffer.device().str()
        );
    }

    if (buffer.sizes().size() != 1) {
        C10_THROW_ERROR(ValueError,
            "`buffer` must be a 1-dimensional tensor"
        );
    }

    if (!buffer.is_contiguous()) {
        C10_THROW_ERROR(ValueError,
            "`buffer` must be a contiguous tensor"
        );
    }

    auto system = metatomic::io::load_buffer(
        buffer.data_ptr<uint8_t>(),
        static_cast<size_t>(buffer.size(0)),
        metatensor_torch::details::create_torch_array
    );

    return SystemHolder::from_metatomic(std::move(system));
}
