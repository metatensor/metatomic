#include <torch/torch.h>

#include "metatomic/torch/model.hpp"
#include "metatomic/torch/version.h"
#include "metatomic/torch/misc.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include <cstring>

#include "./internal/utils.hpp"

#include "metatomic/torch/system.hpp"
#include <metatensor/torch.hpp>

using namespace metatomic_torch;

std::string metatomic_torch::version() {
    return METATOMIC_TORCH_VERSION;
}

using namespace std::string_literals;

static inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static inline bool available_device(const std::string &n) {
    if (n == "cpu") {
        return true;
    } else if (n == "cuda") {
        return torch::cuda::is_available();
    } else  if (n == "mps") {
        return torch::mps::is_available();
    } else if (n == "hip") {
#ifdef TORCH_ENABLE_HIP
        return torch::hip::is_available();
#else
        return false;
#endif
    } else if (
        n == "xla" || n == "ipu" || n == "xpu" || n == "ve" || n == "opencl" ||
        n == "opengl" || n == "vulkan" || n == "mkldnn" || n == "ideep" ||
        n == "mtia" || n == "meta" || n == "hpu"
    ) {
        // For many backends we can't reliably test availability at runtime;
        // assume existence (caller may still fail when using the device).
        return true;
    } else {
        return false;
    }
}

static inline torch::DeviceType map_to_devicetype(const std::string &n) {
    if (n == "cpu") {
        return torch::DeviceType::CPU;
    } else if (n == "cuda") {
        return torch::DeviceType::CUDA;
    } else if (n == "mps") {
        return torch::DeviceType::MPS;
    } else if (n == "hip") {
        return torch::DeviceType::HIP;
    } else if (n == "xla") {
        return torch::DeviceType::XLA;
    } else if (n == "ipu") {
        return torch::DeviceType::IPU;
    } else if (n == "xpu") {
        return torch::DeviceType::XPU;
    } else if (n == "ve") {
        return torch::DeviceType::VE;
    } else if (n == "opencl") {
        return torch::DeviceType::OPENCL;
    } else if (n == "opengl") {
        return torch::DeviceType::OPENGL;
    } else if (n == "vulkan") {
        return torch::DeviceType::Vulkan;
    } else if (n == "mkldnn") {
        return torch::DeviceType::MKLDNN;
    } else if (n == "ideep") {
        return torch::DeviceType::IDEEP;
    } else if (n == "mtia") {
        return torch::DeviceType::MTIA;
    } else if (n == "meta") {
        return torch::DeviceType::Meta;
    } else if (n == "hpu") {
        return torch::DeviceType::HPU;
    } else {
        C10_THROW_ERROR(ValueError, "failed to find a valid device type for '" + n + "'");
    }
}

static inline bool is_known_device(const std::string &n) {
    return (
        n == "cpu" || n == "cuda" || n == "mps" || n == "hip" || n == "xla" ||
        n == "ipu" || n == "xpu" || n == "ve" || n == "opencl" ||
        n == "opengl" || n == "vulkan" || n == "mkldnn" || n == "ideep" ||
        n == "mtia" || n == "meta" || n == "hpu"
    );
}

c10::DeviceType metatomic_torch::pick_device(
    std::vector<std::string> model_devices,
    torch::optional<std::string> desired_device
) {
    // build list of available (normalized) device names in order
    std::vector<std::string> available;
    available.reserve(model_devices.size());
    for (const auto& device : model_devices) {
        std::string lower_device = lower(device);
        if (!is_known_device(lower_device)) {
            TORCH_WARN("ignoring unknown device '" + device + "' from `model_devices`");
            continue;
        }
        if (!available_device(lower_device)) {
            continue;
        }
        available.emplace_back(std::move(lower_device));
    }

    if (available.empty()) {
        C10_THROW_ERROR(ValueError,
            "failed to find a valid device. None of the "
            "model-supported devices are available."
        );
    }

    // if no desired device requested, pick first available
    if (!desired_device.has_value() || desired_device->empty()) {
        return map_to_devicetype(available.front());
    }

    // normalize desired and check
    std::string wanted_str = lower(desired_device.value());
    torch::DeviceType wanted_type;
    try {
        wanted_type = torch::Device(wanted_str).type();
    } catch (const std::exception &) {
        C10_THROW_ERROR(ValueError, "invalid device string: " + desired_device.value());
    }

    for (auto &a : available) {
        if (map_to_devicetype(a) == wanted_type) {
            return wanted_type;
        }
    }

    C10_THROW_ERROR(ValueError,
        "failed to find requested device (" + desired_device.value() +
        "): it is either not supported by this model or not available on this machine"
    );
}

std::string metatomic_torch::pick_output(
    std::string requested_output,
    torch::Dict<std::string, ModelOutput> outputs,
    torch::optional<std::string> desired_variant
) {
    std::vector<std::string> matching_keys;
    bool has_exact = false;

    for (const auto& output: outputs) {
        const auto& key = output.key();

        // match either exact `requested_output` or `requested_output/<variant>`
        if (key == requested_output
            || (key.size() > requested_output.size()
                && key.compare(0, requested_output.size(), requested_output) == 0
                && key[requested_output.size()] == '/')) {
            matching_keys.emplace_back(key);

            if (key == requested_output) {
                has_exact = true;
            }
        }
    }

    if (matching_keys.empty()) {
        C10_THROW_ERROR(ValueError,
            "output '" + requested_output + "' not found in outputs"
        );
    }

    if (desired_variant != torch::nullopt) {
        const auto& output = requested_output + "/" + desired_variant.value();
        auto it = std::find(matching_keys.begin(), matching_keys.end(), output);
        if (it != matching_keys.end()) {
            return *it;
        }
        C10_THROW_ERROR(ValueError,
            "variant '" + desired_variant.value() + "' for output '" + requested_output +
            "' not found in outputs"
        );
    } else if (has_exact) {
        return requested_output;
    } else {
        std::ostringstream oss;
        oss << "output '" << requested_output << "' has no default variant and no `desired_variant` was given. Available variants are:";

        size_t maxlen = 0;
        for (const auto& key: matching_keys) {
            maxlen = std::max(key.size(), maxlen);
        }

        for (const auto& key: matching_keys) {
            auto description = outputs.at(key)->description;
            std::string padding(maxlen - key.size(), ' ');
            oss << "\n - '" << key << "'" << padding << ": " << description;
        }
        C10_THROW_ERROR(ValueError, oss.str());
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
