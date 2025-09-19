#include <torch/torch.h>

#include "metatomic/torch/version.h"
#include "metatomic/torch/misc.hpp"

std::string metatomic_torch::version() {
    return METATOMIC_TORCH_VERSION;
}

std::string metatomic_torch::pick_device(
    std::vector<std::string> model_devices,
    torch::optional<std::string> desired_device
) {
    auto available_devices = std::vector<std::string>();
    std::string selected_device = "cpu";

    for (const auto& device: model_devices) {
        if (device == "cpu") {
            available_devices.emplace_back("cpu");
        } else if (device == "cuda") {
            if (torch::cuda::is_available()) {
                available_devices.emplace_back("cuda");
            }
        } else if (device == "mps") {
            if (torch::mps::is_available()) {
                available_devices.emplace_back("mps");
            }
        } else {
            TORCH_WARN("'model_devices' contains an entry for unknown device (" + torch::str(device)
                + "). It will be ignored.");
        }
    }

    if (available_devices.empty()) {
        C10_THROW_ERROR(ValueError,
            "failed to find a valid device. None of the devices supported by the model ("
            + torch::str(model_devices) + ") where available (" + torch::str(available_devices) + ")."
        );
    }

    if (desired_device == torch::nullopt) {
        // no user request, pick the device the model prefers
        selected_device = available_devices[0];
    } else {
        bool found_desired_device = false;
        for (const auto& device: available_devices) {
            if (device == desired_device) {
                selected_device = device;
                found_desired_device = true;
                break;
            }
        }

        if (!found_desired_device) {
            C10_THROW_ERROR(ValueError,
                "failed to find requested device (" + torch::str(desired_device.value()) + 
                "): it is either not supported by this model or not available on this machine"
            );
        }
    }
    return selected_device;
}
