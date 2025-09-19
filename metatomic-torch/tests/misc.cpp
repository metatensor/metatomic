#include <torch/torch.h>

#include "metatomic/torch.hpp"

#include <catch.hpp>
using namespace Catch::Matchers;

class TestWarningHandler : public torch::WarningHandler {
public:
    std::vector<std::string> messages;

    void process(const torch::Warning& warning) override {
        messages.push_back(warning.msg());
    }
};

TEST_CASE("Version macros") {
    CHECK(std::string(METATOMIC_TORCH_VERSION) == metatomic_torch::version());

    auto version = std::to_string(METATOMIC_TORCH_VERSION_MAJOR) + "."
        + std::to_string(METATOMIC_TORCH_VERSION_MINOR) + "."
        + std::to_string(METATOMIC_TORCH_VERSION_PATCH);

    // METATOMIC_TORCH_VERSION should start with `x.y.z`
    CHECK(std::string(METATOMIC_TORCH_VERSION).find(version) == 0);
}


TEST_CASE("Pick device") {
    // check that first entry in vector is picked, if no desired device is given
    std::vector<std::string> supported_devices = {"cpu", "cuda", "mps"};
    CHECK(metatomic_torch::pick_device(supported_devices) == "cpu");

    // test that desired device is picked if available
    std::vector<std::string> desired_devices = {"cpu"};
    if (torch::cuda::is_available()) {
        desired_devices.emplace_back("cuda");
    }
    if (torch::mps::is_available()) {
        desired_devices.emplace_back("mps");
    }

    for (const auto& desired_device: desired_devices) {
        CHECK(metatomic_torch::pick_device(supported_devices, desired_device) == desired_device);
    }

    // Check that warning is emitted:
    TestWarningHandler handler;
    torch::WarningUtils::WarningHandlerGuard guard(&handler);
    torch::WarningUtils::set_warnAlways(true);

    std::vector<std::string> supported_devices_foo = {"cpu", "fooo"};
    CHECK(metatomic_torch::pick_device(supported_devices_foo) == "cpu");
    REQUIRE_FALSE(handler.messages.empty());
    CHECK(handler.messages[0].find("'model_devices' contains an entry for unknown device") != std::string::npos);

    // check exception raised
    std::vector<std::string> supported_devices_cuda = {"cuda"};
    CHECK_THROWS_WITH(metatomic_torch::pick_device(supported_devices_cuda, "cpu"), StartsWith("failed to find a valid device"));

    std::vector<std::string> supported_devices_cpu = {"cpu"};
    CHECK_THROWS_WITH(metatomic_torch::pick_device(supported_devices_cpu, "cuda"), StartsWith("failed to find requested device"));
}
