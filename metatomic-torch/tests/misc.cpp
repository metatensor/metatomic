#include <c10/util/intrusive_ptr.h>
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
    std::vector<std::string> supported = {"cpu", "cuda", "mps"};

    auto devtype = metatomic_torch::pick_device(supported);
    // devtype must be one of the DeviceType enum values; at minimum CPU should be selectable
    // Catch doesn't allow operator|| inside assertions in a portable way, so evaluate
    // the boolean expression first and assert that.
    bool dev_ok = (devtype == c10::DeviceType::CPU) ||
                  (devtype == c10::DeviceType::CUDA) ||
                  (devtype == c10::DeviceType::MPS);
    REQUIRE(dev_ok);

    // Test requested device selection when available
    if (torch::cuda::is_available()) {
        auto dt_cuda = metatomic_torch::pick_device(supported, std::string("cuda"));
        CHECK(dt_cuda == c10::DeviceType::CUDA);
    }
    if (torch::mps::is_available()) {
        auto dt_mps = metatomic_torch::pick_device(supported, std::string("mps"));
        CHECK(dt_mps == c10::DeviceType::MPS);
    }

    // Check that warning is emitted:
    TestWarningHandler handler;
    torch::WarningUtils::WarningHandlerGuard guard(&handler);
    torch::WarningUtils::set_warnAlways(true);

    std::vector<std::string> supported_devices_foo = {"cpu", "fooo"};
    auto tdevtype = torch::Device(metatomic_torch::pick_device(supported_devices_foo));
    CHECK(tdevtype.str() == "cpu");
    REQUIRE_FALSE(handler.messages.empty());
    CHECK(handler.messages[0].find("'model_devices' contains an entry for unknown device") != std::string::npos);
}

TEST_CASE("Pick device errors") {
    // If the model only supports CUDA and CUDA is not available, throw
    std::vector<std::string> only_cuda = {"cuda"};
    if (!torch::cuda::is_available()) {
        CHECK_THROWS_WITH(metatomic_torch::pick_device(only_cuda), StartsWith("failed to find a valid device"));
    } else {
        // If CUDA is available, requesting CPU when not declared should raise
        std::vector<std::string> only_cpu = {"cpu"};
        CHECK_THROWS_WITH(metatomic_torch::pick_device(only_cpu, std::string("cuda")), StartsWith("failed to find requested device"));
    }
}


TEST_CASE("Pick variant") {
    auto output_base = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
    output_base->description = "my awesome energy";
    output_base->set_quantity("energy");

    auto variantA = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
    variantA->set_quantity("energy");
    variantA->description = "Variant A of the output";

    auto variantfoo = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
    variantfoo->set_quantity("energy");
    variantfoo->description = "Variant foo of the output";

    auto outputs = torch::Dict<std::string, metatomic_torch::ModelOutput>();
    outputs.insert("energy", output_base);
    outputs.insert("energy/A", variantA);
    outputs.insert("energy/foo", variantfoo);

    CHECK(metatomic_torch::pick_output("energy", outputs) == "energy");
    CHECK(metatomic_torch::pick_output("energy", outputs, "A") == "energy/A");
    CHECK_THROWS_WITH(metatomic_torch::pick_output("foo", outputs), StartsWith("output 'foo' not found in outputs"));
    CHECK_THROWS_WITH(metatomic_torch::pick_output("energy", outputs, "C"), StartsWith("variant 'C' for output 'energy' not found in outputs"));

    (void)outputs.erase("energy");
    const auto *err = "output 'energy' has no default variant and no `desired_variant` was given. "
        "Available variants are:\n"
        " - 'energy/A'  : Variant A of the output\n"
        " - 'energy/foo': Variant foo of the output";
    CHECK_THROWS_WITH(metatomic_torch::pick_output("energy", outputs), StartsWith(err));
}
