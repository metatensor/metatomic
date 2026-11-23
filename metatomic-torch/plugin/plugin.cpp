// A metatomic plugin loading models exported with the legacy TorchScript-based
// `metatomic.torch.AtomisticModel` API (typically `.pt` files), and running
// them through `metatomic_torch::LegacyModel`.
//
// The supported options are `extensions_directory`, the directory containing
// the TorchScript extensions the model requires, and `device`, the device on
// which to run the model.

#include <array>
#include <map>
#include <memory>
#include <regex>
#include <string>

#include <caffe2/serialize/inline_container.h>

#include <metatomic.hpp>

#include "./legacy_model.hpp"

/// Check if `version`, a version of metatomic-torch in the form Python's
/// `packaging` gives it, is below 0.2.0. Development builds and release
/// candidates of 0.2.0, like `0.2.0.dev120+git.ae5d859` or `0.2.0rc1`, are.
static bool is_before_0_2_0(const std::string& version) {
    static const auto pattern = std::regex(R"(^(\d+)\.(\d+)(?:\.(\d+))?(.*)$)");
    auto match = std::smatch();
    if (!std::regex_match(version, match, pattern)) {
        return false;
    }

    auto release = std::array<int, 3>{
        std::stoi(match[1]),
        std::stoi(match[2]),
        match[3].matched ? std::stoi(match[3]) : 0,
    };
    if (release != std::array<int, 3>{0, 2, 0}) {
        return release < std::array<int, 3>{0, 2, 0};
    }

    auto suffix = match[4].str();
    for (const auto* before: {"a", "b", "rc", ".dev"}) {
        if (suffix.rfind(before, 0) == 0) {
            return true;
        }
    }
    return false;
}

/// Check if `path` is a file exported by `AtomisticModel` before metatomic-torch
/// v0.2.0, from its `metatomic-version` record. Newer TorchScript models are
/// not legacy models.
static bool is_atomistic_model(const std::string& path) {
    try {
        auto reader = caffe2::serialize::PyTorchStreamReader(path);
        if (!reader.hasRecord("extra/metatomic-version")) {
            return false;
        }

        auto [data, size] = reader.getRecord("extra/metatomic-version");
        return is_before_0_2_0(std::string(static_cast<const char*>(data.get()), size));
    } catch (const std::exception&) {
        // not a file, or not a TorchScript archive
        return false;
    }
}

static std::unique_ptr<metatomic::BaseModel> load_model(
    const std::string& load_from,
    const std::map<std::string, std::string>& options
) {
    if (!is_atomistic_model(load_from)) {
        return nullptr;
    }

    auto extensions_directory = c10::optional<std::string>();
    auto device = c10::optional<std::string>();
    for (const auto& [name, value]: options) {
        if (name == "extensions_directory") {
            extensions_directory = value;
        } else if (name == "device") {
            device = value;
        } else {
            throw metatomic::Error("unknown option for metatomic-torch models: '" + name + "'");
        }
    }

    return std::make_unique<metatomic_torch::LegacyModel>(load_from, extensions_directory, device);
}

MTA_REGISTER_CXX_PLUGIN("metatomic-torch", load_model);
