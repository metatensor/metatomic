// A metatomic plugin loading models exported with the TorchScript-based
// `metatomic.torch.AtomisticModel` API (typically `.pt` files), and running
// them through `metatomic_torch::LegacyModel`.
//
// The only supported option is `extensions_directory`, the directory
// containing the TorchScript extensions the model requires.

#include <map>
#include <memory>
#include <string>

#include <caffe2/serialize/inline_container.h>

#include <metatomic.hpp>
#include <metatomic/torch/legacy.hpp>

/// Check if `path` is a file exported by `AtomisticModel`, using the same
/// record as `metatomic_torch::load_atomistic_model`.
static bool is_atomistic_model(const std::string& path) {
    try {
        auto reader = caffe2::serialize::PyTorchStreamReader(path);
        return reader.hasRecord("extra/metatomic-version");
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
    for (const auto& [name, value]: options) {
        if (name == "extensions_directory") {
            extensions_directory = value;
        } else {
            throw metatomic::Error("unknown option for metatomic-torch models: '" + name + "'");
        }
    }

    return std::make_unique<metatomic_torch::LegacyModel>(load_from, extensions_directory);
}

MTA_REGISTER_CXX_PLUGIN("metatomic-torch", load_model);
