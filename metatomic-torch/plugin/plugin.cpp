// A metatomic plugin loading models exported with the legacy TorchScript-based
// `metatomic.torch.AtomisticModel` API (typically `.pt` files), and running
// them through `metatomic_torch::LegacyModel`.
//
// The supported options are `extensions_directory`, the directory containing
// the TorchScript extensions the model requires, and `device`, the device on
// which to run the model.

#include <map>
#include <memory>
#include <string>

#include <caffe2/serialize/inline_container.h>

#include <metatomic.hpp>

#include "./legacy_model.hpp"

/// Check if `path` is a file exported by `AtomisticModel` with metatomic-torch
/// v0.1, or with a development build of v0.2.0, from its `metatomic-version`
/// record. Newer TorchScript models are not legacy models.
static bool is_atomistic_model(const std::string& path) {
    try {
        auto reader = caffe2::serialize::PyTorchStreamReader(path);
        if (!reader.hasRecord("extra/metatomic-version")) {
            return false;
        }

        auto [data, size] = reader.getRecord("extra/metatomic-version");
        auto version = std::string(static_cast<const char*>(data.get()), size);
        return version.rfind("0.1.", 0) == 0 || version.rfind("0.2.0.dev", 0) == 0;
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

    // the extensions are needed to load the model, the other options are given
    // to `set_options`
    auto model_options = options;
    auto extensions_directory = c10::optional<std::string>();
    if (auto it = model_options.find("extensions_directory"); it != model_options.end()) {
        extensions_directory = it->second;
        model_options.erase(it);
    }

    auto model = std::make_unique<metatomic_torch::LegacyModel>(load_from, extensions_directory);
    model->set_options(model_options);
    return model;
}

MTA_REGISTER_CXX_PLUGIN("metatomic-torch", load_model);
