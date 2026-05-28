#pragma once

#include <cstring>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <metatomic.h>

#include "./model.hpp"
#include "./utils.hpp"

namespace metatomic {

/// Abstract base class for metatomic plugins implemented in C++.
class Plugin {
public:
    virtual ~Plugin() = default;

    /// Name used to identify this plugin.
    virtual std::string name() const = 0;

    /// Load a model from `load_from`, using the provided key/value options.
    virtual Model load_model(
        const std::string& load_from,
        const std::vector<KeyValuePair>& options
    ) = 0;
};

namespace details {
    template<typename PluginT>
    struct PluginRegistration {
        static PluginT* plugin;
        static const char* name;

        static mta_status_t load_model(
            const char* load_from,
            const mta_kv_pair_t* options,
            uintptr_t options_count,
            mta_model_t* model
        ) {
            return details::catch_exceptions([&]() {
                details::check_pointer(plugin);
                details::check_pointer(model);

                auto loaded = plugin->load_model(
                    load_from == nullptr ? "" : load_from,
                    details::from_c_options(options, options_count)
                );

                *model = loaded.release();
            });
        }
    };

    template<typename PluginT>
    PluginT* PluginRegistration<PluginT>::plugin = nullptr;

    template<typename PluginT>
    const char* PluginRegistration<PluginT>::name = nullptr;
} // namespace details

/// Register a C++ plugin.
///
/// Due to the current C plugin ABI, this stores one plugin instance per concrete
/// C++ plugin type. The registered object must outlive all model-loading calls.
template<typename PluginT>
void register_plugin(PluginT& plugin) {
    static_assert(
        std::is_base_of<Plugin, PluginT>::value,
        "register_plugin expects a class derived from metatomic::Plugin"
    );

    details::PluginRegistration<PluginT>::plugin = &plugin;
    const auto name = plugin.name();
    // The C plugin registry keeps this pointer; allocate stable process-lifetime storage.
    auto* name_storage = new char[name.size() + 1];
    std::memcpy(name_storage, name.c_str(), name.size() + 1);
    details::PluginRegistration<PluginT>::name = name_storage;

    auto c_plugin = mta_plugin_t{
        details::PluginRegistration<PluginT>::name,
        &details::PluginRegistration<PluginT>::load_model,
    };

    mta_register_plugin(c_plugin);
}

/// Register a raw C plugin.
inline void register_plugin(mta_plugin_t plugin) {
    mta_register_plugin(plugin);
}

/// Load a plugin dynamic library from the given path.
inline void load_plugin(const std::string& path) {
    details::check_status(mta_load_plugin(path.c_str()));
}

/// Load a model using the given plugin.
inline Model load_model(
    const std::string& plugin_name,
    const std::string& load_from,
    const std::vector<KeyValuePair>& options = {}
) {
    auto c_options = details::to_c_options(options);

    auto model = mta_model_t{};
    details::check_status(mta_load_model(
        plugin_name.c_str(),
        load_from.c_str(),
        c_options.data(),
        c_options.size(),
        &model
    ));

    return Model(model);
}

/// Load a model, letting metatomic pick the plugin.
inline Model load_model(
    const std::string& load_from,
    const std::vector<KeyValuePair>& options = {}
) {
    auto c_options = details::to_c_options(options);

    auto model = mta_model_t{};
    details::check_status(mta_load_model(
        nullptr,
        load_from.c_str(),
        c_options.data(),
        c_options.size(),
        &model
    ));

    return Model(model);
}

} // namespace metatomic
