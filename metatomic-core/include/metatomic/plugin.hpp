#pragma once

#include <string>
#include <type_traits>
#include <utility>

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
        const std::string& options_json = "{}"
    ) = 0;
};

/// Handle to a plugin registered in metatomic's global plugin registry.
class PluginHandle final {
public:
    explicit PluginHandle(std::string name): name_(std::move(name)) {}

    /// Name used to identify this plugin.
    const std::string& name() const {
        return name_;
    }

    /// Load a model from `load_from`, using the provided key/value options.
    Model load_model(
        const std::string& load_from,
        const std::string& options_json = "{}"
    ) const {
        auto model = mta_model_t{};
        details::check_status(mta_load_model(
            name_.c_str(),
            load_from.c_str(),
            options_json.c_str(),
            &model
        ));

        return Model(model);
    }

private:
    std::string name_;
};

namespace details {
    template<typename PluginT>
    struct PluginRegistration {
        static PluginT* plugin;
        static std::string name;

        static mta_status_t load_model(
            const char* load_from,
            const char* options_json,
            mta_model_t* model
        ) {
            return details::catch_exceptions([&]() {
                details::check_pointer(plugin);
                details::check_pointer(model);

                auto loaded = plugin->load_model(
                    load_from == nullptr ? "" : load_from,
                    options_json == nullptr ? "{}" : options_json
                );

                *model = loaded.release();
            });
        }
    };

    template<typename PluginT>
    PluginT* PluginRegistration<PluginT>::plugin = nullptr;

    template<typename PluginT>
    std::string PluginRegistration<PluginT>::name;
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
    details::PluginRegistration<PluginT>::name = plugin.name();

    auto c_plugin = mta_plugin_t{
        details::PluginRegistration<PluginT>::name.c_str(),
        &details::PluginRegistration<PluginT>::load_model,
    };

    mta_register_plugin(c_plugin);
}

/// Load a plugin dynamic library from the given path.
inline void load_plugin(const std::string& path) {
    details::check_status(mta_load_plugin(path.c_str()));
}

/// Get a handle to a plugin in metatomic's global plugin registry.
inline PluginHandle plugin(const std::string& name) {
    return PluginHandle(name);
}

/// Load a model using the given plugin.
inline Model load_model(
    const std::string& plugin_name,
    const std::string& load_from,
    const std::string& options_json = "{}"
) {
    return plugin(plugin_name).load_model(load_from, options_json);
}

/// Load a model, letting metatomic pick the plugin.
inline Model load_model(
    const std::string& load_from,
    const std::string& options_json = "{}"
) {
    auto model = mta_model_t{};
    details::check_status(mta_load_model(
        nullptr,
        load_from.c_str(),
        options_json.c_str(),
        &model
    ));

    return Model(model);
}

} // namespace metatomic
