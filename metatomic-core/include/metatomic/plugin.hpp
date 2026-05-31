#pragma once

#include <map>
#include <string>
#include <type_traits>
#include <utility>

#include <metatomic.h>
#include <nlohmann/json.hpp>

#include "./model.hpp"
#include "./utils.hpp"

namespace metatomic {

/// Abstract base class for metatomic plugins implemented in C++.
class Plugin {
public:
    virtual ~Plugin() = default;

    virtual std::string name() const = 0;

    virtual Model load_model(
        const std::string& load_from,
        const std::map<std::string, std::string>& options = {}
    ) = 0;
};

class PluginHandle final {
public:
    explicit PluginHandle(std::string name): name_(std::move(name)) {}

    const std::string& name() const {
        return name_;
    }

    Model load_model(
        const std::string& load_from,
        const std::map<std::string, std::string>& options = {}
    ) const {
        auto model = mta_model_t{};
        const auto options_json = nlohmann::json(options).dump();
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

        static mta_status_t load_model(
            const char* load_from,
            const char* options_json,
            mta_model_t* model
        ) {
            return details::catch_exceptions([&]() {
                details::check_pointer(plugin);
                details::check_pointer(model);

                auto options = std::map<std::string, std::string>();
                if (options_json != nullptr) {
                    options = nlohmann::json::parse(options_json).get<std::map<std::string, std::string>>();
                }

                auto loaded = plugin->load_model(load_from == nullptr ? "" : load_from, options);
                *model = loaded.release();
            });
        }
    };

    template<typename PluginT>
    PluginT* PluginRegistration<PluginT>::plugin = nullptr;
} // namespace details

template<typename PluginT>
void register_plugin(PluginT& plugin) {
    static_assert(
        std::is_base_of<Plugin, PluginT>::value,
        "register_plugin expects a class derived from metatomic::Plugin"
    );

    details::PluginRegistration<PluginT>::plugin = &plugin;
    const auto name = plugin.name();

    auto c_plugin = mta_plugin_t{
        MTA_ABI_VERSION,
        name.c_str(),
        &details::PluginRegistration<PluginT>::load_model,
    };

    details::check_status(mta_register_plugin(c_plugin));
}

inline void load_plugin(const std::string& path) {
    details::check_status(mta_load_plugin(path.c_str()));
}

inline PluginHandle plugin(const std::string& name) {
    return PluginHandle(name);
}

inline Model load_model(
    const std::string& plugin_name,
    const std::string& load_from,
    const std::map<std::string, std::string>& options = {}
) {
    return plugin(plugin_name).load_model(load_from, options);
}

inline Model load_model(
    const std::string& load_from,
    const std::map<std::string, std::string>& options = {}
) {
    auto model = mta_model_t{};
    const auto options_json = nlohmann::json(options).dump();
    details::check_status(mta_load_model(
        nullptr,
        load_from.c_str(),
        options_json.c_str(),
        &model
    ));

    return Model(model);
}

} // namespace metatomic
