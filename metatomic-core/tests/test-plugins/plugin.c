#include <metatomic.h>


static mta_status_t load_model(const char *load_from, const char *options_json, struct mta_model_t *model) {
    // This plugin can not load any model
    return MTA_MODEL_NOT_SUPPORTED_ERROR;
}


MTA_REGISTER_PLUGIN(register_plugin, {
    mta_plugin_t plugin = {
        .abi_version = MTA_ABI_VERSION,
        .name = "test-c-plugin",
        .load_model = load_model,
    };
    return register_plugin(plugin);
});
