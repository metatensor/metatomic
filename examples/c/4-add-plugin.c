// .. _c-tutorial-add-plugin:
//
// Making a model loadable through a plugin
// =========================================
//
// The previous tutorial constructed a model directly, which requires its
// implementation to be linked with the program using it.
//
// Plugins allow models to be loaded separately. An engine can use different
// model implementations without containing model-specific loading code.
//
// A plugin provides a ``load_model`` callback. ``load_from`` is interpreted by
// the plugin and can, for example, be a generic string (i.e. "model name") or a file path.

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <metatomic.h>

// %%
//
// A minimal plugin
// ----------------
//
// The plugin below can load a single named model. Anything else returns
// :c:enumerator:`MTA_MODEL_NOT_SUPPORTED_ERROR` so other plugins get a
// chance when ``plugin_name`` is left as ``NULL``.
//
// ``unload`` frees the private data allocated in ``load_model``.

static mta_status_t stub_unload(void* model_data) {
    free(model_data);
    return MTA_SUCCESS;
}

// %%
//
// Metadata is the human-facing description of the model: name, authors,
// references. Capabilities list what it can compute. This stub only
// advertises an energy output; it does not implement ``execute_inner``
// (see the :ref:`previous tutorial <c-tutorial-add-model>` for a model
// that does).

static mta_status_t stub_metadata(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create(
        "{"
        "\"type\": \"metatomic_model_metadata\","
        "\"name\": \"plugin-demo\","
        "\"authors\": [\"metatomic C tutorials\"],"
        "\"description\": \"Minimal model used to demonstrate plugins\","
        "\"references\": {"
        "  \"model\": [], \"architecture\": [], \"implementation\": []"
        "},"
        "\"extra\": {}"
        "}"
    );
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t stub_capabilities(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create(
        "{"
        "\"type\": \"metatomic_model_capabilities\","
        "\"outputs\": [{"
        "  \"type\": \"metatomic_quantity\","
        "  \"name\": \"energy\","
        "  \"unit\": \"eV\","
        "  \"gradients\": [],"
        "  \"sample_kind\": \"system\""
        "}],"
        "\"atomic_types\": [1, 6, 8],"
        "\"interaction_range\": 0.0,"
        "\"length_unit\": \"Angstrom\","
        "\"supported_devices\": [\"cpu\"],"
        "\"dtype\": \"float64\""
        "}"
    );
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t stub_empty_list(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("[]");  // no pair lists, no extra inputs
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t stub_execute_inner(
    void* model_data,
    const mta_system_t* const* systems,
    uintptr_t systems_count,
    const mts_labels_t* selected_atoms,
    const char* requested_outputs_json,
    mts_tensormap_t** outputs,
    uintptr_t outputs_count
) {
    (void)model_data;
    (void)systems;
    (void)systems_count;
    (void)selected_atoms;
    (void)requested_outputs_json;
    (void)outputs;
    (void)outputs_count;
    mta_set_last_error(
        "plugin-demo does not implement execute_inner",
        "stub_execute_inner",
        NULL,
        NULL
    );
    return MTA_INTERNAL_ERROR;
}

// %%
//
// ``load_model`` is the plugin's only required callback. It either fills the
// :c:type:`mta_model_t` vtable or returns
// :c:enumerator:`MTA_MODEL_NOT_SUPPORTED_ERROR` so another plugin can try.

static mta_status_t demo_load_model(
    const char* load_from,
    const char* options_json,
    mta_model_t* model
) {
    (void)options_json;
    if (strcmp(load_from, "plugin-demo") != 0) {
        return MTA_MODEL_NOT_SUPPORTED_ERROR;
    }

    model->data = malloc(1);  // non-NULL so unload has something to free
    if (model->data == NULL) {
        mta_set_last_error("out of memory", "demo_load_model", NULL, NULL);
        return MTA_INTERNAL_ERROR;
    }
    model->unload = stub_unload;
    model->metadata = stub_metadata;
    model->capabilities = stub_capabilities;
    model->requested_pair_lists = stub_empty_list;
    model->requested_inputs = stub_empty_list;
    model->execute_inner = stub_execute_inner;
    return MTA_SUCCESS;
}

// %%
//
// Shared-library entry point
// --------------------------
//
// Plugins are typically distributed as shared libraries.
// :c:macro:`MTA_REGISTER_PLUGIN` defines the entry point used by metatomic
// when loading the library.
//
// .. code-block:: C
//
//     MTA_REGISTER_PLUGIN(register_plugin, {
//         mta_plugin_t plugin = {
//             .abi_version = MTA_ABI_VERSION,
//             .name = "tutorial-demo-plugin",
//             .load_model = demo_load_model,
//         };
//         return register_plugin(plugin);
//     });

int main(void) {

// %%
//
// A plugin that is already linked into the program can instead be registered
// directly with :c:func:`mta_register_plugin`.
//
// :c:func:`mta_load_model` selects a plugin and calls its ``load_model``
// callback. The plugin interprets ``load_from`` and fills the ``mta_model_t``
// vtable. Once loaded, the model is used independently of the plugin that
// created it.

mta_plugin_t plugin = {
    .abi_version = MTA_ABI_VERSION,
    .name = "tutorial-demo-plugin",
    .load_model = demo_load_model,
};

assert(mta_register_plugin(plugin) == MTA_SUCCESS);

mta_model_t model = {0};
assert(
    mta_load_model("plugin-demo", "{}", "tutorial-demo-plugin", &model)
    == MTA_SUCCESS
);
assert(model.data != NULL);

// %%
//
// Once loaded, the vtable is a normal :c:type:`mta_model_t`.

mta_string_t capabilities = NULL;
assert(model.capabilities(model.data, &capabilities) == MTA_SUCCESS);
assert(capabilities != NULL);
mta_string_free(capabilities);

// %%
//
// Asking a plugin for a model it does not know must fail. The callback
// returns :c:enumerator:`MTA_MODEL_NOT_SUPPORTED_ERROR`; with a named
// plugin, :c:func:`mta_load_model` wraps that as
// :c:enumerator:`MTA_INVALID_PARAMETER_ERROR`.

mta_model_t missing = {0};
mta_status_t status =
    mta_load_model("not-a-real-model", "{}", "tutorial-demo-plugin", &missing);
assert(status == MTA_INVALID_PARAMETER_ERROR);

assert(model.unload(model.data) == MTA_SUCCESS);
return EXIT_SUCCESS;
}
