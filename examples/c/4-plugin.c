// .. _c-tutorial-plugin:
//
// Creating a plugin
// =================
//
// The :ref:`previous tutorial <c-tutorial-model>` constructed a model directly,
// which requires its implementation to be linked with the program using it.
//
// Plugins allow models to be loaded at runtime using shared libraries. This
// way, simulation engines can use different models without containing any
// model-specific code. The engine will register plugins with
// :c:func:`mta_load_plugin` and then load models with :c:func:`mta_load_model`.
// The latter will then query all loaded plugin to find one that can load the
// requested model.
//
// In the C API, plugins are represented by the :c:type:`mta_plugin_t` struct,
// which contains a :c:func:`mta_plugin_t.load_model` callback.

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <metatomic.h>

// %%
//
// A minimal model
// ---------------
//
// Let's start by defining a stub model that does not actually compute anything,
// but implements the required callbacks. For more details on the model
// callbacks, see the :ref:`previous tutorial <c-tutorial-model>`.

static mta_status_t error(mta_status_t status, const char* message) {
    assert(status != MTA_SUCCESS);
    mta_set_last_error(
        /*message=*/message,
        /*origin=*/"plugin tutorial",
        /*data=*/NULL,
        /*data_deleter=*/NULL
    );
    return status;
}

// mta_model_t.unload implementation
static mta_status_t stub_unload(void* model_data) {
    (void)model_data;
    return MTA_SUCCESS;
}

// mta_model_t.metadata implementation
static mta_status_t stub_metadata(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("{"
        "\"type\": \"metatomic_model_metadata\","
        "\"name\": \"Stub model for tutorials\","
        "\"authors\": [\"metatomic authors\"],"
        "\"references\": {\"model\": [], \"architecture\": [], \"implementation\": []}"
    "}");

    if (*out == NULL) {
        return error(MTA_MEMORY_ERROR, "failed to allocate metadata JSON");
    }
    return MTA_SUCCESS;
}

// mta_model_t.capabilities implementation
static mta_status_t stub_capabilities(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("{"
        "\"type\": \"metatomic_model_capabilities\","
        "\"outputs\": [],"
        "\"atomic_types\": [],"
        "\"interaction_range\": 0.0,"
        "\"length_unit\": \"Angstrom\","
        "\"supported_devices\": [\"cpu\"],"
        "\"dtype\": \"float64\""
    "}");

    if (*out == NULL) {
        return error(MTA_MEMORY_ERROR, "failed to allocate capabilities JSON");
    }
    return MTA_SUCCESS;
}

// mta_model_t.requested_pair_lists implementation
static mta_status_t stub_requested_pair_lists(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("[]");
    if (*out == NULL) {
        return error(MTA_MEMORY_ERROR, "failed to allocate requested pair lists JSON");
    }
    return MTA_SUCCESS;
}
// mta_model_t.requested_inputs implementation
static mta_status_t stub_requested_inputs(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("[]");
    if (*out == NULL) {
        return error(MTA_MEMORY_ERROR, "failed to allocate requested inputs JSON");
    }
    return MTA_SUCCESS;
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
    return error(MTA_MODEL_ERROR, "this model is a stub and does not compute anything");
}

// %%
//
// The plugin
// ----------
//
// Each plugin must implement the ``load_model`` callback. This function is
// called by :c:func:`mta_load_model` to load a model, and given a string
// indicating which model to load and a JSON string with options. The
// ``load_from`` string indicates which model to load, and typically is either a
// model name or the path to a model file. The JSON options contain any model
// and/or plugin parameters as a JSON object.
//
// It MUST either fills the :c:type:`mta_model_t` vtable or returns
// :c:enumerator:`MTA_UNSUPPORTED_MODEL_ERROR` to indicate that the plugin does
// not know how to load the requested model.

static mta_status_t demo_load_model(
    const char* load_from,
    const char* options_json,
    mta_model_t* model
) {
    (void)options_json;
    if (strcmp(load_from, "plugin-tutorial-stub-model") != 0) {
        return MTA_UNSUPPORTED_MODEL_ERROR;
    }

    model->unload = stub_unload;
    model->metadata = stub_metadata;
    model->capabilities = stub_capabilities;
    model->requested_pair_lists = stub_requested_pair_lists;
    model->requested_inputs = stub_requested_inputs;
    model->execute_inner = stub_execute_inner;
    return MTA_SUCCESS;
}

// %%
//
// We then need to make the plugin accessible to metatomic and
// :c:func:`mta_load_plugin`. This is done by the :c:macro:`MTA_REGISTER_PLUGIN`
// macro.
//
// This macro takes the name of a function that will be called to register the
// plugin, and a block that creates the :c:type:`mta_plugin_t` struct, and use
// them to define a registration function. The block must return a status code
// indicating sucess or failure.

MTA_REGISTER_PLUGIN(register_plugin, {
    fprintf(stderr, "registering plugin 'tutorial-plugin'\n");
    mta_plugin_t plugin = {
        .abi_version = MTA_ABI_VERSION,
        .name = "tutorial-plugin",
        .load_model = demo_load_model,
    };
    return register_plugin(plugin);
});

// %%
//

int main(void) {


// %%
//
// We can load the plugin from the current binary by using ``NULL`` as the path
// in :c:func:`mta_load_plugin`.
//
// .. warning::
//
//    To be able to load a plugin directly from an executable, all symbols
//    should be exported. This is the default on macOS and Windows, but on
//    linux we need to link the executable with ``-Wl,--export-dynamic``, or
//    set the ``ENABLE_EXPORTS`` property on the target in CMake.

mta_status_t status = mta_load_plugin(NULL);

if (status != MTA_SUCCESS) {
    const char* error_message = NULL;
    mta_last_error(&error_message, NULL, NULL);
    fprintf(stderr, "failed to load plugin: %s\n", error_message);
    return EXIT_FAILURE;
}

// %%
//
// Once all relevant plugins have been loaded, we can load a model with
// :c:func:`mta_load_model`. The easiest way to do this is to leave
// ``plugin_name=NULL`` and let metatomic find a plugin that can load the model.

mta_model_t model = {0};
status = mta_load_model(
    /*load_from=*/"plugin-tutorial-stub-model",
    /*options_json=*/NULL,
    /*plugin_name=*/NULL,
    &model
);

if (status != MTA_SUCCESS) {
    const char* error_message = NULL;
    mta_last_error(&error_message, NULL, NULL);
    fprintf(stderr, "failed to load model: %s\n", error_message);
    return EXIT_FAILURE;
}

// %%
//
// Once loaded, we can use the model as usual. For example, we can query the
// metadata

mta_string_t metadata = NULL;
status = model.metadata(model.data, &metadata);
if (status != MTA_SUCCESS) {
    const char* error_message = NULL;
    mta_last_error(&error_message, NULL, NULL);
    fprintf(stderr, "failed to get model metadata: %s\n", error_message);
    return EXIT_FAILURE;
}

assert(metadata != NULL);
assert(strstr(mta_string_view(metadata), "Stub model for tutorials") != NULL);
mta_string_free(metadata);

model.unload(model.data);

// %%
//
// We can also specify the plugin to use when loading a model, preventing
// metatomic to try to use a different plugin to load this model:

memset(&model, 0, sizeof(model));
status = mta_load_model(
    /*load_from=*/"plugin-tutorial-stub-model",
    /*options_json=*/NULL,
    /*plugin_name=*/"tutorial-plugin",
    &model
);

if (status != MTA_SUCCESS) {
    const char* error_message = NULL;
    mta_last_error(&error_message, NULL, NULL);
    fprintf(stderr, "failed to load model: %s\n", error_message);
    return EXIT_FAILURE;
}
model.unload(model.data);

// %%
//

return EXIT_SUCCESS;
}
