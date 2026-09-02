// Loading models through plugins
// ==============================
//
// Metatomic discovers models through **plugins**. A plugin is a small
// :c:type:`mta_plugin_t` that knows how to turn a ``load_from`` string (path,
// name, …) into a filled :c:type:`mta_model_t` vtable.
//
// There are two ways to register a plugin:
//
// - in a shared library, export ``mta_plugin_init`` with the
//   :c:macro:`MTA_REGISTER_PLUGIN` macro, then call :c:func:`mta_load_plugin`;
// - in the same process as the engine (handy for tests and tutorials), call
//   :c:func:`mta_register_plugin` directly.
//
// This tutorial uses the in-process path and shows the shared-library pattern
// you would ship in a ``.so`` / ``.dylib`` / ``.dll``.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatomic.h>

// %%
//
// A minimal plugin
// ----------------
//
// The plugin below can load a single named model. Anything else returns
// :c:enumerator:`MTA_MODEL_NOT_SUPPORTED_ERROR` so other plugins get a chance
// when ``plugin_name`` is left as ``NULL``.

static mta_status_t stub_unload(void* model_data) {
    free(model_data);
    return MTA_SUCCESS;
}

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

static mta_status_t stub_supported_outputs(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create(
        "[{\"type\":\"metatomic_quantity\",\"name\":\"energy\",\"unit\":\"eV\","
        "\"gradients\":[],\"sample_kind\":\"system\"}]"
    );
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t stub_empty_list(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("[]");
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
        "plugin-demo does not implement execute_inner (WIP)",
        "stub_execute_inner",
        NULL,
        NULL
    );
    return MTA_INTERNAL_ERROR;
}

static mta_status_t demo_load_model(
    const char* load_from,
    const char* options_json,
    mta_model_t* model
) {
    (void)options_json;
    if (strcmp(load_from, "plugin-demo") != 0) {
        return MTA_MODEL_NOT_SUPPORTED_ERROR;
    }

    model->data = malloc(1);
    if (model->data == NULL) {
        mta_set_last_error("out of memory", "demo_load_model", NULL, NULL);
        return MTA_INTERNAL_ERROR;
    }
    model->unload = stub_unload;
    model->metadata = stub_metadata;
    model->capabilities = stub_capabilities;
    model->supported_outputs = stub_supported_outputs;
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
// The same plugin body can be compiled as a shared library. The macro below
// exports ``mta_plugin_init``, which :c:func:`mta_load_plugin` looks up after
// ``dlopen``. Keep the ABI version at :c:macro:`MTA_ABI_VERSION`.
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

// %%

static void print_last_error(const char* prefix) {
    const char* message = NULL;
    mta_last_error(&message, NULL, NULL);
    printf("%s: %s\n", prefix, message != NULL ? message : "(no message)");
}

int main(void) {
    mta_plugin_t plugin = {
        .abi_version = MTA_ABI_VERSION,
        .name = "tutorial-demo-plugin",
        .load_model = demo_load_model,
    };

    if (mta_register_plugin(plugin) != MTA_SUCCESS) {
        print_last_error("register_plugin failed");
        return EXIT_FAILURE;
    }
    printf("registered plugin '%s'\n", plugin.name);

    mta_model_t model = {0};
    if (mta_load_model("plugin-demo", "{}", "tutorial-demo-plugin", &model)
        != MTA_SUCCESS) {
        print_last_error("load_model failed");
        return EXIT_FAILURE;
    }
    printf("loaded model 'plugin-demo'\n");

    mta_string_t capabilities = NULL;
    if (model.capabilities(model.data, &capabilities) != MTA_SUCCESS) {
        print_last_error("capabilities failed");
        model.unload(model.data);
        return EXIT_FAILURE;
    }
    printf("capabilities: %s\n", mta_string_view(capabilities));
    mta_string_free(capabilities);

    /* Asking a plugin for a model it does not know must fail cleanly. */
    mta_model_t missing = {0};
    mta_status_t status =
        mta_load_model("not-a-real-model", "{}", "tutorial-demo-plugin", &missing);
    if (status == MTA_SUCCESS) {
        fprintf(stderr, "expected load of unknown model to fail\n");
        missing.unload(missing.data);
        model.unload(model.data);
        return EXIT_FAILURE;
    }
    print_last_error("unknown model");

#ifdef PLUGIN_DIR
    /* When this example is built with the CMake test suite, also try loading
       the shared library plugin that ships with the tests. */
    {
        const char* path = PLUGIN_DIR "/test-c-plugin.so";
        FILE* probe = fopen(path, "rb");
        if (probe != NULL) {
            fclose(probe);
            status = mta_load_plugin(path);
            if (status == MTA_SUCCESS) {
                printf("loaded shared library plugin from %s\n", path);
            } else {
                print_last_error("shared plugin");
                model.unload(model.data);
                return EXIT_FAILURE;
            }
        }
    }
#endif

    model.unload(model.data);
    return EXIT_SUCCESS;
}

// %%
//
// Expected output
// ---------------
//
// ::
//
//     registered plugin 'tutorial-demo-plugin'
//     loaded model 'plugin-demo'
//     capabilities: {"type": "metatomic_model_capabilities", ...}
//     unknown model: invalid parameter: failed to load model from
//     'not-a-real-model': plugin 'tutorial-demo-plugin' could not load the model
