// Defining a model (WIP)
// ======================
//
// This tutorial shows how to implement a metatomic model in C by filling an
// :c:type:`mta_model_t` vtable and registering it through a plugin.
//
// .. warning::
//
//     **Work in progress.** ``execute_inner`` is not implemented here. Returning
//     energy TensorMaps (and forces as position gradients) will land in a
//     follow-up tutorial. This file only covers the model vtable, metadata,
//     and in-process plugin registration.
//
// We use Einstein's solid as the running example: each atom is an independent
// harmonic oscillator around an equilibrium position,
//
// .. math::
//
//     E = \sum_i k \left(\vec{r}_i - \vec{r}_i^0\right)^2.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatomic.h>

// %%
//
// Model state
// -----------
//
// The model owns its parameters. Here that is a force constant; a later
// tutorial will also store equilibrium positions and evaluate the energy.

#define FORCE_CONSTANT 10.0 /* eV / Angstrom^2 */

typedef struct {
    double force_constant;
} HarmonicModel;

// %%
//
// Metadata callbacks
// ------------------
//
// Each callback writes a JSON document matching the schemas in
// :ref:`core-json-formats`. Prefer the typed forms
// (``"type": "metatomic_..."``) over older field names.

static mta_status_t harmonic_unload(void* model_data) {
    free(model_data);
    return MTA_SUCCESS;
}

static mta_status_t harmonic_metadata(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create(
        "{"
        "\"type\": \"metatomic_model_metadata\","
        "\"name\": \"harmonic-diamond\","
        "\"authors\": [\"metatomic C tutorials\"],"
        "\"description\": \"Einstein solid on a diamond carbon basis\","
        "\"references\": {"
        "  \"model\": [],"
        "  \"architecture\": [],"
        "  \"implementation\": []"
        "},"
        "\"extra\": {\"potential\": \"harmonic\"}"
        "}"
    );
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t harmonic_capabilities(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create(
        "{"
        "\"type\": \"metatomic_model_capabilities\","
        "\"outputs\": [{"
        "  \"type\": \"metatomic_quantity\","
        "  \"name\": \"energy\","
        "  \"unit\": \"eV\","
        "  \"gradients\": [\"positions\"],"
        "  \"sample_kind\": \"system\""
        "}],"
        "\"atomic_types\": [6],"
        "\"interaction_range\": 0.0,"
        "\"length_unit\": \"Angstrom\","
        "\"supported_devices\": [\"cpu\"],"
        "\"dtype\": \"float64\""
        "}"
    );
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t harmonic_supported_outputs(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create(
        "[{"
        "\"type\": \"metatomic_quantity\","
        "\"name\": \"energy\","
        "\"unit\": \"eV\","
        "\"gradients\": [\"positions\"],"
        "\"sample_kind\": \"system\""
        "}]"
    );
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t harmonic_empty_list(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("[]");
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t harmonic_execute_inner(
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
        "harmonic-diamond execute_inner is not implemented yet (WIP)",
        "harmonic_execute_inner",
        NULL,
        NULL
    );
    return MTA_INTERNAL_ERROR;
}

// %%
//
// Plugin registration
// -------------------
//
// Models are produced by plugins. For a single-file tutorial we register the
// plugin in-process with :c:func:`mta_register_plugin`. Shared-library plugins
// use the :c:macro:`MTA_REGISTER_PLUGIN` macro instead (see the next tutorial).

static mta_status_t harmonic_load_model(
    const char* load_from,
    const char* options_json,
    mta_model_t* model
) {
    (void)options_json;
    if (strcmp(load_from, "harmonic-diamond") != 0) {
        return MTA_MODEL_NOT_SUPPORTED_ERROR;
    }

    HarmonicModel* data = malloc(sizeof(HarmonicModel));
    if (data == NULL) {
        mta_set_last_error("out of memory", "harmonic_load_model", NULL, NULL);
        return MTA_INTERNAL_ERROR;
    }
    data->force_constant = FORCE_CONSTANT;

    model->data = data;
    model->unload = harmonic_unload;
    model->metadata = harmonic_metadata;
    model->capabilities = harmonic_capabilities;
    model->supported_outputs = harmonic_supported_outputs;
    model->requested_pair_lists = harmonic_empty_list;
    model->requested_inputs = harmonic_empty_list;
    model->execute_inner = harmonic_execute_inner;
    return MTA_SUCCESS;
}

static int fail(mta_model_t* model, const char* what) {
    const char* message = NULL;
    mta_last_error(&message, NULL, NULL);
    fprintf(stderr, "%s: %s\n", what, message != NULL ? message : "(no message)");
    if (model != NULL && model->unload != NULL && model->data != NULL) {
        model->unload(model->data);
    }
    return EXIT_FAILURE;
}

// %%

int main(void) {
    static mta_plugin_t PLUGIN = {
        .abi_version = MTA_ABI_VERSION,
        .name = "tutorial-harmonic-plugin",
        .load_model = harmonic_load_model,
    };
    if (mta_register_plugin(PLUGIN) != MTA_SUCCESS) {
        return fail(NULL, "failed to register plugin");
    }

    mta_model_t model = {0};
    if (mta_load_model("harmonic-diamond", "{}", "tutorial-harmonic-plugin", &model)
        != MTA_SUCCESS) {
        return fail(NULL, "failed to load model");
    }

    mta_string_t metadata = NULL;
    if (model.metadata(model.data, &metadata) != MTA_SUCCESS) {
        return fail(&model, "failed to get metadata");
    }
    mta_string_t printed = NULL;
    if (mta_format_metadata(mta_string_view(metadata), &printed) != MTA_SUCCESS) {
        mta_string_free(metadata);
        return fail(&model, "failed to format metadata");
    }
    printf("%s\n", mta_string_view(printed));
    mta_string_free(metadata);
    mta_string_free(printed);

    printf("execute_inner is WIP; energy TensorMaps will be added later\n");

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
//     This is the harmonic-diamond model
//     ==================================
//
//     Einstein solid on a diamond carbon basis
//
//     Model authors
//     -------------
//
//     - metatomic C tutorials
//
//     execute_inner is WIP; energy TensorMaps will be added later
