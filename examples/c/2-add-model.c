// Defining a Lennard-Jones model
// ==============================
//
// This tutorial shows how to implement a metatomic model in C by filling an
// :c:type:`mta_model_t` vtable and registering it through a plugin.
//
// The running example is a **shifted Lennard-Jones** pair potential. The energy
// is a sum over neighbor pairs inside a spherical cutoff
//
// .. math::
//
//     E = \sum_{i<j}^{r_{ij} < r_c} \left[
//         4 \epsilon \left(
//             \left(\frac{\sigma}{r_{ij}}\right)^{12}
//             - \left(\frac{\sigma}{r_{ij}}\right)^{6}
//         \right) - E_{\mathrm{shift}}
//     \right],
//
// with :math:`E_{\mathrm{shift}}` chosen so the pair term is exactly zero at
// :math:`r_c`. Each pair is counted once (a half neighbor list) and half of
// the pair energy is assigned to each atom.
//
// .. note::
//
//     ``execute_inner`` sums shifted pair energies from the engine-provided
//     pair list. Position gradients (forces) are still TODO.
//     :c:func:`mta_execute_model` on this branch is ``todo!()`` (see
//     https://github.com/metatensor/metatomic/pull/306), so this tutorial
//     calls ``execute_inner`` directly.

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatomic.h>
#include "simple_array.h"

// %%
//
// Model state
// -----------
//
// The model owns its LJ parameters. ``cutoff``, ``sigma``, and ``epsilon``
// match a typical argon-scale test setup (Å and eV). ``shift`` is
// :math:`4\epsilon[(\sigma/r_c)^{12} - (\sigma/r_c)^{6}]` so the energy
// goes to zero at the cutoff.

#define LJ_CUTOFF 3.4    /* Angstrom */
#define LJ_SIGMA 1.5     /* Angstrom */
#define LJ_EPSILON 23.0  /* eV */
#define LJ_ATOMIC_TYPE 12

typedef struct {
    double cutoff;
    double sigma;
    double epsilon;
    double shift;
    int32_t atomic_type;
} LennardJonesModel;

static double lj_shift(double cutoff, double sigma, double epsilon) {
    double sigma_rc = sigma / cutoff;
    double sigma_rc_6 = sigma_rc * sigma_rc * sigma_rc;
    sigma_rc_6 *= sigma_rc_6;
    return 4.0 * epsilon * (sigma_rc_6 * sigma_rc_6 - sigma_rc_6);
}

// Pair-list cutoffs in JSON use the IEEE-754 bit pattern as a hex string
// (see :ref:`core-json-pair-options`), so the engine sees the exact ``double``.
static void format_f64_hex(double value, char* buf, size_t n) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    snprintf(buf, n, "0x%" PRIx64, bits);
}

static void lj_format_pair_options(const LennardJonesModel* lj, char* buf, size_t n) {
    char cutoff_hex[32];
    format_f64_hex(lj->cutoff, cutoff_hex, sizeof(cutoff_hex));
    snprintf(
        buf,
        n,
        "{"
        "\"type\": \"metatomic_pair_options\","
        "\"cutoff\": \"%s\","
        "\"full_list\": false,"
        "\"strict\": true,"
        "\"requestors\": [\"lennard-jones\"]"
        "}",
        cutoff_hex
    );
}

static double lj_pair_term(double r2, const LennardJonesModel* lj) {
    if (r2 <= 0.0 || r2 >= lj->cutoff * lj->cutoff) {
        return 0.0;
    }
    double inv2 = (lj->sigma * lj->sigma) / r2;
    double inv6 = inv2 * inv2 * inv2;
    double inv12 = inv6 * inv6;
    return 4.0 * lj->epsilon * (inv12 - inv6) - lj->shift;
}

// %%
//
// Metadata callbacks
// ------------------
//
// Each callback writes a JSON document matching the schemas in
// :ref:`core-json-formats`. Prefer the typed forms
// (``"type": "metatomic_..."``) over older field names.

static mta_status_t lj_unload(void* model_data) {
    free(model_data);
    return MTA_SUCCESS;
}

static mta_status_t lj_metadata(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create(
        "{"
        "\"type\": \"metatomic_model_metadata\","
        "\"name\": \"lennard-jones\","
        "\"authors\": [\"metatomic C tutorials\"],"
        "\"description\": \"Minimal shifted Lennard-Jones potential for engine integration tests\","
        "\"references\": {"
        "  \"model\": [],"
        "  \"architecture\": [],"
        "  \"implementation\": []"
        "},"
        "\"extra\": {\"potential\": \"shifted-lennard-jones\"}"
        "}"
    );
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t lj_capabilities(const void* model_data, mta_string_t* out) {
    const LennardJonesModel* lj = (const LennardJonesModel*)model_data;
    char json[1024];
    snprintf(
        json,
        sizeof(json),
        "{"
        "\"type\": \"metatomic_model_capabilities\","
        "\"outputs\": [{"
        "  \"type\": \"metatomic_quantity\","
        "  \"name\": \"energy\","
        "  \"unit\": \"eV\","
        "  \"gradients\": [\"positions\"],"
        "  \"sample_kind\": \"system\""
        "}],"
        "\"atomic_types\": [%d],"
        "\"interaction_range\": %.17g,"
        "\"length_unit\": \"Angstrom\","
        "\"supported_devices\": [\"cpu\"],"
        "\"dtype\": \"float64\""
        "}",
        lj->atomic_type,
        lj->cutoff
    );
    *out = mta_string_create(json);
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t lj_requested_pair_lists(const void* model_data, mta_string_t* out) {
    const LennardJonesModel* lj = (const LennardJonesModel*)model_data;
    char object[512];
    char json[520];
    lj_format_pair_options(lj, object, sizeof(object));
    snprintf(json, sizeof(json), "[%s]", object);
    *out = mta_string_create(json);
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t lj_requested_inputs(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("[]");
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

// %%
//
// ``execute_inner``
// -----------------
//
// The engine attaches a half neighbor list with :c:func:`mta_system_add_pairs`.
// The model reads pair displacement vectors from that block, sums the shifted
// Lennard-Jones pair terms, and writes a system-level energy TensorMap.
// A missing neighbor list is an error (not a silent zero energy).
// ``simple_array.h`` is a C stand-in for ``metatensor::SimpleDataArray``.

static mts_tensormap_t* scalar_tensormap(double value) {
    double* data = malloc(sizeof(double));
    if (data == NULL) {
        return NULL;
    }
    *data = value;

    const mts_labels_t* samples = labels_single_zero("system");
    const mts_labels_t* properties = labels_single_zero("energy");
    const mts_labels_t* keys = labels_single_zero("_");
    if (samples == NULL || properties == NULL || keys == NULL) {
        mts_labels_free(samples);
        mts_labels_free(properties);
        mts_labels_free(keys);
        free(data);
        return NULL;
    }

    mts_array_t values = tutorial_array_own(data, 1, 1, kDLFloat, 64);
    mts_block_t* block = mts_block(values, samples, NULL, 0, properties);
    if (block == NULL) {
        if (values.destroy != NULL) {
            values.destroy(values.ptr);
        }
        mts_labels_free(samples);
        mts_labels_free(properties);
        mts_labels_free(keys);
        return NULL;
    }

    mts_block_t* blocks[] = {block};
    mts_tensormap_t* tensor = mts_tensormap(keys, blocks, 1);
    mts_labels_free(samples);
    mts_labels_free(properties);
    mts_labels_free(keys);
    if (tensor == NULL) {
        mts_block_free(block);
    }
    return tensor;
}

static mta_status_t lj_fail_mts(const char* what) {
    const char* message = NULL;
    mts_last_error(&message, NULL, NULL);
    mta_set_last_error(
        message != NULL ? message : what,
        "lj_execute_inner",
        NULL,
        NULL
    );
    return MTA_METATENSOR_ERROR;
}

static mta_status_t lj_energy_of_system(
    const LennardJonesModel* lj,
    const mta_system_t* system,
    double* energy
) {
    char options[512];
    lj_format_pair_options(lj, options, sizeof(options));

    const mts_block_t* pairs = NULL;
    mta_status_t status = mta_system_get_pairs(system, options, &pairs);
    if (status != MTA_SUCCESS) {
        /* Missing neighbor lists are an error, not silent zeros. */
        return status;
    }
    if (pairs == NULL) {
        mta_set_last_error(
            "pair list pointer is NULL after a successful get_pairs",
            "lj_execute_inner",
            NULL,
            NULL
        );
        return MTA_INTERNAL_ERROR;
    }
    *energy = 0.0;

    mts_array_t array;
    memset(&array, 0, sizeof(array));
    if (mts_block_data((mts_block_t*)pairs, &array) != MTS_SUCCESS) {
        return lj_fail_mts("failed to read pair list values");
    }
    if (array.shape == NULL || array.as_dlpack == NULL) {
        return lj_fail_mts("pair list array is missing shape/as_dlpack");
    }

    const uintptr_t* shape = NULL;
    uintptr_t ndim = 0;
    if (array.shape(array.ptr, &shape, &ndim) != MTS_SUCCESS || ndim < 1) {
        return lj_fail_mts("failed to get pair list shape");
    }
    if (ndim != 3 || shape[1] != 3 || shape[2] != 1) {
        mta_set_last_error(
            "pair list values must have shape [n_pairs, 3, 1] (xyz displacement, one property)",
            "lj_execute_inner",
            NULL,
            NULL
        );
        return MTA_INVALID_PARAMETER_ERROR;
    }
    uintptr_t n_pairs = shape[0];

    DLManagedTensorVersioned* view = NULL;
    DLDevice cpu = {.device_type = kDLCPU, .device_id = 0};
    DLPackVersion version = {
        .major = DLPACK_MAJOR_VERSION,
        .minor = DLPACK_MINOR_VERSION
    };
    if (array.as_dlpack(array.ptr, &view, cpu, NULL, version) != MTS_SUCCESS
        || view == NULL) {
        return lj_fail_mts("failed to export pair displacements");
    }
    if (view->dl_tensor.dtype.code != kDLFloat
        || view->dl_tensor.dtype.bits != 64) {
        view->deleter(view);
        mta_set_last_error(
            "pair displacements must be float64",
            "lj_execute_inner",
            NULL,
            NULL
        );
        return MTA_DLPACK_ERROR;
    }

    double* vec = (double*)((char*)view->dl_tensor.data + view->dl_tensor.byte_offset);
    for (uintptr_t i = 0; i < n_pairs; i++) {
        double dx = vec[3 * i + 0];
        double dy = vec[3 * i + 1];
        double dz = vec[3 * i + 2];
        *energy += lj_pair_term(dx * dx + dy * dy + dz * dz, lj);
    }
    view->deleter(view);
    return MTA_SUCCESS;
}

static mta_status_t lj_execute_inner(
    void* model_data,
    const mta_system_t* const* systems,
    uintptr_t systems_count,
    const mts_labels_t* selected_atoms,
    const char* requested_outputs_json,
    mts_tensormap_t** outputs,
    uintptr_t outputs_count
) {
    (void)selected_atoms;
    (void)requested_outputs_json;

    if (model_data == NULL) {
        mta_set_last_error("model_data is NULL", "lj_execute_inner", NULL, NULL);
        return MTA_INVALID_PARAMETER_ERROR;
    }
    if (systems_count > 0 && systems == NULL) {
        mta_set_last_error(
            "systems is NULL but systems_count is not 0",
            "lj_execute_inner",
            NULL,
            NULL
        );
        return MTA_INVALID_PARAMETER_ERROR;
    }
    if (outputs_count > 0 && outputs == NULL) {
        mta_set_last_error(
            "outputs is NULL but outputs_count is not 0",
            "lj_execute_inner",
            NULL,
            NULL
        );
        return MTA_INVALID_PARAMETER_ERROR;
    }

    const LennardJonesModel* lj = (const LennardJonesModel*)model_data;

    double energy = 0.0;
    for (uintptr_t s = 0; s < systems_count; s++) {
        double part = 0.0;
        mta_status_t status = lj_energy_of_system(lj, systems[s], &part);
        if (status != MTA_SUCCESS) {
            return status;
        }
        energy += part;
    }

    for (uintptr_t i = 0; i < outputs_count; i++) {
        outputs[i] = scalar_tensormap(energy);
        if (outputs[i] == NULL) {
            for (uintptr_t j = 0; j < i; j++) {
                mts_tensormap_free(outputs[j]);
                outputs[j] = NULL;
            }
            return lj_fail_mts("failed to build energy TensorMap");
        }
    }
    return MTA_SUCCESS;
}

// %%
//
// Plugin registration
// -------------------
//
// Models are produced by plugins. For a single-file tutorial we register the
// plugin in-process with :c:func:`mta_register_plugin`. Shared-library plugins
// use the :c:macro:`MTA_REGISTER_PLUGIN` macro instead (see the next tutorial).

static mta_status_t lj_load_model(
    const char* load_from,
    const char* options_json,
    mta_model_t* model
) {
    (void)options_json;
    if (strcmp(load_from, "lennard-jones") != 0) {
        return MTA_MODEL_NOT_SUPPORTED_ERROR;
    }

    LennardJonesModel* data = malloc(sizeof(LennardJonesModel));
    if (data == NULL) {
        mta_set_last_error("out of memory", "lj_load_model", NULL, NULL);
        return MTA_INTERNAL_ERROR;
    }
    data->cutoff = LJ_CUTOFF;
    data->sigma = LJ_SIGMA;
    data->epsilon = LJ_EPSILON;
    data->shift = lj_shift(LJ_CUTOFF, LJ_SIGMA, LJ_EPSILON);
    data->atomic_type = LJ_ATOMIC_TYPE;

    model->data = data;
    model->unload = lj_unload;
    model->metadata = lj_metadata;
    model->capabilities = lj_capabilities;
    model->requested_pair_lists = lj_requested_pair_lists;
    model->requested_inputs = lj_requested_inputs;
    model->execute_inner = lj_execute_inner;
    return MTA_SUCCESS;
}

static int die(
    mta_model_t* model,
    mta_system_t* system,
    mts_tensormap_t* tensor,
    const char* what
) {
    fprintf(stderr, "assertion failed: %s\n", what);
    mts_tensormap_free(tensor);
    if (system != NULL) {
        mta_system_free(system);
    }
    if (model != NULL && model->unload != NULL && model->data != NULL) {
        model->unload(model->data);
    }
    return EXIT_FAILURE;
}

static int fail(mta_model_t* model, mta_system_t* system, const char* what) {
    const char* message = NULL;
    mta_last_error(&message, NULL, NULL);
    fprintf(stderr, "%s: %s\n", what, message != NULL ? message : "(no message)");
    if (system != NULL) {
        mta_system_free(system);
    }
    if (model != NULL && model->unload != NULL && model->data != NULL) {
        model->unload(model->data);
    }
    return EXIT_FAILURE;
}

static mts_block_t* make_displacement_pair_block(double dx, double dy, double dz) {
    const char* sample_names[] = {
        "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"
    };
    int32_t sample_values[] = {0, 1, 0, 0, 0};
    const char* xyz_name[] = {"xyz"};
    int32_t xyz_values[] = {0, 1, 2};
    const char* distance_name[] = {"distance"};
    int32_t distance_values[] = {0};

    const mts_labels_t* samples = tutorial_labels(sample_names, 5, sample_values, 1);
    const mts_labels_t* xyz = tutorial_labels(xyz_name, 1, xyz_values, 3);
    const mts_labels_t* distance = tutorial_labels(distance_name, 1, distance_values, 1);
    if (samples == NULL || xyz == NULL || distance == NULL) {
        mts_labels_free(samples);
        mts_labels_free(xyz);
        mts_labels_free(distance);
        return NULL;
    }

    double* data = malloc(3 * sizeof(double));
    if (data == NULL) {
        mts_labels_free(samples);
        mts_labels_free(xyz);
        mts_labels_free(distance);
        return NULL;
    }
    data[0] = dx;
    data[1] = dy;
    data[2] = dz;
    uintptr_t shape[] = {1, 3, 1};
    mts_array_t values = tutorial_array_nd(data, shape, 3, kDLFloat, 64);
    const mts_labels_t* components[] = {xyz};
    mts_block_t* block = mts_block(values, samples, components, 1, distance);
    if (block == NULL && values.destroy != NULL) {
        values.destroy(values.ptr);
    }
    mts_labels_free(samples);
    mts_labels_free(xyz);
    mts_labels_free(distance);
    return block;
}

static int read_scalar_energy(mts_tensormap_t* tensor, double* out) {
    mts_block_t* block = NULL;
    if (mts_tensormap_block_by_id(tensor, &block, 0) != MTS_SUCCESS || block == NULL) {
        return -1;
    }
    mts_array_t array;
    memset(&array, 0, sizeof(array));
    if (mts_block_data(block, &array) != MTS_SUCCESS || array.as_dlpack == NULL) {
        return -1;
    }
    DLManagedTensorVersioned* view = NULL;
    DLDevice cpu = {.device_type = kDLCPU, .device_id = 0};
    DLPackVersion version = {
        .major = DLPACK_MAJOR_VERSION,
        .minor = DLPACK_MINOR_VERSION
    };
    if (array.as_dlpack(array.ptr, &view, cpu, NULL, version) != MTS_SUCCESS
        || view == NULL) {
        return -1;
    }
    if (view->dl_tensor.dtype.code != kDLFloat || view->dl_tensor.dtype.bits != 64) {
        view->deleter(view);
        return -1;
    }
    *out = *(double*)((char*)view->dl_tensor.data + view->dl_tensor.byte_offset);
    view->deleter(view);
    return 0;
}

static int run_energy(
    mta_model_t* model,
    const mta_system_t* system,
    uintptr_t n_systems,
    double* energy_out
) {
    static const char* requested =
        "[{\"type\":\"metatomic_quantity\",\"name\":\"energy\","
        "\"unit\":\"eV\",\"gradients\":[],\"sample_kind\":\"system\"}]";
    mts_tensormap_t* energy = NULL;
    const mta_system_t* systems[1] = {system};
    mta_status_t status = model->execute_inner(
        model->data,
        n_systems == 0 ? NULL : systems,
        n_systems,
        NULL,
        requested,
        &energy,
        1
    );
    if (status != MTA_SUCCESS || energy == NULL) {
        mts_tensormap_free(energy);
        return -1;
    }
    if (read_scalar_energy(energy, energy_out) != 0) {
        mts_tensormap_free(energy);
        return -2;
    }
    mts_tensormap_free(energy);
    return 0;
}

// %%

int main(void) {
    static mta_plugin_t PLUGIN = {
        .abi_version = MTA_ABI_VERSION,
        .name = "tutorial-lj-plugin",
        .load_model = lj_load_model,
    };
    if (mta_register_plugin(PLUGIN) != MTA_SUCCESS) {
        return fail(NULL, NULL, "failed to register plugin");
    }

    mta_model_t model = {0};
    if (mta_load_model("lennard-jones", "{}", "tutorial-lj-plugin", &model)
        != MTA_SUCCESS) {
        return fail(NULL, NULL, "failed to load model");
    }
    if (model.data == NULL || model.execute_inner == NULL
        || model.requested_pair_lists == NULL || model.unload == NULL) {
        return die(&model, NULL, NULL, "loaded model is missing vtable entries");
    }

    mta_string_t metadata = NULL;
    if (model.metadata(model.data, &metadata) != MTA_SUCCESS) {
        return fail(&model, NULL, "failed to get metadata");
    }
    mta_string_t printed = NULL;
    if (mta_format_metadata(mta_string_view(metadata), &printed) != MTA_SUCCESS) {
        mta_string_free(metadata);
        return fail(&model, NULL, "failed to format metadata");
    }
    printf("%s\n", mta_string_view(printed));
    if (strstr(mta_string_view(printed), "lennard-jones") == NULL) {
        fprintf(stderr, "formatted metadata missing model name\n");
        mta_string_free(metadata);
        mta_string_free(printed);
        model.unload(model.data);
        return EXIT_FAILURE;
    }
    mta_string_free(metadata);
    mta_string_free(printed);

    mta_string_t pairs = NULL;
    if (model.requested_pair_lists(model.data, &pairs) != MTA_SUCCESS) {
        return fail(&model, NULL, "failed to get requested pair lists");
    }
    const char* pairs_json = mta_string_view(pairs);
    printf("requested pair lists: %s\n", pairs_json);
    if (strstr(pairs_json, "0x400b333333333333") == NULL
        || strstr(pairs_json, "\"full_list\": false") == NULL
        || strstr(pairs_json, "\"strict\": true") == NULL) {
        fprintf(stderr, "unexpected pair-list request: %s\n", pairs_json);
        mta_string_free(pairs);
        model.unload(model.data);
        return EXIT_FAILURE;
    }
    mta_string_free(pairs);

    /* A plugin signals "not my model" with MTA_MODEL_NOT_SUPPORTED_ERROR.
       mta_load_model with a named plugin wraps that as
       MTA_INVALID_PARAMETER_ERROR so the engine can fail the load. */
    mta_model_t scratch = {0};
    if (lj_load_model("einstein-solid", "{}", &scratch)
        != MTA_MODEL_NOT_SUPPORTED_ERROR) {
        return die(
            &model, NULL, NULL,
            "plugin load_model must return MTA_MODEL_NOT_SUPPORTED_ERROR for unknown names"
        );
    }
    printf(
        "plugin load_model status for unknown name: %d\n",
        (int)MTA_MODEL_NOT_SUPPORTED_ERROR
    );

    mta_model_t rejected = {0};
    mta_status_t unsupported = mta_load_model(
        "einstein-solid", "{}", "tutorial-lj-plugin", &rejected
    );
    if (unsupported == MTA_SUCCESS) {
        rejected.unload(rejected.data);
        return die(&model, NULL, NULL, "mta_load_model should reject unknown models");
    }
    if (unsupported != MTA_INVALID_PARAMETER_ERROR) {
        return die(
            &model, NULL, NULL,
            "named-plugin mta_load_model wraps MTA_MODEL_NOT_SUPPORTED_ERROR as MTA_INVALID_PARAMETER_ERROR"
        );
    }
    {
        const char* message = NULL;
        mta_last_error(&message, NULL, NULL);
        if (message == NULL || strstr(message, "tutorial-lj-plugin") == NULL) {
            return die(
                &model, NULL, NULL,
                "mta_load_model error should name the plugin that could not load the model"
            );
        }
    }
    printf("mta_load_model status for unknown name: %d\n", (int)unsupported);

    if (model.execute_inner(NULL, NULL, 0, NULL, "[]", NULL, 0)
        != MTA_INVALID_PARAMETER_ERROR) {
        return die(
            &model, NULL, NULL,
            "execute_inner should reject a NULL model pointer"
        );
    }
    mts_tensormap_t* dummy = NULL;
    if (model.execute_inner(model.data, NULL, 1, NULL, "[]", &dummy, 1)
        != MTA_INVALID_PARAMETER_ERROR) {
        mts_tensormap_free(dummy);
        return die(
            &model, NULL, NULL,
            "execute_inner should reject systems=NULL when n_systems > 0"
        );
    }
    if (model.execute_inner(model.data, NULL, 0, NULL, "[]", NULL, 1)
        != MTA_INVALID_PARAMETER_ERROR) {
        return die(
            &model, NULL, NULL,
            "execute_inner should reject outputs=NULL when n_outputs > 0"
        );
    }
    printf("execute_inner rejects invalid arguments\n");

    int32_t types_data[] = {LJ_ATOMIC_TYPE, LJ_ATOMIC_TYPE};
    double positions_data[] = {
        0.0, 0.0, 0.0,
        LJ_SIGMA, 0.0, 0.0,
    };
    double cell_data[] = {
        20.0, 0.0, 0.0,
        0.0, 20.0, 0.0,
        0.0, 0.0, 20.0,
    };
    bool pbc_data[] = {true, true, true};

    mta_system_t* system = NULL;
    mta_status_t status = mta_system_create(
        "Angstrom",
        tutorial_dlpack_view(
            types_data, 1, (int64_t[]){2},
            (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}
        ),
        tutorial_dlpack_view(
            positions_data, 2, (int64_t[]){2, 3},
            (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}
        ),
        tutorial_dlpack_view(
            cell_data, 2, (int64_t[]){3, 3},
            (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}
        ),
        tutorial_dlpack_view(
            pbc_data, 1, (int64_t[]){3},
            (DLDataType){.code = kDLBool, .bits = 8, .lanes = 1}
        ),
        &system
    );
    if (status != MTA_SUCCESS) {
        return fail(&model, system, "failed to create dimer system");
    }

    uintptr_t n_atoms = 0;
    if (mta_system_size(system, &n_atoms) != MTA_SUCCESS || n_atoms != 2) {
        return die(&model, system, NULL, "dimer should contain 2 atoms");
    }

    double unused = 1.0;
    if (run_energy(&model, system, 1, &unused) == 0) {
        return die(
            &model, system, NULL,
            "execute_inner should fail when the requested pair list is missing"
        );
    }
    {
        const char* message = NULL;
        mta_last_error(&message, NULL, NULL);
        if (message == NULL || strstr(message, "no pair list") == NULL) {
            return die(
                &model, system, NULL,
                "missing neighbor list error should mention 'no pair list'"
            );
        }
    }
    printf("missing pair list is rejected\n");

    mts_block_t* pair_block = make_displacement_pair_block(LJ_SIGMA, 0.0, 0.0);
    if (pair_block == NULL) {
        return fail(&model, system, "failed to build pair list");
    }
    char pair_options[512];
    lj_format_pair_options((const LennardJonesModel*)model.data, pair_options, sizeof(pair_options));
    if (mta_system_add_pairs(system, pair_options, pair_block) != MTA_SUCCESS) {
        mts_block_free(pair_block);
        return fail(&model, system, "failed to add pair list");
    }
    {
        const mts_block_t* got_pairs = NULL;
        if (mta_system_get_pairs(system, pair_options, &got_pairs) != MTA_SUCCESS
            || got_pairs == NULL) {
            return fail(&model, system, "pair list should be retrievable after add_pairs");
        }
    }

    double got = 0.0;
    if (run_energy(&model, system, 1, &got) != 0) {
        return fail(&model, system, "execute_inner failed");
    }
    double expected = -((const LennardJonesModel*)model.data)->shift;
    printf("energy at r=sigma: %.12f eV\n", got);
    double err = got - expected;
    if (err < 0.0) {
        err = -err;
    }
    if (err > 1e-10) {
        fprintf(stderr, "expected %.12f eV, got %.12f eV\n", expected, got);
        return die(&model, system, NULL, "shifted LJ energy at r=sigma should be -E_shift");
    }

    /* Do not call add_pairs twice with the same options. That is an error, and
       on current metatomic-core a failed add_pairs drops the C-owned system so
       mta_system_free heap-corrupts (Windows MSVC STATUS_HEAP_CORRUPTION). */

    mta_system_free(system);
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
//     This is the lennard-jones model
//     ===============================
//
//     Minimal shifted Lennard-Jones potential for engine integration tests
//
//     Model authors
//     -------------
//
//     - metatomic C tutorials
//
//     requested pair lists: [{"type": "metatomic_pair_options","cutoff": "0x400b333333333333","full_list": false,"strict": true,"requestors": ["lennard-jones"]}]
//     plugin load_model status for unknown name: 6
//     mta_load_model status for unknown name: 1
//     execute_inner rejects invalid arguments
//     missing pair list is rejected
//     energy at r=sigma: 0.673360663351 eV
