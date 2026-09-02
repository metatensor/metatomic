// Defining a harmonic model
// =========================
//
// This tutorial shows how to implement a metatomic model in C by filling an
// :c:type:`mta_model_t` vtable, registering it through a plugin, and evaluating
// energy and forces for a small system.
//
// We use Einstein's solid: each atom is an independent harmonic oscillator
// around an equilibrium position,
//
// .. math::
//
//     E = \sum_i k \left(\vec{r}_i - \vec{r}_i^0\right)^2,
//
// so the force on atom :math:`i` is :math:`\vec{F}_i = -2k(\vec{r}_i -
// \vec{r}_i^0)`. In metatomic, forces are returned as the negative of the
// ``"positions"`` energy gradient (see :ref:`energy-quantity-gradients`).

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatomic.h>
#include <metatensor/dlpack/dlpack.h>

// Dense CPU arrays for building metatensor Labels / TensorMaps
// ------------------------------------------------------------
//
// The helpers below implement a small ``mts_array_t`` backend so this
// example can construct energy TensorMaps (and position gradients)
// without depending on the C++ ``SimpleDataArray`` helpers.

/* Minimal CPU dense arrays for metatensor TensorMaps in C tutorials.
 *
 * Only the callbacks needed to build Labels / TensorBlocks and read values
 * back are fully implemented. Advanced ops return an error.
 */

typedef struct {
    uintptr_t* shape;
    uintptr_t shape_count;
    uint8_t* data;
    uintptr_t n_bytes;
    DLDataType dtype;
} DenseArray;

typedef struct {
    DenseArray* array;
    int64_t* shape;
    int64_t* strides;
} DenseDLPackContext;

static mts_data_origin_t dense_array_origin_id(void) {
    static mts_data_origin_t origin = 0;
    if (origin == 0) {
        mts_register_data_origin("metatomic::examples::DenseArray", &origin);
    }
    return origin;
}

static void dense_array_free(DenseArray* array) {
    if (array == NULL) {
        return;
    }
    free(array->shape);
    free(array->data);
    free(array);
}

static DenseArray* dense_array_new(
    const uintptr_t* shape,
    uintptr_t shape_count,
    DLDataType dtype,
    const void* values,
    uintptr_t n_bytes
) {
    DenseArray* array = calloc(1, sizeof(DenseArray));
    if (array == NULL) {
        return NULL;
    }

    array->shape_count = shape_count;
    array->dtype = dtype;
    array->n_bytes = n_bytes;

    if (shape_count > 0) {
        array->shape = malloc(shape_count * sizeof(uintptr_t));
        if (array->shape == NULL) {
            free(array);
            return NULL;
        }
        memcpy(array->shape, shape, shape_count * sizeof(uintptr_t));
    }

    array->data = malloc(n_bytes == 0 ? 1 : n_bytes);
    if (array->data == NULL) {
        free(array->shape);
        free(array);
        return NULL;
    }
    if (values != NULL && n_bytes > 0) {
        memcpy(array->data, values, n_bytes);
    } else if (n_bytes > 0) {
        memset(array->data, 0, n_bytes);
    }
    return array;
}

static void dense_dlpack_deleter(DLManagedTensorVersioned* self) {
    if (self == NULL) {
        return;
    }
    DenseDLPackContext* ctx = (DenseDLPackContext*)self->manager_ctx;
    if (ctx != NULL) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
    }
    free(self);
}

static void dense_array_destroy(void* ptr) {
    dense_array_free((DenseArray*)ptr);
}

static mts_status_t dense_array_origin(const void* ptr, mts_data_origin_t* origin) {
    (void)ptr;
    *origin = dense_array_origin_id();
    return MTS_SUCCESS;
}

static mts_status_t dense_array_device(const void* ptr, DLDevice* device) {
    (void)ptr;
    device->device_type = kDLCPU;
    device->device_id = 0;
    return MTS_SUCCESS;
}

static mts_status_t dense_array_dtype(const void* ptr, DLDataType* dtype) {
    *dtype = ((const DenseArray*)ptr)->dtype;
    return MTS_SUCCESS;
}

static mts_status_t dense_array_shape(
    const void* ptr,
    const uintptr_t** shape,
    uintptr_t* shape_count
) {
    const DenseArray* array = (const DenseArray*)ptr;
    *shape = array->shape;
    *shape_count = array->shape_count;
    return MTS_SUCCESS;
}

static mts_status_t dense_array_as_dlpack(
    void* ptr,
    DLManagedTensorVersioned** out,
    DLDevice device,
    const int64_t* stream,
    DLPackVersion max_version
) {
    DenseArray* array = (DenseArray*)ptr;

    if (device.device_type != kDLCPU) {
        mts_set_last_error("DenseArray only supports CPU", "dense_array", NULL, NULL);
        return MTS_CALLBACK_ERROR;
    }
    if (stream != NULL) {
        mts_set_last_error("DenseArray does not support streams", "dense_array", NULL, NULL);
        return MTS_CALLBACK_ERROR;
    }
    if (max_version.major != DLPACK_MAJOR_VERSION) {
        mts_set_last_error("unsupported DLPack version", "dense_array", NULL, NULL);
        return MTS_CALLBACK_ERROR;
    }

    DenseDLPackContext* ctx = calloc(1, sizeof(DenseDLPackContext));
    DLManagedTensorVersioned* tensor = calloc(1, sizeof(DLManagedTensorVersioned));
    if (ctx == NULL || tensor == NULL) {
        free(ctx);
        free(tensor);
        mts_set_last_error("out of memory", "dense_array", NULL, NULL);
        return MTS_CALLBACK_ERROR;
    }

    if (array->shape_count > 0) {
        ctx->shape = malloc(array->shape_count * sizeof(int64_t));
        ctx->strides = malloc(array->shape_count * sizeof(int64_t));
        if (ctx->shape == NULL || ctx->strides == NULL) {
            free(ctx->shape);
            free(ctx->strides);
            free(ctx);
            free(tensor);
            mts_set_last_error("out of memory", "dense_array", NULL, NULL);
            return MTS_CALLBACK_ERROR;
        }
        int64_t stride = 1;
        for (uintptr_t i = array->shape_count; i-- > 0;) {
            ctx->shape[i] = (int64_t)array->shape[i];
            ctx->strides[i] = stride;
            stride *= ctx->shape[i];
        }
    }

    ctx->array = array;
    tensor->version.major = DLPACK_MAJOR_VERSION;
    tensor->version.minor = DLPACK_MINOR_VERSION;
    tensor->manager_ctx = ctx;
    tensor->deleter = dense_dlpack_deleter;
    tensor->dl_tensor.data = array->data;
    tensor->dl_tensor.device = device;
    tensor->dl_tensor.ndim = (int32_t)array->shape_count;
    tensor->dl_tensor.dtype = array->dtype;
    tensor->dl_tensor.shape = ctx->shape;
    tensor->dl_tensor.strides = ctx->strides;
    tensor->dl_tensor.byte_offset = 0;

    *out = tensor;
    return MTS_SUCCESS;
}

static mts_status_t dense_array_unsupported(void) {
    mts_set_last_error(
        "operation not implemented for tutorial DenseArray",
        "dense_array",
        NULL,
        NULL
    );
    return MTS_CALLBACK_ERROR;
}

static mts_status_t dense_array_reshape(void* array, const uintptr_t* shape, uintptr_t shape_count) {
    (void)array;
    (void)shape;
    (void)shape_count;
    return dense_array_unsupported();
}

static mts_status_t dense_array_swap_axes(void* array, uintptr_t axis_1, uintptr_t axis_2) {
    (void)array;
    (void)axis_1;
    (void)axis_2;
    return dense_array_unsupported();
}

static mts_status_t dense_array_create(
    const void* array,
    const uintptr_t* shape,
    uintptr_t shape_count,
    struct mts_array_t fill_value,
    struct mts_array_t* new_array
) {
    (void)array;
    (void)shape;
    (void)shape_count;
    (void)fill_value;
    (void)new_array;
    return dense_array_unsupported();
}

static mts_status_t dense_array_copy(const void* array, DLDevice device, struct mts_array_t* new_array) {
    (void)array;
    (void)device;
    (void)new_array;
    return dense_array_unsupported();
}

static mts_status_t dense_array_from_dlpack(
    const void* array,
    DLManagedTensorVersioned* dl_managed_tensor,
    struct mts_array_t* new_array
) {
    (void)array;
    (void)dl_managed_tensor;
    (void)new_array;
    return dense_array_unsupported();
}

static mts_status_t dense_array_move_data(
    void* output,
    const void* input,
    const struct mts_data_movement_t* movements,
    uintptr_t movements_count
) {
    (void)output;
    (void)input;
    (void)movements;
    (void)movements_count;
    return dense_array_unsupported();
}

static struct mts_array_t dense_array_to_mts(DenseArray* array) {
    struct mts_array_t mts = {
        .ptr = array,
        .destroy = dense_array_destroy,
        .origin = dense_array_origin,
        .device = dense_array_device,
        .dtype = dense_array_dtype,
        .as_dlpack = dense_array_as_dlpack,
        .from_dlpack = dense_array_from_dlpack,
        .shape = dense_array_shape,
        .reshape = dense_array_reshape,
        .swap_axes = dense_array_swap_axes,
        .create = dense_array_create,
        .copy = dense_array_copy,
        .move_data = dense_array_move_data,
    };
    return mts;
}

static struct mts_array_t dense_f64(const uintptr_t* shape, uintptr_t ndim, const double* values) {
    uintptr_t n = 1;
    for (uintptr_t i = 0; i < ndim; i++) {
        n *= shape[i];
    }
    DenseArray* array = dense_array_new(
        shape,
        ndim,
        (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1},
        values,
        n * sizeof(double)
    );
    return dense_array_to_mts(array);
}

static struct mts_array_t dense_i32(const uintptr_t* shape, uintptr_t ndim, const int32_t* values) {
    uintptr_t n = 1;
    for (uintptr_t i = 0; i < ndim; i++) {
        n *= shape[i];
    }
    DenseArray* array = dense_array_new(
        shape,
        ndim,
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1},
        values,
        n * sizeof(int32_t)
    );
    return dense_array_to_mts(array);
}

static const mts_labels_t* labels_from_i32(
    const char* const* names,
    uintptr_t n_names,
    const int32_t* values,
    uintptr_t n_entries
) {
    uintptr_t shape[2] = {n_entries, n_names};
    return mts_labels(names, n_names, dense_i32(shape, 2, values));
}


// %%
//
// Model state
// -----------
//
// The model owns its parameters. Here that is a force constant and the
// equilibrium positions (diamond carbon basis, cubic cell :math:`a = 3.567`
// Å). The initial geometry below is displaced by 0.05 Å so the energy and
// forces are non-zero but easy to check by hand.

#define N_ATOMS 2
#define FORCE_CONSTANT 10.0 /* eV / Angstrom^2 */
#define CELL_A 3.567

typedef struct {
    double force_constant;
    double r0[N_ATOMS][3];
} HarmonicModel;

static const double EQUILIBRIUM[N_ATOMS][3] = {
    {0.0, 0.0, 0.0},
    {CELL_A / 4.0, CELL_A / 4.0, CELL_A / 4.0},
};

/* Displaced diamond basis: atom 0 along +x, atom 1 along +z. */
static const double POSITIONS0[N_ATOMS][3] = {
    {0.05, 0.0, 0.0},
    {CELL_A / 4.0, CELL_A / 4.0, CELL_A / 4.0 + 0.05},
};

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

static mta_status_t harmonic_requested_pair_lists(const void* model_data, mta_string_t* out) {
    (void)model_data;
    /* Non-interacting oscillators: no neighbor lists. */
    *out = mta_string_create("[]");
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t harmonic_requested_inputs(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("[]");
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

// %%
//
// Building the energy TensorMap
// -----------------------------
//
// :c:func:`execute_inner` must return one :c:type:`mts_tensormap_t` per
// requested output. For system-level energy the block has samples
// ``["system"]``, no components, and properties ``["energy"]``. Position
// gradients use samples ``["sample", "system", "atom"]`` and component
// ``["xyz"]``.

static mts_tensormap_t* energy_with_forces(
    double energy,
    const double forces[N_ATOMS][3]
) {
    /* values: shape (n_systems=1, n_properties=1) */
    double values_data[1] = {energy};
    uintptr_t values_shape[2] = {1, 1};
    int32_t sample_values[1] = {0};
    int32_t property_values[1] = {0};
    int32_t key_values[1] = {0};

    const char* sample_names[] = {"system"};
    const char* property_names[] = {"energy"};
    const char* key_names[] = {"_"};

    const mts_labels_t* samples = labels_from_i32(sample_names, 1, sample_values, 1);
    const mts_labels_t* properties = labels_from_i32(property_names, 1, property_values, 1);
    const mts_labels_t* keys = labels_from_i32(key_names, 1, key_values, 1);
    if (samples == NULL || properties == NULL || keys == NULL) {
        return NULL;
    }

    mts_block_t* block = mts_block(
        dense_f64(values_shape, 2, values_data),
        samples,
        NULL,
        0,
        properties
    );
    mts_labels_free(samples);
    mts_labels_free(properties);
    if (block == NULL) {
        mts_labels_free(keys);
        return NULL;
    }

    /* gradients: ∂E/∂r = -F, shape (n_atoms, 3, 1) */
    double grad_data[N_ATOMS * 3];
    int32_t grad_samples_data[N_ATOMS * 3];
    for (int i = 0; i < N_ATOMS; i++) {
        grad_data[3 * i + 0] = -forces[i][0];
        grad_data[3 * i + 1] = -forces[i][1];
        grad_data[3 * i + 2] = -forces[i][2];
        grad_samples_data[3 * i + 0] = 0; /* sample */
        grad_samples_data[3 * i + 1] = 0; /* system */
        grad_samples_data[3 * i + 2] = i; /* atom */
    }

    uintptr_t grad_shape[3] = {N_ATOMS, 3, 1};
    int32_t xyz_data[3] = {0, 1, 2};
    const char* grad_sample_names[] = {"sample", "system", "atom"};
    const char* xyz_names[] = {"xyz"};

    const mts_labels_t* grad_samples =
        labels_from_i32(grad_sample_names, 3, grad_samples_data, N_ATOMS);
    const mts_labels_t* xyz = labels_from_i32(xyz_names, 1, xyz_data, 3);
    const mts_labels_t* grad_properties =
        labels_from_i32(property_names, 1, property_values, 1);
    if (grad_samples == NULL || xyz == NULL || grad_properties == NULL) {
        mts_block_free(block);
        mts_labels_free(keys);
        return NULL;
    }

    const mts_labels_t* components[1] = {xyz};
    mts_block_t* gradient = mts_block(
        dense_f64(grad_shape, 3, grad_data),
        grad_samples,
        components,
        1,
        grad_properties
    );
    mts_labels_free(grad_samples);
    mts_labels_free(xyz);
    mts_labels_free(grad_properties);
    if (gradient == NULL) {
        mts_block_free(block);
        mts_labels_free(keys);
        return NULL;
    }

    if (mts_block_add_gradient(block, "positions", gradient) != MTS_SUCCESS) {
        mts_block_free(gradient);
        mts_block_free(block);
        mts_labels_free(keys);
        return NULL;
    }

    mts_block_t* blocks[1] = {block};
    mts_tensormap_t* tensor = mts_tensormap(keys, blocks, 1);
    mts_labels_free(keys);
    return tensor;
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
    HarmonicModel* model = (HarmonicModel*)model_data;
    (void)requested_outputs_json;

    if (selected_atoms != NULL) {
        mta_set_last_error(
            "selected_atoms is not supported by harmonic-diamond",
            "harmonic_execute_inner",
            NULL,
            NULL
        );
        return MTA_INVALID_PARAMETER_ERROR;
    }
    if (systems_count != 1 || outputs_count != 1) {
        mta_set_last_error(
            "this tutorial model expects one system and one output",
            "harmonic_execute_inner",
            NULL,
            NULL
        );
        return MTA_INVALID_PARAMETER_ERROR;
    }

    DLManagedTensorVersioned* positions = NULL;
    mta_status_t status = mta_system_get_data(
        systems[0], MTA_SYSTEM_DATA_POSITIONS, &positions
    );
    if (status != MTA_SUCCESS) {
        return status;
    }

    double* xyz = (double*)((char*)positions->dl_tensor.data + positions->dl_tensor.byte_offset);
    double energy = 0.0;
    double forces[N_ATOMS][3];
    for (int i = 0; i < N_ATOMS; i++) {
        for (int a = 0; a < 3; a++) {
            double dr = xyz[3 * i + a] - model->r0[i][a];
            energy += model->force_constant * dr * dr;
            forces[i][a] = -2.0 * model->force_constant * dr;
        }
    }
    positions->deleter(positions);

    outputs[0] = energy_with_forces(energy, forces);
    if (outputs[0] == NULL) {
        const char* message = NULL;
        mts_last_error(&message, NULL, NULL);
        mta_set_last_error(
            message != NULL ? message : "failed to build energy TensorMap",
            "harmonic_execute_inner",
            NULL,
            NULL
        );
        return MTA_METATENSOR_ERROR;
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
    memcpy(data->r0, EQUILIBRIUM, sizeof(EQUILIBRIUM));

    model->data = data;
    model->unload = harmonic_unload;
    model->metadata = harmonic_metadata;
    model->capabilities = harmonic_capabilities;
    model->supported_outputs = harmonic_supported_outputs;
    model->requested_pair_lists = harmonic_requested_pair_lists;
    model->requested_inputs = harmonic_requested_inputs;
    model->execute_inner = harmonic_execute_inner;
    return MTA_SUCCESS;
}

// %%
//
// Helpers to build the system
// ---------------------------

typedef struct {
    int64_t* shape;
    int64_t* strides;
} SystemDLPackContext;

static void system_dlpack_deleter(DLManagedTensorVersioned* self) {
    if (self == NULL) {
        return;
    }
    SystemDLPackContext* ctx = (SystemDLPackContext*)self->manager_ctx;
    if (ctx != NULL) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
    }
    free(self);
}

static DLManagedTensorVersioned* system_tensor_from_data(
    void* data,
    int32_t ndim,
    const int64_t* shape,
    DLDataType dtype
) {
    SystemDLPackContext* ctx = malloc(sizeof(SystemDLPackContext));
    DLManagedTensorVersioned* tensor = calloc(1, sizeof(*tensor));
    if (ctx == NULL || tensor == NULL) {
        free(ctx);
        free(tensor);
        return NULL;
    }
    ctx->shape = malloc((size_t)ndim * sizeof(int64_t));
    ctx->strides = malloc((size_t)ndim * sizeof(int64_t));
    if (ctx->shape == NULL || ctx->strides == NULL) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
        free(tensor);
        return NULL;
    }
    memcpy(ctx->shape, shape, (size_t)ndim * sizeof(int64_t));
    int64_t stride = 1;
    for (int32_t i = ndim - 1; i >= 0; i--) {
        ctx->strides[i] = stride;
        stride *= shape[i];
    }

    tensor->version.major = DLPACK_MAJOR_VERSION;
    tensor->version.minor = DLPACK_MINOR_VERSION;
    tensor->manager_ctx = ctx;
    tensor->deleter = system_dlpack_deleter;
    tensor->flags = DLPACK_FLAG_BITMASK_READ_ONLY;
    tensor->dl_tensor.data = data;
    tensor->dl_tensor.device.device_type = kDLCPU;
    tensor->dl_tensor.device.device_id = 0;
    tensor->dl_tensor.dtype = dtype;
    tensor->dl_tensor.ndim = ndim;
    tensor->dl_tensor.shape = ctx->shape;
    tensor->dl_tensor.strides = ctx->strides;
    return tensor;
}

static int fail(mta_system_t* system, mta_model_t* model, const char* what) {
    const char* message = NULL;
    mta_last_error(&message, NULL, NULL);
    fprintf(stderr, "%s: %s\n", what, message != NULL ? message : "(no message)");
    if (model != NULL && model->unload != NULL && model->data != NULL) {
        model->unload(model->data);
    }
    if (system != NULL) {
        mta_system_free(system);
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
        return fail(NULL, NULL, "failed to register plugin");
    }

    mta_model_t model = {0};
    if (mta_load_model("harmonic-diamond", "{}", "tutorial-harmonic-plugin", &model)
        != MTA_SUCCESS) {
        return fail(NULL, NULL, "failed to load model");
    }

    mta_string_t metadata = NULL;
    if (model.metadata(model.data, &metadata) != MTA_SUCCESS) {
        return fail(NULL, &model, "failed to get metadata");
    }
    mta_string_t printed = NULL;
    if (mta_format_metadata(mta_string_view(metadata), &printed) != MTA_SUCCESS) {
        mta_string_free(metadata);
        return fail(NULL, &model, "failed to format metadata");
    }
    printf("%s\n", mta_string_view(printed));
    mta_string_free(metadata);
    mta_string_free(printed);

    int32_t types_data[N_ATOMS] = {6, 6};
    double positions_data[N_ATOMS * 3];
    double cell_data[9] = {
        CELL_A, 0.0, 0.0,
        0.0, CELL_A, 0.0,
        0.0, 0.0, CELL_A,
    };
    bool pbc_data[3] = {true, true, true};
    memcpy(positions_data, POSITIONS0, sizeof(POSITIONS0));

    DLManagedTensorVersioned* types = system_tensor_from_data(
        types_data, 1, (int64_t[]){N_ATOMS},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}
    );
    DLManagedTensorVersioned* positions = system_tensor_from_data(
        positions_data, 2, (int64_t[]){N_ATOMS, 3},
        (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}
    );
    DLManagedTensorVersioned* cell = system_tensor_from_data(
        cell_data, 2, (int64_t[]){3, 3},
        (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}
    );
    DLManagedTensorVersioned* pbc = system_tensor_from_data(
        pbc_data, 1, (int64_t[]){3},
        (DLDataType){.code = kDLBool, .bits = 8, .lanes = 1}
    );

    mta_system_t* system = NULL;
    if (mta_system_create("Angstrom", types, positions, cell, pbc, &system)
        != MTA_SUCCESS) {
        return fail(NULL, &model, "failed to create system");
    }

    /* Call execute_inner directly. Engines normally use mta_execute_model,
       which adds validation and unit conversion on top of this callback. */
    mts_tensormap_t* outputs[1] = {NULL};
    const char* requested =
        "[{\"type\":\"metatomic_quantity\",\"name\":\"energy\",\"unit\":\"eV\","
        "\"gradients\":[\"positions\"],\"sample_kind\":\"system\"}]";
    if (model.execute_inner(
            model.data, (const mta_system_t* const*)&system, 1, NULL, requested, outputs, 1
        )
        != MTA_SUCCESS) {
        return fail(system, &model, "execute_inner failed");
    }

    mts_block_t* block = NULL;
    if (mts_tensormap_block_by_id(outputs[0], &block, 0) != MTS_SUCCESS) {
        mts_tensormap_free(outputs[0]);
        return fail(system, &model, "failed to get energy block");
    }

    mts_array_t values;
    if (mts_block_data(block, &values) != MTS_SUCCESS) {
        mts_tensormap_free(outputs[0]);
        return fail(system, &model, "failed to get energy values");
    }

    DLManagedTensorVersioned* values_dl = NULL;
    DLDevice cpu = {.device_type = kDLCPU, .device_id = 0};
    DLPackVersion ver = {.major = DLPACK_MAJOR_VERSION, .minor = DLPACK_MINOR_VERSION};
    if (values.as_dlpack(values.ptr, &values_dl, cpu, NULL, ver) != MTS_SUCCESS) {
        mts_tensormap_free(outputs[0]);
        return fail(system, &model, "failed to export energy values");
    }
    double energy = ((double*)((char*)values_dl->dl_tensor.data + values_dl->dl_tensor.byte_offset))[0];
    values_dl->deleter(values_dl);

    mts_block_t* gradient = NULL;
    if (mts_block_gradient(block, "positions", &gradient) != MTS_SUCCESS) {
        mts_tensormap_free(outputs[0]);
        return fail(system, &model, "failed to get positions gradient");
    }
    mts_array_t grad_values;
    if (mts_block_data(gradient, &grad_values) != MTS_SUCCESS) {
        mts_tensormap_free(outputs[0]);
        return fail(system, &model, "failed to get gradient values");
    }
    DLManagedTensorVersioned* grad_dl = NULL;
    if (grad_values.as_dlpack(grad_values.ptr, &grad_dl, cpu, NULL, ver) != MTS_SUCCESS) {
        mts_tensormap_free(outputs[0]);
        return fail(system, &model, "failed to export gradient values");
    }
    double* grad = (double*)((char*)grad_dl->dl_tensor.data + grad_dl->dl_tensor.byte_offset);

    printf("energy: %.6f eV\n", energy);
    printf("forces (eV/Angstrom):\n");
    for (int i = 0; i < N_ATOMS; i++) {
        /* F = -∂E/∂r */
        double fx = -grad[3 * i + 0];
        double fy = -grad[3 * i + 1];
        double fz = -grad[3 * i + 2];
        if (fabs(fx) < 1e-12) { fx = 0.0; }
        if (fabs(fy) < 1e-12) { fy = 0.0; }
        if (fabs(fz) < 1e-12) { fz = 0.0; }
        printf("  atom %d: %8.3f %8.3f %8.3f\n", i, fx, fy, fz);
    }
    grad_dl->deleter(grad_dl);
    mts_tensormap_free(outputs[0]);

    model.unload(model.data);
    mta_system_free(system);
    return EXIT_SUCCESS;
}

// %%
//
// Expected output
// ---------------
//
// With :math:`k = 10` eV/Å² and 0.05 Å displacements, each atom contributes
// :math:`10 \times 0.05^2 = 0.025` eV, so the total energy is 0.05 eV. The
// non-zero force components have magnitude :math:`2k\times 0.05 = 1` eV/Å::
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
//     energy: 0.050000 eV
//     forces (eV/Angstrom):
//       atom 0:   -1.000    0.000    0.000
//       atom 1:    0.000    0.000   -1.000
