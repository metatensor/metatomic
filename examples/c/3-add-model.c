// .. _c-tutorial-add-model:
//
// Defining a model: energy and forces
// ====================================
//
// The previous two tutorials show the *engine* side of the C API: building a
// :c:type:`mta_system_t` and reading data back out of it. This one shows the
// other side -- implementing an :c:type:`mta_model_t` -- and, unlike earlier
// drafts of this tutorial, does not stop at the energy: it also fills in the
// ``"positions"`` gradient block, since that gradient *is* the force, and a
// real engine cannot run molecular dynamics without it.
//
// The running example is a **shifted Lennard-Jones** pair potential. The
// energy is
//
// .. math::
//
//     E = 4 \epsilon \left[
//         \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}
//     \right] - E_{\mathrm{shift}},
//
// with :math:`E_{\mathrm{shift}}` chosen so the energy is exactly zero at the
// cutoff :math:`r_c`.

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatomic.h>
#include <metatensor/dlpack/dlpack.h>

// %%
//
// .. raw:: html
//
//   <details><summary>DLPack and <code>mts_array_t</code> helpers (same as the previous two tutorials)</summary>

typedef struct CustomDLPackContext {
    int64_t* shape;
    int64_t* strides;
} CustomDLPackContext;

static void dlpack_deleter(DLManagedTensorVersioned *self) {
    if (!self) {
        return;
    }
    CustomDLPackContext* ctx = (CustomDLPackContext*)self->manager_ctx;
    if (ctx) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
    }
    free(self);
}

static DLManagedTensorVersioned* tensor_from_data(
    void *data,
    int32_t ndim,
    const int64_t *shape,
    DLDataType dtype
) {
    CustomDLPackContext* ctx = malloc(sizeof(CustomDLPackContext));
    if (!ctx) {
        return NULL;
    }
    ctx->shape = malloc(ndim * sizeof(int64_t));
    ctx->strides = malloc(ndim * sizeof(int64_t));
    if (!ctx->shape || !ctx->strides) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
        return NULL;
    }
    memcpy(ctx->shape, shape, ndim * sizeof(int64_t));

    int64_t stride = 1;
    for (int32_t i = ndim - 1; i >= 0; i--) {
        ctx->strides[i] = stride;
        stride *= shape[i];
    }

    DLManagedTensorVersioned* tensor = calloc(1, sizeof(*tensor));
    if (!tensor) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
        return NULL;
    }
    tensor->version.major = DLPACK_MAJOR_VERSION;
    tensor->version.minor = DLPACK_MINOR_VERSION;
    tensor->manager_ctx = ctx;
    tensor->deleter = dlpack_deleter;
    tensor->flags = DLPACK_FLAG_BITMASK_READ_ONLY;
    tensor->dl_tensor.data = data;
    tensor->dl_tensor.byte_offset = 0;
    tensor->dl_tensor.device.device_type = kDLCPU;
    tensor->dl_tensor.device.device_id = 0;
    tensor->dl_tensor.dtype = dtype;
    tensor->dl_tensor.ndim = ndim;
    tensor->dl_tensor.shape = ctx->shape;
    tensor->dl_tensor.strides = ctx->strides;
    return tensor;
}

typedef struct BasicMtsArray {
    void* data;
    uintptr_t ndim;
    uintptr_t shape[4];
    DLDataType dtype;
    uintptr_t size;
} BasicMtsArray;

static mts_data_origin_t BASIC_MTS_ARRAY_ORIGIN = 0;

static void array_destroy(void* array) {
    BasicMtsArray* a = (BasicMtsArray*)array;
    free(a->data);
    free(a);
}

static mts_status_t array_origin(const void* array, mts_data_origin_t* origin) {
    (void)array;
    if (BASIC_MTS_ARRAY_ORIGIN == 0) {
        mts_register_data_origin("tutorial-mts-array", &BASIC_MTS_ARRAY_ORIGIN);
    }
    *origin = BASIC_MTS_ARRAY_ORIGIN;
    return MTS_SUCCESS;
}

static mts_status_t array_device(const void* array, DLDevice* device) {
    (void)array;
    device->device_type = kDLCPU;
    device->device_id = 0;
    return MTS_SUCCESS;
}

static mts_status_t array_dtype(const void* array, DLDataType* dtype) {
    *dtype = ((const BasicMtsArray*)array)->dtype;
    return MTS_SUCCESS;
}

static mts_status_t array_as_dlpack(
    void* array,
    DLManagedTensorVersioned** tensor,
    DLDevice device,
    const int64_t* stream,
    DLPackVersion max_version
) {
    (void)stream;
    (void)max_version;
    BasicMtsArray* a = (BasicMtsArray*)array;
    if (device.device_type != kDLCPU) {
        return MTS_CALLBACK_ERROR;
    }
    *tensor = tensor_from_data(a->data, (int32_t)a->ndim, (const int64_t*)a->shape, a->dtype);
    return MTS_SUCCESS;
}

static mts_status_t array_shape(
    const void* array,
    const uintptr_t** shape,
    uintptr_t* shape_count
) {
    const BasicMtsArray* a = (const BasicMtsArray*)array;
    *shape = a->shape;
    *shape_count = a->ndim;
    return MTS_SUCCESS;
}

static struct mts_array_t make_mts_array(void* data, uintptr_t ndim, const uintptr_t* shape, DLDataType dtype, uintptr_t size) {
    BasicMtsArray* raw = malloc(sizeof(BasicMtsArray));
    size_t data_size = size * (dtype.bits / 8);
    raw->data = malloc(data_size);
    memcpy(raw->data, data, data_size);
    raw->ndim = ndim;
    for (uintptr_t i = 0; i < ndim; i++) {
        raw->shape[i] = shape[i];
    }
    raw->dtype = dtype;
    raw->size = size;

    struct mts_array_t result = {0};
    result.ptr = raw;
    result.destroy = array_destroy;
    result.origin = array_origin;
    result.device = array_device;
    result.dtype = array_dtype;
    result.as_dlpack = array_as_dlpack;
    result.shape = array_shape;
    return result;
}

// read a block's data as a flat float64 buffer; the returned view must be
// released with `view->deleter(view)`.
static DLManagedTensorVersioned* block_f64_view(const mts_block_t* block) {
    struct mts_array_t array = {0};
    if (mts_block_data((mts_block_t*)block, &array) != MTS_SUCCESS) {
        return NULL;
    }
    DLManagedTensorVersioned* view = NULL;
    DLDevice cpu = {.device_type = kDLCPU, .device_id = 0};
    DLPackVersion version = {.major = DLPACK_MAJOR_VERSION, .minor = DLPACK_MINOR_VERSION};
    if (array.as_dlpack(array.ptr, &view, cpu, NULL, version) != MTS_SUCCESS) {
        return NULL;
    }
    return view;
}

static double* block_f64_data(DLManagedTensorVersioned* view) {
    return (double*)((uint8_t*)view->dl_tensor.data + view->dl_tensor.byte_offset);
}

// %%
//
// .. raw:: html
//
//   </details>
//
// The model's physics
// --------------------
//
// This is independent of the C API entirely: given a pair displacement, it
// returns the pair energy and the force it puts on the *first* atom of the
// pair (the force on the second atom is the same, negated).

typedef struct {
    double sigma;
    double epsilon;
    double cutoff;
    double shift;
} LennardJones;

static double lj_shift(double cutoff, double sigma, double epsilon) {
    double x = sigma / cutoff;
    double x6 = x * x * x * x * x * x;
    return 4.0 * epsilon * (x6 * x6 - x6);
}

static void lj_pair(
    double dx, double dy, double dz,
    const LennardJones* lj,
    double* energy,
    double force_on_first[3]
) {
    double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 <= 0.0 || r2 >= lj->cutoff * lj->cutoff) {
        *energy = 0.0;
        force_on_first[0] = force_on_first[1] = force_on_first[2] = 0.0;
        return;
    }

    double inv2 = (lj->sigma * lj->sigma) / r2;
    double inv6 = inv2 * inv2 * inv2;
    double inv12 = inv6 * inv6;
    *energy = 4.0 * lj->epsilon * (inv12 - inv6) - lj->shift;

    // dE/d(r^2) = (12 epsilon / r^2) (inv6 - 2 inv12); force on the first
    // atom is -dE/d(pos_1), and since r^2 = |pos_2 - pos_1|^2,
    // d(r^2)/d(pos_1) = -2 d, giving F_1 = 2 dE/d(r^2) d.
    double dedr2 = (12.0 * lj->epsilon / r2) * (inv6 - 2.0 * inv12);
    force_on_first[0] = 2.0 * dedr2 * dx;
    force_on_first[1] = 2.0 * dedr2 * dy;
    force_on_first[2] = 2.0 * dedr2 * dz;
}

// %%
//
// ``mta_model_t`` callbacks
// --------------------------
//
// Metadata, capabilities, and the (empty) extra-inputs list are the same
// kind of small JSON-returning callbacks as any other model. Note that
// ``capabilities`` declares ``"gradients": ["positions"]`` for the energy
// output -- this is the model promising it can compute forces, which
// ``execute_inner`` below has to actually deliver on.

static mta_status_t lj_unload(void* model_data) {
    free(model_data);
    return MTA_SUCCESS;
}

static mta_status_t lj_metadata(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create(
        "{"
        "\"type\": \"metatomic_model_metadata\","
        "\"name\": \"tutorial-lennard-jones\","
        "\"authors\": [\"metatomic C tutorials\"],"
        "\"references\": {\"model\": [], \"architecture\": [], \"implementation\": []}"
        "}"
    );
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t lj_capabilities(const void* model_data, mta_string_t* out) {
    const LennardJones* lj = (const LennardJones*)model_data;
    char json[512];
    snprintf(
        json, sizeof(json),
        "{"
        "\"type\": \"metatomic_model_capabilities\","
        "\"outputs\": [{"
        "  \"type\": \"metatomic_quantity\","
        "  \"name\": \"energy\","
        "  \"unit\": \"eV\","
        "  \"gradients\": [\"positions\"],"
        "  \"sample_kind\": \"system\""
        "}],"
        "\"atomic_types\": [1],"
        "\"interaction_range\": %.17g,"
        "\"length_unit\": \"Angstrom\","
        "\"supported_devices\": [\"cpu\"],"
        "\"dtype\": \"float64\""
        "}",
        lj->cutoff
    );
    *out = mta_string_create(json);
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static void format_pair_options(const LennardJones* lj, char* buf, size_t n) {
    uint64_t bits;
    memcpy(&bits, &lj->cutoff, sizeof(bits));
    char cutoff_hex[32];
    snprintf(cutoff_hex, sizeof(cutoff_hex), "0x%" PRIx64, bits);
    snprintf(
        buf, n,
        "{\"type\": \"metatomic_pair_options\", \"cutoff\": \"%s\","
        " \"full_list\": false, \"strict\": true, \"requestors\": [\"tutorial\"]}",
        cutoff_hex
    );
}

static mta_status_t lj_requested_pair_lists(const void* model_data, mta_string_t* out) {
    char options[512];
    format_pair_options((const LennardJones*)model_data, options, sizeof(options));
    char json[520];
    snprintf(json, sizeof(json), "[%s]", options);
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
// Filling in the gradient
// -------------------------
//
// The ``"energy"`` output is a :c:type:`mts_tensormap_t` with a single block
// (one ``"system"`` sample, one ``"energy"`` property). The ``"positions"``
// gradient attached to it follows the recipe from
// :ref:`the energy quantity docs <energy-quantity>`: samples
// ``["sample", "system", "atom"]`` (one row per atom), one component
// ``"xyz"``, properties matching the parent block, and values equal to the
// **negative** of the force (:math:`\partial E/\partial r_j = -F_j`).

static mts_block_t* positions_gradient(const double* forces, uintptr_t n_atoms) {
    int32_t* sample_values = malloc(sizeof(int32_t) * n_atoms * 3);
    for (uintptr_t j = 0; j < n_atoms; j++) {
        sample_values[3 * j + 0] = 0;          // sample: row 0 of the energy block
        sample_values[3 * j + 1] = 0;          // system
        sample_values[3 * j + 2] = (int32_t)j; // atom
    }
    const char* sample_dims[] = {"sample", "system", "atom"};
    struct mts_array_t samples_array = make_mts_array(
        sample_values, 2, (uintptr_t[]){n_atoms, 3},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, n_atoms * 3
    );
    free(sample_values);
    const mts_labels_t* samples = mts_labels(sample_dims, 3, samples_array);

    int32_t xyz_values[] = {0, 1, 2};
    struct mts_array_t xyz_array = make_mts_array(
        xyz_values, 2, (uintptr_t[]){3, 1},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 3
    );
    const char* xyz_dims[] = {"xyz"};
    const mts_labels_t* xyz = mts_labels(xyz_dims, 1, xyz_array);
    const mts_labels_t* components[] = {xyz};

    int32_t energy_value = 0;
    struct mts_array_t energy_prop_array = make_mts_array(
        &energy_value, 2, (uintptr_t[]){1, 1},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 1
    );
    const char* energy_dims[] = {"energy"};
    const mts_labels_t* properties = mts_labels(energy_dims, 1, energy_prop_array);

    double* data = malloc(sizeof(double) * n_atoms * 3);
    for (uintptr_t j = 0; j < n_atoms; j++) {
        for (int k = 0; k < 3; k++) {
            data[j * 3 + k] = -forces[j * 3 + k]; // gradient = -force
        }
    }
    struct mts_array_t values = make_mts_array(
        data, 3, (uintptr_t[]){n_atoms, 3, 1},
        (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}, n_atoms * 3
    );
    free(data);

    mts_block_t* gradient = mts_block(values, samples, components, 1, properties);
    mts_labels_free(samples);
    mts_labels_free(xyz);
    mts_labels_free(properties);
    return gradient;
}

static mts_tensormap_t* energy_tensormap(double energy, const double* forces, uintptr_t n_atoms) {
    struct mts_array_t values = make_mts_array(
        &energy, 2, (uintptr_t[]){1, 1},
        (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}, 1
    );
    int32_t zero = 0;
    struct mts_array_t samples_array = make_mts_array(
        &zero, 2, (uintptr_t[]){1, 1},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 1
    );
    const char* system_dims[] = {"system"};
    const mts_labels_t* samples = mts_labels(system_dims, 1, samples_array);

    struct mts_array_t prop_array = make_mts_array(
        &zero, 2, (uintptr_t[]){1, 1},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 1
    );
    const char* energy_dims[] = {"energy"};
    const mts_labels_t* properties = mts_labels(energy_dims, 1, prop_array);

    struct mts_array_t key_array = make_mts_array(
        &zero, 2, (uintptr_t[]){1, 1},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 1
    );
    const char* key_dims[] = {"_"};
    const mts_labels_t* keys = mts_labels(key_dims, 1, key_array);

    mts_block_t* block = mts_block(values, samples, NULL, 0, properties);
    assert(block != NULL);

    mts_block_t* gradient = positions_gradient(forces, n_atoms);
    mts_status_t status = mts_block_add_gradient(block, "positions", gradient);
    assert(status == MTS_SUCCESS);

    mts_block_t* blocks[] = {block};
    mts_tensormap_t* tensor = mts_tensormap(keys, blocks, 1);
    mts_labels_free(samples);
    mts_labels_free(properties);
    mts_labels_free(keys);
    return tensor;
}

// %%
//
// ``execute_inner``
// -------------------
//
// Reads the displacement vector and the ``first_atom``/``second_atom``
// sample labels straight out of the pair list the engine attached, sums the
// pair energy and force, and hands both back through
// :c:func:`energy_tensormap`.

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
    assert(systems_count == 1);

    const LennardJones* lj = (const LennardJones*)model_data;
    const mta_system_t* system = systems[0];

    uintptr_t n_atoms = 0;
    mta_system_size(system, &n_atoms);

    char options[512];
    format_pair_options(lj, options, sizeof(options));
    const mts_block_t* pairs = NULL;
    mta_status_t status = mta_system_get_pairs(system, options, &pairs);
    if (status != MTA_SUCCESS) {
        return status;
    }

    DLManagedTensorVersioned* disp_view = block_f64_view(pairs);
    assert(disp_view != NULL);
    double* disp = block_f64_data(disp_view);
    uintptr_t n_pairs = disp_view->dl_tensor.shape[0];

    const mts_labels_t* pair_samples = mts_block_labels(pairs, 0);
    const int32_t* sample_values = NULL;
    uintptr_t sample_count = 0, sample_size = 0;
    mts_labels_values_cpu(pair_samples, &sample_values, &sample_count, &sample_size);
    assert(sample_count == n_pairs);

    double* forces = calloc(n_atoms * 3, sizeof(double));
    double energy = 0.0;
    for (uintptr_t p = 0; p < n_pairs; p++) {
        double pair_energy;
        double force_on_first[3];
        lj_pair(disp[3 * p + 0], disp[3 * p + 1], disp[3 * p + 2], lj, &pair_energy, force_on_first);
        energy += pair_energy;

        int32_t i = sample_values[p * sample_size + 0];
        int32_t j = sample_values[p * sample_size + 1];
        for (int k = 0; k < 3; k++) {
            forces[(uintptr_t)i * 3 + k] += force_on_first[k];
            forces[(uintptr_t)j * 3 + k] -= force_on_first[k];
        }
    }

    mts_labels_free(pair_samples);
    disp_view->deleter(disp_view);

    for (uintptr_t idx = 0; idx < outputs_count; idx++) {
        outputs[idx] = energy_tensormap(energy, forces, n_atoms);
    }
    free(forces);
    return MTA_SUCCESS;
}

static mta_status_t lj_load_model(const char* load_from, const char* options_json, mta_model_t* model) {
    (void)options_json;
    if (strcmp(load_from, "tutorial-lennard-jones") != 0) {
        return MTA_MODEL_NOT_SUPPORTED_ERROR;
    }
    LennardJones* data = malloc(sizeof(LennardJones));
    data->sigma = 1.0;
    data->epsilon = 1.0;
    data->cutoff = 3.0;
    data->shift = lj_shift(data->cutoff, data->sigma, data->epsilon);

    model->data = data;
    model->unload = lj_unload;
    model->metadata = lj_metadata;
    model->capabilities = lj_capabilities;
    model->requested_pair_lists = lj_requested_pair_lists;
    model->requested_inputs = lj_requested_inputs;
    model->execute_inner = lj_execute_inner;
    return MTA_SUCCESS;
}

static mta_status_t plugin_load_model(const char* load_from, const char* options_json, struct mta_model_t* model) {
    return lj_load_model(load_from, options_json, model);
}

// %%
//
// Putting it together
// ---------------------
//
// A two-atom system, ``distance`` apart along *z*, with the single pair
// between them attached by hand (as tutorial 2 does; a real engine would
// compute this from a neighbor search instead).

static mta_system_t* build_two_atom_system(double distance) {
    static double positions_data[6];
    positions_data[0] = 0.0; positions_data[1] = 0.0; positions_data[2] = 0.0;
    positions_data[3] = 0.0; positions_data[4] = 0.0; positions_data[5] = distance;

    // non-periodic (pbc = false below), so the cell must be all zeros
    static double cell_data[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    static int32_t types_data[] = {1, 1};
    static bool pbc_data[] = {false, false, false};

    DLManagedTensorVersioned* positions = tensor_from_data(
        positions_data, 2, (int64_t[]){2, 3}, (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}
    );
    DLManagedTensorVersioned* cell = tensor_from_data(
        cell_data, 2, (int64_t[]){3, 3}, (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}
    );
    DLManagedTensorVersioned* types = tensor_from_data(
        types_data, 1, (int64_t[]){2}, (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}
    );
    DLManagedTensorVersioned* pbc = tensor_from_data(
        pbc_data, 1, (int64_t[]){3}, (DLDataType){.code = kDLBool, .bits = 8, .lanes = 1}
    );

    mta_system_t* system = NULL;
    mta_status_t create_status = mta_system_create("Angstrom", types, positions, cell, pbc, &system);
    if (create_status != MTA_SUCCESS) {
        const char* error_message = NULL;
        mta_last_error(&error_message, NULL, NULL);
        fprintf(stderr, "failed to create system: %s\n", error_message);
        return NULL;
    }

    int32_t pair_samples[] = {0, 1, 0, 0, 0};
    struct mts_array_t samples_array = make_mts_array(
        pair_samples, 2, (uintptr_t[]){1, 5}, (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 5
    );
    const char* sample_dims[] = {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"};
    const mts_labels_t* samples = mts_labels(sample_dims, 5, samples_array);

    int32_t xyz_values[] = {0, 1, 2};
    struct mts_array_t xyz_array = make_mts_array(
        xyz_values, 2, (uintptr_t[]){3, 1}, (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 3
    );
    const char* xyz_dims[] = {"xyz"};
    const mts_labels_t* xyz = mts_labels(xyz_dims, 1, xyz_array);
    const mts_labels_t* components[] = {xyz};

    int32_t zero = 0;
    struct mts_array_t prop_array = make_mts_array(
        &zero, 2, (uintptr_t[]){1, 1}, (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 1
    );
    const char* distance_dims[] = {"distance"};
    const mts_labels_t* properties = mts_labels(distance_dims, 1, prop_array);

    double disp_data[] = {0.0, 0.0, distance};
    struct mts_array_t values = make_mts_array(
        disp_data, 3, (uintptr_t[]){1, 3, 1}, (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}, 3
    );
    mts_block_t* pairs = mts_block(values, samples, components, 1, properties);

    LennardJones lj = {.sigma = 1.0, .epsilon = 1.0, .cutoff = 3.0};
    char options[512];
    format_pair_options(&lj, options, sizeof(options));
    mta_system_add_pairs(system, options, pairs);

    mts_labels_free(samples);
    mts_labels_free(xyz);
    mts_labels_free(properties);
    return system;
}

int main(void) {

// %%
//
// Register the plugin in-process (a real, shared-library plugin would use
// the :c:macro:`MTA_REGISTER_PLUGIN` macro instead -- see the next tutorial),
// and load the model.

struct mta_plugin_t plugin = {
    .abi_version = MTA_ABI_VERSION,
    .name = "tutorial-lj-plugin",
    .load_model = plugin_load_model,
};
mta_register_plugin(plugin);

mta_model_t model;
mta_status_t status = mta_load_model("tutorial-lennard-jones", "{}", "tutorial-lj-plugin", &model);
if (status != MTA_SUCCESS) {
    fprintf(stderr, "failed to load model\n");
    return EXIT_FAILURE;
}

// %%
//
// Run it
// ------
//
// Two atoms 1.3 sigma apart -- just past the potential's minimum, so they
// pull on each other.

mta_system_t* system = build_two_atom_system(/*distance=*/1.3);
const mta_system_t* systems[] = {system};

const char* requested_outputs =
    "[{\"type\": \"metatomic_quantity\", \"name\": \"energy\", \"unit\": \"eV\","
    " \"gradients\": [\"positions\"], \"sample_kind\": \"system\"}]";

mts_tensormap_t* output = NULL;
status = mta_execute_model(model, systems, 1, NULL, requested_outputs, /*check_consistency=*/true, &output, 1);
if (status != MTA_SUCCESS) {
    const char* error_message = NULL;
    mta_last_error(&error_message, NULL, NULL);
    fprintf(stderr, "failed to run model: %s\n", error_message);
    return EXIT_FAILURE;
}

mts_block_t* block = NULL;
mts_tensormap_block_by_id(output, &block, 0);

DLManagedTensorVersioned* energy_view = block_f64_view(block);
double energy = block_f64_data(energy_view)[0];
energy_view->deleter(energy_view);
printf("energy: %g\n", energy);

mts_block_t* gradient = NULL;
mts_block_gradient(block, "positions", &gradient);
DLManagedTensorVersioned* grad_view = block_f64_view(gradient);
double* grad = block_f64_data(grad_view);

// gradient = -force, and by construction row 1 (the second atom) is the
// one moving when we perturb `distance` below.
double force_z_atom1 = -grad[1 * 3 + 2];
printf("force on atom 1, z component: %g\n", force_z_atom1);
grad_view->deleter(grad_view);

// %%
//
// Checking the gradient against a finite difference
// ----------------------------------------------------
//
// The whole point of computing a gradient analytically is that it should
// match the numerical derivative of the energy -- so let's check that,
// running the *entire* system -> pairs -> ``execute_inner`` pipeline twice
// more, rather than trusting the model's own math about itself.

double eps = 1e-6;
mta_system_t* system_plus = build_two_atom_system(1.3 + eps);
mta_system_t* system_minus = build_two_atom_system(1.3 - eps);

double energies[2];
for (int s = 0; s < 2; s++) {
    const mta_system_t* one_system[] = {s == 0 ? system_plus : system_minus};
    mts_tensormap_t* out = NULL;
    mta_execute_model(model, one_system, 1, NULL, requested_outputs, true, &out, 1);
    mts_block_t* b = NULL;
    mts_tensormap_block_by_id(out, &b, 0);
    DLManagedTensorVersioned* v = block_f64_view(b);
    energies[s] = block_f64_data(v)[0];
    v->deleter(v);
    mts_tensormap_free(out);
}

double numerical_force_z = -(energies[0] - energies[1]) / (2.0 * eps);
printf("finite-difference force on atom 1, z component: %g\n", numerical_force_z);

assert(fabs(numerical_force_z - force_z_atom1) < 1e-5);
printf("analytic and finite-difference forces agree to 1e-5\n");

// %%
//
// Cleanup

mts_tensormap_free(output);
mta_system_free(system);
mta_system_free(system_plus);
mta_system_free(system_minus);
model.unload(model.data);

return EXIT_SUCCESS; }
