// .. _c-tutorial-use-system:
//
// Using ``mta_system_t``
// ======================
//
// This tutorial explores how to access data stored inside a
// :c:type:`mta_system_t`. We look at retrieving the basic tensors (positions,
// cell, types, pbc), working with pair lists (neighbor lists), and storing
// custom per-system data.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatomic.h>
#include <metatensor/dlpack/dlpack.h>

// Function to create a system that we will use in this tutorial
static mta_system_t* create_system_for_tutorial();

// %%
//
// This tutorial uses the same code as the :ref:`previous one
// <c-tutorial-create-system>` to create a system. In practice, the
// :c:type:`mta_system_t` is created by a simulation engine, and then passed to
// the model, which can acess the data inside in a similar way regardless of
// wether the system was created from C, Python, or any other supported
// language.
//
// .. raw:: html
//
//   <details><summary>Implementation of <code>create_system_for_tutorial()</code></summary>

typedef struct CustomDLPackContext {
    int64_t* shape;
    int64_t* strides;
} CustomDLPackContext;

void dlpack_deleter(DLManagedTensorVersioned *self) {
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

// %%
//
// To work with metatensor's ``mts_block_t`` and ``mts_tensormap_t`` in C, we
// need an ``mts_array_t`` - a vtable-based abstraction over n-dimensional
// arrays. Below is a minimal implementation backed by a flat data buffer that
// the array owns: the data is copied into a heap allocation when the array is
// created, and released when the array is destroyed.

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

    // copy the data into a buffer owned by the array
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
    result.from_dlpack = NULL;
    result.reshape = NULL;
    result.swap_axes = NULL;
    result.create = NULL;
    result.copy = NULL;
    result.move_data =  NULL;
    return result;
}

// %%
//
// In a simulation, the engine would create a system and attach the pair lists
// and custom data to it. In this tutorial, the function
// ``create_system_for_tutorial()`` plays this role: it creates the basic system
// data, then creates a pair list and some custom data, attaches both to the
// system, and returns the fully-populated system.

static mta_system_t* create_system_for_tutorial() {
    // The basic tensor data (positions, cell, types, pbc) is referenced
    // directly by the DLPack tensors, so it must stay alive as long as the
    // system. We use ``static`` arrays to keep it alive without polluting the
    // global scope.
    static double POSITIONS_DATA[] = {
        0.0, 0.0, 0.0,
        0.5, 0.5, 0.0,
        0.5, 0.0, 0.5,
        0.0, 0.5, 0.5,
    };

    static double CELL_DATA[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };

    static int32_t TYPES_DATA[] = {1, 1, 6, 6};

    static bool PBC_DATA[] = {true, true, true};

    const int64_t n_atoms = 4;
    DLManagedTensorVersioned* positions = tensor_from_data(
        POSITIONS_DATA, 2, (int64_t[]){n_atoms, 3},
        (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}
    );

    DLManagedTensorVersioned* cell = tensor_from_data(
        CELL_DATA, 2, (int64_t[]){3, 3},
        (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}
    );

    DLManagedTensorVersioned* types = tensor_from_data(
        TYPES_DATA, 1, (int64_t[]){n_atoms},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}
    );

    DLManagedTensorVersioned* pbc = tensor_from_data(
        PBC_DATA, 1, (int64_t[]){3},
        (DLDataType){.code = kDLBool, .bits = 8, .lanes = 1}
    );

    mta_system_t* system = NULL;
    mta_status_t status = mta_system_create(
        "Angstrom", types, positions, cell, pbc, &system
    );

    if (status != MTA_SUCCESS) {
        const char* error_message = NULL;
        mta_last_error(&error_message, NULL, NULL);
        fprintf(stderr, "failed to create system: %s\n", error_message);
        return NULL;
    }

    int32_t pair_samples[] = {
        0, 1, 0, 0, 0,
        0, 2, 0, 0, 0,
        1, 3, 0, 0, 0,
    };

    double pair_distances[] = {
        0.5, 0.5, 0.0,
        0.5, 0.0, 0.5,
        0.0, 0.5, 0.5,
    };

    int32_t xyz_values[] = {0, 1, 2};
    int32_t distance_values[] = {0};

    const char* sample_dimensions[] = {
        "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"
    };
    struct mts_array_t samples_array = make_mts_array(
        pair_samples, 2, (uintptr_t[]){3, 5},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 15
    );
    const mts_labels_t* samples = mts_labels(sample_dimensions, 5, samples_array);

    const char* component_dimensions[] = {"xyz"};
    struct mts_array_t comp_array = make_mts_array(
        xyz_values, 2, (uintptr_t[]){3, 1},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 3
    );
    const mts_labels_t* component = mts_labels(component_dimensions, 1, comp_array);
    const mts_labels_t* components[] = {component};

    const char* properties_dimensions[] = {"distance"};
    struct mts_array_t prop_array = make_mts_array(
        distance_values, 2, (uintptr_t[]){1, 1},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 1
    );
    const mts_labels_t* properties = mts_labels(properties_dimensions, 1, prop_array);

    struct mts_array_t values_array = make_mts_array(
        pair_distances, 3, (uintptr_t[]){3, 3, 1},
        (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}, 9
    );
    mts_block_t* pairs = mts_block(
        values_array, samples, components, 1, properties
    );

    const char* options =
        "{\"type\": \"metatomic_pair_options\","
        " \"cutoff\": \"0x4008000000000000\","
        " \"full_list\": true,"
        " \"strict\": false,"
        " \"requestors\": [\"tutorial\"]}";

    status = mta_system_add_pairs(system, options, pairs);
    if (status != MTA_SUCCESS) {
        const char* error_message = NULL;
        mta_last_error(&error_message, NULL, NULL);
        fprintf(stderr, "failed to add pairs to system: %s\n", error_message);
        mts_labels_free(samples);
        mts_labels_free(component);
        mts_labels_free(properties);
        mta_system_free(system);
        return NULL;
    }

    mts_labels_free(samples);
    mts_labels_free(component);
    mts_labels_free(properties);

    // ------------------------------------------------------------------ //
    // Custom data: a TensorMap with a single block of per-atom values

    double custom_values[] = {0.42, -0.31, 0.15, -0.08};
    int32_t custom_keys[] = {0};
    int32_t custom_samples[] = {0, 1, 2, 3};
    int32_t custom_properties[] = {0};

    const char* key_dims[] = {"_"};
    struct mts_array_t key_array = make_mts_array(
        custom_keys, 2, (uintptr_t[]){1, 1},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 1
    );
    const mts_labels_t* keys = mts_labels(key_dims, 1, key_array);

    const char* custom_sample_dims[] = {"atom"};
    struct mts_array_t custom_samples_array = make_mts_array(
        custom_samples, 2, (uintptr_t[]){4, 1},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 4
    );
    const mts_labels_t* custom_labels = mts_labels(
        custom_sample_dims, 1, custom_samples_array
    );

    const char* custom_prop_dims[] = {"property"};
    struct mts_array_t custom_prop_array = make_mts_array(
        custom_properties, 2, (uintptr_t[]){1, 1},
        (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}, 1
    );
    const mts_labels_t* custom_labels_props = mts_labels(
        custom_prop_dims, 1, custom_prop_array
    );

    struct mts_array_t values_array_custom = make_mts_array(
        custom_values, 2, (uintptr_t[]){4, 1},
        (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}, 4
    );
    mts_block_t* block = mts_block(
        values_array_custom, custom_labels, NULL, 0, custom_labels_props
    );
    assert(block != NULL);

    mts_block_t* blocks[] = {block};
    mts_tensormap_t* custom = mts_tensormap(keys, blocks, 1);
    assert(custom != NULL);

    status = mta_system_add_custom_data(system, "tutorial::charges", custom);
    if (status != MTA_SUCCESS) {
        const char* error_message = NULL;
        mta_last_error(&error_message, NULL, NULL);
        fprintf(stderr, "failed to add custom data: %s\n", error_message);
        mts_labels_free(custom_labels);
        mts_labels_free(custom_labels_props);
        mta_system_free(system);
        return NULL;
    }

    mts_labels_free(custom_labels);
    mts_labels_free(custom_labels_props);

    return system;
}

// %%
//
// .. raw:: html
//
//   </details>

int main(void) {

// %%
//
// Let's get a system from somewhere, and have a look as what's inside

mta_system_t* system = create_system_for_tutorial();

if (system == NULL) {
    fprintf(stderr, "Failed to create tutorial system\n");
    return EXIT_FAILURE;
}

// %%
//
// Global information
// ------------------
//
// First, we can access some global information about a system, such as the
// number of atoms making up the system:

uintptr_t size = 0;
mta_status_t status = mta_system_size(system, &size);

if (status != MTA_SUCCESS) {
    fprintf(stderr, "failed to get system size\n");
    mta_system_free(system);
    return EXIT_FAILURE;
}
assert(size == 4);
printf("this system contains %lu atoms\n", (unsigned long)size);

// %%
//
// We can also access the unit used for all length data in this system.
//
// This is returned as a newly allocated :c:type:`mta_string_t`. Callers of
// functions that return :c:type:`mta_string_t` can get a view (i.e. a ``const
// char*`` to a null-terminated string) and are responsible for freeing the
// memory.

mta_string_t length_unit = NULL;
status = mta_system_get_length_unit(system, &length_unit);
if (status != MTA_SUCCESS) {
    fprintf(stderr, "failed to get length unit\n");
    mta_system_free(system);
    return EXIT_FAILURE;
}
printf("length unit: %s\n", mta_string_view(length_unit));
assert(strcmp(mta_string_view(length_unit), "Angstrom") == 0);

mta_string_free(length_unit);

// %%
//
// Tensor data
// -----------
//
// The main tensors in the system (positions, cell, atomic types and pbc) can
// all be accessed with :c:func:`mta_system_get_data`, passing a different
// :c:enum:`mta_system_data_kind` for each tensor.

DLManagedTensorVersioned* positions = NULL;
status = mta_system_get_data(system, MTA_SYSTEM_DATA_POSITIONS, &positions);

// this tensor contains 64-bit floats
assert(positions->dl_tensor.dtype.code == kDLFloat);
assert(positions->dl_tensor.dtype.bits == 64);
assert(positions->dl_tensor.dtype.lanes == 1);

double* pos_ptr = (double*)(positions->dl_tensor.data + positions->dl_tensor.byte_offset);

assert(positions->dl_tensor.ndim == 2);
assert(positions->dl_tensor.shape[0] == 4);
assert(positions->dl_tensor.shape[1] == 3);

// check that the code is contiguous and row-major
if (positions->dl_tensor.strides != NULL) {
    assert(positions->dl_tensor.strides[0] == 3);
    assert(positions->dl_tensor.strides[1] == 1);
}

// Acces the data in a linear fashion
assert(pos_ptr[0] == 0.0 && pos_ptr[1] == 0.0 && pos_ptr[2] == 0.0);
assert(pos_ptr[3] == 0.5 && pos_ptr[4] == 0.5 && pos_ptr[5] == 0.0);
assert(pos_ptr[6] == 0.5 && pos_ptr[7] == 0.0 && pos_ptr[8] == 0.5);
assert(pos_ptr[9] == 0.0 && pos_ptr[10] == 0.5 && pos_ptr[11] == 0.5);

// %%
//
// When done with a dlpack tensor, one must release it
if (positions->deleter) {
    positions->deleter(positions);
}

// %%
//
// We can do something similar with the pbc data:

DLManagedTensorVersioned* pbc = NULL;
status = mta_system_get_data(system, MTA_SYSTEM_DATA_PBC, &pbc);

// this tensor contains booleans
assert(pbc->dl_tensor.dtype.code == kDLBool);
assert(pbc->dl_tensor.dtype.bits == 8);
assert(pbc->dl_tensor.dtype.lanes == 1);

bool* pbc_ptr = (bool*)(pbc->dl_tensor.data + pbc->dl_tensor.byte_offset);

assert(pbc->dl_tensor.ndim == 1);
assert(pbc->dl_tensor.shape[0] == 3);

// check that the code is contiguous and row-major
if (pbc->dl_tensor.strides != NULL) {
    assert(pbc->dl_tensor.strides[0] == 1);
}

// Acces the data in a linear fashion
assert(pbc_ptr[0] == true && pbc_ptr[1] == true && pbc_ptr[2] == true);

// release the dlpack tensor
if (pbc->deleter) {
    pbc->deleter(pbc);
}

// %%
//
// Pair lists
// ----------
//
// A pair list (neighbor list) is a :c:type:`mts_block_t` where each sample
// represents a pair of atoms, and the values contains the distance vector
// between them.
//
// The pair list is identified by its JSON-serialized pair list options, which
// are typically declared by a model through
// :c:func:`mta_model_t.requested_pair_lists`.

const char* pair_options =
    "{\"type\": \"metatomic_pair_options\","
    " \"cutoff\": \"0x4008000000000000\","
    " \"full_list\": true,"
    " \"strict\": false,"
    " \"requestors\": [\"tutorial\"]}";

// The block returned by this function is a borrowed view, do not free it.
const mts_block_t* pairs = NULL;
status = mta_system_get_pairs(system, pair_options, &pairs);
if (status != MTA_SUCCESS) {
    fprintf(stderr, "failed to get pairs from system\n");
    mta_system_free(system);
    return EXIT_FAILURE;
}
assert(pairs != NULL);
printf("successfully retrieved pair list\n");

// %%
//
// We can also list all known pair lists in the system:

mta_string_t known = NULL;
status = mta_system_known_pairs(system, &known);
if (status == MTA_SUCCESS && known != NULL) {
    printf("known pair lists: %s\n", mta_string_view(known));
    mta_string_free(known);
}

// %%
//
// Custom data
// -----------
//
// Custom data allows models to attach arbitrary per-system data to a system,
// stored as a named :c:type:`mts_tensormap_t`. The name must follow the usual
// quantity naming convention.


// The returned tensor map is a borrowed view, do not free it.
const mts_tensormap_t* retrieved = NULL;
status = mta_system_get_custom_data(system, "tutorial::charges", &retrieved);
if (status != MTA_SUCCESS) {
    fprintf(stderr, "failed to get custom data from system\n");
    mta_system_free(system);
    return EXIT_FAILURE;
}
assert(retrieved != NULL);
printf("successfully retrieved custom data 'tutorial::charges'\n");

// %%
//
// And we can list all known custom data names:

mta_string_t names = NULL;
status = mta_system_known_custom_data(system, &names);
if (status == MTA_SUCCESS && names != NULL) {
    printf("known custom data: %s\n", mta_string_view(names));
    mta_string_free(names);
}

// %%
//
// Once done, we can cleanup the system.

status = mta_system_free(system);
if (status != MTA_SUCCESS) {
    fprintf(stderr, "failed to free system memory\n");
    return EXIT_FAILURE;
};

return EXIT_SUCCESS; }
