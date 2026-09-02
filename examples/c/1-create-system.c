// Creating ``mta_system_t``
// =========================
//
// When integrating metatomic into an existing simulation code, the atomic
// types, positions, cell, and periodic boundary conditions are usually
// already stored in memory as plain arrays — for example as ``double**``
// pointers. This example shows how to wrap such existing data into DLPack
// tensors, and use them to create a :c:type:`mta_system_t`.
//
// The same approach works for any data layout, as long as you can describe
// it with a DLPack tensor.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatomic.h>


// %%
//
// DLPack tensors
// --------------
//
// This tutorial shows a basic way to create DLPack tensors from existing data.
// You should also explore the corresponding documentation in the DLPack header
// file, which describes the full DLPack API and options.
//
// We get the dlpack header from the vendored version in the metatensor package,
// which is the same that metatomic uses internally. You can also bring your own
// copy of the DLPack header, or use the one from your framework (PyTorch,
// TensorFlow, …) as long at it is at least version 1.0.

#include <metatensor/dlpack/dlpack.h>

// %%
//
// We'll need a context to store the shape and strides of the DLPack tensor. The
// context is owned by the DLPack tensor, and will be freed when the tensor is
// freed. The whole DLManagedTensorVersioned is passed to a custom deleter
// function when the tensor is no longer needed, which can free the context and
// the tensor itself.

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

// %%
//
// We then define a helper function to create a DLPack tensor from a flat data
// buffer. The tensor is created as a `row-major, contiguous` tensor on CPU, with
// the specified shape and data type. The caller owns the data buffer, and is
// responsible for freeing it after the tensor is no longer needed.

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

    // copy the shape into a new buffer owned by the DLPack tensor.
    ctx->shape = malloc(ndim * sizeof(int64_t));
    ctx->strides = malloc(ndim * sizeof(int64_t));
    if (!ctx->shape || !ctx->strides) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
        return NULL;
    }
    memcpy(ctx->shape, shape, ndim * sizeof(int64_t));

    // set the strides to indicate a contiguous row-major tensor
    int64_t stride = 1;
    for (int32_t i = ndim - 1; i >= 0; i--) {
        ctx->strides[i] = stride;
        stride *= shape[i];
    }

    // Create the DLPack tensor
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

    // Set the flags to indicate that the tensor is read-only.
    tensor->flags = DLPACK_FLAG_BITMASK_READ_ONLY;

    tensor->dl_tensor.data = data;
    // offset in bytes from the beginning of the data buffer to the first
    // element of the tensor.
    tensor->dl_tensor.byte_offset = 0;

    // device the tensor is on. Here we use CPU, device 0 (the only CPU device).
    tensor->dl_tensor.device.device_type = kDLCPU;
    tensor->dl_tensor.device.device_id = 0;

    // data type of the tensor
    tensor->dl_tensor.dtype = dtype;

    // number of dimensions, shape, and strides, re-using the buffers we
    // allocated above.
    tensor->dl_tensor.ndim = ndim;
    tensor->dl_tensor.shape = ctx->shape;
    tensor->dl_tensor.strides = ctx->strides;

    return tensor;
}

// %%

int main(void) {

// %%
//
// Build the ``positions`` and ``cell`` tensor
// -------------------------------------------
//
// The positions and cell tensors can be either ``float32`` or ``float64``. They
// would typically wrap existing data from the simulation code, here we create
// them inline for demonstration purposes.

const int64_t n_atoms = 4;
double positions_data[] = {
    0.0, 0.0, 0.0,
    0.5, 0.5, 0.0,
    0.5, 0.0, 0.5,
    0.0, 0.5, 0.5,
};

double cell_data[] = {
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
};

DLManagedTensorVersioned* positions = tensor_from_data(
    /*data=*/ positions_data,
    /*ndim=*/ 2,
    /*shape=*/(int64_t[]){n_atoms, 3},
    /*dtype=*/(DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}
);

DLManagedTensorVersioned* cell = tensor_from_data(
    /*data=*/ cell_data,
    /*ndim=*/ 2,
    /*shape=*/(int64_t[]){3, 3},
    /*dtype=*/(DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1}
);

// %%
//
// Build the ``types`` tensor
// --------------------------
//
// Atomic types must be an ``int32`` tensor of shape ``(n_atoms,)``. Note that
// the atomic types are not necessarily the same as the atomic numbers, and can
// be any integer values that the model understands. It can be useful to let
// users provide a mapping from the atomic tags used in the simulation code to
// the atomic types used by the model.

int32_t types_data[] = {
    1, 1, 6, 6
};

DLManagedTensorVersioned *types = tensor_from_data(
    /*data=*/ types_data,
    /*ndim=*/ 1,
    /*shape=*/(int64_t[]){n_atoms},
    /*dtype=*/(DLDataType){.code = kDLInt, .bits = 32, .lanes = 1}
);

// %%
//
// Build the ``pbc`` tensor
// ------------------------
//
// Periodic boundary conditions are a ``bool`` tensor of shape ``(3,)``,
// one entry per axis. For a fully periodic system all three are ``true``.

bool pbc_data[] = {true, true, true};
int64_t pbc_shape[] = {3};
DLManagedTensorVersioned *pbc = tensor_from_data(
    /*data=*/ pbc_data,
    /*ndim=*/ 1,
    /*shape=*/(int64_t[]){3},
    /*dtype=*/(DLDataType){.code = kDLBool, .bits = 8, .lanes = 1}
);

// %%
//
// Create the system
// -----------------
//
// :c:func:`mta_system_create` takes ownership of the four DLPack tensors;
// they must not be used afterwards. The returned :c:type:`mta_system_t`
// must be freed with :c:func:`mta_system_free` once you are done with it.

mta_system_t* system = NULL;
mta_status_t status = mta_system_create(
    "Angstrom", types, positions, cell, pbc, &system
);

if (status != MTA_SUCCESS) {
    const char* error_message = NULL;
    mta_last_error(&error_message, /*origin=*/NULL, /*data=*/NULL);
    fprintf(stderr, "failed to create system: %s\n", error_message);
    return EXIT_FAILURE;
}

// %%
//
// Use the system
// --------------
//
// Now that we have a :c:type:`mta_system_t`, we can query its attributes with
// the rest of the metatomic API. A system always stores atomic types,
// positions, the cell, and periodic boundary conditions. Forces and energies
// are **not** part of the system: they are model outputs, obtained later by
// calling :c:func:`mta_execute_model`.
//
// Start by checking the number of atoms and the length unit:

uintptr_t size = 0;
status = mta_system_size(system, &size);
if (status != MTA_SUCCESS) {
    fprintf(stderr, "failed to get system size\n");
    mta_system_free(system);
    return EXIT_FAILURE;
}
printf("created system with %lu atoms\n", (unsigned long)size);

mta_string_t length_unit = NULL;
status = mta_system_get_length_unit(system, &length_unit);
if (status != MTA_SUCCESS) {
    fprintf(stderr, "failed to get length unit\n");
    mta_system_free(system);
    return EXIT_FAILURE;
}
printf("length unit: %s\n", mta_string_view(length_unit));
mta_string_free(length_unit);

// %%
//
// Query types, positions, cell, and PBC
// -------------------------------------
//
// :c:func:`mta_system_get_data` returns a **borrowed** DLPack view of the
// requested data. You must call the tensor's ``deleter`` when you are done;
// do not modify the underlying buffer.

DLManagedTensorVersioned* data = NULL;

status = mta_system_get_data(system, MTA_SYSTEM_DATA_TYPES, &data);
if (status != MTA_SUCCESS) {
    fprintf(stderr, "failed to get types\n");
    mta_system_free(system);
    return EXIT_FAILURE;
}
int32_t* types_view = (int32_t*)((char*)data->dl_tensor.data + data->dl_tensor.byte_offset);
printf("types:");
for (uintptr_t i = 0; i < size; i++) {
    printf(" %d", types_view[i]);
}
printf("\n");
data->deleter(data);

status = mta_system_get_data(system, MTA_SYSTEM_DATA_POSITIONS, &data);
if (status != MTA_SUCCESS) {
    fprintf(stderr, "failed to get positions\n");
    mta_system_free(system);
    return EXIT_FAILURE;
}
double* positions_view = (double*)((char*)data->dl_tensor.data + data->dl_tensor.byte_offset);
printf("positions:\n");
for (uintptr_t i = 0; i < size; i++) {
    printf("  %.3f %.3f %.3f\n",
           positions_view[3 * i + 0],
           positions_view[3 * i + 1],
           positions_view[3 * i + 2]);
}
data->deleter(data);

status = mta_system_get_data(system, MTA_SYSTEM_DATA_CELL, &data);
if (status != MTA_SUCCESS) {
    fprintf(stderr, "failed to get cell\n");
    mta_system_free(system);
    return EXIT_FAILURE;
}
double* cell_view = (double*)((char*)data->dl_tensor.data + data->dl_tensor.byte_offset);
printf("cell:\n");
for (int i = 0; i < 3; i++) {
    printf("  %.3f %.3f %.3f\n",
           cell_view[3 * i + 0],
           cell_view[3 * i + 1],
           cell_view[3 * i + 2]);
}
data->deleter(data);

status = mta_system_get_data(system, MTA_SYSTEM_DATA_PBC, &data);
if (status != MTA_SUCCESS) {
    fprintf(stderr, "failed to get pbc\n");
    mta_system_free(system);
    return EXIT_FAILURE;
}
uint8_t* pbc_view = (uint8_t*)((char*)data->dl_tensor.data + data->dl_tensor.byte_offset);
printf("pbc: %s %s %s\n",
       pbc_view[0] ? "true" : "false",
       pbc_view[1] ? "true" : "false",
       pbc_view[2] ? "true" : "false");
data->deleter(data);

// %%
//
// Running this program prints::
//
//     created system with 4 atoms
//     length unit: Angstrom
//     types: 1 1 6 6
//     positions:
//       0.000 0.000 0.000
//       0.500 0.500 0.000
//       0.500 0.000 0.500
//       0.000 0.500 0.500
//     cell:
//       1.000 0.000 0.000
//       0.000 1.000 0.000
//       0.000 0.000 1.000
//     pbc: true true true

// %%
//
// Cleanup
// -------
//
// Free the system once it is no longer needed. The DLPack tensors have already
// been consumed by ``mta_system_create`` and must not be freed again.

mta_system_free(system);

// %%

return EXIT_SUCCESS; }
