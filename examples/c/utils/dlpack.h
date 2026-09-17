#ifndef MTA_EXAMPLE_UTILS_DLPACK_H
#define MTA_EXAMPLE_UTILS_DLPACK_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <metatensor/dlpack/dlpack.h>

// Context for a DLPack tensor, which keeps the allocations for shape and
// strides.
typedef struct CustomDLPackContext {
    int64_t* shape;
    int64_t* strides;
} CustomDLPackContext;

// Deleter for a DLPack tensor, which frees the context and the tensor itself.
// We do not free the data buffer, as it is owned by the caller.
static inline void dlpack_deleter(DLManagedTensorVersioned *self) {
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

// Create a DLPack tensor from a flat data buffer. The tensor is created as a
// row-major, contiguous tensor on CPU, with the specified shape and data type.
// The caller owns the data buffer, and is responsible for freeing it after the
// tensor is no longer needed.
static inline DLManagedTensorVersioned* tensor_from_data(
    void *data,
    const int64_t *shape,
    int32_t ndim,
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

#endif // MTA_EXAMPLE_UTILS_DLPACK_H
