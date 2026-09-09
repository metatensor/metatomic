#ifndef METATOMIC_EXAMPLES_C_SIMPLE_ARRAY_H
#define METATOMIC_EXAMPLES_C_SIMPLE_ARRAY_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <metatensor.h>
#include <metatensor/dlpack/dlpack.h>

/* Stand-in for metatensor::SimpleDataArray used by the C++ C-API tests. */

#define TUTORIAL_ARRAY_MAX_DIM 3

typedef struct {
    void* data;
    uintptr_t shape[TUTORIAL_ARRAY_MAX_DIM];
    int64_t dl_shape[TUTORIAL_ARRAY_MAX_DIM];
    int64_t dl_strides[TUTORIAL_ARRAY_MAX_DIM];
    uintptr_t ndim;
    uint8_t dtype_code;
    uint8_t dtype_bits;
} TutorialArray;

static mts_data_origin_t tutorial_array_origin_id = 0;

static mts_status_t tutorial_array_fail(const char* fn) {
    mts_set_last_error(
        "not implemented on the tutorial SimpleDataArray stand-in",
        fn,
        NULL,
        NULL
    );
    return MTS_CALLBACK_ERROR;
}

static void tutorial_array_destroy(void* array) {
    TutorialArray* self = (TutorialArray*)array;
    if (self == NULL) {
        return;
    }
    free(self->data);
    free(self);
}

static mts_status_t tutorial_array_origin(const void* array, mts_data_origin_t* origin) {
    (void)array;
    if (tutorial_array_origin_id == 0) {
        if (mts_register_data_origin("c-tutorial-simple-array", &tutorial_array_origin_id)
            != MTS_SUCCESS) {
            return MTS_CALLBACK_ERROR;
        }
    }
    *origin = tutorial_array_origin_id;
    return MTS_SUCCESS;
}

static mts_status_t tutorial_array_device(const void* array, DLDevice* device) {
    (void)array;
    device->device_type = kDLCPU;
    device->device_id = 0;
    return MTS_SUCCESS;
}

static mts_status_t tutorial_array_dtype(const void* array, DLDataType* dtype) {
    const TutorialArray* self = (const TutorialArray*)array;
    dtype->code = self->dtype_code;
    dtype->bits = self->dtype_bits;
    dtype->lanes = 1;
    return MTS_SUCCESS;
}

static mts_status_t tutorial_array_shape(
    const void* array,
    const uintptr_t** shape,
    uintptr_t* shape_count
) {
    const TutorialArray* self = (const TutorialArray*)array;
    *shape = self->shape;
    *shape_count = self->ndim;
    return MTS_SUCCESS;
}

static void tutorial_dlpack_deleter(DLManagedTensorVersioned* tensor) {
    if (tensor == NULL) {
        return;
    }
    free(tensor->manager_ctx);
    free(tensor);
}

static mts_status_t tutorial_array_as_dlpack(
    void* array,
    DLManagedTensorVersioned** out,
    DLDevice device,
    const int64_t* stream,
    DLPackVersion max_version
) {
    (void)stream;
    (void)max_version;
    TutorialArray* self = (TutorialArray*)array;
    if (device.device_type != kDLCPU) {
        return tutorial_array_fail("as_dlpack");
    }

    DLManagedTensorVersioned* tensor = calloc(1, sizeof(*tensor));
    if (tensor == NULL) {
        mts_set_last_error("out of memory", "as_dlpack", NULL, NULL);
        return MTS_CALLBACK_ERROR;
    }

    tensor->version.major = DLPACK_MAJOR_VERSION;
    tensor->version.minor = DLPACK_MINOR_VERSION;
    tensor->deleter = tutorial_dlpack_deleter;
    tensor->dl_tensor.data = self->data;
    tensor->dl_tensor.device.device_type = kDLCPU;
    tensor->dl_tensor.device.device_id = 0;
    tensor->dl_tensor.ndim = (int32_t)self->ndim;
    tensor->dl_tensor.dtype.code = self->dtype_code;
    tensor->dl_tensor.dtype.bits = self->dtype_bits;
    tensor->dl_tensor.dtype.lanes = 1;
    tensor->dl_tensor.shape = self->dl_shape;
    tensor->dl_tensor.strides = self->dl_strides;
    *out = tensor;
    return MTS_SUCCESS;
}

static mts_status_t tutorial_array_from_dlpack(
    const void* array,
    DLManagedTensorVersioned* tensor,
    mts_array_t* new_array
) {
    (void)array;
    (void)tensor;
    (void)new_array;
    return tutorial_array_fail("from_dlpack");
}

static mts_status_t tutorial_array_reshape(
    void* array,
    const uintptr_t* shape,
    uintptr_t shape_count
) {
    (void)array;
    (void)shape;
    (void)shape_count;
    return tutorial_array_fail("reshape");
}

static mts_status_t tutorial_array_swap_axes(void* array, uintptr_t axis_1, uintptr_t axis_2) {
    (void)array;
    (void)axis_1;
    (void)axis_2;
    return tutorial_array_fail("swap_axes");
}

static mts_status_t tutorial_array_create(
    const void* array,
    const uintptr_t* shape,
    uintptr_t shape_count,
    mts_array_t fill_value,
    mts_array_t* new_array
) {
    (void)array;
    (void)shape;
    (void)shape_count;
    (void)fill_value;
    (void)new_array;
    return tutorial_array_fail("create");
}

static mts_status_t tutorial_array_copy(
    const void* array,
    DLDevice device,
    mts_array_t* new_array
) {
    (void)array;
    (void)device;
    (void)new_array;
    return tutorial_array_fail("copy");
}

static mts_status_t tutorial_array_move_data(
    void* output,
    const void* input,
    const mts_data_movement_t* movements,
    uintptr_t movements_count
) {
    (void)output;
    (void)input;
    (void)movements;
    (void)movements_count;
    return tutorial_array_fail("move_data");
}

static mts_array_t tutorial_array_nd(
    void* data,
    const uintptr_t* shape,
    uintptr_t ndim,
    uint8_t dtype_code,
    uint8_t dtype_bits
) {
    mts_array_t array;
    memset(&array, 0, sizeof(array));
    if (data == NULL || shape == NULL || ndim == 0 || ndim > TUTORIAL_ARRAY_MAX_DIM) {
        free(data);
        return array;
    }

    TutorialArray* self = calloc(1, sizeof(TutorialArray));
    if (self == NULL) {
        free(data);
        return array;
    }
    self->data = data;
    self->ndim = ndim;
    self->dtype_code = dtype_code;
    self->dtype_bits = dtype_bits;

    int64_t stride = 1;
    for (uintptr_t i = ndim; i > 0; i--) {
        uintptr_t dim = i - 1;
        self->shape[dim] = shape[dim];
        self->dl_shape[dim] = (int64_t)shape[dim];
        self->dl_strides[dim] = stride;
        stride *= (int64_t)shape[dim];
    }

    array.ptr = self;
    array.destroy = tutorial_array_destroy;
    array.origin = tutorial_array_origin;
    array.device = tutorial_array_device;
    array.dtype = tutorial_array_dtype;
    array.as_dlpack = tutorial_array_as_dlpack;
    array.from_dlpack = tutorial_array_from_dlpack;
    array.shape = tutorial_array_shape;
    array.reshape = tutorial_array_reshape;
    array.swap_axes = tutorial_array_swap_axes;
    array.create = tutorial_array_create;
    array.copy = tutorial_array_copy;
    array.move_data = tutorial_array_move_data;
    return array;
}

static mts_array_t tutorial_array_own(
    void* data,
    uintptr_t rows,
    uintptr_t cols,
    uint8_t dtype_code,
    uint8_t dtype_bits
) {
    uintptr_t shape[2] = {rows, cols};
    return tutorial_array_nd(data, shape, 2, dtype_code, dtype_bits);
}

static const mts_labels_t* tutorial_labels(
    const char* const* names,
    uintptr_t n_names,
    const int32_t* values,
    uintptr_t n_rows
) {
    size_t n = (size_t)n_rows * (size_t)n_names;
    int32_t* copy = malloc(n * sizeof(int32_t));
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, values, n * sizeof(int32_t));
    mts_array_t array = tutorial_array_own(copy, n_rows, n_names, kDLInt, 32);
    if (array.ptr == NULL) {
        return NULL;
    }
    const mts_labels_t* labels = mts_labels(names, n_names, array);
    if (labels == NULL && array.destroy != NULL) {
        array.destroy(array.ptr);
    }
    return labels;
}

static const mts_labels_t* labels_single_zero(const char* name) {
    int32_t zero = 0;
    const char* names[] = {name};
    return tutorial_labels(names, 1, &zero, 1);
}

typedef struct {
    int64_t* shape;
    int64_t* strides;
} TutorialSystemDLPackContext;

static void tutorial_system_dlpack_deleter(DLManagedTensorVersioned* self) {
    if (self == NULL) {
        return;
    }
    TutorialSystemDLPackContext* ctx = (TutorialSystemDLPackContext*)self->manager_ctx;
    if (ctx != NULL) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
    }
    free(self);
}

/* Wrap an existing buffer as a CPU DLPack tensor. The caller keeps the buffer. */
static DLManagedTensorVersioned* tutorial_dlpack_view(
    void* data,
    int32_t ndim,
    const int64_t* shape,
    DLDataType dtype
) {
    TutorialSystemDLPackContext* ctx = malloc(sizeof(*ctx));
    if (ctx == NULL) {
        return NULL;
    }
    ctx->shape = malloc((size_t)ndim * sizeof(int64_t));
    ctx->strides = malloc((size_t)ndim * sizeof(int64_t));
    if (ctx->shape == NULL || ctx->strides == NULL) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
        return NULL;
    }
    memcpy(ctx->shape, shape, (size_t)ndim * sizeof(int64_t));
    int64_t stride = 1;
    for (int32_t i = ndim - 1; i >= 0; i--) {
        ctx->strides[i] = stride;
        stride *= shape[i];
    }

    DLManagedTensorVersioned* tensor = calloc(1, sizeof(*tensor));
    if (tensor == NULL) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
        return NULL;
    }
    tensor->version.major = DLPACK_MAJOR_VERSION;
    tensor->version.minor = DLPACK_MINOR_VERSION;
    tensor->manager_ctx = ctx;
    tensor->deleter = tutorial_system_dlpack_deleter;
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

#endif /* METATOMIC_EXAMPLES_C_SIMPLE_ARRAY_H */
