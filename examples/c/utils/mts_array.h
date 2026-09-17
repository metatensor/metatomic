#ifndef MTA_EXAMPLE_UTILS_ARRAY_H
#define MTA_EXAMPLE_UTILS_ARRAY_H

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <metatensor.h>

#include "./dlpack.h"

// To work with metatensor's ``mts_block_t`` and ``mts_tensormap_t`` in C, we
// need an ``mts_array_t`` - a vtable-based abstraction over n-dimensional
// arrays. Below is a minimal implementation backed by a flat data buffer that
// the array owns: the data is copied into a heap allocation when the array is
// created, and released when the array is destroyed.

// Context data for the array, which is stored in the ``ptr`` field of the
// ``mts_array_t``. The context is owned by the array, and will be freed when
// the array is destroyed.
typedef struct BasicMtsArray {
    void* data;
    uintptr_t ndim;
    uintptr_t shape[4];
    DLDataType dtype;
} BasicMtsArray;

// Destroy the array, freeing the data buffer and the context itself.
static inline void array_destroy(void* array) {
    BasicMtsArray* a = (BasicMtsArray*)array;
    free(a->data);
    free(a);
}

// Return the origin of the array.
static inline mts_status_t array_origin(const void* array, mts_data_origin_t* origin) {
    static mts_data_origin_t BASIC_MTS_ARRAY_ORIGIN = 0;

    (void)array;
    if (BASIC_MTS_ARRAY_ORIGIN == 0) {
        mts_register_data_origin("tutorial-mts-array", &BASIC_MTS_ARRAY_ORIGIN);
    }
    *origin = BASIC_MTS_ARRAY_ORIGIN;
    return MTS_SUCCESS;
}

// Return the device of the array.
static inline mts_status_t array_device(const void* array, DLDevice* device) {
    (void)array;
    device->device_type = kDLCPU;
    device->device_id = 0;
    return MTS_SUCCESS;
}

// Return the data type of the array.
static inline mts_status_t array_dtype(const void* array, DLDataType* dtype) {
    *dtype = ((const BasicMtsArray*)array)->dtype;
    return MTS_SUCCESS;
}

// Return the array as a DLPack tensor. The DLpack tensor must be released with
// `tensor->deleter(tensor)` when done.
static inline mts_status_t array_as_dlpack(
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
    static_assert(sizeof(uintptr_t) == sizeof(int64_t), "int64_t and uintptr_t must be the same size");
    *tensor = tensor_from_data(a->data, (const int64_t*)a->shape, (int32_t)a->ndim, a->dtype);
    return MTS_SUCCESS;
}

// Get the shape of the array.
static inline mts_status_t array_shape(
    const void* array,
    const uintptr_t** shape,
    uintptr_t* shape_count
) {
    const BasicMtsArray* a = (const BasicMtsArray*)array;
    *shape = a->shape;
    *shape_count = a->ndim;
    return MTS_SUCCESS;
}

// Create a new ``mts_array_t`` backed by a flat data buffer. The array will
// copy the data into a heap allocation, and will free it when the array is
// destroyed. The array will have the specified shape and data type.
static inline struct mts_array_t make_mts_array(
    const void* data, const uintptr_t* shape, uintptr_t ndim, DLDataType dtype
) {
    BasicMtsArray* raw = malloc(sizeof(BasicMtsArray));

    // copy the data into a buffer owned by the array
    size_t data_size = 1;
    for (uintptr_t i = 0; i < ndim; i++) {
        data_size *= shape[i];
    }
    data_size *= (dtype.bits / 8);
    raw->data = malloc(data_size);
    memcpy(raw->data, data, data_size);

    raw->ndim = ndim;
    for (uintptr_t i = 0; i < ndim; i++) {
        raw->shape[i] = shape[i];
    }
    raw->dtype = dtype;

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

#endif // MTA_EXAMPLE_UTILS_ARRAY_H
