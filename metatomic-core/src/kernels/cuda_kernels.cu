typedef signed long long i64;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;


#define MAX_NDIM 7

/// Multi-dimensional strided index (up to MAX_NDIM dimensions).
/// Decomposes a flat linear index into multi-dimensional coordinates from the
/// shape, then computes the strided memory offset using the stride array.
///
/// WARNING: any change here needs to be reflected in the Rust and Metal sources.
struct StridedNDIndex {
    i64 ndim;
    i64 shape[MAX_NDIM];
    i64 strides[MAX_NDIM];

    /// Get the offset from the start of the array for a given flat index
    __device__ i64 offset(i64 flat_idx) const {
        i64 off = 0;
        for (int d = this->ndim - 1; d >= 0; d--) {
            i64 coord = flat_idx % this->shape[d];
            flat_idx /= this->shape[d];
            off += coord * this->strides[d];
        }
        return off;
    }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

extern "C" __global__ void is_equal_i32(
    const int* values,
    StridedNDIndex values_idx,
    const int* reference,
    StridedNDIndex reference_idx,
    i64 n,
    int* mismatch
) {
    i64 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        i64 value_offset = values_idx.offset(i);
        i64 reference_offset = reference_idx.offset(i);
        if (values[value_offset] != reference[reference_offset]) {
            atomicMax(mismatch, 1);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ void validate_cell_pbc_impl(
    const bool* pbc,
    StridedNDIndex pbc_idx,
    const T* cell,
    StridedNDIndex cell_idx,
    int* mismatch_idx
) {
    int i = threadIdx.x;
    if (i < 3) {
        if (!pbc[pbc_idx.offset(i)]) {
            if (
                cell[cell_idx.offset(i * 3 + 0)] != T(0) ||
                cell[cell_idx.offset(i * 3 + 1)] != T(0) ||
                cell[cell_idx.offset(i * 3 + 2)] != T(0)
            ) {
                atomicMax(mismatch_idx, i + 1);
            }
        }
    }
}

extern "C" __global__ void validate_cell_pbc_f32(
    const bool* pbc,
    StridedNDIndex pbc_idx,
    const float* cell,
    StridedNDIndex cell_idx,
    int* mismatch_idx
) {
    validate_cell_pbc_impl<float>(pbc, pbc_idx, cell, cell_idx, mismatch_idx);
}

extern "C" __global__ void validate_cell_pbc_f64(
    const bool* pbc,
    StridedNDIndex pbc_idx,
    const double* cell,
    StridedNDIndex cell_idx,
    int* mismatch_idx
) {
    validate_cell_pbc_impl<double>(pbc, pbc_idx, cell, cell_idx, mismatch_idx);
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ void scale_inplace_impl(
    T* tensor,
    StridedNDIndex tensor_idx,
    i64 n,
    double factor
) {
    i64 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        i64 offset = tensor_idx.offset(i);
        tensor[offset] = static_cast<T>(static_cast<double>(tensor[offset]) * factor);
    }
}

extern "C" __global__ void scale_f32(
    float* tensor,
    StridedNDIndex tensor_idx,
    i64 n,
    double factor
) {
    scale_inplace_impl<float>(tensor, tensor_idx, n, factor);
}

extern "C" __global__ void scale_f64(
    double* tensor,
    StridedNDIndex tensor_idx,
    i64 n,
    double factor
) {
    scale_inplace_impl<double>(tensor, tensor_idx, n, factor);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Copy `n` elements from `src` (which can use arbitrary strides, described by
/// `src_idx`) to `dst`, which must be able to store `n` contiguous elements.
///
/// The kernels below are instantiated for each element size instead of each
/// data type, since only the size of the elements matters when moving data
/// around.
template <typename T>
__device__ void copy_to_contiguous_impl(
    const T* src,
    StridedNDIndex src_idx,
    T* dst,
    i64 n
) {
    i64 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[src_idx.offset(i)];
    }
}

extern "C" __global__ void copy_to_contiguous_8bit(
    const u8* src,
    StridedNDIndex src_idx,
    u8* dst,
    i64 n
) {
    copy_to_contiguous_impl<u8>(src, src_idx, dst, n);
}

extern "C" __global__ void copy_to_contiguous_16bit(
    const u16* src,
    StridedNDIndex src_idx,
    u16* dst,
    i64 n
) {
    copy_to_contiguous_impl<u16>(src, src_idx, dst, n);
}

extern "C" __global__ void copy_to_contiguous_32bit(
    const u32* src,
    StridedNDIndex src_idx,
    u32* dst,
    i64 n
) {
    copy_to_contiguous_impl<u32>(src, src_idx, dst, n);
}

extern "C" __global__ void copy_to_contiguous_64bit(
    const u64* src,
    StridedNDIndex src_idx,
    u64* dst,
    i64 n
) {
    copy_to_contiguous_impl<u64>(src, src_idx, dst, n);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

extern "C" __global__ void check_atomic_types(
    const int* types,
    StridedNDIndex types_idx,
    i64 n_atoms,
    const int* valid_types,
    i64 n_valid_types,
    int* invalid_count
) {
    i64 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_atoms) {
        int atom_type = types[types_idx.offset(i)];
        bool found = false;
        for (i64 j = 0; j < n_valid_types; j++) {
            if (valid_types[j] == atom_type) {
                found = true;
                break;
            }
        }
        if (!found) {
            atomicAdd(invalid_count, 1);
        }
    }
}
