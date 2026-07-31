#define MAX_NDIM 7

/// Multi-dimensional strided index (up to MAX_NDIM dimensions).
/// Decomposes a flat linear index into multi-dimensional coordinates from the
/// shape, then computes the strided memory offset using the stride array.
///
/// WARNING: any change here needs to be reflected in the Rust and Metal sources.
struct StridedNDIndex {
    int64_t ndim;
    int64_t shape[MAX_NDIM];
    int64_t strides[MAX_NDIM];

    /// Get the offset from the start of the array for a given flat index
    __device__ int64_t offset(int64_t flat_idx) const {
        int64_t off = 0;
        for (int d = this->ndim - 1; d >= 0; d--) {
            int64_t coord = flat_idx % this->shape[d];
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
    int64_t n,
    int* mismatch
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t value_offset = values_idx.offset(i);
        int64_t reference_offset = reference_idx.offset(i);
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
    int64_t n,
    double factor
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t offset = tensor_idx.offset(i);
        tensor[offset] = static_cast<T>(static_cast<double>(tensor[offset]) * factor);
    }
}

extern "C" __global__ void scale_f32(
    float* tensor,
    StridedNDIndex tensor_idx,
    int64_t n,
    double factor
) {
    scale_inplace_impl<float>(tensor, tensor_idx, n, factor);
}

extern "C" __global__ void scale_f64(
    double* tensor,
    StridedNDIndex tensor_idx,
    int64_t n,
    double factor
) {
    scale_inplace_impl<double>(tensor, tensor_idx, n, factor);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

extern "C" __global__ void check_atomic_types(
    const int* types,
    StridedNDIndex types_idx,
    int64_t n_atoms,
    const int* valid_types,
    int64_t n_valid_types,
    int* invalid_count
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_atoms) {
        int atom_type = types[types_idx.offset(i)];
        bool found = false;
        for (int64_t j = 0; j < n_valid_types; j++) {
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
