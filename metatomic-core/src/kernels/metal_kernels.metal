#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Multi-dimensional strided index helper (up to MAX_NDIM dimensions).
//
// Decomposes a flat linear index into multi-dimensional coordinates based on
// the shape and then computes the strided memory offset using the stride
// array.
//
// WARNING: the layout of this struct must match both the CUDA
// (cuda_kernels.cu) and Rust (kernels/mod.rs) definitions.
// ---------------------------------------------------------------------------
constant long MAX_NDIM [[maybe_unused]] = 7;

struct StridedNDIndex {
    long ndim;
    long shape[MAX_NDIM];
    long strides[MAX_NDIM];

    long offset(long flat_idx) const {
        long off = 0;
        for (int d = ndim - 1; d >= 0; d--) {
            long coord = flat_idx % shape[d];
            flat_idx /= shape[d];
            off += coord * strides[d];
        }
        return off;
    }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

kernel void is_equal_i32(
    [[buffer(0)]] device const int* values,
    [[buffer(1)]] constant StridedNDIndex& values_idx,
    [[buffer(2)]] device const int* reference,
    [[buffer(3)]] constant StridedNDIndex& reference_idx,
    [[buffer(4)]] constant uint64_t& n,
    [[buffer(5)]] device atomic_int* mismatch,
    [[thread_position_in_grid]] uint gid
) {
    if (gid < n) {
        long v_off = values_idx.offset(gid);
        long r_off = reference_idx.offset(gid);
        if (values[v_off] != reference[r_off]) {
            atomic_fetch_max_explicit(mismatch, 1, memory_order_relaxed);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Validate cell vectors against PBC flags (f32 only on Metal).
kernel void validate_cell_pbc_f32(
    [[buffer(0)]] device const bool* pbc,
    [[buffer(1)]] constant StridedNDIndex& pbc_idx,
    [[buffer(2)]] device const float* cell,
    [[buffer(3)]] constant StridedNDIndex& cell_idx,
    [[buffer(4)]] device atomic_int* mismatch_idx,
    [[thread_position_in_threadgroup]] uint tid
) {
    if (tid < 3) {
        if (!pbc[pbc_idx.offset(tid)]) {
            if (
                cell[cell_idx.offset(tid * 3 + 0)] != 0.0f ||
                cell[cell_idx.offset(tid * 3 + 1)] != 0.0f ||
                cell[cell_idx.offset(tid * 3 + 2)] != 0.0f
            ) {
                atomic_fetch_max_explicit(mismatch_idx, int(tid + 1), memory_order_relaxed);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Scale all elements of a tensor in place by `factor` (f32 only on Metal).
kernel void scale_f32(
    [[buffer(0)]] device float* tensor,
    [[buffer(1)]] constant StridedNDIndex& tensor_idx,
    [[buffer(2)]] constant uint64_t& n,
    [[buffer(3)]] constant float& factor,
    [[thread_position_in_grid]] uint gid
) {
    if (gid < n) {
        long offset = tensor_idx.offset(gid);
        tensor[offset] = tensor[offset] * factor;
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Check that all atomic types in `types` are present in `valid_types`.
/// Increments `invalid_count` for each invalid type found.
/// `invalid_count` is initialized to 0 by the caller.
kernel void check_atomic_types(
    [[buffer(0)]] device const int* types,
    [[buffer(1)]] constant StridedNDIndex& types_idx,
    [[buffer(2)]] constant uint64_t& n_atoms,
    [[buffer(3)]] device const int* valid_types,
    [[buffer(4)]] constant uint64_t& n_valid_types,
    [[buffer(5)]] device atomic_int* invalid_count,
    [[thread_position_in_grid]] uint gid
) {
    if (gid < n_atoms) {
        int atom_type = types[types_idx.offset(gid)];
        bool found = false;
        for (uint64_t j = 0; j < n_valid_types; j++) {
            if (valid_types[j] == atom_type) {
                found = true;
                break;
            }
        }
        if (!found) {
            atomic_fetch_add_explicit(invalid_count, 1, memory_order_relaxed);
        }
    }
}
