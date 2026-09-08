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
};

/// Get the offset from the start of the array for a given flat index.
///
/// This is a free function instead of a member function of `StridedNDIndex`,
/// since MSL does not allow calling member functions on objects living in the
/// `constant` address space.
static long strided_offset(constant StridedNDIndex& index, long flat_idx) {
    long off = 0;
    for (int d = index.ndim - 1; d >= 0; d--) {
        long coord = flat_idx % index.shape[d];
        flat_idx /= index.shape[d];
        off += coord * index.strides[d];
    }
    return off;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

kernel void is_equal_i32(
    [[buffer(0)]] device const int* values,
    [[buffer(1)]] constant StridedNDIndex& values_idx,
    [[buffer(2)]] device const int* reference,
    [[buffer(3)]] constant StridedNDIndex& reference_idx,
    [[buffer(4)]] constant uint& n,
    [[buffer(5)]] device atomic_int* mismatch,
    [[thread_position_in_grid]] uint gid
) {
    if (gid < n) {
        long v_off = strided_offset(values_idx, gid);
        long r_off = strided_offset(reference_idx, gid);
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
        if (!pbc[strided_offset(pbc_idx, tid)]) {
            if (
                cell[strided_offset(cell_idx, tid * 3 + 0)] != 0.0f ||
                cell[strided_offset(cell_idx, tid * 3 + 1)] != 0.0f ||
                cell[strided_offset(cell_idx, tid * 3 + 2)] != 0.0f
            ) {
                atomic_fetch_max_explicit(mismatch_idx, int(tid + 1), memory_order_relaxed);
            }
        }
    }
}
