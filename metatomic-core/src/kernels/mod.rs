use std::sync::OnceLock;

use cudarc::driver::CudaSlice;
use dlpk::sys::DLDeviceType;
use dlpk::DLPackTensorRef;
use ndarray::{ArrayD, ArrayViewD};

use crate::Error;

mod cpu;
mod cuda;

#[cfg(target_os = "macos")]
mod metal;

const MAX_NDIM: usize = 7;

/// Multi-dimensional strided index (up to MAX_NDIM dimensions).
///
/// Decomposes a flat linear index into multi-dimensional coordinates from the
/// shape, then computes the strided memory offset using the stride array.
///
/// WARNING: any change here needs to be reflected in the CUDA and Metal sources.
#[repr(C)]
pub(crate) struct StridedNDIndex {
    pub(crate) ndim: i64,
    pub(crate) shape: [i64; MAX_NDIM],
    pub(crate) strides: [i64; MAX_NDIM],
}

#[allow(clippy::cast_possible_wrap)]
impl StridedNDIndex {
    /// Create a `StridedNDIndex` from a DLPack tensor's shape and strides.
    pub(crate) fn from_dlpack(tensor: &DLPackTensorRef<'_>) -> Self {
        Self::from_shape_strides(tensor.shape(), tensor.strides())
    }

    /// Create a `StridedNDIndex` from an ndarray view's shape and strides.
    pub(crate) fn from_ndarray<T>(array: &ArrayViewD<'_, T>) -> Self {
        let shape: Vec<i64> = array.shape().iter().map(|&s| s as i64).collect();
        let strides: Vec<i64> = array.strides().iter().map(|&s| s as i64).collect();
        Self::from_shape_strides(&shape, Some(&strides))
    }

    /// Create a `StridedNDIndex` from shape and optional strides.
    ///
    /// If strides is `None`, the strides are computed as if the array were
    /// contiguous (row-major / C-contiguous).
    pub(crate) fn from_shape_strides(shape: &[i64], strides: Option<&[i64]>) -> Self {
        let ndim = shape.len();
        assert!(
            ndim <= MAX_NDIM,
            "StridedNDIndex only supports up to {MAX_NDIM} dimensions, got {ndim}"
        );
        let mut shape_arr = [0i64; MAX_NDIM];
        let mut strides_arr = [0i64; MAX_NDIM];

        // Contiguous fallback strides (row-major / C-contiguous)
        let mut acc: i64 = 1;
        for i in (0..ndim).rev() {
            shape_arr[i] = shape[i];
            strides_arr[i] = acc;
            acc *= shape[i];
        }

        if let Some(strides) = strides {
            strides_arr[..ndim].copy_from_slice(&strides[..ndim]);
        }
        StridedNDIndex { ndim: ndim as i64, shape: shape_arr, strides: strides_arr }
    }
}

/// Store and cache reference values for different backends (CPU, CUDA, Metal).
pub struct ReferenceValue<T> {
    /// The reference values stored on the CPU, always there
    pub(crate) cpu: ArrayD<T>,
    /// Reference values stored on CUDA, intialized on first use from the CPU values
    pub(crate) cuda: OnceLock<(CudaSlice<T>, StridedNDIndex)>,
    #[cfg(target_os = "macos")]
    /// Reference values stored on Metal, intialized on first use from the CPU values
    pub(crate) metal: OnceLock<(metal::MetalBuffer, StridedNDIndex)>,
}

impl<T> ReferenceValue<T> {
    pub(crate) fn new(cpu: ArrayD<T>) -> Self {
        Self {
            cpu,
            cuda: OnceLock::new(),
            #[cfg(target_os = "macos")]
            metal: OnceLock::new(),
        }
    }
}

/// Check that the values of an i32 DLPack tensor match the expected reference.
///
/// This dispatches to the appropriate backend based on the device of `tensor`.
///
/// # Parameters
/// - `tensor`: DLPack tensor with i32 data type
/// - `reference`: expected values with the same shape as the tensor
pub(crate) fn is_equal_i32(tensor: DLPackTensorRef<'_>, reference: &ReferenceValue<i32>) -> Result<bool, Error> {
    match tensor.device().device_type {
        DLDeviceType::kDLCPU | DLDeviceType::kDLCUDAHost | DLDeviceType::kDLROCMHost => {
            cpu::is_equal_i32(tensor, reference)
        }
        DLDeviceType::kDLCUDA | DLDeviceType::kDLCUDAManaged => {
            cuda::is_equal_i32(tensor, reference)
        }
        DLDeviceType::kDLMetal => {
            #[cfg(target_os = "macos")] {
                metal::is_equal_i32(tensor, reference)
            }
            #[cfg(not(target_os = "macos"))] {
                Err(Error::Internal(
                    "Metal backend is only available on macOS".into(),
                ))
            }
        }
        _ => {
            eprintln!(
                "is_equal_i32 for device {:?} is not implemented",
                tensor.device()
            );
            Ok(true)
        }
    }
}

/// Validate that cell vectors are zero for non-periodic dimensions.
///
/// This dispatches to the appropriate backend based on the device of `pbc`.
///
/// # Parameters
/// - `pbc`: 1D boolean tensor of length 3 (periodic boundary condition flags)
/// - `cell`: 3x3 tensor (unit cell vectors as rows)
pub(crate) fn validate_cell_pbc(pbc: DLPackTensorRef<'_>, cell: DLPackTensorRef<'_>) -> Result<(), Error> {
    debug_assert!(
        pbc.device() == cell.device(),
        "pbc and cell must be on the same device"
    );

    match pbc.device().device_type {
        DLDeviceType::kDLCPU | DLDeviceType::kDLCUDAHost | DLDeviceType::kDLROCMHost => {
            cpu::validate_cell_pbc(pbc, cell)
        }
        DLDeviceType::kDLCUDA | DLDeviceType::kDLCUDAManaged => {
            cuda::validate_cell_pbc(pbc, cell)
        }
        DLDeviceType::kDLMetal => {
            #[cfg(target_os = "macos")] {
                metal::validate_cell_pbc(pbc, cell)
            }
            #[cfg(not(target_os = "macos"))] {
                Err(Error::Internal(
                    "Metal backend is only available on macOS".into(),
                ))
            }
        }
        _ => {
            eprintln!(
                "Cell/PBC validation for device {:?} is not implemented",
                pbc.device()
            );
            Ok(())
        }
    }
}
