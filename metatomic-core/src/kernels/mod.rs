use std::sync::{Arc, OnceLock};

use cudarc::driver::safe::{CudaContext, CudaStream, DeviceRepr};
use cudarc::driver::CudaSlice;
use dlpk::sys::DLDeviceType;
use dlpk::{DLPackTensorRef, DLPackTensorRefMut};
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
    pub(crate) fn from_dlpack(tensor: DLPackTensorRef<'_>) -> Self {
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

type CudaArray<T> = (CudaSlice<T>, StridedNDIndex);
#[cfg(target_os = "macos")]
type MetalArray = (metal::MetalBuffer, StridedNDIndex);

/// Store and cache reference values for different backends (CPU, CUDA, Metal).
///
/// The CPU copy is always present. Device-resident copies are lazily uploaded
/// on first use per device: the outer `OnceLock` initializes a `Vec` with one
/// entry per device (sized from the device count), and each inner `OnceLock`
/// is independently initialized on first access for that specific device.
pub struct ReferenceValue<T> {
    /// The reference values stored on the CPU, always there
    pub(crate) cpu: ArrayD<T>,
    /// Reference values stored on CUDA, one `OnceLock` per device (lazily sized)
    pub(crate) cuda: OnceLock<Vec<OnceLock<CudaArray<T>>>>,
    #[cfg(target_os = "macos")]
    /// Reference values stored on Metal, one `OnceLock` per device (lazily sized)
    pub(crate) metal: OnceLock<Vec<OnceLock<MetalArray>>>,
}

impl std::fmt::Debug for ReferenceValue<i32> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReferenceValue")
            .field("cpu", &self.cpu)
            .finish()
    }
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

impl<T: DeviceRepr> ReferenceValue<T> {
    /// Get the CUDA-resident copy of the reference values for `device_id`,
    /// uploading from CPU on first use for this device.
    ///
    /// The returned reference is tied to `&self` and valid for the lifetime of
    /// this `ReferenceValue`.
    pub(crate) fn cuda_data(
        &self,
        device_id: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<&(CudaSlice<T>, StridedNDIndex), Error> {
        let entries = self.cuda.get_or_init(|| {
            let count = usize::try_from(CudaContext::device_count().unwrap_or(0)).expect("got negative device count");
            (0..count).map(|_| OnceLock::new()).collect()
        });

        if device_id >= entries.len() {
            return Err(Error::Internal(format!(
                "CUDA device {device_id} does not exist (only {} devices available)",
                entries.len()
            )));
        }

        Ok(entries[device_id].get_or_init(|| {
            let slice = stream
                .clone_htod(self.cpu.as_slice().expect("reference should be contiguous"))
                .expect("clone_htod reference failed");
            let idx = StridedNDIndex::from_ndarray(&self.cpu.view());
            (slice, idx)
        }))
    }
}

#[cfg(target_os = "macos")]
impl<T> ReferenceValue<T> {
    /// Get the Metal-resident copy of the reference values for `device_id`,
    /// uploading from CPU on first use for this device.
    ///
    /// The returned reference is tied to `&self` and valid for the lifetime of
    /// this `ReferenceValue`.
    pub(crate) fn metal_data(
        &self,
        device_id: usize,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
    ) -> Result<&(metal::MetalBuffer, StridedNDIndex), Error> {
        use objc2_metal::{MTLCopyAllDevices, MTLDevice};

        let entries = self.metal.get_or_init(|| {
            let count = MTLCopyAllDevices().count();
            (0..count).map(|_| OnceLock::new()).collect()
        });

        if device_id >= entries.len() {
            return Err(Error::Internal(format!(
                "Metal device {device_id} does not exist (only {} devices available)",
                entries.len()
            )));
        }

        Ok(entries[device_id].get_or_init(|| {
            let ref_bytes = self.cpu.len() * std::mem::size_of::<T>();
            let ref_ptr: *const std::ffi::c_void = self.cpu
                .as_slice()
                .expect("reference should be contiguous")
                .as_ptr()
                .cast();
            let buf = unsafe {
                use std::ptr::NonNull;
                use objc2_metal::MTLResourceOptions;
                device.newBufferWithBytes_length_options(
                    NonNull::new(ref_ptr.cast_mut()).expect("reference pointer must not be null"),
                    ref_bytes,
                    MTLResourceOptions::empty(),
                ).expect("failed to create reference buffer")
            };
            let idx = StridedNDIndex::from_ndarray(&self.cpu.view());
            (metal::MetalBuffer(buf), idx)
        }))
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

/// Scale all elements of `tensor` in place by `factor`.
///
/// This dispatches to the appropriate backend based on the device of `tensor`.
/// Only 32-bit and 64-bit floating point tensors are supported.
///
/// # Parameters
/// - `tensor`: a mutable DLPack tensor with f32 or f64 data type
/// - `factor`: the multiplicative factor to apply to every element
pub(crate) fn scale_inplace(tensor: DLPackTensorRefMut<'_>, factor: f64) -> Result<(), Error> {
    match tensor.device().device_type {
        DLDeviceType::kDLCPU | DLDeviceType::kDLCUDAHost | DLDeviceType::kDLROCMHost => {
            cpu::scale_inplace(tensor, factor)
        }
        DLDeviceType::kDLCUDA | DLDeviceType::kDLCUDAManaged => {
            cuda::scale_inplace(tensor, factor)
        }
        DLDeviceType::kDLMetal => {
            #[cfg(target_os = "macos")] {
                metal::scale_inplace(tensor, factor)
            }
            #[cfg(not(target_os = "macos"))] {
                Err(Error::Internal(
                    "Metal backend is only available on macOS".into(),
                ))
            }
        }
        _ => {
            Err(Error::Internal(format!(
                "scale_inplace is not implemented for device {:?}",
                tensor.device()
            )))
        }
    }
}
