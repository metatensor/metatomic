use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, LazyLock};

use cudarc::driver::safe::DeviceRepr;
use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use dlpk::DLPackTensorRef;

use crate::Error;
use super::{ReferenceValue, StridedNDIndex};

// CUDA kernel source compiled at runtime via NVRTC for the exact GPU
const KERNEL_SRC: &str = include_str!("cuda_kernels.cu");

unsafe impl DeviceRepr for StridedNDIndex {}

/// Zero-cost wrapper to pass an existing device pointer as a CUDA kernel
/// argument.
///
/// Does NOT own the memory — the caller (DLPack tensor) is responsible for
/// lifetime and must ensure the pointer remains valid for the duration of the
/// kernel launch.
///
/// The `#[repr(transparent)]` wrapper over `cudarc::driver::sys::CUdeviceptr`
/// is passed to `PushKernelArg::arg()` which pushes the address of this struct
/// on the host stack. CUDA reads 8 bytes from that address as the kernel
/// parameter value, giving the kernel the correct device pointer.
#[repr(transparent)]
struct DevicePtrArg {
    ptr: cudarc::driver::sys::CUdeviceptr,
}

unsafe impl DeviceRepr for DevicePtrArg {}

/// Per-device cached resources: context, module, and kernel function handles.
struct CudaKernelCache {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    is_equal_i32: CudaFunction,
    validate_cell_pbc_f32: CudaFunction,
    validate_cell_pbc_f64: CudaFunction,
}

impl CudaKernelCache {
    fn new(device_id: usize) -> Result<Self, Error> {
        let ctx = CudaContext::new(device_id)
            .map_err(|e| Error::Internal(format!("CudaContext::new({device_id}): {e}")))?;
        let ptx = compile_ptx(KERNEL_SRC)
            .map_err(|e| Error::Internal(format!("NVRTC compile failed: {e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| Error::Internal(format!("PTX load failed: {e}")))?;
        let is_equal_i32 = module
            .load_function("is_equal_i32")
            .map_err(|e| Error::Internal(format!("load_function(is_equal_i32): {e}")))?;
        let validate_cell_pbc_f32 = module
            .load_function("validate_cell_pbc_f32")
            .map_err(|e| Error::Internal(format!("load_function(validate_cell_pbc_f32): {e}")))?;
        let validate_cell_pbc_f64 = module
            .load_function("validate_cell_pbc_f64")
            .map_err(|e| Error::Internal(format!("load_function(validate_cell_pbc_f64): {e}")))?;
        Ok(Self {
            ctx,
            module,
            is_equal_i32,
            validate_cell_pbc_f32,
            validate_cell_pbc_f64,
        })
    }
}

static CUDA_CACHE: LazyLock<Mutex<HashMap<usize, CudaKernelCache>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn get_or_init(device_id: usize) -> Result<Arc<CudaStream>, Error> {
    let mut cache = CUDA_CACHE.lock().expect("failed to lock CUDA_CACHE");
    let entry = match cache.entry(device_id) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => entry.insert(CudaKernelCache::new(device_id)?),
    };
    Ok(entry.ctx.default_stream())
}

/// Extract a `CUdeviceptr` from a DLPack tensor's raw `data` + `byte_offset`.
///
/// # Safety
///
/// The returned `CUdeviceptr` is only valid as long as the DLPack tensor's
/// backing memory is alive. The caller must ensure the tensor is not dropped
/// before the kernel finishes execution.
unsafe fn dlpack_to_device_ptr(tensor: &DLPackTensorRef<'_>) -> cudarc::driver::sys::CUdeviceptr {
    debug_assert!(
        tensor.device().device_type == dlpk::sys::DLDeviceType::kDLCUDA,
        "dlpack_to_device_ptr called on non-CUDA tensor"
    );
    let raw_ptr = tensor.raw.data as u64;
    (raw_ptr + tensor.raw.byte_offset) as cudarc::driver::sys::CUdeviceptr
}

/// Check that the values of a CUDA-resident i32 DLPack tensor match an expected
/// reference array.
///
/// The comparison is performed entirely on-device: the existing GPU pointer
/// from `tensor` is wrapped as a `DevicePtrArg`, the reference is uploaded to
/// the GPU (and cached for subsequent calls), and a single-element result flag
/// (`0` = ok, `1` = mismatch) is read back.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub(crate) fn is_equal_i32(tensor: DLPackTensorRef<'_>, reference: &ReferenceValue<i32>) -> Result<bool, Error> {
    debug_assert!(
        tensor.device().device_type == dlpk::sys::DLDeviceType::kDLCUDA,
        "is_equal_i32 called on non-CUDA tensor"
    );
    debug_assert!(tensor.device().device_id >= 0, "is_equal_i32 called on invalid device_id");

    let device_id = tensor.device().device_id as usize;
    let stream = get_or_init(device_id)?;
    let cache = CUDA_CACHE.lock().expect("failed to lock CUDA_CACHE");
    let entry = &cache[&device_id];

    let n_elements: i64 = tensor.shape().iter().product();

    // Build strided index from the DLPack tensor (preserves actual strides)
    let values_idx = StridedNDIndex::from_dlpack(&tensor);

    // Wrap the existing GPU-allocated tensor pointer
    let tensor_ptr = unsafe { DevicePtrArg { ptr: dlpack_to_device_ptr(&tensor) } };

    // Upload reference values to GPU (cached after first call, per device)
    let (ref_dev, reference_idx) = reference.cuda_data(device_id, &stream)?;

    // Allocate result flag (initialized to 0 = no mismatch)
    let mut result = stream.alloc_zeros::<i32>(1)
        .map_err(|e| Error::Internal(format!("alloc_zeros: {e}")))?;

    unsafe {
        stream.launch_builder(&entry.is_equal_i32)
            .arg(&tensor_ptr)
            .arg(&values_idx)
            .arg(ref_dev)
            .arg(reference_idx)
            .arg(&n_elements)
            .arg(&mut result)
            .launch(LaunchConfig::for_num_elems(u32::try_from(n_elements).expect("tensor too large for CUDA kernel")))
            .map_err(|e| Error::Internal(format!("kernel launch (is_equal_i32): {e}")))?;
    }

    stream.synchronize()
        .map_err(|e| Error::Internal(format!("device sync: {e}")))?;

    let host = stream.clone_dtoh(&result)
        .map_err(|e| Error::Internal(format!("clone_dtoh result: {e}")))?;

    return Ok(host[0] == 0);
}

/// Validate that cell vectors are zero for non-periodic dimensions, on CUDA device.
#[allow(clippy::cast_sign_loss)]
pub(crate) fn validate_cell_pbc(
    pbc: DLPackTensorRef<'_>,
    cell: DLPackTensorRef<'_>,
) -> Result<(), Error> {
    debug_assert!(
        pbc.device().device_type == dlpk::sys::DLDeviceType::kDLCUDA,
        "validate_cell_pbc called on non-CUDA tensor"
    );
    debug_assert!(pbc.device().device_id >= 0, "validate_cell_pbc called on invalid device_id");
    debug_assert!(cell.device() == pbc.device(), "pbc and cell must be on the same device");


    let device_id = pbc.device().device_id as usize;
    let stream = get_or_init(device_id)?;
    let cache = CUDA_CACHE.lock().expect("failed to lock CUDA_CACHE");
    let entry = &cache[&device_id];

    let pbc_ptr = unsafe { DevicePtrArg { ptr: dlpack_to_device_ptr(&pbc) } };
    let cell_ptr = unsafe { DevicePtrArg { ptr: dlpack_to_device_ptr(&cell) } };

    let pbc_idx = StridedNDIndex::from_dlpack(&pbc);
    let cell_idx = StridedNDIndex::from_dlpack(&cell);

    let mut result = stream.alloc_zeros::<i32>(1)
        .map_err(|e| Error::Internal(format!("alloc_zeros: {e}")))?;

    if cell.dtype().bits == 32 {
        unsafe {
            stream.launch_builder(&entry.validate_cell_pbc_f32)
                .arg(&pbc_ptr)
                .arg(&pbc_idx)
                .arg(&cell_ptr)
                .arg(&cell_idx)
                .arg(&mut result)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (3, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| Error::Internal(format!("kernel launch (f32): {e}")))?;
        }
    } else {
        assert_eq!(cell.dtype().bits, 64, "validate_cell_pbc: unsupported cell dtype");
        unsafe {
            stream.launch_builder(&entry.validate_cell_pbc_f64)
                .arg(&pbc_ptr)
                .arg(&pbc_idx)
                .arg(&cell_ptr)
                .arg(&cell_idx)
                .arg(&mut result)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (3, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| Error::Internal(format!("kernel launch (f64): {e}")))?;
        }
    }

    stream.synchronize()
        .map_err(|e| Error::Internal(format!("device sync: {e}")))?;

    let host = stream.clone_dtoh(&result)
        .map_err(|e| Error::Internal(format!("clone_dtoh result: {e}")))?;

    if host[0] != 0 {
        let dim = host[0] - 1;
        return Err(Error::InvalidParameter(format!(
            "invalid cell: for non-periodic dimensions, the corresponding \
             cell vector must be zero, but cell[{}] contains non-zero values",
            dim
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use dlpk::{DLDevice, GetDLPackDataType};
    use ndarray::ArrayD;

    /// Check whether a CUDA device is available on this machine. The tests
    /// below are skipped when there is none.
    fn cuda_available() -> bool {
        // this requires the `dynamic-loading` feature of cudarc, without which
        // the tests would fail to link on machines without CUDA anyway
        if !unsafe {cudarc::driver::sys::is_culib_present() } {
            return false;
        }
        return CudaContext::device_count().unwrap_or(0) > 0;
    }

    macro_rules! skip_without_cuda {
        () => {
            if !cuda_available() {
                eprintln!("no CUDA device available, skipping this test");
                return;
            }
        };
    }

    /// A DLPack tensor with CUDA-resident data, used to test the kernels above.
    struct CudaTensor {
        ptr: cudarc::driver::sys::CUdeviceptr,
        shape: Vec<i64>,
        strides: Vec<i64>,
        dtype: dlpk::sys::DLDataType,
    }

    impl CudaTensor {
        /// Create a new CUDA tensor with the given `shape` and `strides`,
        /// containing a copy of `data`.
        ///
        /// `data` is the full memory span of the tensor, including any gap
        /// between the elements actually part of the tensor.
        fn new<T: GetDLPackDataType + Copy>(data: &[T], shape: &[i64], strides: &[i64]) -> Self {
            let stream = get_or_init(0).expect("failed to initialize CUDA device 0");
            stream.context().bind_to_thread().expect("bind_to_thread failed");

            assert!(!data.is_empty());
            let ptr = unsafe {
                cudarc::driver::result::malloc_sync(std::mem::size_of_val(data))
            }.expect("malloc_sync failed");

            if !data.is_empty() {
                unsafe {
                    cudarc::driver::result::memcpy_htod_sync(ptr, data)
                }.expect("memcpy_htod_sync failed");
            }

            CudaTensor {
                ptr,
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                dtype: T::get_dlpack_data_type(),
            }
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        fn as_ref(&self) -> DLPackTensorRef<'_> {
            unsafe {
                DLPackTensorRef::from_raw(dlpk::sys::DLTensor {
                    data: self.ptr as *mut std::ffi::c_void,
                    device: DLDevice {
                        device_type: dlpk::sys::DLDeviceType::kDLCUDA,
                        device_id: 0,
                    },
                    ndim: self.shape.len() as i32,
                    dtype: self.dtype,
                    shape: self.shape.as_ptr().cast_mut(),
                    strides: self.strides.as_ptr().cast_mut(),
                    byte_offset: 0,
                })
            }
        }
    }

    impl Drop for CudaTensor {
        fn drop(&mut self) {
            unsafe {
                let _ = cudarc::driver::result::free_sync(self.ptr);
            }
        }
    }

    #[test]
    fn is_equal_i32_kernel() {
        skip_without_cuda!();

        let reference = ReferenceValue::new(
            ArrayD::<i32>::from_shape_vec(vec![3, 1], vec![0, 1, 2]).unwrap()
        );

        // matching values
        let tensor = CudaTensor::new(&[0_i32, 1, 2], &[3, 1], &[1, 1]);
        assert!(is_equal_i32(tensor.as_ref(), &reference).unwrap());

        // mismatching values
        let tensor = CudaTensor::new(&[0_i32, 42, 2], &[3, 1], &[1, 1]);
        assert!(!is_equal_i32(tensor.as_ref(), &reference).unwrap());

        // matching values in a non-contiguous tensor: every other element of
        // [0, -1, 1, -1, 2, -1]
        let tensor = CudaTensor::new(&[0_i32, -1, 1, -1, 2, -1], &[3, 1], &[2, 1]);
        assert!(is_equal_i32(tensor.as_ref(), &reference).unwrap());
    }

    #[test]
    fn validate_cell_pbc_kernel() {
        skip_without_cuda!();

        // fully periodic: any cell is valid
        let pbc = CudaTensor::new(&[true, true, true], &[3], &[1]);
        let cell = CudaTensor::new(
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3], &[3, 1]
        );
        validate_cell_pbc(pbc.as_ref(), cell.as_ref()).unwrap();

        // non-periodic dimension with a zero cell vector: valid
        let pbc = CudaTensor::new(&[true, false, true], &[3], &[1]);
        let cell = CudaTensor::new(
            &[10.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0], &[3, 3], &[3, 1]
        );
        validate_cell_pbc(pbc.as_ref(), cell.as_ref()).unwrap();

        // non-periodic dimension with a non-zero cell vector: invalid
        let cell = CudaTensor::new(
            &[10.0_f32, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 10.0], &[3, 3], &[3, 1]
        );
        let err = validate_cell_pbc(pbc.as_ref(), cell.as_ref()).unwrap_err();
        assert!(err.to_string().contains("cell[1] contains non-zero values"), "{err}");

        // the same checks with f64 data, using the last dimension as the
        // non-periodic one
        let pbc = CudaTensor::new(&[true, true, false], &[3], &[1]);
        let cell = CudaTensor::new(
            &[10.0_f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0], &[3, 3], &[3, 1]
        );
        validate_cell_pbc(pbc.as_ref(), cell.as_ref()).unwrap();

        let cell = CudaTensor::new(
            &[10.0_f64, 0.0, 0.0, 0.0, 10.0, 0.0, 3.0, 0.0, 10.0], &[3, 3], &[3, 1]
        );
        let err = validate_cell_pbc(pbc.as_ref(), cell.as_ref()).unwrap_err();
        assert!(err.to_string().contains("cell[2] contains non-zero values"), "{err}");
    }
}
