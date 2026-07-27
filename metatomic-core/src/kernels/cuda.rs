use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, LazyLock};

use cudarc::driver::safe::DeviceRepr;
use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use dlpk::{DLPackTensorRef, DLPackTensorRefMut};

use crate::Error;
use super::{ReferenceValue, StridedNDIndex};

// CUDA kernel source compiled at runtime via NVRTC for the exact GPU
const KERNEL_SRC: &str = include_str!("cuda_kernels.cu");

unsafe impl DeviceRepr for StridedNDIndex {}

/// Create a [`LaunchConfig`] for `n_elements` with 64-bit element counts.
///
/// This replaces `LaunchConfig::for_num_elems` which only accepts `u32`.
/// CUDA's gridDim.x supports up to 2^31 - 1 blocks; with a block size of
/// 1024 this covers up to ~2.2 × 10¹² elements.
#[allow(clippy::cast_possible_truncation)]
fn launch_config_for_elems(n_elements: u64) -> LaunchConfig {
    const NUM_THREADS: u64 = 1024;
    const MAX_GRID_X: u64 = (1u64 << 31) - 1;
    let num_blocks = std::cmp::min(n_elements.div_ceil(NUM_THREADS), MAX_GRID_X);
    LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (NUM_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

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
struct DLPackDevicePtr<'a> {
    ptr: cudarc::driver::sys::CUdeviceptr,
    _phantom: std::marker::PhantomData<&'a [u8]>,
}

unsafe impl DeviceRepr for DLPackDevicePtr<'_> {}

impl<'a> DLPackDevicePtr<'a> {
    /// Wrap a CUDA-resident DLPack tensor's device pointer for use as a kernel
    /// argument.
    ///
    /// The returned `DlpackDevicePtr` borrows the tensor's lifetime, ensuring the
    /// backing memory stays alive as long as the argument is in use.
    fn from_ref(tensor: DLPackTensorRef<'a>) -> Self {
        Self {
            ptr: unsafe { dlpack_to_device_ptr(tensor) },
            _phantom: std::marker::PhantomData,
        }
    }

    /// Wrap a CUDA-resident DLPack tensor's device pointer for use as a kernel
    /// argument (mutable variant).
    ///
    /// The returned `DlpackDevicePtr` borrows the tensor's lifetime, ensuring the
    /// backing memory stays alive as long as the argument is in use.
    fn from_mut(tensor: DLPackTensorRefMut<'_>) -> Self {
        Self {
            ptr: unsafe { dlpack_to_device_ptr(tensor.as_ref()) },
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Per-device cached resources: context, module, and kernel function handles.
struct CudaKernelCache {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    is_equal_i32: CudaFunction,
    validate_cell_pbc_f32: CudaFunction,
    validate_cell_pbc_f64: CudaFunction,
    scale_f32: CudaFunction,
    scale_f64: CudaFunction,
}

impl CudaKernelCache {
    fn new(device_id: usize) -> Result<Self, Error> {
        let ctx = CudaContext::new(device_id)
            .map_err(|e| Error::Internal(format!("CudaContext::new({device_id}): {e}")))?;

        let ptx = compile_ptx(KERNEL_SRC)
            .map_err(|e| Error::Internal(format!("NVRTC compile failed: {e}")))?;

        let module = ctx.load_module(ptx)
            .map_err(|e| Error::Internal(format!("PTX load failed: {e}")))?;

        let is_equal_i32 = module.load_function("is_equal_i32")
            .map_err(|e| Error::Internal(format!("load_function(is_equal_i32): {e}")))?;

        let validate_cell_pbc_f32 = module.load_function("validate_cell_pbc_f32")
            .map_err(|e| Error::Internal(format!("load_function(validate_cell_pbc_f32): {e}")))?;

        let validate_cell_pbc_f64 = module.load_function("validate_cell_pbc_f64")
            .map_err(|e| Error::Internal(format!("load_function(validate_cell_pbc_f64): {e}")))?;

        let scale_f32 = module.load_function("scale_f32")
            .map_err(|e| Error::Internal(format!("load_function(scale_f32): {e}")))?;

        let scale_f64 = module.load_function("scale_f64")
            .map_err(|e| Error::Internal(format!("load_function(scale_f64): {e}")))?;

        Ok(Self {
            ctx,
            module,
            is_equal_i32,
            validate_cell_pbc_f32,
            validate_cell_pbc_f64,
            scale_f32,
            scale_f64,
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
unsafe fn dlpack_to_device_ptr(tensor: DLPackTensorRef<'_>) -> cudarc::driver::sys::CUdeviceptr {
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
/// from `tensor` is wrapped as a `DlpackDevicePtr`, the reference is uploaded to
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
    let values_idx = StridedNDIndex::from_dlpack(tensor);

    // Wrap the existing GPU-allocated tensor pointer
    let tensor_ptr = DLPackDevicePtr::from_ref(tensor);

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
            .launch(launch_config_for_elems(n_elements as u64))
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

    let pbc_ptr = DLPackDevicePtr::from_ref(pbc);
    let cell_ptr = DLPackDevicePtr::from_ref(cell);

    let pbc_idx = StridedNDIndex::from_dlpack(pbc);
    let cell_idx = StridedNDIndex::from_dlpack(cell);

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


/// Scale all elements of `tensor` in place by `factor`, on CUDA device.
///
/// The tensor must be a 32-bit or 64-bit floating point tensor residing on a
/// CUDA device. The scaling is performed entirely on-device, in place.
#[allow(clippy::cast_sign_loss)]
pub(crate) fn scale_inplace(
    tensor: DLPackTensorRefMut<'_>,
    factor: f64,
) -> Result<(), Error> {
    debug_assert!(
        tensor.device().device_type == dlpk::sys::DLDeviceType::kDLCUDA,
        "scale_inplace called on non-CUDA tensor"
    );
    debug_assert!(tensor.device().device_id >= 0, "scale_inplace called on invalid device_id");

    let device_id = tensor.device().device_id as usize;
    let stream = get_or_init(device_id)?;
    let cache = CUDA_CACHE.lock().expect("failed to lock CUDA_CACHE");
    let entry = &cache[&device_id];

    let n_elements: i64 = tensor.shape().iter().product();
    if n_elements == 0 {
        return Ok(());
    }

    let dtype = tensor.dtype();
    let tensor_idx = StridedNDIndex::from_dlpack(tensor.as_ref());
    let tensor_ptr = DLPackDevicePtr::from_mut(tensor);

    if dtype.code == dlpk::sys::DLDataTypeCode::kDLFloat && dtype.bits == 32 {
        unsafe {
            stream.launch_builder(&entry.scale_f32)
                .arg(&tensor_ptr)
                .arg(&tensor_idx)
                .arg(&n_elements)
                .arg(&factor)
                .launch(launch_config_for_elems(n_elements as u64))
                .map_err(|e| Error::Internal(format!("kernel launch (scale_f32): {e}")))?;
        }
    } else if dtype.code == dlpk::sys::DLDataTypeCode::kDLFloat && dtype.bits == 64 {
        unsafe {
            stream.launch_builder(&entry.scale_f64)
                .arg(&tensor_ptr)
                .arg(&tensor_idx)
                .arg(&n_elements)
                .arg(&factor)
                .launch(launch_config_for_elems(n_elements as u64))
                .map_err(|e| Error::Internal(format!("kernel launch (scale_f64): {e}")))?;
        }
    } else {
        return Err(Error::InvalidParameter(format!(
            "scale_inplace only supports 32-bit or 64-bit floats, got {}-bit {:?}",
            dtype.bits, dtype.code
        )));
    }

    stream.synchronize()
        .map_err(|e| Error::Internal(format!("device sync: {e}")))?;

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

            unsafe {
                cudarc::driver::result::memcpy_htod_sync(ptr, data)
            }.expect("memcpy_htod_sync failed");

            CudaTensor {
                ptr,
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                dtype: T::get_dlpack_data_type(),
            }
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        fn dl_tensor(&self) -> dlpk::sys::DLTensor {
            dlpk::sys::DLTensor {
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
            }
        }

        fn as_ref(&self) -> DLPackTensorRef<'_> {
            unsafe { DLPackTensorRef::from_raw(self.dl_tensor()) }
        }

        fn as_mut(&mut self) -> DLPackTensorRefMut<'_> {
            unsafe { DLPackTensorRefMut::from_raw(self.dl_tensor()) }
        }

        /// Read the first `n` elements of this tensor's memory span back to the
        /// CPU
        fn data<T: Copy + Default>(&self, n: usize) -> Vec<T> {
            let mut host = vec![T::default(); n];
            unsafe {
                cudarc::driver::result::memcpy_dtoh_sync(host.as_mut_slice(), self.ptr)
            }.expect("memcpy_dtoh_sync failed");

            return host;
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

    #[test]
    fn scale_inplace_kernel() {
        skip_without_cuda!();

        // f32
        let mut tensor = CudaTensor::new(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &[3, 1]);
        scale_inplace(tensor.as_mut(), 2.5).unwrap();
        assert_eq!(tensor.data::<f32>(6), [2.5, 5.0, 7.5, 10.0, 12.5, 15.0]);

        // f64
        let mut tensor = CudaTensor::new(&[1.0_f64, -2.0, 3.5, 0.0], &[2, 2], &[2, 1]);
        scale_inplace(tensor.as_mut(), 0.5).unwrap();
        assert_eq!(tensor.data::<f64>(4), [0.5, -1.0, 1.75, 0.0]);

        // non-contiguous tensor: only the 2x2 block in the top left corner of
        // this 3x4 array is scaled, the rest of the data is left alone
        let data: Vec<f32> = (0..12_i16).map(f32::from).collect();
        let mut tensor = CudaTensor::new(&data, &[2, 2], &[4, 1]);
        scale_inplace(tensor.as_mut(), 10.0).unwrap();
        assert_eq!(tensor.data::<f32>(12), [
            0.0, 10.0, 2.0, 3.0,
            40.0, 50.0, 6.0, 7.0,
            8.0, 9.0, 10.0, 11.0,
        ]);

        // empty tensors are left alone (the allocation still has one element,
        // since CUDA does not allow zero-sized allocations, and it should not
        // be touched by the kernel)
        let mut tensor = CudaTensor::new(&[3.0_f32], &[0], &[1]);
        scale_inplace(tensor.as_mut(), 2.0).unwrap();
        assert_eq!(tensor.data::<f32>(1), [3.0]);

        // integers are not supported
        let mut tensor = CudaTensor::new(&[1_i32, 2, 3], &[3], &[1]);
        let err = scale_inplace(tensor.as_mut(), 2.0).unwrap_err();
        assert!(err.to_string().contains("only supports 32-bit or 64-bit floats"), "{err}");
    }
}
