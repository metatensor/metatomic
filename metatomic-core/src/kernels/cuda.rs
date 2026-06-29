use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, LazyLock};

use cudarc::driver::safe::DeviceRepr;
use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use dlpk::DLPackTensorRef;
use ndarray::ArrayViewD;


use crate::Error;

// CUDA kernel source compiled at runtime via NVRTC for the exact GPU
const KERNEL_SRC: &str = include_str!("cuda_kernels.cu");

const MAX_NDIM: usize = 7;

/// Multi-dimensional strided index matching the CUDA `StridedNDIndex` struct.
/// Supports up to 7 dimensions.
///
/// WARNING: any change here needs to be reflected in the CUDA source
#[repr(C)]
struct StridedNDIndex {
    ndim: i64,
    shape: [i64; MAX_NDIM],
    strides: [i64; MAX_NDIM],
}

unsafe impl DeviceRepr for StridedNDIndex {}

impl StridedNDIndex {
    fn from_dlpack(tensor: &DLPackTensorRef<'_>) -> Self {
        return StridedNDIndex::from_shape_strides(tensor.shape(), tensor.strides())
    }

    #[allow(clippy::cast_possible_wrap)]
    fn from_ndarray<T>(array: &ArrayViewD<'_, T>) -> Self {
        let shape = array.shape().iter().map(|&s| s as i64).collect::<Vec<i64>>();
        let strides = array.strides().iter().map(|&s| s as i64).collect::<Vec<i64>>();
        return Self::from_shape_strides(&shape, Some(&strides));
    }

    #[allow(clippy::cast_possible_wrap)]
    fn from_shape_strides(shape: &[i64], strides: Option<&[i64]>) -> Self {
        let ndim = shape.len();
        assert!(ndim <= MAX_NDIM, "StridedNDIndex only supports up to {MAX_NDIM} dimensions, got {ndim}");
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
/// The comparison is performed entirely on-device: the existing GPU pointer from
/// `tensor` is wrapped as a `DevicePtrArg`, the reference is uploaded to the GPU,
/// and a single-element result flag (`0` = ok, `1` = mismatch) is read back.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub(crate) fn is_equal_i32(tensor: DLPackTensorRef<'_>, reference: ArrayViewD<'_, i32>) -> Result<bool, Error> {
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

    // Build strided index from the ndarray view (preserves actual strides)
    let reference_idx = StridedNDIndex::from_ndarray(&reference);

    // Wrap the existing GPU-allocated tensor pointer
    let tensor_ptr = unsafe { DevicePtrArg { ptr: dlpack_to_device_ptr(&tensor) } };

    // Upload reference values to GPU
    let ref_dev = stream.clone_htod(reference.as_slice().expect("reference should be contiguous"))
        .map_err(|e| Error::Internal(format!("clone_htod reference: {e}")))?;

    // Allocate result flag (initialized to 0 = no mismatch)
    let mut result = stream.alloc_zeros::<i32>(1)
        .map_err(|e| Error::Internal(format!("alloc_zeros: {e}")))?;

    unsafe {
        stream.launch_builder(&entry.is_equal_i32)
            .arg(&tensor_ptr)
            .arg(&values_idx)
            .arg(&ref_dev)
            .arg(&reference_idx)
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
