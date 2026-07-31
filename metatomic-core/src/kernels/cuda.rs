use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, LazyLock};

use cudarc::driver::safe::DeviceRepr;
use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::driver::sys;
use cudarc::nvrtc::compile_ptx;
use dlpk::{DLDevice, DLPackTensor, DLPackTensorRef, DLPackTensorRefMut};

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
    check_atomic_types: CudaFunction,
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

        let check_atomic_types = module.load_function("check_atomic_types")
            .map_err(|e| Error::Internal(format!("load_function(check_atomic_types): {e}")))?;

        Ok(Self {
            ctx,
            module,
            is_equal_i32,
            validate_cell_pbc_f32,
            validate_cell_pbc_f64,
            scale_f32,
            scale_f64,
            check_atomic_types,
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

fn check_valid_device(function: &str, device: DLDevice) {
    assert_eq!(
        device.device_type, dlpk::sys::DLDeviceType::kDLCUDA,
        "{} called on non-CUDA tensor", function
    );
    assert!(device.device_id >= 0, "{} called on invalid device_id", function);
}

/// Extract a `CUdeviceptr` from a DLPack tensor's raw `data` + `byte_offset`.
///
/// # Safety
///
/// The returned `CUdeviceptr` is only valid as long as the DLPack tensor's
/// backing memory is alive. The caller must ensure the tensor is not dropped
/// before the kernel finishes execution.
unsafe fn dlpack_to_device_ptr(tensor: DLPackTensorRef<'_>) -> cudarc::driver::sys::CUdeviceptr {
    debug_assert_eq!(
        tensor.device().device_type, dlpk::sys::DLDeviceType::kDLCUDA,
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
pub(super) fn is_equal_i32(tensor: DLPackTensorRef<'_>, reference: &ReferenceValue<i32>) -> Result<bool, Error> {
    check_valid_device("is_equal_i32", tensor.device());

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
pub(super) fn validate_cell_pbc(
    pbc: DLPackTensorRef<'_>,
    cell: DLPackTensorRef<'_>,
) -> Result<(), Error> {
    debug_assert_eq!(cell.device(), pbc.device(), "pbc and cell must be on the same device");
    check_valid_device("validate_cell_pbc", pbc.device());

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
pub(super) fn scale_inplace(
    tensor: DLPackTensorRefMut<'_>,
    factor: f64,
) -> Result<(), Error> {
    check_valid_device("scale_inplace", tensor.device());

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

/// Check that all atomic types in `types` are present in `valid_types`, on CUDA.
///
/// The check runs on-device. If invalid types are found (count > 0), a CPU
/// fallback scan identifies the specific invalid type for the error message.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub(super) fn check_atomic_types(
    types: DLPackTensorRef<'_>,
    valid_types: &ReferenceValue<i32>,
) -> Result<(), Error> {
    check_valid_device("check_atomic_types", types.device());
    assert!(
        valid_types.cpu.is_standard_layout(),
        "valid_types reference must be C-contiguous"
    );
    assert_eq!(
        types.n_dims(), 1,
        "check_atomic_types expects a 1D types tensor"
    );

    let device_id = types.device().device_id as usize;
    let stream = get_or_init(device_id)?;
    let cache = CUDA_CACHE.lock().expect("failed to lock CUDA_CACHE");
    let entry = &cache[&device_id];

    let n_atoms: i64 = types.shape().iter().product();
    if n_atoms == 0 {
        return Ok(());
    }

    let types_idx = StridedNDIndex::from_dlpack(types);
    let types_ptr = DLPackDevicePtr::from_ref(types);

    let (valid_types_device, _) = valid_types.cuda_data(device_id, &stream)?;
    let n_valid_types = i64::try_from(valid_types.cpu.len()).expect("could not cast n_valid_types to i64");

    // Allocate result counter (initialized to 0)
    let mut result = stream.alloc_zeros::<i32>(1)
        .map_err(|e| Error::Internal(format!("alloc_zeros: {e}")))?;

    unsafe {
        stream.launch_builder(&entry.check_atomic_types)
            .arg(&types_ptr)
            .arg(&types_idx)
            .arg(&n_atoms)
            .arg(valid_types_device)
            .arg(&n_valid_types)
            .arg(&mut result)
            .launch(launch_config_for_elems(n_atoms as u64))
            .map_err(|e| Error::Internal(format!("kernel launch (check_atomic_types): {e}")))?;
    }

    stream.synchronize()
        .map_err(|e| Error::Internal(format!("device sync: {e}")))?;

    let host = stream.clone_dtoh(&result)
        .map_err(|e| Error::Internal(format!("clone_dtoh result: {e}")))?;

    if host[0] > 0 {
        // Invalid types found — copy types to CPU and scan for the specific
        // invalid types. The tensor may be non-contiguous, so we copy the full
        // byte span and index using types_idx.
        let n_atoms_usize = n_atoms as usize;
        let elem_size = std::mem::size_of::<i32>();
        let n_bytes = match types.strides() {
            None => n_atoms_usize * elem_size,
            Some(strides) => {
                let max_offset: i64 = types.shape().iter()
                    .zip(strides.iter())
                    .map(|(&s, &st)| (s - 1) * st)
                    .sum();
                (max_offset as usize + 1) * elem_size
            }
        };
        let n_elements = n_bytes / elem_size;
        let mut host_types = vec![0i32; n_elements];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(host_types.as_mut_slice(), dlpack_to_device_ptr(types))
        }.map_err(|e| Error::Internal(format!("memcpy_dtoh_sync types: {e}")))?;

        super::cpu::check_atomic_types_buffer(&host_types, &types_idx, n_atoms_usize, valid_types)?;
    }

    Ok(())
}

/// Context held by the deleter of a cloned CUDA `DLManagedTensorVersioned`.
///
/// Stores the `CUdeviceptr` and the stream it was allocated on, so it can be
/// freed when the DLPack tensor is dropped.
struct CudaCloneContext {
    ptr: sys::CUdeviceptr,
    stream: Arc<CudaStream>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

/// Deleter for a cloned CUDA DLPack tensor.
///
/// Frees the device memory and the boxed `DLManagedTensorVersioned`.
unsafe extern "C" fn cuda_clone_deleter(tensor: *mut dlpk::sys::DLManagedTensorVersioned) {
    unsafe {
        let ctx = (*tensor).manager_ctx.cast::<CudaCloneContext>();
        let ctx = Box::from_raw(ctx);

        // free the device memory
        let _ = sys::cuMemFreeAsync(ctx.ptr, ctx.stream.cu_stream());

        // also drop the tensor itself
        let _ = Box::from_raw(tensor);
    }
}

/// Clone a DLPack tensor on CUDA, copying the underlying device memory.
///
/// The returned `DLPackTensor` owns its own CUDA memory allocation and is
/// independent of the original tensor.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
pub(super) fn clone_tensor(tensor: &DLPackTensorRef<'_>) -> Result<DLPackTensor, Error> {
    check_valid_device("clone_tensor", tensor.device());

    let device_id = tensor.device().device_id as usize;
    let stream = get_or_init(device_id)?;

    let src_ptr = unsafe { dlpack_to_device_ptr(*tensor) };
    let n_elements: i64 = tensor.shape().iter().product();
    let elem_size = tensor.dtype().bits as usize / 8;
    let num_bytes = n_elements as usize * elem_size;

    let shape: Vec<i64> = tensor.shape().to_vec();
    let strides: Vec<i64> = if let Some(s) = tensor.strides() {
        s.to_vec()
    } else {
        // contiguous: compute row-major strides
        let mut strides = vec![0i64; shape.len()];
        let mut acc: i64 = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
        strides
    };

    // allocate device memory
    let mut dst_ptr: sys::CUdeviceptr = 0;
    stream.context().bind_to_thread()
        .map_err(|e| Error::Internal(format!("bind_to_thread: {e}")))?;
    unsafe {
        sys::cuMemAllocAsync(&mut dst_ptr, num_bytes, stream.cu_stream())
            .result()
            .map_err(|e| Error::Internal(format!("cuMemAllocAsync: {e}")))?;
    }

    // copy data
    if num_bytes > 0 {
        unsafe {
            sys::cuMemcpyDtoDAsync_v2(dst_ptr, src_ptr, num_bytes, stream.cu_stream())
                .result()
                .map_err(|e| Error::Internal(format!("cuMemcpyDtoDAsync_v2: {e}")))?;
        }
    }

    stream.synchronize().map_err(|e| Error::Internal(format!("device sync: {e}")))?;

    // build the DLManagedTensorVersioned
    let ctx = Box::new(CudaCloneContext {
        ptr: dst_ptr,
        stream: stream.clone(),
        shape: shape,
        strides: strides,
    });

    let ndim = ctx.shape.len() as i32;
    let dl_tensor = dlpk::sys::DLTensor {
        data: dst_ptr as *mut std::ffi::c_void,
        device: tensor.device(),
        ndim,
        dtype: tensor.dtype(),
        shape: ctx.shape.as_ptr().cast_mut(),
        strides: ctx.strides.as_ptr().cast_mut(),
        byte_offset: 0,
    };

    let managed = Box::new(dlpk::sys::DLManagedTensorVersioned {
        version: dlpk::sys::DLPackVersion::current(),
        manager_ctx: Box::into_raw(ctx).cast(),
        deleter: Some(cuda_clone_deleter),
        flags: dlpk::sys::DLPACK_FLAG_BITMASK_IS_COPIED,
        dl_tensor,
    });

    let ptr = Box::into_raw(managed);
    Ok(unsafe { DLPackTensor::from_ptr(ptr) })
}
