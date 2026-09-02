use std::collections::{HashMap, hash_map::Entry};
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::LazyLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;

use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCopyAllDevices, MTLCompileOptions,
    MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use dlpk::{DLDevice, DLPackTensor, DLPackTensorRef, DLPackTensorRefMut};

use crate::Error;
use super::{ReferenceValue, StridedNDIndex};

// Small wrapper around MTLBuffer to implement Send and Sync, since the data is
// read-only after initialization.
pub(crate) struct MetalBuffer(pub(super) Retained<ProtocolObject<dyn MTLBuffer>>);

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl std::ops::Deref for MetalBuffer {
    type Target = ProtocolObject<dyn MTLBuffer>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A Metal buffer that borrows the lifetime of the data it points to.
///
/// Created by wrapping an existing memory region (e.g. a DLPack tensor's data)
/// with `newBufferWithBytesNoCopy`, so the buffer does not own the memory and
/// must not outlive it.
pub(super) struct MetalBufferRef<'a> {
    buffer: MetalBuffer,
    _phantom: std::marker::PhantomData<&'a [u8]>,
}

impl std::ops::Deref for MetalBufferRef<'_> {
    type Target = MetalBuffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<'a> MetalBufferRef<'a> {
    /// Wrap a DLPack tensor's existing memory in a Metal buffer without copying.
    ///
    /// Uses `newBufferWithBytesNoCopy:length:options:deallocator:` with no
    /// deallocator, since the DLPack tensor (or its owner) retains ownership of
    /// the memory. The returned buffer borrows the tensor's lifetime and must
    /// not outlive the tensor's backing memory.
    pub(super) fn from_dlpack(
        device: &ProtocolObject<dyn MTLDevice>,
        tensor: DLPackTensorRef<'a>,
    ) -> Result<Self, Error> {
        let ptr = dlpack_data_ptr(tensor);
        let length = dlpack_num_bytes(tensor);

        let nonnull = NonNull::new(ptr.cast_mut())
            .ok_or_else(|| Error::Internal("tensor data pointer is null".into()))?;

        let buffer = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                nonnull,
                length,
                MTLResourceOptions::empty(),
                None,
            )
        };

        let buffer = buffer.ok_or_else(|| Error::Internal(
            "failed to create Metal buffer from DLPack tensor (newBufferWithBytesNoCopy returned nil)".into()
        ))?;

        Ok(Self {
            buffer: MetalBuffer(buffer),
            _phantom: std::marker::PhantomData,
        })
    }
}

const KERNEL_SRC: &str = include_str!("metal_kernels.metal");

/// Cached metal ressources: device, command queue, and pipeline states for kernels.
struct MetalKernelCache {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    is_equal_i32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    validate_cell_pbc_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    scale_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    check_atomic_types: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

/// All Metal devices on this system, queried once on first access.
static METAL_DEVICES: LazyLock<Vec<Retained<ProtocolObject<dyn MTLDevice>>>> = LazyLock::new(|| MTLCopyAllDevices().to_vec());

impl MetalKernelCache {
    fn new(device_id: usize) -> Result<Self, Error> {
        let device = METAL_DEVICES
            .get(device_id)
            .ok_or_else(|| Error::Internal(format!("no Metal device with id {device_id}")))?
            .clone();

        let library = device
            .newLibraryWithSource_options_error(
                ns_string!(KERNEL_SRC),
                Some(&MTLCompileOptions::new()),
            )
            .map_err(|e| Error::Internal(format!("MSL compile failed: {e}")))?;

        let is_equal_i32 = make_pipeline(&device, &library, "is_equal_i32")?;
        let validate_cell_pbc_f32 = make_pipeline(&device, &library, "validate_cell_pbc_f32")?;
        let scale_f32 = make_pipeline(&device, &library, "scale_f32")?;
        let check_atomic_types = make_pipeline(&device, &library, "check_atomic_types")?;

        let queue = device
            .newCommandQueue()
            .ok_or_else(|| Error::Internal("failed to create command queue".into()))?;

        Ok(Self {
            device,
            queue,
            is_equal_i32,
            validate_cell_pbc_f32,
            scale_f32,
            check_atomic_types,
        })
    }
}

fn make_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, Error> {
    use objc2_foundation::NSString;

    let ns_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| Error::Internal(format!("get_function({name}): not found")))?;

    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| Error::Internal(format!("pipeline state ({name}): {e}")))
}

static METAL_CACHE: LazyLock<Mutex<HashMap<usize, MetalKernelCache>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn get_or_init(cache: &mut HashMap<usize, MetalKernelCache>, device_id: usize) -> Result<&MetalKernelCache, Error> {
    let entry = match cache.entry(device_id) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => entry.insert(MetalKernelCache::new(device_id)?),
    };
    Ok(entry)
}

fn check_valid_device(function: &str, device: DLDevice) {
    assert_eq!(
        device.device_type, dlpk::sys::DLDeviceType::kDLMetal,
        "{} called on non-metal tensor", function
    );
    assert!(device.device_id >= 0, "{} called on invalid device_id", function);
}

/// Compute the byte span of a DLPack tensor's data (including gaps from
/// strides).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn dlpack_num_bytes(tensor: DLPackTensorRef<'_>) -> usize {
    let elem_size = tensor.dtype().bits as usize / 8;
    let shape = tensor.shape();
    match tensor.strides() {
        None => shape.iter().map(|&s| s as usize).product::<usize>() * elem_size,
        Some(strides) => {
            let max_idx: i64 = shape.iter()
                .zip(strides.iter())
                .map(|(&s, &st)| (s - 1) * st)
                .sum();
            (max_idx as usize + 1) * elem_size
        }
    }
}

/// Extract a raw pointer to the tensor's data, accounting for byte_offset.
///
/// # Safety
///
/// The returned pointer is only valid as long as the DLPack tensor's backing
/// memory is alive.
#[allow(clippy::cast_possible_truncation)]
fn dlpack_data_ptr(tensor: DLPackTensorRef<'_>) -> *const std::ffi::c_void {
    unsafe {
        tensor.raw.data.cast::<u8>().add(tensor.raw.byte_offset as usize).cast()
    }
}

/// Check that the values of a Metal-resident i32 DLPack tensor match an expected
/// reference array.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub(super) fn is_equal_i32(tensor: DLPackTensorRef<'_>, reference: &ReferenceValue<i32>) -> Result<bool, Error> {
    check_valid_device("is_equal_i32", tensor.device());

    let device_id = tensor.device().device_id as usize;
    let mut lock = METAL_CACHE.lock().expect("failed to lock METAL_CACHE");
    let cache = get_or_init(&mut lock, device_id)?;

    let n_elements: usize = tensor.shape().iter().map(|&s| s as usize).product();
    let ref_bytes = n_elements * std::mem::size_of::<i32>();

    // Build strided index for the values
    let values_idx = StridedNDIndex::from_dlpack(tensor);

    // Upload reference values to Metal (cached after first call, per device)
    let (ref_buf, reference_idx) = reference.metal_data(device_id, &cache.device)?;

    let values_buf = MetalBufferRef::from_dlpack(&cache.device, tensor)?;
    let result_buf = unsafe {
        cache.device.newBufferWithBytes_length_options(
            NonNull::from(&0i32).cast(),
            std::mem::size_of::<i32>(),
            MTLResourceOptions::empty(),
        ).expect("failed to create result buffer")
    };

    objc2::rc::autoreleasepool(|_| {
        let cmd_buf = cache.queue.commandBuffer().expect("failed to create command buffer");
        let encoder = cmd_buf.computeCommandEncoder().expect("failed to create compute encoder");

        encoder.setComputePipelineState(&cache.is_equal_i32);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*values_buf), 0, 0);

            encoder.setBytes_length_atIndex(
                NonNull::from(&values_idx).cast(),
                std::mem::size_of::<StridedNDIndex>(),
                1,
            );

            encoder.setBuffer_offset_atIndex(Some(&*ref_buf), 0, 2);

            encoder.setBytes_length_atIndex(
                NonNull::from(&reference_idx).cast(),
                std::mem::size_of::<StridedNDIndex>(),
                3,
            );

            encoder.setBytes_length_atIndex(
                NonNull::from(&(n_elements as u64)).cast(),
                std::mem::size_of::<u64>(),
                4,
            );

            encoder.setBuffer_offset_atIndex(Some(&*result_buf), 0, 5);
        }

        let tg_size = 32;
        let tg_count = n_elements.div_ceil(tg_size);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: tg_count, height: 1, depth: 1 },
            MTLSize { width: tg_size, height: 1, depth: 1 },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    });

    let result = unsafe {
        *result_buf.contents().as_ptr().cast::<i32>()
    };
    return Ok(result == 0);
}

/// Validate that cell vectors are zero for non-periodic dimensions on Metal.
#[allow(clippy::cast_sign_loss)]
pub(super) fn validate_cell_pbc(
    pbc: DLPackTensorRef<'_>,
    cell: DLPackTensorRef<'_>,
) -> Result<(), Error> {
    debug_assert_eq!(cell.device(), pbc.device(), "pbc and cell must be on the same device");
    check_valid_device("validate_cell_pbc", cell.device());

    let device_id = pbc.device().device_id as usize;
    let mut lock = METAL_CACHE.lock().expect("failed to lock METAL_CACHE");
    let cache = get_or_init(&mut lock, device_id)?;

    let pbc_idx = StridedNDIndex::from_dlpack(pbc);
    let cell_idx = StridedNDIndex::from_dlpack(cell);

    let pbc_buf = MetalBufferRef::from_dlpack(&cache.device, pbc)?;
    let cell_buf = MetalBufferRef::from_dlpack(&cache.device, cell)?;
    let result_buf = unsafe {
        cache.device.newBufferWithBytes_length_options(
            NonNull::from(&0i32).cast(),
            std::mem::size_of::<i32>(),
            MTLResourceOptions::empty(),
        ).expect("failed to create result buffer")
    };

    objc2::rc::autoreleasepool(|_| {
        let cmd_buf = cache.queue.commandBuffer().expect("failed to create command buffer");
        let encoder = cmd_buf.computeCommandEncoder().expect("failed to create compute encoder");

        assert!(cell.dtype().bits == 32, "only float32 is supported on Metal");

        encoder.setComputePipelineState(&cache.validate_cell_pbc_f32);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*pbc_buf), 0, 0);

            encoder.setBytes_length_atIndex(
                NonNull::from(&pbc_idx).cast(),
                std::mem::size_of::<StridedNDIndex>(),
                1,
            );

            encoder.setBuffer_offset_atIndex(Some(&*cell_buf), 0, 2);

            encoder.setBytes_length_atIndex(
                NonNull::from(&cell_idx).cast(),
                std::mem::size_of::<StridedNDIndex>(),
                3,
            );

            encoder.setBuffer_offset_atIndex(Some(&*result_buf), 0, 4);
        }

        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: 3, height: 1, depth: 1 },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    });

    let result = unsafe {
        *result_buf.contents().as_ptr().cast::<i32>()
    };

    if result != 0 {
        let dim = result - 1;
        return Err(Error::InvalidParameter(format!(
            "invalid cell: for non-periodic dimensions, the corresponding \
             cell vector must be zero, but cell[{}] contains non-zero values",
            dim
        )));
    }
    Ok(())
}

/// Scale all elements of `tensor` in place by `factor`, on Metal device.
///
/// Only 32-bit floating point tensors are supported on Metal.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub(super) fn scale_inplace(
    tensor: DLPackTensorRefMut<'_>,
    factor: f64,
) -> Result<(), Error> {
    check_valid_device("scale_inplace", tensor.device());

    let device_id = tensor.device().device_id as usize;
    let mut lock = METAL_CACHE.lock().expect("failed to lock METAL_CACHE");
    let cache = get_or_init(&mut lock, device_id)?;

    let dtype = tensor.dtype();
    if dtype.code != dlpk::sys::DLDataTypeCode::kDLFloat || dtype.bits != 32 {
        return Err(Error::InvalidParameter(format!(
            "scale_inplace on Metal only supports 32-bit floats, got {}-bit {:?}",
            dtype.bits, dtype.code
        )));
    }

    let n_elements: usize = tensor.shape().iter().map(|&s| s as usize).product();
    if n_elements == 0 {
        return Ok(());
    }

    let tensor_idx = StridedNDIndex::from_dlpack(tensor.as_ref());
    let factor_f32 = factor as f32;

    let tensor_buf = MetalBufferRef::from_dlpack(&cache.device, tensor.as_ref())?;

    objc2::rc::autoreleasepool(|_| {
        let cmd_buf = cache.queue.commandBuffer().expect("failed to create command buffer");
        let encoder = cmd_buf.computeCommandEncoder().expect("failed to create compute encoder");

        encoder.setComputePipelineState(&cache.scale_f32);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*tensor_buf), 0, 0);

            encoder.setBytes_length_atIndex(
                NonNull::from(&tensor_idx).cast(),
                std::mem::size_of::<StridedNDIndex>(),
                1,
            );

            encoder.setBytes_length_atIndex(
                NonNull::from(&(n_elements as u64)).cast(),
                std::mem::size_of::<u64>(),
                2,
            );

            encoder.setBytes_length_atIndex(
                NonNull::from(&factor_f32).cast(),
                std::mem::size_of::<f32>(),
                3,
            );
        }

        let tg_size = 32;
        let tg_count = n_elements.div_ceil(tg_size);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: tg_count, height: 1, depth: 1 },
            MTLSize { width: tg_size, height: 1, depth: 1 },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    });

    return Ok(());
}

/// Check that all atomic types in `types` are present in `valid_types`, on Metal.
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
    let mut lock = METAL_CACHE.lock().expect("failed to lock METAL_CACHE");
    let cache = get_or_init(&mut lock, device_id)?;

    let n_atoms: usize = types.shape().iter().map(|&s| s as usize).product();
    if n_atoms == 0 {
        return Ok(());
    }

    let types_idx = StridedNDIndex::from_dlpack(types);

    // Upload valid types to Metal (cached after first call, per device)
    let (valid_types_buffer, _) = valid_types.metal_data(device_id, &cache.device)?;
    let n_valid_types = valid_types.cpu.len() as u64;

    let types_buf = unsafe {
        cache.device.newBufferWithBytes_length_options(
            NonNull::new(dlpack_data_ptr(types).cast_mut()).expect("types pointer must not be null"),
            dlpack_num_bytes(types),
            MTLResourceOptions::empty(),
        ).expect("failed to create types buffer")
    };
    let result_buf = unsafe {
        cache.device.newBufferWithBytes_length_options(
            NonNull::from(&0i32).cast(),
            std::mem::size_of::<i32>(),
            MTLResourceOptions::empty(),
        ).expect("failed to create result buffer")
    };

    objc2::rc::autoreleasepool(|_| {
        let cmd_buf = cache.queue.commandBuffer().expect("failed to create command buffer");
        let encoder = cmd_buf.computeCommandEncoder().expect("failed to create compute encoder");

        encoder.setComputePipelineState(&cache.check_atomic_types);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*types_buf), 0, 0);

            encoder.setBytes_length_atIndex(
                NonNull::from(&types_idx).cast(),
                std::mem::size_of::<StridedNDIndex>(),
                1,
            );

            encoder.setBytes_length_atIndex(
                NonNull::from(&(n_atoms as u64)).cast(),
                std::mem::size_of::<u64>(),
                2,
            );

            encoder.setBuffer_offset_atIndex(Some(&*valid_types_buffer), 0, 3);

            encoder.setBytes_length_atIndex(
                NonNull::from(&n_valid_types).cast(),
                std::mem::size_of::<u64>(),
                4,
            );

            encoder.setBuffer_offset_atIndex(Some(&*result_buf), 0, 5);
        }

        let tg_size = 32;
        let tg_count = n_atoms.div_ceil(tg_size);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: tg_count, height: 1, depth: 1 },
            MTLSize { width: tg_size, height: 1, depth: 1 },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    });

    let result = unsafe {
        *result_buf.contents().as_ptr().cast::<i32>()
    };

    if result > 0 {
        // Invalid types found — read types from the Metal buffer and scan on CPU.
        let n_bytes = dlpack_num_bytes(types);
        let n_elements = n_bytes / std::mem::size_of::<i32>();
        let host_types: Vec<i32> = unsafe {
            std::slice::from_raw_parts(
                types_buf.contents().as_ptr().cast::<i32>(),
                n_elements,
            ).to_vec()
        };
        super::cpu::check_atomic_types_buffer(&host_types, &types_idx, n_atoms, valid_types)?;
    }

    Ok(())
}

/// Context held by the deleter of a cloned Metal `DLManagedTensorVersioned`.
struct MetalCloneContext {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe impl Send for MetalCloneContext {}
unsafe impl Sync for MetalCloneContext {}

/// Deleter for a cloned Metal DLPack tensor.
///
/// Drops the context (which releases the MTLBuffer) and the boxed
/// `DLManagedTensorVersioned`.
unsafe extern "C" fn metal_clone_deleter(tensor: *mut dlpk::sys::DLManagedTensorVersioned) {
    unsafe {
        let ctx = (*tensor).manager_ctx.cast::<MetalCloneContext>();
        let _ = Box::from_raw(ctx);
        let _ = Box::from_raw(tensor);
    }
}

/// Clone a DLPack tensor on Metal, copying the underlying device memory.
///
/// The returned `DLPackTensor` owns its own Metal buffer and is independent of
/// the original tensor.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub(super) fn clone_tensor(tensor: &DLPackTensorRef<'_>) -> Result<DLPackTensor, Error> {
    check_valid_device("clone_tensor", tensor.device());

    let device_id = tensor.device().device_id as usize;
    let mut lock = METAL_CACHE.lock().expect("failed to lock METAL_CACHE");
    let cache = get_or_init(&mut lock, device_id)?;

    let src_ptr = dlpack_data_ptr(*tensor);
    let length = dlpack_num_bytes(*tensor);

    let shape: Vec<i64> = tensor.shape().to_vec();
    let strides: Vec<i64> = if let Some(s) = tensor.strides() {
        s.to_vec()
    } else {
        let mut strides = vec![0i64; shape.len()];
        let mut acc: i64 = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
        strides
    };

    // allocate a new Metal buffer and copy the data into it
    let buffer = if length == 0 {
        cache.device.newBufferWithLength_options(0, MTLResourceOptions::empty())
            .expect("failed to create empty Metal buffer")
    } else {
        unsafe {
            cache.device.newBufferWithBytes_length_options(
                NonNull::new(src_ptr.cast_mut()).expect("source pointer must not be null"),
                length,
                MTLResourceOptions::empty(),
            ).expect("failed to create Metal buffer (copy)")
        }
    };

    let ctx = Box::new(MetalCloneContext {
        buffer,
        shape: shape,
        strides: strides,
    });

    let ndim = ctx.shape.len() as i32;
    let data_ptr = ctx.buffer.contents().as_ptr();

    let dl_tensor = dlpk::sys::DLTensor {
        data: data_ptr.cast::<std::ffi::c_void>(),
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
        deleter: Some(metal_clone_deleter),
        flags: dlpk::sys::DLPACK_FLAG_BITMASK_IS_COPIED,
        dl_tensor,
    });

    let ptr = Box::into_raw(managed);
    Ok(unsafe { DLPackTensor::from_ptr(ptr) })
}
