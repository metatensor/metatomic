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

use dlpk::{DLPackTensor, DLPackTensorRef, DLPackTensorRefMut};

use crate::Error;
use super::{ReferenceValue, StridedNDIndex};

// Small wrapper around MTLBuffer to implement Send and Sync, since the data is
// read-only after initialization.
pub(crate) struct MetalBuffer(pub(crate) Retained<ProtocolObject<dyn MTLBuffer>>);

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl std::ops::Deref for MetalBuffer {
    type Target = ProtocolObject<dyn MTLBuffer>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Get the size of a memory page, which is the alignment required by
/// `newBufferWithBytesNoCopy`.
fn page_size() -> usize {
    unsafe extern "C" {
        fn getpagesize() -> std::ffi::c_int;
    }

    let size = unsafe { getpagesize() };
    return usize::try_from(size).expect("got a negative page size");
}

/// A Metal buffer that borrows the lifetime of the data it points to.
///
/// Created by wrapping an existing memory region (e.g. a DLPack tensor's data)
/// with `newBufferWithBytesNoCopy`, so the buffer does not own the memory and
/// must not outlive it.
///
/// The buffer starts at the beginning of the memory page containing the data,
/// so [`MetalBufferRef::offset`] must be used when binding it to a kernel.
pub(crate) struct MetalBufferRef<'a> {
    buffer: MetalBuffer,
    offset: usize,
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
    ///
    /// `newBufferWithBytesNoCopy` requires a page-aligned pointer, but the data
    /// of a tensor can start anywhere: Metal itself sub-allocates small buffers
    /// inside a single page, and tensors can be views inside a larger
    /// allocation. We thus wrap the whole page-aligned memory range containing
    /// the data, and the offset of the data inside this range is available with
    /// [`MetalBufferRef::offset`].
    pub(crate) fn from_dlpack(
        device: &ProtocolObject<dyn MTLDevice>,
        tensor: DLPackTensorRef<'a>,
    ) -> Result<Self, Error> {
        let ptr = dlpack_data_ptr(tensor);
        if ptr.is_null() {
            return Err(Error::Internal("tensor data pointer is null".into()));
        }

        let page_size = page_size();
        let offset = ptr as usize % page_size;
        // the length must also be a multiple of the page size
        let length = std::cmp::max(
            (offset + dlpack_num_bytes(tensor)).next_multiple_of(page_size),
            page_size,
        );

        let base = unsafe { ptr.cast::<u8>().sub(offset) };
        let nonnull = NonNull::new(base.cast_mut())
            .expect("the start of the page can not be null")
            .cast();

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
            offset,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Offset in bytes of the tensor data inside this buffer
    pub(crate) fn offset(&self) -> usize {
        self.offset
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
    copy_to_contiguous_8bit: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    copy_to_contiguous_16bit: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    copy_to_contiguous_32bit: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    copy_to_contiguous_64bit: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
        let copy_to_contiguous_8bit = make_pipeline(&device, &library, "copy_to_contiguous_8bit")?;
        let copy_to_contiguous_16bit = make_pipeline(&device, &library, "copy_to_contiguous_16bit")?;
        let copy_to_contiguous_32bit = make_pipeline(&device, &library, "copy_to_contiguous_32bit")?;
        let copy_to_contiguous_64bit = make_pipeline(&device, &library, "copy_to_contiguous_64bit")?;

        let queue = device
            .newCommandQueue()
            .ok_or_else(|| Error::Internal("failed to create command queue".into()))?;

        Ok(Self {
            device,
            queue,
            is_equal_i32,
            validate_cell_pbc_f32,
            scale_f32,
            copy_to_contiguous_8bit,
            copy_to_contiguous_16bit,
            copy_to_contiguous_32bit,
            copy_to_contiguous_64bit,
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
pub(crate) fn is_equal_i32(tensor: DLPackTensorRef<'_>, reference: &ReferenceValue<i32>) -> Result<bool, Error> {
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
            encoder.setBuffer_offset_atIndex(Some(&*values_buf), values_buf.offset(), 0);

            encoder.setBytes_length_atIndex(
                NonNull::<StridedNDIndex>::from(&values_idx).cast(),
                std::mem::size_of::<StridedNDIndex>(),
                1,
            );

            encoder.setBuffer_offset_atIndex(Some(&*ref_buf), 0, 2);

            encoder.setBytes_length_atIndex(
                NonNull::<StridedNDIndex>::from(reference_idx).cast(),
                std::mem::size_of::<StridedNDIndex>(),
                3,
            );

            encoder.setBytes_length_atIndex(
                NonNull::from(&(n_elements as u32)).cast(),
                std::mem::size_of::<u32>(),
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
pub(crate) fn validate_cell_pbc(
    pbc: DLPackTensorRef<'_>,
    cell: DLPackTensorRef<'_>,
) -> Result<(), Error> {
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
            encoder.setBuffer_offset_atIndex(Some(&*pbc_buf), pbc_buf.offset(), 0);

            encoder.setBytes_length_atIndex(
                NonNull::<StridedNDIndex>::from(&pbc_idx).cast(),
                std::mem::size_of::<StridedNDIndex>(),
                1,
            );

            encoder.setBuffer_offset_atIndex(Some(&*cell_buf), cell_buf.offset(), 2);

            encoder.setBytes_length_atIndex(
                NonNull::<StridedNDIndex>::from(&cell_idx).cast(),
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
pub(crate) fn scale_inplace(
    tensor: DLPackTensorRefMut<'_>,
    factor: f64,
) -> Result<(), Error> {
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
            encoder.setBuffer_offset_atIndex(Some(&*tensor_buf), tensor_buf.offset(), 0);

            encoder.setBytes_length_atIndex(
                NonNull::<StridedNDIndex>::from(&tensor_idx).cast(),
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
/// the original tensor. The clone is always C-contiguous, even when the
/// original tensor is not: the data is gathered with the `copy_to_contiguous`
/// kernel instead of a plain buffer copy.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub(crate) fn clone_tensor(tensor: &DLPackTensorRef<'_>) -> Result<DLPackTensor, Error> {
    let device_id = tensor.device().device_id as usize;
    let mut lock = METAL_CACHE.lock().expect("failed to lock METAL_CACHE");
    let cache = get_or_init(&mut lock, device_id)?;

    let element_size_bits = super::element_size(tensor.dtype())?;
    let n_elements: usize = tensor.shape().iter().map(|&s| s as usize).product();
    let num_bytes = n_elements * element_size_bits / 8;

    let shape: Vec<i64> = tensor.shape().to_vec();
    // the clone stores the data contiguously, regardless of the strides used by
    // the original tensor
    let strides = super::contiguous_strides(&shape);

    // pick the kernel matching the element size before allocating anything
    let pipeline = match element_size_bits {
        8 => &cache.copy_to_contiguous_8bit,
        16 => &cache.copy_to_contiguous_16bit,
        32 => &cache.copy_to_contiguous_32bit,
        64 => &cache.copy_to_contiguous_64bit,
        _ => {
            return Err(Error::InvalidParameter(format!(
                "clone_tensor does not support {} tensors on Metal",
                tensor.dtype()
            )));
        }
    };

    // allocate a new buffer, only big enough for the contiguous data. Metal
    // does not allow zero-sized buffers, so we always allocate at least one
    // byte (which is never read, since the tensor is then empty).
    let buffer = cache.device
        .newBufferWithLength_options(std::cmp::max(num_bytes, 1), MTLResourceOptions::empty())
        .ok_or_else(|| Error::Internal("failed to allocate Metal buffer for the clone".into()))?;

    if n_elements > 0 {
        // gather the (possibly strided) data from the original tensor into the
        // contiguous allocation
        let src_idx = StridedNDIndex::from_dlpack(*tensor);
        let src_buf = MetalBufferRef::from_dlpack(&cache.device, *tensor)?;

        objc2::rc::autoreleasepool(|_| {
            let cmd_buf = cache.queue.commandBuffer().expect("failed to create command buffer");
            let encoder = cmd_buf.computeCommandEncoder().expect("failed to create compute encoder");

            encoder.setComputePipelineState(pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&*src_buf), src_buf.offset(), 0);

                encoder.setBytes_length_atIndex(
                    NonNull::<StridedNDIndex>::from(&src_idx).cast(),
                    std::mem::size_of::<StridedNDIndex>(),
                    1,
                );

                encoder.setBuffer_offset_atIndex(Some(&*buffer), 0, 2);

                encoder.setBytes_length_atIndex(
                    NonNull::from(&(n_elements as u64)).cast(),
                    std::mem::size_of::<u64>(),
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
    }

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


#[cfg(test)]
mod tests {
    use super::*;

    use dlpk::{DLDevice, GetDLPackDataType};
    use ndarray::ArrayD;

    /// A DLPack tensor with Metal-resident data, used to test the kernels above.
    struct MetalTensor {
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        shape: Vec<i64>,
        strides: Vec<i64>,
        dtype: dlpk::sys::DLDataType,
    }

    impl MetalTensor {
        /// Create a new Metal tensor with the given `shape` and `strides`,
        /// containing a copy of `data`.
        ///
        /// `data` is the full memory span of the tensor, including any gap
        /// between the elements actually part of the tensor.
        fn new<T: GetDLPackDataType + Copy>(data: &[T], shape: &[i64], strides: &[i64]) -> Self {
            let device = METAL_DEVICES.first().expect("no Metal device available");

            // Metal does not allow zero-sized buffers
            assert!(!data.is_empty());
            let buffer = device
                .newBufferWithLength_options(std::mem::size_of_val(data), MTLResourceOptions::empty())
                .expect("failed to allocate Metal buffer");

            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buffer.contents().as_ptr().cast::<T>(),
                    data.len(),
                );
            }

            MetalTensor {
                buffer,
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                dtype: T::get_dlpack_data_type(),
            }
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        fn dl_tensor(&self) -> dlpk::sys::DLTensor {
            dlpk::sys::DLTensor {
                data: self.buffer.contents().as_ptr(),
                device: DLDevice {
                    device_type: dlpk::sys::DLDeviceType::kDLMetal,
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

        /// Read the first `n` elements of this tensor's memory span
        fn data<T: Copy>(&self, n: usize) -> Vec<T> {
            return read_metal(self.as_ref(), n);
        }

        /// Overwrite the data in this tensor's Metal buffer
        fn overwrite<T: Copy>(&self, data: &[T]) {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    self.buffer.contents().as_ptr().cast::<T>(),
                    data.len(),
                );
            }
        }
    }

    /// Read the first `n` elements of the data of any Metal-resident tensor
    fn read_metal<T: Copy>(tensor: DLPackTensorRef<'_>, n: usize) -> Vec<T> {
        unsafe {
            std::slice::from_raw_parts(dlpack_data_ptr(tensor).cast::<T>(), n).to_vec()
        }
    }

    #[test]
    fn is_equal_i32_kernel() {
        let reference = ReferenceValue::new(
            ArrayD::<i32>::from_shape_vec(vec![3, 1], vec![0, 1, 2]).unwrap()
        );

        // matching values
        let tensor = MetalTensor::new(&[0_i32, 1, 2], &[3, 1], &[1, 1]);
        assert!(is_equal_i32(tensor.as_ref(), &reference).unwrap());

        // mismatching values
        let tensor = MetalTensor::new(&[0_i32, 42, 2], &[3, 1], &[1, 1]);
        assert!(!is_equal_i32(tensor.as_ref(), &reference).unwrap());

        // matching values in a non-contiguous tensor: every other element of
        // [0, -1, 1, -1, 2, -1]
        let tensor = MetalTensor::new(&[0_i32, -1, 1, -1, 2, -1], &[3, 1], &[2, 1]);
        assert!(is_equal_i32(tensor.as_ref(), &reference).unwrap());
    }

    #[test]
    fn validate_cell_pbc_kernel() {
        // fully periodic: any cell is valid
        let pbc = MetalTensor::new(&[true, true, true], &[3], &[1]);
        let cell = MetalTensor::new(
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3], &[3, 1]
        );
        validate_cell_pbc(pbc.as_ref(), cell.as_ref()).unwrap();

        // non-periodic dimension with a zero cell vector: valid
        let pbc = MetalTensor::new(&[true, false, true], &[3], &[1]);
        let cell = MetalTensor::new(
            &[10.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0], &[3, 3], &[3, 1]
        );
        validate_cell_pbc(pbc.as_ref(), cell.as_ref()).unwrap();

        // non-periodic dimension with a non-zero cell vector: invalid
        let cell = MetalTensor::new(
            &[10.0_f32, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 10.0], &[3, 3], &[3, 1]
        );
        let err = validate_cell_pbc(pbc.as_ref(), cell.as_ref()).unwrap_err();
        assert!(err.to_string().contains("cell[1] contains non-zero values"), "{err}");

        // the last dimension being non-periodic is reported correctly too
        let pbc = MetalTensor::new(&[true, true, false], &[3], &[1]);
        let cell = MetalTensor::new(
            &[10.0_f32, 0.0, 0.0, 0.0, 10.0, 0.0, 3.0, 0.0, 10.0], &[3, 3], &[3, 1]
        );
        let err = validate_cell_pbc(pbc.as_ref(), cell.as_ref()).unwrap_err();
        assert!(err.to_string().contains("cell[2] contains non-zero values"), "{err}");
    }

    #[test]
    fn scale_inplace_kernel() {
        let mut tensor = MetalTensor::new(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &[3, 1]);
        scale_inplace(tensor.as_mut(), 2.5).unwrap();
        assert_eq!(tensor.data::<f32>(6), [2.5, 5.0, 7.5, 10.0, 12.5, 15.0]);

        // non-contiguous tensor: only the 2x2 block in the top left corner of
        // this 3x4 array is scaled, the rest of the data is left alone
        let data: Vec<f32> = (0..12_i16).map(f32::from).collect();
        let mut tensor = MetalTensor::new(&data, &[2, 2], &[4, 1]);
        scale_inplace(tensor.as_mut(), 10.0).unwrap();
        assert_eq!(tensor.data::<f32>(12), [
            0.0, 10.0, 2.0, 3.0,
            40.0, 50.0, 6.0, 7.0,
            8.0, 9.0, 10.0, 11.0,
        ]);

        // empty tensors are left alone (the buffer still has one element,
        // since Metal does not allow zero-sized buffers, and it should not be
        // touched by the kernel)
        let mut tensor = MetalTensor::new(&[3.0_f32], &[0], &[1]);
        scale_inplace(tensor.as_mut(), 2.0).unwrap();
        assert_eq!(tensor.data::<f32>(1), [3.0]);

        // only 32-bit floats are supported on Metal
        let mut tensor = MetalTensor::new(&[1.0_f64, 2.0], &[2], &[1]);
        let err = scale_inplace(tensor.as_mut(), 2.0).unwrap_err();
        assert!(err.to_string().contains("only supports 32-bit floats"), "{err}");
    }

    #[test]
    fn clone_contiguous() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = MetalTensor::new(&data, &[2, 3], &[3, 1]);

        let cloned = clone_tensor(&tensor.as_ref()).unwrap();

        assert_eq!(cloned.shape(), [2, 3]);
        assert_eq!(cloned.strides(), Some(&[3, 1][..]));
        assert_eq!(read_metal::<f32>(cloned.as_ref(), 6), data);

        // the clone is independent from the original
        tensor.overwrite(&[42.0_f32; 6]);
        assert_eq!(read_metal::<f32>(cloned.as_ref(), 6), data);
    }

    #[test]
    fn clone_non_contiguous() {
        // 2x2 block in the top left corner of a 3x4 array
        let data: Vec<f32> = (0..12_i16).map(f32::from).collect();
        let tensor = MetalTensor::new(&data, &[2, 2], &[4, 1]);

        let cloned = clone_tensor(&tensor.as_ref()).unwrap();

        assert_eq!(cloned.shape(), [2, 2]);
        assert_eq!(cloned.strides(), Some(&[2, 1][..]));
        assert_eq!(read_metal::<f32>(cloned.as_ref(), 4), [0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn clone_transposed() {
        // 2x3 array in column-major order (i.e. the transpose of a 3x2 array)
        let data: Vec<i64> = (0..6).collect();
        let tensor = MetalTensor::new(&data, &[2, 3], &[1, 2]);

        let cloned = clone_tensor(&tensor.as_ref()).unwrap();

        assert_eq!(cloned.shape(), [2, 3]);
        assert_eq!(cloned.strides(), Some(&[3, 1][..]));
        assert_eq!(read_metal::<i64>(cloned.as_ref(), 6), [0, 2, 4, 1, 3, 5]);
    }

    #[test]
    fn clone_element_sizes() {
        // 8-bit elements, every other one
        let data: Vec<u8> = (0..6).collect();
        let tensor = MetalTensor::new(&data, &[3], &[2]);
        let cloned = clone_tensor(&tensor.as_ref()).unwrap();
        assert_eq!(read_metal::<u8>(cloned.as_ref(), 3), [0, 2, 4]);

        // 16-bit elements, every other one
        let data: Vec<u16> = (0..6).collect();
        let tensor = MetalTensor::new(&data, &[3], &[2]);
        let cloned = clone_tensor(&tensor.as_ref()).unwrap();
        assert_eq!(read_metal::<u16>(cloned.as_ref(), 3), [0, 2, 4]);

        // bool elements
        let data = vec![true, false, true, true];
        let tensor = MetalTensor::new(&data, &[2], &[2]);
        let cloned = clone_tensor(&tensor.as_ref()).unwrap();
        assert_eq!(read_metal::<bool>(cloned.as_ref(), 2), [true, true]);
    }

    #[test]
    fn clone_empty() {
        // the buffer still has one element, since Metal does not allow
        // zero-sized buffers, but the tensor itself is empty
        let tensor = MetalTensor::new(&[3.0_f32], &[0], &[1]);

        let cloned = clone_tensor(&tensor.as_ref()).unwrap();

        assert_eq!(cloned.shape(), [0]);
    }
}
