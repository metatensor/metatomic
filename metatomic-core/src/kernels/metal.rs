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
    MTLCreateSystemDefaultDevice, MTLCompileOptions,
    MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use dlpk::DLPackTensorRef;

use crate::Error;
use super::{ReferenceValue, StridedNDIndex};

// Small wrapper around MTLBuffer to implement Send and Sync, since the data is
// read-only after initialization.
pub(crate) struct MetalBuffer(Retained<ProtocolObject<dyn MTLBuffer>>);

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl std::ops::Deref for MetalBuffer {
    type Target = ProtocolObject<dyn MTLBuffer>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

const KERNEL_SRC: &str = include_str!("metal_kernels.metal");

/// Cached metal ressources: device, command queue, and pipeline states for kernels.
struct MetalKernelCache {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    is_equal_i32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    validate_cell_pbc_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MetalKernelCache {
    fn new(device_id: usize) -> Result<Self, Error> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| Error::Internal(format!("no Metal device found for id {device_id}")))?;

        let library = device
            .newLibraryWithSource_options_error(
                ns_string!(KERNEL_SRC),
                Some(&MTLCompileOptions::new()),
            )
            .map_err(|e| Error::Internal(format!("MSL compile failed: {e}")))?;

        let is_equal_i32 = make_pipeline(&device, &library, "is_equal_i32")?;
        let validate_cell_pbc_f32 = make_pipeline(&device, &library, "validate_cell_pbc_f32")?;

        let queue = device
            .newCommandQueue()
            .ok_or_else(|| Error::Internal("failed to create command queue".into()))?;

        Ok(Self {
            device,
            queue,
            is_equal_i32,
            validate_cell_pbc_f32,
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
fn tensor_num_bytes(tensor: &DLPackTensorRef<'_>) -> usize {
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
fn dlpack_data_ptr(tensor: &DLPackTensorRef<'_>) -> *const std::ffi::c_void {
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
    let values_idx = StridedNDIndex::from_dlpack(&tensor);

    // Upload reference values to Metal (cached after first call)
    let (ref_buf, reference_idx) = reference.metal.get_or_init(|| {
        let ref_bytes = reference.cpu.len() * std::mem::size_of::<i32>();
        let ref_ptr: *const std::ffi::c_void = reference.cpu.as_slice()
            .expect("reference should be contiguous")
            .as_ptr()
            .cast();
        let buf = unsafe {
            cache.device.newBufferWithBytes_length_options(
                NonNull::new(ref_ptr.cast_mut()).expect("reference pointer must not be null"),
                ref_bytes,
                MTLResourceOptions::empty(),
            ).expect("failed to create reference buffer")
        };
        let idx = StridedNDIndex::from_ndarray(&reference.cpu.view());
        (MetalBuffer(buf), idx)
    });

    let values_buf = unsafe {
        cache.device.newBufferWithBytes_length_options(
            NonNull::new(dlpack_data_ptr(&tensor).cast_mut()).expect("values pointer must not be null"),
            tensor_num_bytes(&tensor),
            MTLResourceOptions::empty(),
        ).expect("failed to create values buffer")
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

    let pbc_idx = StridedNDIndex::from_dlpack(&pbc);
    let cell_idx = StridedNDIndex::from_dlpack(&cell);

    let pbc_buf = unsafe {
        cache.device.newBufferWithBytes_length_options(
            NonNull::new(dlpack_data_ptr(&pbc).cast_mut()).expect("pbc pointer must not be null"),
            tensor_num_bytes(&pbc),
            MTLResourceOptions::empty(),
        ).expect("failed to create pbc buffer")
    };
    let cell_buf = unsafe {
        cache.device.newBufferWithBytes_length_options(
            NonNull::new(dlpack_data_ptr(&cell).cast_mut()).expect("cell pointer must not be null"),
            tensor_num_bytes(&cell),
            MTLResourceOptions::empty(),
        ).expect("failed to create cell buffer")
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
