use dlpk::sys::DLDeviceType;
use dlpk::DLPackTensorRef;
use ndarray::ArrayViewD;

use crate::Error;

mod cpu;

/// Check that the values of an i32 DLPack tensor match the expected reference.
///
/// This dispatches to the appropriate backend based on the device of `tensor`.
///
/// # Parameters
/// - `tensor`: DLPack tensor with i32 data type
/// - `reference`: expected values with the same shape as the tensor
pub(crate) fn is_equal_i32(tensor: DLPackTensorRef<'_>, reference: ArrayViewD<'_, i32>) -> Result<bool, Error> {
    match tensor.device().device_type {
        DLDeviceType::kDLCPU
        | DLDeviceType::kDLCUDAHost
        | DLDeviceType::kDLROCMHost => {
            cpu::is_equal_i32(tensor, reference)
        }
        _ => {
            eprintln!(
                "is_equal_i32 for non-CPU devices is not implemented, \
                got data on device: {:?}", tensor.device()
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
        DLDeviceType::kDLCPU
        | DLDeviceType::kDLCUDAHost
        | DLDeviceType::kDLROCMHost => {
            cpu::validate_cell_pbc(pbc, cell)
        }
        _ => {
            eprintln!(
                "Cell/PBC validation for non-CPU devices is not implemented, \
                got data on device: {:?}", pbc.device()
            );
            Ok(())
        }
    }
}
