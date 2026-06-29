use dlpk::DLPackTensorRef;
use ndarray::{ArrayView1, ArrayView2, ArrayViewD};

use crate::Error;

/// Check that the values of an i32 DLPack tensor match the expected reference.
///
/// The tensor is converted to an ndarray view and compared element-wise and
/// shape-wise against `reference`. The `description` is used verbatim in the
/// error message on mismatch.
///
/// # Parameters
/// - `tensor`: DLPack tensor with i32 data type
/// - `reference`: expected values with the same shape as the tensor
pub(crate) fn is_equal_i32(
    tensor: DLPackTensorRef<'_>,
    reference: ArrayViewD<'_, i32>,
) -> Result<bool, Error> {
    let values: ArrayViewD<i32> = tensor.try_into()?;
    return Ok(values == reference);
}

macro_rules! validate_cell {
    ($T: ty, $pbc: expr, $cell: expr) => {
        let pbc_array: ArrayView1<bool> = $pbc.try_into()?;
        let cell_array: ArrayView2<$T> = $cell.try_into()?;
        for i in 0..3 {
            if !pbc_array[i] && !cell_array.row(i).iter().all(|&x| x == 0.0) {
                return Err(Error::InvalidParameter(format!(
                    "invalid cell: for non-periodic dimensions, the corresponding \
                    cell vector must be zero, but cell[{}] contains non-zero values",
                    i
                )));
            }
        }
    };
}

/// Validate that cell vectors are zero for non-periodic dimensions on CPU.
///
/// Converts the DLPack tensors to ndarray views and checks that for every
/// dimension where `pbc` is false, the corresponding row of `cell` contains
/// only zeros.
///
/// # Parameters
/// - `pbc`: 1D boolean tensor of length 3 (periodic boundary condition flags)
/// - `cell`: 3x3 tensor (unit cell vectors as rows)
pub(crate) fn validate_cell_pbc(
    pbc: DLPackTensorRef<'_>,
    cell: DLPackTensorRef<'_>,
) -> Result<(), Error> {
    let dtype = cell.dtype();
    if dtype.bits == 32 {
        validate_cell!(f32, pbc, cell);
    } else {
        assert_eq!(dtype.bits, 64);
        validate_cell!(f64, pbc, cell);
    }
    return Ok(());
}
