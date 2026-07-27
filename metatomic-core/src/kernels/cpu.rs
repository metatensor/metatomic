use dlpk::{DLPackTensorRef, DLPackTensorRefMut};
use ndarray::{ArrayView1, ArrayView2, ArrayViewD, ArrayViewMutD};

use crate::Error;
use super::ReferenceValue;

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
    reference: &ReferenceValue<i32>,
) -> Result<bool, Error> {
    let values: ArrayViewD<i32> = tensor.try_into()?;
    return Ok(values == reference.cpu.view());
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

/// Scale all elements of `tensor` in place by `factor` on CPU.
///
/// Supports 32-bit and 64-bit floating point tensors. The tensor is converted
/// to a mutable ndarray view and scaled element-wise.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn scale_inplace(
    tensor: DLPackTensorRefMut<'_>,
    factor: f64,
) -> Result<(), Error> {
    let dtype = tensor.dtype();
    if dtype.code == dlpk::sys::DLDataTypeCode::kDLFloat && dtype.bits == 32 {
        let mut view: ArrayViewMutD<f32> = tensor.try_into()?;
        view *= factor as f32;
    } else if dtype.code == dlpk::sys::DLDataTypeCode::kDLFloat && dtype.bits == 64 {
        let mut view: ArrayViewMutD<f64> = tensor.try_into()?;
        view *= factor;
    } else {
        return Err(Error::InvalidParameter(format!(
            "scale_inplace only supports 32-bit or 64-bit floats, got {}-bit {:?}",
            dtype.bits, dtype.code
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ReferenceValue;

    use dlpk::DLPackTensor;
    use ndarray::{Array1, Array2, ArrayD};

    #[test]
    fn test_is_equal_i32() {
        let data = ArrayD::<i32>::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        let tensor: DLPackTensor = data.try_into().unwrap();
        let reference = ReferenceValue::new(ArrayD::<i32>::from_shape_vec(
            vec![2, 3],
            vec![1, 2, 3, 4, 5, 6],
        ).unwrap());

        assert!(is_equal_i32(tensor.as_ref(), &reference).unwrap());

        let data = ArrayD::<i32>::from_shape_vec(vec![2, 3], vec![1, 2, 3, 42, 5, 6]).unwrap();
        let tensor: DLPackTensor = data.try_into().unwrap();
        assert!(!is_equal_i32(tensor.as_ref(), &reference).unwrap());

        // shape mismatch
        let data = ArrayD::<i32>::from_shape_vec(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
        let tensor: DLPackTensor = data.try_into().unwrap();
        let reference = ReferenceValue::new(ArrayD::<i32>::from_shape_vec(
            vec![4],
            vec![1, 2, 3, 4],
        ).unwrap());

        assert!(!is_equal_i32(tensor.as_ref(), &reference).unwrap());

        // empty arrays
        let data = ArrayD::<i32>::from_shape_vec(vec![0], vec![]).unwrap();
        let tensor: DLPackTensor = data.try_into().unwrap();
        let reference = ReferenceValue::new(ArrayD::<i32>::from_shape_vec(vec![0], vec![]).unwrap());
        assert!(is_equal_i32(tensor.as_ref(), &reference).unwrap());
    }

    #[test]
    fn test_validate_cell_pbc() {
        // helper: pbc flags + 3x3 cell (row-major) => expected Ok or error substring
        fn check(pbc: &[bool], cell: &[f64]) -> Result<(), Error> {
            let pbc = Array1::<bool>::from_vec(pbc.to_vec());
            let cell = Array2::<f64>::from_shape_vec((3, 3), cell.to_vec()).unwrap();

            let pbc: DLPackTensor = pbc.try_into().unwrap();
            let cell: DLPackTensor = cell.try_into().unwrap();

            validate_cell_pbc(pbc.as_ref(), cell.as_ref())
        }

        // fully periodic — any cell is fine
        check(&[true, true, true], &[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]).unwrap();

        // fully periodic — non-diagonal cell is fine too
        check(&[true, true, true], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

        // non-periodic dim with zero cell vector — ok
        check(&[true, false, true], &[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]).unwrap();

        // non-periodic dim with nonzero cell vector — error
        let err = check(
            &[true, false, true],
            &[10.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 10.0],
        ).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid cell: for non-periodic dimensions, \
            the corresponding cell vector must be zero, but cell[1] contains non-zero values"
        );

        // first dim non-periodic with nonzero cell
        let err = check(
            &[false, true, true],
            &[1.0, 2.0, 3.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
        ).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid cell: for non-periodic dimensions, \
            the corresponding cell vector must be zero, but cell[0] contains non-zero values"
        );

        // last dim non-periodic with nonzero cell
        let err = check(
            &[true, true, false],
            &[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 7.0, 8.0, 9.0],
        ).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid cell: for non-periodic dimensions, \
            the corresponding cell vector must be zero, but cell[2] contains non-zero values"
        );

        // all non-periodic with zero cell — ok
        check(&[false, false, false], &[0.0; 9]).unwrap();

        // f32 path
        {
            let pbc = Array1::<bool>::from_vec(vec![true, false, true]);
            let cell = Array2::<f32>::from_shape_vec(
                (3, 3),
                vec![10.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 10.0],
            ).unwrap();

            let pbc: DLPackTensor = pbc.try_into().unwrap();
            let cell: DLPackTensor = cell.try_into().unwrap();

            let err = validate_cell_pbc(pbc.as_ref(), cell.as_ref()).unwrap_err();
            assert!(err.to_string().contains("cell[1] contains non-zero values"));
        }
    }

    #[test]
    fn test_scale_inplace() {
        // f32 2D
        {
            let data = ArrayD::<f32>::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
            let mut tensor: DLPackTensor = data.try_into().unwrap();

            scale_inplace(tensor.as_mut(), 2.5).unwrap();

            let view: ArrayViewD<f32> = tensor.as_ref().try_into().unwrap();
            assert_eq!(view, ndarray::arr2(&[[2.5_f32, 5.0, 7.5], [10.0, 12.5, 15.0]]).into_dyn());
        }

        // f64 2D
        {
            let data = ArrayD::<f64>::from_shape_vec(vec![2, 2], vec![1.0, -2.0, 3.5, 0.0]).unwrap();
            let mut tensor: DLPackTensor = data.try_into().unwrap();

            scale_inplace(tensor.as_mut(), 0.5).unwrap();

            let view: ArrayViewD<f64> = tensor.as_ref().try_into().unwrap();
            assert_eq!(view, ndarray::arr2(&[[0.5_f64, -1.0], [1.75, 0.0]]).into_dyn());
        }

        // zero factor
        {
            let data = ArrayD::<f32>::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
            let mut tensor: DLPackTensor = data.try_into().unwrap();

            scale_inplace(tensor.as_mut(), 0.0).unwrap();

            let view: ArrayViewD<f32> = tensor.as_ref().try_into().unwrap();
            assert_eq!(view, ndarray::arr1(&[0.0_f32, 0.0, 0.0]).into_dyn());
        }

        // 1D f64
        {
            let data = ArrayD::<f64>::from_shape_vec(vec![4], vec![2.0, 4.0, 8.0, 16.0]).unwrap();
            let mut tensor: DLPackTensor = data.try_into().unwrap();

            scale_inplace(tensor.as_mut(), 0.25).unwrap();

            let view: ArrayViewD<f64> = tensor.as_ref().try_into().unwrap();
            assert_eq!(view, ndarray::arr1(&[0.5_f64, 1.0, 2.0, 4.0]).into_dyn());
        }
    }
}
