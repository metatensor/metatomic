use metatensor::{TensorBlock, TensorMap};

use crate::kernels;
use crate::Error;

/// Scale all values and gradients in a `TensorMap` by `factor`.
///
/// If a block's values or gradients are marked as read-only in DLPack, a copy
/// is made via `mts_array_t.copy` and the copy is scaled; otherwise the data is
/// scaled in place. This is a no-op when `factor == 1.0`.
pub(crate) fn scale_tensormap(tensor: TensorMap, factor: f64) -> Result<TensorMap, Error> {
    #[allow(clippy::float_cmp)]
    if factor == 1.0 {
        return Ok(tensor);
    }

    let dlpack_version = dlpk::sys::DLPackVersion::current();

    // Check if all blocks' values and gradients are writable
    let mut all_writable = true;
    for block in tensor.blocks() {
        let device = block.values().device()?;
        let dlpack = block.values().as_dlpack(device, None, dlpack_version)?;
        if dlpack.is_read_only() {
            all_writable = false;
            break;
        }
        for (_, gradient) in block.gradients() {
            let grad_device = gradient.values().device()?;
            let grad_dlpack = gradient.values().as_dlpack(grad_device, None, dlpack_version)?;
            if grad_dlpack.is_read_only() {
                all_writable = false;
                break;
            }
        }
        if !all_writable {
            break;
        }
    }

    if all_writable {
        // Scale in place
        let mut tensor = tensor;
        for mut block in tensor.blocks_mut() {
            let device = block.values().device()?;
            let mut dlpack = block.values_mut().as_dlpack(device, None, dlpack_version)?;
            kernels::scale_inplace(dlpack.as_mut(), factor)?;

            for (_, mut gradient) in block.gradients_mut() {
                let device = gradient.values().device()?;
                let mut dlpack = gradient.values_mut().as_dlpack(device, None, dlpack_version)?;
                kernels::scale_inplace(dlpack.as_mut(), factor)?;
            }
        }
        Ok(tensor)
    } else {
        // At least one block is read-only: build new blocks with copied + scaled data
        let keys = tensor.keys().clone();
        let mut new_blocks = Vec::new();

        for block in tensor.blocks() {
            // copy values, then scale the copy in place
            let device = block.values().device()?;
            let values_copy = block.values().copy(device)?;
            let mut dlpack = values_copy.as_dlpack(device, None, dlpack_version)?;
            assert!(!dlpack.is_read_only(), "copy of the value is still read only");
            kernels::scale_inplace(dlpack.as_mut(), factor)?;

            let samples = block.samples();
            let components = block.components();
            let properties = block.properties();
            let mut new_block = TensorBlock::new(
                values_copy,
                &samples,
                &components,
                &properties,
            )?;

            // copy and scale all gradients
            for (parameter, gradient) in block.gradients() {
                let grad_device = gradient.values().device()?;
                let grad_copy = gradient.values().copy(grad_device)?;
                let mut dlpack = grad_copy.as_dlpack(device, None, dlpack_version)?;
                assert!(!dlpack.is_read_only(), "copy of the gradients is still read only");
                kernels::scale_inplace(dlpack.as_mut(), factor)?;

                let grad_samples = gradient.samples();
                let grad_components = gradient.components();
                let grad_properties = gradient.properties();
                let new_gradient = TensorBlock::new(
                    grad_copy,
                    &grad_samples,
                    &grad_components,
                    &grad_properties,
                )?;

                new_block.add_gradient(parameter, new_gradient)?;
            }

            new_blocks.push(new_block);
        }

        TensorMap::new(keys, new_blocks).map_err(Error::from)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use metatensor::Labels;

    #[test]
    fn test_scale_tensormap() {
        // build a TensorMap with one block containing values and a gradient
        let samples = Labels::new(["system"], [[0i32]]);
        let components = vec![];
        let properties = Labels::new(["energy"], [[0i32]]);
        let values = ndarray::ArrayD::<f32>::from_shape_vec(vec![1, 1], vec![1.0]).unwrap();
        let mut block = TensorBlock::new(values, &samples, &components, &properties).unwrap();

        // add a gradient w.r.t. positions
        let grad_samples = Labels::new(["sample"], [[0i32]]);
        let grad_components = vec![Labels::new(["xyz"], [[0i32], [1i32], [2]])];
        let grad_properties = Labels::new(["energy"], [[0i32]]);
        let grad_values = ndarray::ArrayD::<f32>::from_shape_vec(vec![1, 3, 1], vec![2.0, 4.0, 6.0]).unwrap();
        let gradient = TensorBlock::new(grad_values, &grad_samples, &grad_components, &grad_properties).unwrap();
        block.add_gradient("positions", gradient).unwrap();

        let keys = Labels::new(["_"], [[0i32]]);
        let tensor_map = TensorMap::new(keys, vec![block]).unwrap();

        // scale by 2.5
        let scaled = scale_tensormap(tensor_map, 2.5).unwrap();

        // check values
        let block = scaled.block_by_id(0);
        let device = block.values().device().unwrap();
        let dlpack = block.values().as_dlpack(device, None, dlpk::sys::DLPackVersion::current()).unwrap();
        let values: ndarray::ArrayViewD<f32> = dlpack.as_ref().try_into().unwrap();
        assert_eq!(values, ndarray::arr2(&[[2.5_f32]]).into_dyn());

        // check gradient
        let gradient = block.gradient("positions").unwrap();
        let device = gradient.values().device().unwrap();
        let dlpack = gradient.values().as_dlpack(device, None, dlpk::sys::DLPackVersion::current()).unwrap();
        let grad_values: ndarray::ArrayViewD<f32> = dlpack.as_ref().try_into().unwrap();
        assert_eq!(grad_values, ndarray::arr1(&[5.0_f32, 10.0, 15.0]).to_shape(vec![1, 3, 1]).unwrap());
    }
}
