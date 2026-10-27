use metatensor::{TensorBlock, TensorMap};

use crate::kernels;
use crate::Error;

/// Scale all values and gradients in a `TensorMap` by `factor`. Gradients with
/// respect to positions are additionally scaled by `positions_gradient_factor`,
/// to account for a change of length unit.
///
/// If a block's values or gradients are marked as read-only in DLPack, a copy
/// is made via `mts_array_t.copy` and the copy is scaled; otherwise the data is
/// scaled in place. Copies of immutable arrays (e.g. JAX) are read-only too: they
/// are scaled in a clone owned by metatomic, and given back to the array's library
/// with `mts_array_t.from_dlpack`. This is a no-op when both factors are `1.0`.
pub(crate) fn scale_tensormap(
    tensor: TensorMap,
    factor: f64,
    positions_gradient_factor: f64,
) -> Result<TensorMap, Error> {
    #[allow(clippy::float_cmp)]
    if factor == 1.0 && positions_gradient_factor == 1.0 {
        return Ok(tensor);
    }

    let gradient_factor = |parameter: &str| {
        match parameter {
            "positions" => factor * positions_gradient_factor,
            "strain" => factor,
            _ => panic!("Unknown parameter: {}", parameter),
        }
    };

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
            let grad_dlpack = gradient.values().as_dlpack(device, None, dlpack_version)?;
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

            for (parameter, mut gradient) in block.gradients_mut() {
                let mut dlpack = gradient.values_mut().as_dlpack(device, None, dlpack_version)?;
                kernels::scale_inplace(dlpack.as_mut(), gradient_factor(parameter))?;
            }
        }
        Ok(tensor)
    } else {
        // At least one block is read-only: build new blocks with copied + scaled data
        let keys = tensor.keys().clone();
        let mut new_blocks = Vec::new();

        // Scale `copy` in place. Some libraries only create read-only arrays (e.g.
        // JAX, where all arrays are immutable): then scale a clone owned by
        // metatomic, and give it back to the array's library with `from_dlpack`.
        let scale_copy = |copy: metatensor::MtsArray, device, factor| -> Result<metatensor::MtsArray, Error> {
            let mut dlpack = copy.as_dlpack(device, None, dlpack_version)?;
            if !dlpack.is_read_only() {
                kernels::scale_inplace(dlpack.as_mut(), factor)?;
                drop(dlpack);
                return Ok(copy);
            }

            let mut owned = kernels::clone_tensor(&dlpack.as_ref())?;
            kernels::scale_inplace(owned.as_mut(), factor)?;
            Ok(copy.from_dlpack(owned)?)
        };

        for block in tensor.blocks() {
            // copy values, then scale the copy in place
            let device = block.values().device()?;
            let values_copy = scale_copy(block.values().copy(device)?, device, factor)?;

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
                let grad_copy = scale_copy(
                    gradient.values().copy(device)?,
                    device,
                    gradient_factor(parameter),
                )?;

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

        // add a gradient w.r.t. strain
        let strain_components = vec![
            Labels::new(["xyz_1"], [[0i32], [1i32], [2]]),
            Labels::new(["xyz_2"], [[0i32], [1i32], [2]]),
        ];
        let strain_values = ndarray::ArrayD::<f32>::from_elem(vec![1, 3, 3, 1], 1.0);
        let strain = TensorBlock::new(strain_values, &grad_samples, &strain_components, &grad_properties).unwrap();
        block.add_gradient("strain", strain).unwrap();

        let keys = Labels::new(["_"], [[0i32]]);
        let tensor_map = TensorMap::new(keys, vec![block]).unwrap();

        // scale by 2.5, and positions gradients by an additional 10.0
        let scaled = scale_tensormap(tensor_map, 2.5, 10.0).unwrap();

        // check values
        let block = scaled.block_by_id(0);
        let device = block.values().device().unwrap();
        let dlpack = block.values().as_dlpack(device, None, dlpk::sys::DLPackVersion::current()).unwrap();
        let values: ndarray::ArrayViewD<f32> = dlpack.as_ref().try_into().unwrap();
        assert_eq!(values, ndarray::arr2(&[[2.5_f32]]).into_dyn());

        // check positions gradient
        let gradient = block.gradient("positions").unwrap();
        let device = gradient.values().device().unwrap();
        let dlpack = gradient.values().as_dlpack(device, None, dlpk::sys::DLPackVersion::current()).unwrap();
        let grad_values: ndarray::ArrayViewD<f32> = dlpack.as_ref().try_into().unwrap();
        assert_eq!(grad_values, ndarray::arr1(&[50.0_f32, 100.0, 150.0]).to_shape(vec![1, 3, 1]).unwrap());

        // check strain gradient, which is only scaled by the main factor
        let gradient = block.gradient("strain").unwrap();
        let device = gradient.values().device().unwrap();
        let dlpack = gradient.values().as_dlpack(device, None, dlpk::sys::DLPackVersion::current()).unwrap();
        let grad_values: ndarray::ArrayViewD<f32> = dlpack.as_ref().try_into().unwrap();
        assert_eq!(grad_values, ndarray::ArrayD::<f32>::from_elem(vec![1, 3, 3, 1], 2.5));
    }

    /// An immutable CPU array, like JAX arrays: DLPack views are marked as
    /// read-only, `copy` returns another read-only array, and `from_dlpack` wraps
    /// the data in a new read-only array.
    struct ReadOnlyArray(ndarray::ArcArray<f32, ndarray::IxDyn>);

    impl metatensor::Array for ReadOnlyArray {
        fn as_any(&self) -> &dyn std::any::Any { self }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        fn create(&self, _: &[usize], _: metatensor::MtsArray) -> Box<dyn metatensor::Array> {
            unimplemented!()
        }
        fn copy(&self, _: dlpk::sys::DLDevice) -> Box<dyn metatensor::Array> {
            Box::new(ReadOnlyArray(self.0.to_owned().into_shared()))
        }
        fn shape(&self) -> Vec<usize> { self.0.shape().to_vec() }
        fn reshape(&mut self, _: &[usize]) { unimplemented!() }
        fn swap_axes(&mut self, _: usize, _: usize) { unimplemented!() }
        fn move_data(&mut self, _: &dyn metatensor::Array, _: &[metatensor::c_api::mts_data_movement_t]) {
            unimplemented!()
        }
        fn device(&self) -> dlpk::sys::DLDevice { dlpk::sys::DLDevice::cpu() }
        fn dtype(&self) -> dlpk::sys::DLDataType {
            <f32 as dlpk::GetDLPackDataType>::get_dlpack_data_type()
        }
        fn as_dlpack(
            &self,
            _: dlpk::sys::DLDevice,
            _: Option<i64>,
            _: dlpk::sys::DLPackVersion,
        ) -> Result<dlpk::DLPackTensor, metatensor::Error> {
            Ok(dlpk::DLPackTensor::try_from(self.0.clone()).expect("DLPack conversion"))
        }
        fn from_dlpack(&self, tensor: dlpk::DLPackTensor) -> Result<Box<dyn metatensor::Array>, metatensor::Error> {
            let array = ndarray::ArrayD::<f32>::try_from(tensor).expect("DLPack conversion");
            Ok(Box::new(ReadOnlyArray(array.into_shared())))
        }
    }

    #[test]
    fn test_scale_read_only_tensormap() {
        let read_only = |shape: Vec<usize>, data: Vec<f32>| {
            ReadOnlyArray(ndarray::ArrayD::from_shape_vec(shape, data).unwrap().into_shared())
        };
        let samples = Labels::new(["system"], [[0i32]]);
        let properties = Labels::new(["energy"], [[0i32]]);
        let mut block = TensorBlock::new(read_only(vec![1, 1], vec![1.0]), &samples, &[], &properties).unwrap();
        let gradient = TensorBlock::new(
            read_only(vec![1, 3, 1], vec![2.0, 4.0, 6.0]),
            &Labels::new(["sample"], [[0i32]]),
            &[Labels::new(["xyz"], [[0i32], [1i32], [2]])],
            &properties,
        ).unwrap();
        block.add_gradient("positions", gradient).unwrap();
        let tensor_map = TensorMap::new(Labels::new(["_"], [[0i32]]), vec![block]).unwrap();

        let scaled = scale_tensormap(tensor_map, 2.5, 10.0).unwrap();

        let block = scaled.block_by_id(0);
        let device = block.values().device().unwrap();
        let dlpack = block.values().as_dlpack(device, None, dlpk::sys::DLPackVersion::current()).unwrap();
        // the result comes back through `from_dlpack`, as an array of the same kind
        assert!(dlpack.is_read_only());
        let values: ndarray::ArrayViewD<f32> = dlpack.as_ref().try_into().unwrap();
        assert_eq!(values, ndarray::arr2(&[[2.5_f32]]).into_dyn());

        let gradient = block.gradient("positions").unwrap();
        let dlpack = gradient.values().as_dlpack(device, None, dlpk::sys::DLPackVersion::current()).unwrap();
        let values: ndarray::ArrayViewD<f32> = dlpack.as_ref().try_into().unwrap();
        assert_eq!(values, ndarray::arr1(&[50.0_f32, 100.0, 150.0]).to_shape(vec![1, 3, 1]).unwrap());
    }
}
