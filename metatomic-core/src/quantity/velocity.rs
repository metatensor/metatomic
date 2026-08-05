use metatensor::{Labels, TensorMap};

use super::Quantity;
use super::checks::{self, ExpectedLabels, SINGLE_LABELS_REFERENCE, XYZ_LABELS_REFERENCE};

use crate::{Error, SampleKind, System};



/// Check the layout of the "velocity" quantity.
pub(super) fn check(
    request: &Quantity,
    value: &TensorMap,
    systems: &[System],
    selected_atoms: Option<&Labels>
) -> Result<(), Error> {
    assert!(!request.name.is_custom() && request.name.base() == "velocity");

    let context = format!("'{}'", request.name.full());
    checks::it_should_have_valid_sample_kind(&context, request.sample_kind, &[SampleKind::Atom])?;

    checks::it_should_have_a_single_block(&context, value)?;
    let block = value.block_by_id(0);

    checks::it_should_have_valid_samples(&context, request.sample_kind, block, systems, selected_atoms)?;
    checks::it_should_have_expected_components(&context, block, &[
        ExpectedLabels {
            names: &["xyz"],
            values: &XYZ_LABELS_REFERENCE,
            values_message: "[[0], [1], [2]]",
        },
    ])?;

    let expected_properties = ExpectedLabels {
        names: &["velocity"],
        values: &SINGLE_LABELS_REFERENCE,
        values_message: "[[0]]"
    };
    checks::it_should_have_expected_labels(&context, "properties", &block.properties(), expected_properties)?;
    checks::it_should_have_expected_gradients(&context, request, block, &[])?;

    return Ok(());
}

#[cfg(test)]
mod tests {
    use metatensor::{Labels, TensorBlock, TensorMap};
    use ndarray::{Array1, Array2, ArrayD};
    use dlpk::DLPackTensor;

    use crate::{Quantity, QuantityName, SampleKind, System};

    use super::check;

    fn system(n_atoms: usize) -> System {
        let types: DLPackTensor = Array1::<i32>::from_vec(vec![1; n_atoms]).try_into().unwrap();
        let positions: DLPackTensor = Array2::<f32>::from_shape_vec((n_atoms, 3), vec![0.0; n_atoms * 3]).unwrap().try_into().unwrap();
        let cell: DLPackTensor = Array2::<f32>::from_shape_vec(
            (3, 3),
            vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
        ).unwrap().try_into().unwrap();

        let pbc: DLPackTensor = Array1::<bool>::from_vec(vec![true, true, true]).try_into().unwrap();
        System::new("Angstrom".into(), types, positions, cell, pbc).unwrap()
    }

    fn valid_request() -> Quantity {
        Quantity {
            name: QuantityName::new("velocity".into()).unwrap(),
            unit: "Angstrom/ps".into(),
            description: None,
            gradients: vec![],
            sample_kind: SampleKind::Atom,
        }
    }

    fn valid_xyz_component() -> Labels {
        Labels::new(["xyz"], [[0], [1], [2]])
    }

    fn valid_block() -> TensorBlock {
        let samples = Labels::new(
            ["system", "atom"],
            [[0, 0], [0, 1], [0, 2]],
        );
        let properties = Labels::new(["velocity"], [[0]]);
        let values = ArrayD::<f32>::from_shape_vec(vec![3, 3, 1], vec![1.0; 9]).unwrap();
        TensorBlock::new(values, &samples, &[valid_xyz_component()], &properties).unwrap()
    }

    fn valid_velocity() -> TensorMap {
        let keys = Labels::new(["_"], [[0]]);
        TensorMap::new(keys, vec![valid_block()]).unwrap()
    }

    #[test]
    fn ok() {
        check(&valid_request(), &valid_velocity(), &[system(3)], None).unwrap();
    }

    #[test]
    fn empty_systems() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![0, 3, 1], vec![]).unwrap(),
            &Labels::new(
                ["system", "atom"],
                Array2::<i32>::from_shape_vec((0, 2), vec![]).unwrap(),
            ),
            &[valid_xyz_component()],
            &Labels::new(["velocity"], [[0]]),
        ).unwrap();
        let velocity = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        check(&valid_request(), &velocity, &[], None).unwrap();

        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![0, 3, 1], vec![]).unwrap(),
            &Labels::new(
                ["system", "atom"],
                Array2::<i32>::from_shape_vec((0, 2), vec![]).unwrap(),
            ),
            &[valid_xyz_component()],
            &Labels::new(["velocity"], [[0]]),
        ).unwrap();
        let velocity = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        check(&valid_request(), &velocity, &[system(0)], None).unwrap();
    }

    #[test]
    fn selected_atoms() {
        let selected_atoms = Labels::new(["system", "atom"], [[0, 0], [0, 1], [1, 0]]);
        let systems = [system(3), system(1)];

        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [1, 0]]),
            &[valid_xyz_component()],
            &Labels::new(["velocity"], [[0]]),
        ).unwrap();
        let velocity = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        check(&valid_request(), &velocity, &systems, Some(&selected_atoms)).unwrap();
    }

    #[test]
    fn multiple_systems() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![4, 3, 1], vec![1.0; 12]).unwrap(),
            &Labels::new(
                ["system", "atom"],
                [[0, 0], [0, 1], [0, 2], [1, 0]],
            ),
            &[valid_xyz_component()],
            &Labels::new(["velocity"], [[0]]),
        ).unwrap();

        let velocity = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        check(&valid_request(), &velocity, &[system(3), system(1)], None).unwrap();
    }

    #[test]
    fn invalid_sample_kind() {
        let mut request = valid_request();
        request.sample_kind = SampleKind::System;
        let err = check(&request, &valid_velocity(), &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid sample_kind for 'velocity': expected one of [atom], got 'system'"
        );
    }

    #[test]
    fn wrong_number_of_blocks() {
        let velocity = TensorMap::new(Labels::empty(vec!["_"]), vec![]).unwrap();

        let err = check(&valid_request(), &velocity, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid 'velocity': expected a single block, but found 0 blocks"
        );

        let velocity = TensorMap::new(
            Labels::new(["_"], [[0], [1]]),
            vec![valid_block(), valid_block()]
        ).unwrap();

        let err = check(&valid_request(), &velocity, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid 'velocity': expected a single block, but found 2 blocks"
        );
    }

    #[test]
    fn wrong_key() {
        let velocity = TensorMap::new(Labels::new(["foo"], [[0]]), vec![valid_block()]).unwrap();
        let err = check(&valid_request(), &velocity, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid 'velocity': expected a single block with key '_', but found key names [foo]"
        );

        let velocity = TensorMap::new(Labels::new(["_"], [[1]]), vec![valid_block()]).unwrap();
        let err = check(&valid_request(), &velocity, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid 'velocity': expected a single block with key value 0"
        );
    }

    #[test]
    fn wrong_property() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[valid_xyz_component()],
            &Labels::new(["wrong"], [[0]]),
        ).unwrap();

        let velocity = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &velocity, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid properties for 'velocity': expected names [velocity], got [wrong]"
        );

        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[valid_xyz_component()],
            &Labels::new(["velocity"], [[1]]),
        ).unwrap();

        let velocity = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &velocity, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid properties values for 'velocity': expected [[0]]"
        );
    }

    #[test]
    fn missing_components() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 1], vec![1.0; 3]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[],
            &Labels::new(["velocity"], [[0]]),
        ).unwrap();

        let velocity = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();

        let err = check(&valid_request(), &velocity, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid components for 'velocity': expected 1 component(s), got 0"
        );
    }

    #[test]
    fn wrong_component() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[Labels::new(["abc"], [[0], [1], [2]])],
            &Labels::new(["velocity"], [[0]]),
        ).unwrap();

        let velocity = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &velocity, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid components for 'velocity': expected names [xyz], got [abc]"
        );

        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[Labels::new(["xyz"], [[1], [2], [3]])],
            &Labels::new(["velocity"], [[0]]),
        ).unwrap();

        let velocity = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &velocity, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid components values for 'velocity': expected [[0], [1], [2]]"
        );
    }

    #[test]
    fn extra_component() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 3, 3, 1], vec![1.0; 27]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[
                valid_xyz_component(),
                Labels::new(["abc"], [[0], [1], [2]]),
            ],
            &Labels::new(["velocity"], [[0]]),
        ).unwrap();

        let velocity = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();

        let err = check(&valid_request(), &velocity, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid components for 'velocity': expected 1 component(s), got 2"
        );
    }

    #[test]
    fn wrong_sample_names() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 3, 1], vec![1.0, 2.0, 3.0]).unwrap(),
            &Labels::new(["system"], [[0]]),
            &[valid_xyz_component()],
            &Labels::new(["velocity"], [[0]])
        ).unwrap();

        let velocity = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();

        let err = check(&valid_request(), &velocity, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid sample names for 'velocity': expected [system, atom], got [system]"
        );
    }

    #[test]
    fn gradients() {
        let mut block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[valid_xyz_component()],
            &Labels::new(["velocity"], [[0]]),
        ).unwrap();

        let gradient = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 3, 1], vec![0.1, 0.2, 0.3]).unwrap(),
            &Labels::new(["sample", "system", "atom"], [[0, 0, 0]]),
            &[valid_xyz_component()],
            &Labels::new(["velocity"], [[0]]),
        ).unwrap();

        block.add_gradient("positions", gradient).unwrap();

        let velocity = TensorMap::new(
            Labels::new(["_"], [[0]]),
            vec![block]
        ).unwrap();

        let err = check(&valid_request(), &velocity, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid gradients for 'velocity': expected no gradients, but found gradients with respect to [positions]"
        );
    }

    #[test]
    fn selected_atoms_error() {
        let selected_atoms = Labels::new(["system", "atom"], [[0, 0], [0, 1]]);

        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 3, 1], vec![1.0; 9]).unwrap(),
            // samples that are not in the selected_atoms
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[valid_xyz_component()],
            &Labels::new(["velocity"], [[0]]),
        ).unwrap();
        let velocity = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &velocity, &[system(3)], Some(&selected_atoms)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid samples for 'velocity', they do not match the `systems` and `selected_atoms`"
        );
    }
}
