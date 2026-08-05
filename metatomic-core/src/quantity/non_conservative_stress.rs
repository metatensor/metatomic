use metatensor::{Labels, TensorMap};

use super::Quantity;
use super::checks::{self, ExpectedLabels, XYZ_LABELS_REFERENCE, SINGLE_LABELS_REFERENCE};

use crate::{Error, SampleKind, System};

/// Check the layout of the "non_conservative_stress" quantity.
pub(super) fn check(
    request: &Quantity,
    value: &TensorMap,
    systems: &[System],
    selected_atoms: Option<&Labels>
) -> Result<(), Error> {
    assert!(!request.name.is_custom() && request.name.base() == "non_conservative_stress");

    let context = format!("'{}'", request.name.full());
    checks::it_should_have_valid_sample_kind(&context, request.sample_kind, &[SampleKind::System])?;

    checks::it_should_have_a_single_block(&context, value)?;
    let block = value.block_by_id(0);

    checks::it_should_have_valid_samples(&context, request.sample_kind, block, systems, selected_atoms)?;
    checks::it_should_have_expected_components(&context, block, &[
        ExpectedLabels {
            names: &["xyz_1"],
            values: &XYZ_LABELS_REFERENCE,
            values_message: "[[0], [1], [2]]",
        },
        ExpectedLabels {
            names: &["xyz_2"],
            values: &XYZ_LABELS_REFERENCE,
            values_message: "[[0], [1], [2]]",
        },
    ])?;

    let expected_properties = ExpectedLabels {
        names: &["non_conservative_stress"],
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
            name: QuantityName::new("non_conservative_stress".into()).unwrap(),
            unit: "eV/Angstrom^3".into(),
            description: None,
            gradients: vec![],
            sample_kind: SampleKind::System,
        }
    }

    fn valid_xyz_components() -> Vec<Labels> {
        vec![
            Labels::new(["xyz_1"], [[0], [1], [2]]),
            Labels::new(["xyz_2"], [[0], [1], [2]]),
        ]
    }

    fn valid_block() -> TensorBlock {
        let samples = Labels::new(["system"], [[0]]);
        let properties = Labels::new(["non_conservative_stress"], [[0]]);
        let values = ArrayD::<f32>::from_shape_vec(vec![1, 3, 3, 1], vec![1.0; 9]).unwrap();
        TensorBlock::new(values, &samples, &valid_xyz_components(), &properties).unwrap()
    }

    fn valid_non_conservative_stress() -> TensorMap {
        let keys = Labels::new(["_"], [[0]]);
        TensorMap::new(keys, vec![valid_block()]).unwrap()
    }

    #[test]
    fn ok() {
        check(&valid_request(), &valid_non_conservative_stress(), &[system(3)], None).unwrap();
    }

    #[test]
    fn empty_systems() {
        // Empty systems slice, per-system output
        let mut request = valid_request();
        request.sample_kind = SampleKind::System;
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![0, 3, 3, 1], vec![]).unwrap(),
            &Labels::new(
                ["system"],
                Array2::<i32>::from_shape_vec((0, 1), vec![]).unwrap(),
            ),
            &valid_xyz_components(),
            &Labels::new(["non_conservative_stress"], [[0]]),
        ).unwrap();
        let non_conservative_stress = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        check(&request, &non_conservative_stress, &[], None).unwrap();
    }

    #[test]
    fn selected_atoms() {
        let selected_atoms = Labels::new(["system", "atom"], [[0, 0], [0, 1], [1, 0]]);
        let systems = [system(3), system(1)];

        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![2, 3, 3, 1], vec![1.0; 18]).unwrap(),
            &Labels::new(["system"], [[0], [1]]),
            &valid_xyz_components(),
            &Labels::new(["non_conservative_stress"], [[0]]),
        ).unwrap();
        let non_conservative_stress = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        check(&valid_request(), &non_conservative_stress, &systems, Some(&selected_atoms)).unwrap();
    }

    #[test]
    fn multiple_systems() {
        let samples = Labels::new(
            ["system"],
            [[0], [1]],
        );
        let properties = Labels::new(["non_conservative_stress"], [[0]]);
        let values = ArrayD::<f32>::from_shape_vec(vec![2, 3, 3, 1], vec![1.0; 18]).unwrap();
        let block = TensorBlock::new(values, &samples, &valid_xyz_components(), &properties).unwrap();

        let non_conservative_stress = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        check(&valid_request(), &non_conservative_stress, &[system(3), system(1)], None).unwrap();
    }

    #[test]
    fn invalid_sample_kind() {
        let mut request = valid_request();
        request.sample_kind = SampleKind::Atom;
        let err = check(&request, &valid_non_conservative_stress(), &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid sample_kind for 'non_conservative_stress': expected one of [system], got 'atom'"
        );
    }

    #[test]
    fn wrong_number_of_blocks() {
        let non_conservative_stress = TensorMap::new(Labels::empty(vec!["_"]), vec![]).unwrap();

        let err = check(&valid_request(), &non_conservative_stress, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid 'non_conservative_stress': expected a single block, but found 0 blocks"
        );

        let non_conservative_stress = TensorMap::new(
            Labels::new(["_"], [[0], [1]]),
            vec![valid_block(), valid_block()]
        ).unwrap();

        let err = check(&valid_request(), &non_conservative_stress, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid 'non_conservative_stress': expected a single block, but found 2 blocks"
        );
    }

    #[test]
    fn wrong_key() {
        let non_conservative_stress = TensorMap::new(Labels::new(["foo"], [[0]]), vec![valid_block()]).unwrap();
        let err = check(&valid_request(), &non_conservative_stress, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid 'non_conservative_stress': expected a single block with key '_', but found key names [foo]"
        );

        let non_conservative_stress = TensorMap::new(Labels::new(["_"], [[1]]), vec![valid_block()]).unwrap();
        let err = check(&valid_request(), &non_conservative_stress, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid 'non_conservative_stress': expected a single block with key value 0"
        );
    }

    #[test]
    fn wrong_property() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system"], [[0]]),
            &valid_xyz_components(),
            &Labels::new(["wrong"], [[0]]),
        ).unwrap();

        let non_conservative_stress = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &non_conservative_stress, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid properties for 'non_conservative_stress': expected names [non_conservative_stress], got [wrong]"
        );

        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system"], [[0]]),
            &valid_xyz_components(),
            &Labels::new(["non_conservative_stress"], [[1]]),
        ).unwrap();

        let non_conservative_stress = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &non_conservative_stress, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid properties values for 'non_conservative_stress': expected [[0]]"
        );
    }

    #[test]
    fn missing_components() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 1], vec![1.0]).unwrap(),
            &Labels::new(["system"], [[0]]),
            &[],
            &Labels::new(["non_conservative_stress"], [[0]]),
        ).unwrap();

        let non_conservative_stress = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();

        let err = check(&valid_request(), &non_conservative_stress, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid components for 'non_conservative_stress': expected 2 component(s), got 0"
        );
    }

    #[test]
    fn wrong_component() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system"], [[0]]),
            &[Labels::new(["abc"], [[0], [1], [2]]), Labels::new(["xyz_2"], [[0], [1], [2]])],
            &Labels::new(["non_conservative_stress"], [[0]]),
        ).unwrap();

        let non_conservative_stress = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &non_conservative_stress, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid components for 'non_conservative_stress': expected names [xyz_1], got [abc]"
        );

        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system"], [[0]]),
            &[Labels::new(["xyz_1"], [[1], [2], [3]]), Labels::new(["xyz_2"], [[0], [1], [2]])],
            &Labels::new(["non_conservative_stress"], [[0]]),
        ).unwrap();

        let non_conservative_stress = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &non_conservative_stress, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid components values for 'non_conservative_stress': expected [[0], [1], [2]]"
        );
    }

    #[test]
    fn extra_component() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 3, 3, 3, 1], vec![1.0; 27]).unwrap(),
            &Labels::new(["system"], [[0]]),
            &[
                Labels::new(["xyz_1"], [[0], [1], [2]]),
                Labels::new(["xyz_2"], [[0], [1], [2]]),
                Labels::new(["abc"], [[0], [1], [2]]),
            ],
            &Labels::new(["non_conservative_stress"], [[0]]),
        ).unwrap();

        let non_conservative_stress = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();

        let err = check(&valid_request(), &non_conservative_stress, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid components for 'non_conservative_stress': expected 2 component(s), got 3"
        );
    }

    #[test]
    fn wrong_sample_names() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0]]),
            &valid_xyz_components(),
            &Labels::new(["non_conservative_stress"], [[0]])
        ).unwrap();

        let non_conservative_stress = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();

        let err = check(&valid_request(), &non_conservative_stress, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid sample names for 'non_conservative_stress': expected [system], got [system, atom]"
        );
    }

    #[test]
    fn gradients() {
        let mut block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system"], [[0]]),
            &valid_xyz_components(),
            &Labels::new(["non_conservative_stress"], [[0]]),
        ).unwrap();

        let gradient = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["sample"], [[0]]),
            &valid_xyz_components(),
            &Labels::new(["non_conservative_stress"], [[0]]),
        ).unwrap();

        block.add_gradient("positions", gradient).unwrap();

        let non_conservative_stress = TensorMap::new(
            Labels::new(["_"], [[0]]),
            vec![block]
        ).unwrap();

        let err = check(&valid_request(), &non_conservative_stress, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid gradients for 'non_conservative_stress': expected no gradients, but found gradients with respect to [positions]"
        );
    }

    #[test]
    fn selected_atoms_error() {
        let selected_atoms = Labels::new(["system", "atom"], [[0, 0], [0, 1]]);

        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![2, 3, 3, 1], vec![1.0; 18]).unwrap(),
            // systems that are not in the selected_atoms
            &Labels::new(["system"], [[0], [1]]),
            &valid_xyz_components(),
            &Labels::new(["non_conservative_stress"], [[0]]),
        ).unwrap();
        let non_conservative_stress = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &non_conservative_stress, &[system(3), system(3)], Some(&selected_atoms)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid samples for 'non_conservative_stress', they do not match the `systems` and `selected_atoms`"
        );
    }
}
