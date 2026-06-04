use metatensor::{Labels, TensorMap};

use super::Quantity;
use super::checks::{self, ExpectedLabels, SINGLE_LABELS_REFERENCE};

use crate::{Error, SampleKind, System};


/// Check the layout of the "mass" quantity.
pub(super) fn check(
    request: &Quantity,
    value: &TensorMap,
    systems: &[System],
    selected_atoms: Option<&Labels>
) -> Result<(), Error> {
    assert!(!request.name.is_custom() && request.name.base() == "mass");

    let context = format!("'{}'", request.name.full());
    checks::it_should_have_valid_sample_kind(&context, request.sample_kind, &[SampleKind::Atom])?;

    checks::it_should_have_a_single_block(&context, value)?;
    let block = value.block_by_id(0);

    checks::it_should_have_valid_samples(&context, request.sample_kind, block, systems, selected_atoms)?;
    checks::it_should_have_expected_components(&context, block, &[])?;

    let expected_properties = ExpectedLabels {
        names: &["mass"],
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
            name: QuantityName::new("mass".into()).unwrap(),
            unit: "dalton".into(),
            description: None,
            gradients: vec![],
            sample_kind: SampleKind::Atom,
        }
    }

    fn valid_block() -> TensorBlock {
        let samples = Labels::new(
            ["system", "atom"],
            [[0, 0], [0, 1], [0, 2]],
        );
        let properties = Labels::new(["mass"], [[0]]);
        let values = ArrayD::<f32>::from_shape_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
        TensorBlock::new(values, &samples, &[], &properties).unwrap()
    }

    fn valid_mass() -> TensorMap {
        let keys = Labels::new(["_"], [[0]]);
        TensorMap::new(keys, vec![valid_block()]).unwrap()
    }

    #[test]
    fn ok() {
        check(&valid_request(), &valid_mass(), &[system(3)], None).unwrap();
    }

    #[test]
    fn empty_systems() {
        // Empty systems slice, per-atom output
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![0, 1], vec![]).unwrap(),
            &Labels::new(
                ["system", "atom"],
                Array2::<i32>::from_shape_vec((0, 2), vec![]).unwrap(),
            ),
            &[],
            &Labels::new(["mass"], [[0]]),
        ).unwrap();
        let mass = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        check(&valid_request(), &mass, &[], None).unwrap();

        // System with 0 atoms, per-atom output
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![0, 1], vec![]).unwrap(),
            &Labels::new(
                ["system", "atom"],
                Array2::<i32>::from_shape_vec((0, 2), vec![]).unwrap(),
            ),
            &[],
            &Labels::new(["mass"], [[0]]),
        ).unwrap();
        let mass = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        check(&valid_request(), &mass, &[system(0)], None).unwrap();
    }

    #[test]
    fn selected_atoms() {
        // Per-atom output with selected_atoms across multiple systems
        let selected_atoms = Labels::new(["system", "atom"], [[0, 0], [0, 1], [1, 0]]);
        let systems = [system(3), system(1)];

        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [1, 0]]),
            &[],
            &Labels::new(["mass"], [[0]]),
        ).unwrap();
        let mass = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        check(&valid_request(), &mass, &systems, Some(&selected_atoms)).unwrap();
    }

    #[test]
    fn multiple_systems() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![4, 1], vec![1.0; 4]).unwrap(),
            &Labels::new(
                ["system", "atom"],
                [[0, 0], [0, 1], [0, 2], [1, 0]],
            ),
            &[],
            &Labels::new(["mass"], [[0]]),
        ).unwrap();

        let mass = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        check(&valid_request(), &mass, &[system(3), system(1)], None).unwrap();
    }

    #[test]
    fn invalid_sample_kind() {
        let mut request = valid_request();
        request.sample_kind = SampleKind::System;
        let err = check(&request, &valid_mass(), &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid sample_kind for 'mass': expected one of [atom], got 'system'"
        );
    }

    #[test]
    fn wrong_number_of_blocks() {
        let mass = TensorMap::new(Labels::empty(vec!["_"]), vec![]).unwrap();

        let err = check(&valid_request(), &mass, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid 'mass': expected a single block, but found 0 blocks"
        );

        let mass = TensorMap::new(
            Labels::new(["_"], [[0], [1]]),
            vec![valid_block(), valid_block()]
        ).unwrap();

        let err = check(&valid_request(), &mass, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid 'mass': expected a single block, but found 2 blocks"
        );
    }

    #[test]
    fn wrong_key() {
        let mass = TensorMap::new(Labels::new(["foo"], [[0]]), vec![valid_block()]).unwrap();
        let err = check(&valid_request(), &mass, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid 'mass': expected a single block with key '_', but found key names [foo]"
        );

        let mass = TensorMap::new(Labels::new(["_"], [[1]]), vec![valid_block()]).unwrap();
        let err = check(&valid_request(), &mass, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid 'mass': expected a single block with key value 0"
        );
    }

    #[test]
    fn wrong_property() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[],
            &Labels::new(["wrong"], [[0]]),
        ).unwrap();

        let mass = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &mass, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid properties for 'mass': expected names [mass], got [wrong]"
        );

        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[],
            &Labels::new(["mass"], [[1]]),
        ).unwrap();

        let mass = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &mass, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid properties values for 'mass': expected [[0]]"
        );
    }

    #[test]
    fn has_components() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 3, 1], vec![1.0; 9]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[Labels::new(["xyz"], [[0], [1], [2]])],
            &Labels::new(["mass"], [[0]]),
        ).unwrap();

        let mass = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();

        let err = check(&valid_request(), &mass, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: components for 'mass' should be empty"
        );
    }

    #[test]
    fn wrong_sample_names() {
        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 1], vec![1.0]).unwrap(),
            &Labels::new(["system"], [[0]]),
            &[],
            &Labels::new(["mass"], [[0]])
        ).unwrap();

        let mass = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();

        let err = check(&valid_request(), &mass, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid sample names for 'mass': expected [system, atom], got [system]"
        );
    }

    #[test]
    fn gradients() {
        let mut block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap(),
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[],
            &Labels::new(["mass"], [[0]]),
        ).unwrap();

        let gradient = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![1, 3, 1], vec![0.1, 0.2, 0.3]).unwrap(),
            &Labels::new(["sample", "system", "atom"], [[0, 0, 0]]),
            &[Labels::new(["xyz"], [[0], [1], [2]])],
            &Labels::new(["mass"], [[0]]),
        ).unwrap();

        block.add_gradient("positions", gradient).unwrap();

        let mass = TensorMap::new(
            Labels::new(["_"], [[0]]),
            vec![block]
        ).unwrap();

        let err = check(&valid_request(), &mass, &[system(3)], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid gradients for 'mass': expected no gradients, but found gradients with respect to [positions]"
        );
    }

    #[test]
    fn selected_atoms_error() {
        let selected_atoms = Labels::new(["system", "atom"], [[0, 0], [0, 1]]);

        let block = TensorBlock::new(
            ArrayD::<f32>::from_shape_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap(),
            // samples that are not in the selected_atoms
            &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
            &[],
            &Labels::new(["mass"], [[0]]),
        ).unwrap();
        let mass = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
        let err = check(&valid_request(), &mass, &[system(3)], Some(&selected_atoms)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid samples for 'mass', they do not match the `systems` and `selected_atoms`"
        );
    }
}
