use metatensor::{Labels, TensorMap};

use super::Quantity;
use super::checks::{self, ExpectedLabels, SINGLE_LABELS_REFERENCE};

use crate::{Error, SampleKind, System};
use crate::kernels::ReferenceValue;


/// Check the layout of one of the energy-related quantities ("energy",
/// "energy_ensemble", "energy_uncertainty").
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub(super) fn check(
    request: &Quantity,
    value: &TensorMap,
    systems: &[System],
    selected_atoms: Option<&Labels>
) -> Result<(), Error> {
    let name = &request.name;
    assert!(!name.is_custom());
    assert!(name.base() == "energy" || name.base() == "energy_ensemble" || name.base() == "energy_uncertainty");

    let context = format!("'{}'", name.full());
    checks::it_should_have_valid_sample_kind(&context, request.sample_kind, &[SampleKind::System, SampleKind::Atom])?;

    checks::it_should_have_a_single_block(&context, value)?;
    let block = value.block_by_id(0);

    checks::it_should_have_valid_samples(&context, request.sample_kind, block, systems, selected_atoms)?;
    checks::it_should_have_expected_components(&context, block, &[])?;

    if name.base() == "energy" || name.base() == "energy_uncertainty" {
        checks::it_should_have_expected_labels(
            &context,
           "properties",
            &block.properties(),
            ExpectedLabels {
                names: &["energy"],
                values: &SINGLE_LABELS_REFERENCE,
                values_message: "[[0]]",
            }
        )?;
    } else {
        let n_ensemble_members = *block.values().shape()?.last().expect("energy block has an empty shape");
        let reference  = ReferenceValue::new(ndarray::ArrayD::from_shape_vec(
            vec![n_ensemble_members, 1], (0..n_ensemble_members as i32).collect()
        ).expect("created invalid array for energy_ensemble properties"));
        checks::it_should_have_expected_labels(
            &context,
            "properties",
            &block.properties(),
            ExpectedLabels {
                names: &["energy"],
                values: &reference,
                values_message: "[[0, ..., n]]",
            }
        )?;
    }

    checks::it_should_have_expected_gradients(&context, request, block, &["strain", "positions"])?;
    return Ok(());
}

#[cfg(test)]
mod tests {
    // use a macro to generate the test code for all three energy-related quantities
    macro_rules! energy_tests {
        ($base_name: ident) => {
            mod $base_name {
                use metatensor::{Labels, TensorBlock, TensorMap};
                use ndarray::{Array1, Array2, ArrayD};
                use dlpk::DLPackTensor;

                use crate::{Gradients, Quantity, QuantityName, SampleKind, System};

                use super::super::check;

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
                        name: QuantityName::new(String::from(stringify!($base_name))).unwrap(),
                        unit: "eV".into(),
                        description: None,
                        gradients: vec![Gradients::Positions, Gradients::Strain],
                        sample_kind: SampleKind::Atom,
                    }
                }

                fn n_properties() -> usize {
                    if stringify!($base_name) == "energy_ensemble" { 2 } else { 1 }
                }

                fn property_labels() -> Labels {
                    if stringify!($base_name) == "energy_ensemble" {
                        Labels::new(["energy"], [[0], [1]])
                    } else {
                        Labels::new(["energy"], [[0]])
                    }
                }

                fn valid_block() -> TensorBlock {
                    let samples = Labels::new(
                        ["system", "atom"],
                        [[0, 0], [0, 1], [0, 2]],
                    );
                    let n_props = n_properties();
                    let values = ArrayD::<f32>::from_shape_vec(
                        vec![3, n_props],
                        vec![1.0; 3 * n_props],
                    ).unwrap();
                    TensorBlock::new(values, &samples, &[], &property_labels()).unwrap()
                }

                fn with_gradients(block: &mut TensorBlock) {
                    let n_props = n_properties();
                    let props = property_labels();

                    let pos_gradient = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(
                            vec![1, 3, n_props],
                            vec![0.1; 3 * n_props],
                        ).unwrap(),
                        &Labels::new(["sample", "system", "atom"], [[0, 0, 0]]),
                        &[Labels::new(["xyz"], [[0], [1], [2]])],
                        &props,
                    ).unwrap();
                    block.add_gradient("positions", pos_gradient).unwrap();

                    let strain_gradient = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(
                            vec![1, 3, 3, n_props],
                            vec![0.1; 9 * n_props],
                        ).unwrap(),
                        &Labels::new(["sample"], [[0]]),
                        &[
                            Labels::new(["xyz_1"], [[0], [1], [2]]),
                            Labels::new(["xyz_2"], [[0], [1], [2]]),
                        ],
                        &props,
                    ).unwrap();
                    block.add_gradient("strain", strain_gradient).unwrap();
                }

                fn valid_energy() -> TensorMap {
                    let mut block = valid_block();
                    with_gradients(&mut block);
                    let keys = Labels::new(["_"], [[0]]);
                    TensorMap::new(keys, vec![block]).unwrap()
                }

                fn system_energy() -> TensorMap {
                    let samples = Labels::new(["system"], [[0]]);
                    let n_props = n_properties();
                    let values = ArrayD::<f32>::from_shape_vec(
                        vec![1, n_props],
                        vec![2.0; n_props],
                    ).unwrap();
                    let mut block = TensorBlock::new(values, &samples, &[], &property_labels()).unwrap();
                    with_gradients(&mut block);
                    let keys = Labels::new(["_"], [[0]]);
                    TensorMap::new(keys, vec![block]).unwrap()
                }

                #[test]
                fn ok() {
                    check(&valid_request(), &valid_energy(), &[system(3)], None).unwrap();

                    let mut request = valid_request();
                    request.sample_kind = SampleKind::System;
                    check(&request, &system_energy(), &[system(3)], None).unwrap();
                }

                #[test]
                fn empty_systems() {
                    // Empty systems slice, per-atom output
                    let n_props = n_properties();
                    let block = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(vec![0, n_props], vec![]).unwrap(),
                        &Labels::new(
                            ["system", "atom"],
                            Array2::<i32>::from_shape_vec((0, 2), vec![]).unwrap(),
                        ),
                        &[],
                        &property_labels(),
                    ).unwrap();
                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    check(&valid_request(), &energy, &[], None).unwrap();

                    // System with 0 atoms, per-atom output
                    let block = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(vec![0, n_props], vec![]).unwrap(),
                        &Labels::new(
                            ["system", "atom"],
                            Array2::<i32>::from_shape_vec((0, 2), vec![]).unwrap(),
                        ),
                        &[],
                        &property_labels(),
                    ).unwrap();
                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    check(&valid_request(), &energy, &[system(0)], None).unwrap();

                    // Empty systems slice, per-system output
                    let mut request = valid_request();
                    request.sample_kind = SampleKind::System;
                    let block = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(vec![0, n_props], vec![]).unwrap(),
                        &Labels::new(
                            ["system"],
                            Array2::<i32>::from_shape_vec((0, 1), vec![]).unwrap(),
                        ),
                        &[],
                        &property_labels(),
                    ).unwrap();
                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    check(&request, &energy, &[], None).unwrap();
                }

                #[test]
                fn selected_atoms() {
                    let selected_atoms = Labels::new(["system", "atom"], [[0, 0], [0, 1], [1, 0]]);
                    let systems = [system(3), system(1)];

                    let n_props = n_properties();
                    let block = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(vec![3, n_props], vec![1.0; 3 * n_props]).unwrap(),
                        &Labels::new(["system", "atom"], [[0, 0], [0, 1], [1, 0]]),
                        &[],
                        &property_labels(),
                    ).unwrap();
                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    check(&valid_request(), &energy, &systems, Some(&selected_atoms)).unwrap();

                    let mut request = valid_request();
                    request.sample_kind = SampleKind::System;
                    let block = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(vec![2, n_props], vec![2.0; 2 * n_props]).unwrap(),
                        &Labels::new(["system"], [[0], [1]]),
                        &[],
                        &property_labels(),
                    ).unwrap();
                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    check(&request, &energy, &systems, Some(&selected_atoms)).unwrap();
                }

                #[test]
                fn multiple_systems() {
                    let samples = Labels::new(
                        ["system", "atom"],
                        [[0, 0], [0, 1], [0, 2], [1, 0]],
                    );
                    let n_props = n_properties();
                    let values = ArrayD::<f32>::from_shape_vec(
                        vec![4, n_props],
                        vec![1.0; 4 * n_props],
                    ).unwrap();
                    let mut block = TensorBlock::new(values, &samples, &[], &property_labels()).unwrap();
                    with_gradients(&mut block);
                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    check(&valid_request(), &energy, &[system(3), system(1)], None).unwrap();
                }

                #[test]
                fn invalid_sample_kind() {
                    let mut request = valid_request();
                    request.sample_kind = SampleKind::AtomPair;
                    let err = check(&request, &valid_energy(), &[system(3)], None).unwrap_err();
                    assert_eq!(
                        err.to_string(),
                        format!(
                            "invalid parameter: invalid sample_kind for '{}': expected one of [system, atom], got 'atom_pair'",
                            stringify!($base_name)
                        )
                    );
                }

                #[test]
                fn wrong_number_of_blocks() {
                    let energy = TensorMap::new(Labels::empty(vec!["_"]), vec![]).unwrap();
                    let err = check(&valid_request(), &energy, &[system(3)], None).unwrap_err();
                    assert_eq!(
                        err.to_string(),
                        format!(
                            "invalid parameter: invalid '{}': expected a single block, but found 0 blocks",
                            stringify!($base_name)
                        )
                    );

                    let energy = TensorMap::new(
                        Labels::new(["_"], [[0], [1]]),
                        vec![valid_block(), valid_block()]
                    ).unwrap();
                    let err = check(&valid_request(), &energy, &[system(3)], None).unwrap_err();
                    assert_eq!(
                        err.to_string(),
                        format!(
                            "invalid parameter: invalid '{}': expected a single block, but found 2 blocks",
                            stringify!($base_name)
                        )
                    );
                }

                #[test]
                fn wrong_key() {
                    let energy = TensorMap::new(Labels::new(["foo"], [[0]]), vec![valid_block()]).unwrap();
                    let err = check(&valid_request(), &energy, &[system(3)], None).unwrap_err();
                    assert_eq!(
                        err.to_string(),
                        format!(
                            "invalid parameter: invalid '{}': expected a single block with key '_', but found key names [foo]",
                            stringify!($base_name)
                        )
                    );

                    let energy = TensorMap::new(Labels::new(["_"], [[1]]), vec![valid_block()]).unwrap();
                    let err = check(&valid_request(), &energy, &[system(3)], None).unwrap_err();
                    assert_eq!(
                        err.to_string(),
                        format!(
                            "invalid parameter: invalid '{}': expected a single block with key value 0",
                            stringify!($base_name)
                        )
                    );
                }

                #[test]
                fn wrong_property() {
                    let samples = Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]);
                    let n_props = n_properties();
                    let values = ArrayD::<f32>::from_shape_vec(
                        vec![3, n_props],
                        vec![1.0; 3 * n_props],
                    ).unwrap();

                    let props_wrong = if stringify!($base_name) == "energy_ensemble" {
                        Labels::new(["wrong"], [[0], [1]])
                    } else {
                        Labels::new(["wrong"], [[0]])
                    };

                    let block = TensorBlock::new(values, &samples, &[], &props_wrong).unwrap();
                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    let err = check(&valid_request(), &energy, &[system(3)], None).unwrap_err();
                    assert_eq!(
                        err.to_string(),
                        format!(
                            "invalid parameter: invalid properties for '{}': expected names [energy], got [wrong]",
                            stringify!($base_name)
                        )
                    );

                    let props_wrong = if stringify!($base_name) == "energy_ensemble" {
                        Labels::new(["energy"], [[1], [0]])
                    } else {
                        Labels::new(["energy"], [[1]])
                    };
                    let values = ArrayD::<f32>::from_shape_vec(
                        vec![3, n_props],
                        vec![1.0; 3 * n_props],
                    ).unwrap();

                    let block = TensorBlock::new(values, &samples, &[], &props_wrong).unwrap();
                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    let err = check(&valid_request(), &energy, &[system(3)], None).unwrap_err();

                    let expected_msg = if stringify!($base_name) == "energy_ensemble" {
                        format!("invalid parameter: invalid properties values for '{}': expected [[0, ..., n]]", stringify!($base_name))
                    } else {
                        format!("invalid parameter: invalid properties values for '{}': expected [[0]]", stringify!($base_name))
                    };
                    assert_eq!(err.to_string(), expected_msg);
                }

                #[test]
                fn has_components() {
                    let n_props = n_properties();
                    let props = property_labels();
                    let values = ArrayD::<f32>::from_shape_vec(
                        vec![3, 3, n_props],
                        vec![1.0; 9 * n_props],
                    ).unwrap();
                    let block = TensorBlock::new(
                        values,
                        &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
                        &[Labels::new(["xyz"], [[0], [1], [2]])],
                        &props,
                    ).unwrap();
                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    let err = check(&valid_request(), &energy, &[system(3)], None).unwrap_err();
                    assert_eq!(
                        err.to_string(),
                        format!(
                            "invalid parameter: components for '{}' should be empty",
                            stringify!($base_name)
                        )
                    );
                }

                #[test]
                fn wrong_sample_names() {
                    let n_props = n_properties();
                    let props = property_labels();
                    let values = ArrayD::<f32>::from_shape_vec(
                        vec![1, n_props],
                        vec![1.0; n_props],
                    ).unwrap();
                    let block = TensorBlock::new(
                        values,
                        &Labels::new(["system"], [[0]]),
                        &[],
                        &props,
                    ).unwrap();
                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    let err = check(&valid_request(), &energy, &[system(3)], None).unwrap_err();
                    assert_eq!(
                        err.to_string(),
                        format!(
                            "invalid parameter: invalid sample names for '{}': expected [system, atom], got [system]",
                            stringify!($base_name)
                        )
                    );
                }

                #[test]
                fn gradients_dummy() {
                    let n_props = n_properties();
                    let props = property_labels();
                    let mut block = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(
                            vec![3, n_props],
                            vec![1.0; 3 * n_props],
                        ).unwrap(),
                        &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
                        &[],
                        &props,
                    ).unwrap();

                    let dummy_gradient = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(
                            vec![1, n_props],
                            vec![0.1; n_props],
                        ).unwrap(),
                        &Labels::new(["sample"], [[0]]),
                        &[],
                        &props,
                    ).unwrap();
                    block.add_gradient("dummy", dummy_gradient).unwrap();

                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    let err = check(&valid_request(), &energy, &[system(3)], None).unwrap_err();
                    assert_eq!(
                        err.to_string(),
                        format!(
                            "invalid parameter: invalid gradient 'dummy' for '{}': expected one of [strain, positions]",
                            stringify!($base_name)
                        )
                    );
                }

                #[test]
                fn gradients_position_not_requested() {
                    let mut request = valid_request();
                    request.gradients = vec![];

                    let n_props = n_properties();
                    let props = property_labels();
                    let mut block = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(
                            vec![3, n_props],
                            vec![1.0; 3 * n_props],
                        ).unwrap(),
                        &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
                        &[],
                        &props,
                    ).unwrap();

                    let pos_gradient = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(
                            vec![1, 3, n_props],
                            vec![0.1; 3 * n_props],
                        ).unwrap(),
                        &Labels::new(["sample", "system", "atom"], [[0, 0, 0]]),
                        &[Labels::new(["xyz"], [[0], [1], [2]])],
                        &props,
                    ).unwrap();
                    block.add_gradient("positions", pos_gradient).unwrap();

                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    let err = check(&request, &energy, &[system(3)], None).unwrap_err();
                    assert_eq!(
                        err.to_string(),
                        format!(
                            "invalid parameter: invalid gradient 'positions' for '{}': these gradients were not requested",
                            stringify!($base_name)
                        )
                    );
                }

                #[test]
                fn selected_atoms_error() {
                    let selected_atoms = Labels::new(["system", "atom"], [[0, 0], [0, 1]]);

                    let n_props = n_properties();
                    let block = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(vec![3, n_props], vec![1.0; 3 * n_props]).unwrap(),
                        // samples that are not in the selected_atoms
                        &Labels::new(["system", "atom"], [[0, 0], [0, 1], [0, 2]]),
                        &[],
                        &property_labels(),
                    ).unwrap();
                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    let err = check(&valid_request(), &energy, &[system(3)], Some(&selected_atoms)).unwrap_err();
                    assert_eq!(
                        err.to_string(),
                        format!(
                            "invalid parameter: invalid samples for '{}', they do not match the `systems` and `selected_atoms`",
                            stringify!($base_name)
                        )
                    );

                    let mut request = valid_request();
                    request.sample_kind = SampleKind::System;
                    let block = TensorBlock::new(
                        ArrayD::<f32>::from_shape_vec(vec![2, n_props], vec![2.0; 2 * n_props]).unwrap(),
                        // systems that are not in the selected_atoms
                        &Labels::new(["system"], [[0], [1]]),
                        &[],
                        &property_labels(),
                    ).unwrap();
                    let energy = TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).unwrap();
                    let err = check(&request, &energy, &[system(3), system(3)], Some(&selected_atoms)).unwrap_err();
                    assert_eq!(
                        err.to_string(),
                        format!(
                            "invalid parameter: invalid samples for '{}', they do not match the `systems` and `selected_atoms`",
                            stringify!($base_name)
                        )
                    );
                }
            }
        };
    }

    energy_tests!(energy);
    energy_tests!(energy_ensemble);
    energy_tests!(energy_uncertainty);
}
