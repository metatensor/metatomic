use std::{iter::zip};

use dlpk::DLDataType;
use dlpk::sys::DLDataTypeCode::kDLFloat;
use metatensor::{Labels, LabelsBuilder, TensorBlockRef, TensorMap};

use crate::{Error, SampleKind, System, metadata::DType::{self, Float32, Float64}, quantity::Quantity};

const ENERGY_BASES: &[&str] = &[
    "energy",
    "energy_ensemble",
    "energy_uncertainty"
];
const ENERGY_GRADIENTS: &[&str] = &["strain", "positions"];

fn join_names(names: &[&str]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("'{}'", n)).collect();
    format!("[{}]", quoted.join(", "))
}

fn create_list_string(size: usize) -> String {
    if size > 3 {
        let rep = String::from("[[0], ..., [n]]");
        rep
    } else {
        let mut rep = String::from("[");
        for i in 0..size {
            if i != size - 1 {
                rep += format!("[{}], ", i).as_str();
            } else {
                rep += format!("[{}]", i).as_str();
            }
        }
        rep += "]";
        rep
    }
}

/// Ensure the TensorMap has a single block with the expected key
fn validate_single_block(name:&str, value: &TensorMap) -> Result<(), Error> {
    let expected_label = Labels::new(["_"], &[[0]]);
    if *value.keys() != expected_label {
        return Err(Error::InvalidParameter(format!("invalid keys for '{}': expected `Labels('_', [[0]])`", name)));
    }
    Ok(())
}

/// Validates the sample labels against the expected structure
fn validate_atomic_samples(
    name: &str,
    value: &TensorMap,
    systems: &[System],
    request: &Quantity,
    selected_atoms: &Option<metatensor::Labels>,
) -> Result<(), Error> {
    let block = value.block_by_id(0);

    // Check if the sample names are as expected based on the sample_kind
    let expected_samples_names: &[&str] = match request.sample_kind {
        SampleKind::Atom => &["system", "atom"],
        SampleKind::AtomPair => &[
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        SampleKind::System => &["system"],
    };
    if block.samples().names() != expected_samples_names {
        return Err(Error::InvalidParameter(format!("invalid sample names for '{}': expected {}, got {}", name, join_names(expected_samples_names), join_names(&block.samples().names()))));
    }

    // Check if the samples match the systems and selected_atoms
    let mut builder = LabelsBuilder::new(expected_samples_names.to_vec());
    let expected_samples: Labels = match request.sample_kind {
        SampleKind::Atom => {
            for s in 0..systems.len() {
                for a in 0..systems[s].size() {
                    builder.add(&[s, a]);
                }
            }
            let mut labels = builder.finish_assume_unique();
            if let Some(selected) = selected_atoms {
                labels = labels.intersection(selected, None, None)?;
            }
            labels
        },
        SampleKind::AtomPair => {
            for [system, first_atom, second_atom, _, _, _] in block.samples().iter_fixed_size::<6>() {
                let s = system.i32();
                let a1 = first_atom.i32();
                let a2 = second_atom.i32();
                if s < 0 || s >= systems.len().try_into().unwrap() {
                    return Err(Error::InvalidParameter(format!("invalid system index in samples for '{}': {} is out of bounds", name, s)))
                }
                let n_atoms = systems[s as usize].size() as i32;
                if a1 < 0 || a1 >= n_atoms {
                    return Err(Error::InvalidParameter(format!("invalid first_atom index in samples for '{}': {} is out of bounds for system {}", name, a1, s)))
                }
                if a2 < 0 || a2 >= n_atoms {
                    return Err(Error::InvalidParameter(format!("invalid second_atom index in samples for '{}': {} is out of bounds for system {}", name, a2, s)))
                }
            }
            return Ok(());
        },
        SampleKind::System => {
            for s in 0..systems.len() {
                builder.add(&[s]);
            }
            let mut labels = builder.finish_assume_unique();
            let mut selected_systems_idx: Vec<i32> = [].to_vec();
            if let Some(selected) = selected_atoms {
                for [s, a] in selected.iter_fixed_size::<2>() {
                    selected_systems_idx.extend([s.i32()]);
                }
                selected_systems_idx.sort();
                selected_systems_idx.dedup();
                let mut builder_system = LabelsBuilder::new(["system"].to_vec());
                for s in selected_systems_idx {
                    builder_system.add(&[s]);
                }
                let selected_system = builder_system.finish_assume_unique();
                labels = labels.intersection(&selected_system, None, None)?;
            }
            labels
        },
    };

    if expected_samples.union(&block.samples(), None, None)?.size() != expected_samples.size() {
        return Err(Error::InvalidParameter(format!("invalid samples entries for '{}', they do not match the `systems` and `selected_atoms`. Expected samples:\n{:?}", name, expected_samples)));
    }
    Ok(())
}

fn validate_components(
    name: &str,
    components: &[Labels],
    expected_components: &[Labels],
) -> Result<(), Error> {
    let actual_n_components = components.len();
    let expected_n_components = expected_components.len();
    if actual_n_components != expected_n_components {
        if expected_n_components == 0 {
            return Err(Error::InvalidParameter(format!("invalid components for {}: `components` should be empty", name)));
        }
        return Err(Error::InvalidParameter(format!("invalid components for {}: expected {} component(s)", name, expected_n_components)));
    }
    for (component, expected_component) in zip(components, expected_components) {
        if *component != *expected_component {
           let expected_labels = format!("Labels('{}', {})", join_names(&expected_component.names()), create_list_string(expected_component.size()));
           return Err(Error::InvalidParameter(format!("invalid components for '{}': expected `{}`", name, expected_labels)));
        }
    }
    Ok(())
}

fn validate_properties(
    name: &str,
    block: &TensorBlockRef,
    expected_properties: &Labels,
) -> Result<(), Error> {
    if &block.properties() != expected_properties {
        let expected_labels = format!("Labels('{}', {})", join_names(&expected_properties.names()), create_list_string(expected_properties.size()));
        return Err(Error::InvalidParameter(format!("invalid properties for '{}': expected `{}`", name, expected_labels)));
    }
    Ok(())
}

fn validate_gradient(
    name: &str,
    parameter: &str,
    gradient: &TensorBlockRef,
    expected_samples_names: &[&str],
    expected_components: &[Labels]
) -> Result<(), Error> {
    if gradient.samples().names() != expected_samples_names {
        return Err(Error::InvalidParameter(format!("invalid samples for '{}' gradients with respect to '{}': expected the names to be {}, got {}", name, parameter, join_names(expected_samples_names), join_names(&gradient.samples().names()))))
    }
    validate_components(format!("{} '{}' gradients", name, parameter).as_str(), gradient.components().as_slice(), expected_components)?;
    Ok(())
}

fn validate_no_gradients(name: &str, gradient_list: Vec<&str>,) -> Result<(), Error> {
    if !gradient_list.is_empty() {
        return Err(Error::InvalidParameter(format!("invalid gradients for '{}': expected no gradients, found {}", name, join_names(&gradient_list))));
    }
    Ok(())
}
fn check_energy_like(
    name: &str,
    value: &TensorMap,
    systems: &[System],
    request: &Quantity,
    selected_atoms: &Option<metatensor::Labels>
) -> Result<(), Error> {
    validate_single_block(name, value)?;
    validate_atomic_samples(name, value, systems, request, selected_atoms)?;
    let energy_block = value.block_by_id(0);
    validate_components(name, energy_block.components().as_slice(), &[])?;

    let expected_properties: Labels;
    if name == "energy" || name == "energy_uncertainty" {
        expected_properties = Labels::new(["energy"], &[[0]]);
    } else {
        let n_ensemble_members = *energy_block.values().shape()?.last().unwrap();
        let mut builder = LabelsBuilder::new(["energy"].to_vec());
        for i in 0..n_ensemble_members {
            builder.add(&[i]);
        }
        expected_properties = builder.finish();
    }
    validate_properties(name, &energy_block, &expected_properties)?;

    let gradients = energy_block.gradients();
    for (parameter, gradient) in gradients {
        let expected_samples_names: Vec<&str>;
        let expected_components: Vec<Labels>;
        let xyz = [[0], [1], [2]];
        match parameter {
            "strain" => {
                expected_samples_names = vec!["sample"];
                expected_components = vec![
                    Labels::new(["xyz_1"], &xyz),
                    Labels::new(["xyz_2"], &xyz),
                ];
            },
            "positions" => {
                expected_samples_names = vec!["sample", "system", "atom"];
                expected_components = vec![
                    Labels::new(["xyz"], &[[0], [1], [2]])
                ];
            },
            _ => {
                return Err(Error::InvalidParameter(format!("unexpected gradient {} of energy", parameter)));
            }
        }
        validate_gradient(name, parameter, &gradient, expected_samples_names.as_slice(), expected_components.as_slice())?;
    }
    Ok(())
}

fn check_feature(
    value: &TensorMap,
    systems: &[System],
    request: &Quantity,
    selected_atoms: &Option<metatensor::Labels>
) -> Result<(), Error> {
    validate_single_block("feature", value)?;
    validate_atomic_samples("feature", value, systems, request, selected_atoms)?;
    let feature_block = value.block_by_id(0);
    validate_components("feature", &feature_block.components(), [].as_slice())?;
    validate_no_gradients("feature", feature_block.gradient_list())?;
    Ok(())
}

fn check_non_conservative_force(
    value: &TensorMap,
    systems: &[System],
    request: &Quantity,
    selected_atoms: &Option<metatensor::Labels>
) -> Result<(), Error> {
    validate_single_block("non_conservative_force", value)?;
    validate_atomic_samples("non_conservative_force", value, systems, request, selected_atoms)?;
    let force_block = value.block_by_id(0);
    validate_components("non_conservative_force", &force_block.components(), [Labels::new(["xyz"], &[[0], [1], [2]])].as_slice())?;
    validate_properties("non_conservative_force", &force_block, &Labels::new(["non_conservative_force"], &[[0]]))?;
    validate_no_gradients("non_conservative_force", force_block.gradient_list())?;
    Ok(())
}

fn check_non_conservative_stress(
    value: &TensorMap,
    systems: &[System],
    request: &Quantity,
    selected_atoms: &Option<metatensor::Labels>
) -> Result<(), Error> {
    validate_single_block("non_conservative_stress", value)?;
    validate_atomic_samples("non_conservative_stress", value, systems, request, selected_atoms)?;
    let stress_block = value.block_by_id(0);
    validate_components("non_conservative_stress", &stress_block.components(), [
        Labels::new(["xyz_1"], &[[0], [1], [2]]),
        Labels::new(["xyz_2"], &[[0], [1], [2]]),
    ].as_slice())?;
    validate_properties("non_conservative_stress", &stress_block, &Labels::new(["non_conservative_stress"], &[[0]]))?;
    validate_no_gradients("non_conservative_stress", stress_block.gradient_list())?;
    Ok(())
}


fn check_position(
    value: &TensorMap,
    systems: &[System],
    request: &Quantity,
    selected_atoms: &Option<metatensor::Labels>
) -> Result<(), Error> {
    validate_single_block("position", value)?;
    validate_atomic_samples("position", value, systems, request, selected_atoms)?;
    let position_block = value.block_by_id(0);
    validate_components("position", &position_block.components(), [
        Labels::new(["xyz"], &[[0], [1], [2]]),
    ].as_slice())?;
    validate_properties("position", &position_block, &Labels::new(["position"], &[[0]]))?;
    validate_no_gradients("position", position_block.gradient_list())?;
    Ok(())
}


fn check_momentum(
    value: &TensorMap,
    systems: &[System],
    request: &Quantity,
    selected_atoms: &Option<metatensor::Labels>
) -> Result<(), Error> {
    validate_single_block("momentum", value)?;
    validate_atomic_samples("momentum", value, systems, request, selected_atoms)?;
    let momentum_block = value.block_by_id(0);
    validate_components("momentum", &momentum_block.components(), [
        Labels::new(["xyz"], &[[0], [1], [2]]),
    ].as_slice())?;
    validate_properties("momentum", &momentum_block, &Labels::new(["momentum"], &[[0]]))?;
    validate_no_gradients("momentum", momentum_block.gradient_list())?;
    Ok(())
}


fn check_mass(
    value: &TensorMap,
    systems: &[System],
    request: &Quantity,
    selected_atoms: &Option<metatensor::Labels>
) -> Result<(), Error> {
    validate_single_block("mass", value)?;
    validate_atomic_samples("mass", value, systems, request, selected_atoms)?;
    let mass_block = value.block_by_id(0);
    validate_components("mass", &mass_block.components(), [
        Labels::new(["xyz"], &[[0], [1], [2]]),
    ].as_slice())?;
    validate_properties("mass", &mass_block, &Labels::new(["mass"], &[[0]]))?;
    validate_no_gradients("mass", mass_block.gradient_list())?;
    Ok(())
}


fn check_velocity(
    value: &TensorMap,
    systems: &[System],
    request: &Quantity,
    selected_atoms: &Option<metatensor::Labels>
) -> Result<(), Error> {
    validate_single_block("velocity", value)?;
    validate_atomic_samples("velocity", value, systems, request, selected_atoms)?;
    let velocity_block = value.block_by_id(0);
    validate_components("velocity", &velocity_block.components(), [
        Labels::new(["xyz"], &[[0], [1], [2]]),
    ].as_slice())?;
    validate_properties("velocity", &velocity_block, &Labels::new(["velocity"], &[[0]]))?;
    validate_no_gradients("velocity", velocity_block.gradient_list())?;
    Ok(())
}


fn check_charge(
    value: &TensorMap,
    systems: &[System],
    request: &Quantity,
) -> Result<(), Error> {
    validate_single_block("charge", value)?;
    validate_atomic_samples("charge", value, systems, request, &None)?;
    let charge_block = value.block_by_id(0);
    validate_components("charge", &charge_block.components(), [].as_slice())?;
    validate_properties("charge", &charge_block, &Labels::new(["charge"], &[[0]]))?;
    validate_no_gradients("charge", charge_block.gradient_list())?;
    Ok(())
}


fn check_heat_flux(
    value: &TensorMap,
    systems: &[System],
    request: &Quantity,
) -> Result<(), Error> {
    validate_single_block("heat_flux", value)?;
    if request.sample_kind != SampleKind::System {
        return Err(Error::InvalidParameter(format!("invalid 'heat_flux': heat_flux is a per-system quantity, but the request indicates `sample_kind='{}'`", request.sample_kind)));
    }
    validate_atomic_samples("heat_flux", value, systems, request, &None)?;
    let heat_flux_block = value.block_by_id(0);
    validate_components("heat_flux", &heat_flux_block.components(), [
        Labels::new(["xyz"], &[[0], [1], [2]]),
    ].as_slice())?;
    validate_properties("heat_flux", &heat_flux_block, &Labels::new(["heat_flux"], &[[0]]))?;
    validate_no_gradients("heat_flux", heat_flux_block.gradient_list())?;
    Ok(())
}


fn check_spin_multiplicity(
    value: &TensorMap,
    systems: &[System],
    request: &Quantity,
) -> Result<(), Error> {
    validate_single_block("spin_multiplicity", value)?;
    if request.sample_kind != SampleKind::System {
        return Err(Error::InvalidParameter(format!("invalid 'spin_multiplicity': spin_multiplicity is a per-system quantity, but the request indicates `sample_kind='{}'`", request.sample_kind)));
    }
    validate_atomic_samples("spin_multiplicity", value, systems, request, &None)?;
    let spin_multiplicity_block = value.block_by_id(0);
    validate_components("spin_multiplicity", &spin_multiplicity_block.components(), [].as_slice())?;
    validate_properties("spin_multiplicity", &spin_multiplicity_block, &Labels::new(["spin_multiplicity"], &[[0]]))?;
    validate_no_gradients("spin_multiplicity", spin_multiplicity_block.gradient_list())?;
    Ok(())
}

pub fn check_quantities(
    systems: &[System],
    requested: &[Quantity],
    selected_atoms: &Option<metatensor::Labels>,
    values: &[TensorMap],
    model_dtype: &DType,
    inputs_or_outputs: &String
) -> Result<(), Error> {
    if inputs_or_outputs != "inputs" && inputs_or_outputs != "outputs" {
        return Err(Error::InvalidParameter(format!("internal error: inputs_or_outputs should be 'inputs' or 'outputs', got '{}'", inputs_or_outputs)));
    }

    for (request, value) in zip(requested, values) {
        let name = &request.name;
        if value.keys().count() != 0 {
            let dldata_type = value.block_by_id(0).values().dtype()?;
            let DLDataType{code, bits, ..} = dldata_type;
            if code != kDLFloat || (*model_dtype == Float32 && bits != 32) || (*model_dtype == Float64 && bits != 64) {
                return Err(Error::InvalidParameter(format!("wrong dtype for '{}': the model dtype is {} but the data uses {}", name, model_dtype, dldata_type)))
            }
        }

        let base = name.split('/').next().unwrap();
        if ENERGY_BASES.contains(&base) {
            check_energy_like(base, value, systems, request, selected_atoms)?;
        } else if base == "feature" {
            check_feature(value, systems, request, selected_atoms)?;
        } else if base == "non_conservative_force" {
            check_non_conservative_force(value, systems, request, selected_atoms)?;
        } else if base == "non_conservative_stress" {
            check_non_conservative_stress(value, systems, request, selected_atoms)?;
        } else if base == "position" {
            check_position(value, systems, request, selected_atoms)?;
        } else if base == "momentum" {
            check_momentum(value, systems, request, selected_atoms)?;
        } else if base == "mass" {
            check_mass(value, systems, request, selected_atoms)?;
        } else if base == "velocity" {
            check_velocity(value, systems, request, selected_atoms)?;
        } else if base == "charge" {
            check_charge(value, systems, request)?;
        } else if base == "heat_flux" {
            check_heat_flux(value, systems, request)?;
        } else if base == "spin_multiplicity" {
            check_spin_multiplicity(value, systems, request)?;
        } else if name.contains("::") {
            // this is a non-standard quantity, there is nothing to check
        } else {
            return Err(Error::InvalidParameter(format!("Invalid quantity name: '{}'. Variants should look like '<quantity>/<variant>'. Non-standard quantity names should look like '<domain>::<quantity>[/<variant>]'.", name)))
        }
    }
    Ok(())
}
