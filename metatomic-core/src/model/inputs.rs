use std::sync::Arc;

use dlpk::DLDataTypeCode::kDLFloat;
use metatensor::Labels;

use crate::{Error, ModelCapabilities, PairListOptions, Quantity, System};
use crate::metadata::DType;

/// Validate that the requested outputs match the model's capabilities.
///
/// This checks that each requested output is in the model's capabilities, with
/// compatible `sample_kind` and `explicit_gradients`.
pub(crate) fn check_requested_outputs(
    capabilities: &ModelCapabilities,
    requested_outputs: &[Quantity],
) -> Result<(), Error> {
    for requested in requested_outputs {
        // find all capability entries for this output name (a model may declare
        // the same output with different sample_kinds, e.g. "energy" as both
        // system-level and per-atom)
        let possible = capabilities.outputs.iter()
            .filter(|q| q.name == requested.name)
            .find(|q| q.sample_kind == requested.sample_kind);

        let possible = if let Some(quantity) = possible {
            quantity
        } else {
            // check if the output name exists at all (with any sample_kind)
            let name_exists = capabilities.outputs.iter().any(|q| q.name == requested.name);
            if name_exists {
                let available_kinds: Vec<_> = capabilities.outputs.iter()
                    .filter(|q| q.name == requested.name)
                    .map(|q| q.sample_kind.to_string())
                    .collect();
                return Err(Error::InvalidParameter(format!(
                    "this model can not compute '{}' with sample kind '{}', only with sample kind{} [{}]",
                    requested.name,
                    requested.sample_kind,
                    if available_kinds.len() > 1 { "s" } else { "" },
                    available_kinds.join(", ")
                )));
            } else {
                return Err(Error::InvalidParameter(format!(
                    "this model can not compute '{}', the implemented outputs are [{}]",
                    requested.name,
                    capabilities.outputs.iter().map(|q| q.name.full()).collect::<Vec<_>>().join(", ")
                )));
            }
        };

        // check explicit gradients
        for gradient in &requested.gradients {
            if !possible.gradients.contains(gradient) {
                return Err(Error::InvalidParameter(format!(
                    "this model can not compute explicit gradients of '{}' with respect to '{}'",
                    requested.name, gradient
                )));
            }
        }
    }

    Ok(())
}

/// Validate that the inputs to a model are consistent with its capabilities.
///
/// This checks that:
/// - all systems are on the same device and have the same dtype, matching the
///   model's expected dtype
/// - `selected_atoms` (if provided) has the right names, device, and only
///   contains entries that correspond to actual atoms in the systems
/// - all requested neighbor lists are present on every system
/// - all requested inputs are present on every system
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub(crate) fn check_inputs(
    capabilities: &ModelCapabilities,
    requested_neighbor_lists: &[PairListOptions],
    requested_inputs: &[Quantity],
    systems: &[Arc<System>],
    selected_atoms: Option<&Labels>,
) -> Result<(), Error> {
    if systems.is_empty() {
        return Ok(());
    }

    let global_device = systems[0].device();
    let global_dtype = systems[0].dtype();

    // check dtype matches the model's expected dtype
    let expected_dlpack_dtype = match capabilities.dtype {
        DType::Float32 => <f32 as dlpk::GetDLPackDataType>::get_dlpack_data_type(),
        DType::Float64 => <f64 as dlpk::GetDLPackDataType>::get_dlpack_data_type(),
    };

    if global_dtype != expected_dlpack_dtype {
        let actual_dtype = if global_dtype.code == kDLFloat && global_dtype.bits == 32 {
            "float32"
        } else if global_dtype.code == kDLFloat && global_dtype.bits == 64 {
            "float64"
        } else {
            // unknown float type; display the raw DLDataType
            &global_dtype.to_string()
        };
        return Err(Error::InvalidParameter(format!(
            "wrong dtype for the systems: the model wants {}, we got {}",
            capabilities.dtype, actual_dtype
        )));
    }

    // check selected_atoms
    if let Some(selected) = selected_atoms {
        if selected.device() != global_device {
            return Err(Error::InvalidParameter(format!(
                "expected selected_atoms to be on the same device as the systems, got {} and {}",
                selected.device(), global_device
            )));
        }

        if selected.names() != ["system", "atom"] {
            return Err(Error::InvalidParameter(format!(
                "invalid names for selected_atoms: expected ['system', 'atom'], got {:?}",
                selected.names()
            )));
        }

        // build the set of all possible (system, atom) pairs
        let total_atoms: usize = systems.iter().map(|s| s.size()).sum();
        let mut possible_values = ndarray::Array2::from_elem((total_atoms, 2), 0i32);
        let mut index = 0;
        for (system_i, system) in systems.iter().enumerate() {
            for atom_i in 0..system.size() {
                possible_values[[index, 0]] = system_i as i32;
                possible_values[[index, 1]] = atom_i as i32;
                index += 1;
            }
        }
        let possible_atoms = metatensor::Labels::new_assume_unique(["system", "atom"], possible_values);

        let intersection = selected.intersection(&possible_atoms, None, None)?;
        if intersection.count() != selected.count() {
            return Err(Error::InvalidParameter(
                "invalid selected_atoms: there are entries that are not possible for the current systems".into()
            ));
        }
    }

    // check each system
    for system in systems {
        if system.device() != global_device {
            return Err(Error::InvalidParameter(format!(
                "expected all systems to be on the same device, got {} and {}",
                global_device, system.device()
            )));
        }

        if system.dtype() != global_dtype {
            return Err(Error::InvalidParameter(format!(
                "expected all systems to have the same dtype, got {} and {}",
                global_dtype, system.dtype()
            )));
        }

        // check neighbor lists
        for request in requested_neighbor_lists {
            if let Some(pairs) = system.get_pairs(request) {
                if !pairs.as_ref().gradient_list().is_empty() {
                    return Err(Error::InvalidParameter(format!(
                        "neighbors list for {:?} contain gradients, which are \
                        not supported",
                        request
                    )));
                }
            } else {
                return Err(Error::InvalidParameter(format!(
                    "missing neighbors list in the system: the model requested \
                    a list for {:?}, but it was not provided in the system",
                    request
                )));
            }


        }

        // check additional inputs
        let known_data = system.known_custom_data();
        for request in requested_inputs {
            let found = known_data.iter().any(|known| *known == request.name.full());
            if !found {
                return Err(Error::InvalidParameter(format!(
                    "missing additional input in the system: the model requested \
                    '{}' as an extra input, but it was not provided in the system",
                    request.name
                )));
            }
        }
    }

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    use crate::quantity::{Gradients, Quantity, QuantityName, SampleKind};
    use crate::system::test_system;

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_check_requested_outputs() {
        let capabilities = ModelCapabilities {
            outputs: vec![Quantity {
                name: QuantityName::new("energy".into()).unwrap(),
                unit: "eV".into(),
                description: None,
                gradients: vec![Gradients::Positions],
                sample_kind: SampleKind::System,
            }],
            atomic_types: vec![1, 6, 8],
            interaction_range: 5.0,
            length_unit: "nm".into(),
            supported_devices: vec![crate::Device::cpu()],
            dtype: DType::Float32,
        };

        // happy path
        let requested_outputs = vec![Quantity {
            name: QuantityName::new("energy".into()).unwrap(),
            unit: "eV".into(),
            description: None,
            gradients: vec![],
            sample_kind: SampleKind::System,
        }];
        check_requested_outputs(&capabilities, &requested_outputs).unwrap();

        // requested output not in capabilities
        let bad_outputs = vec![Quantity {
            name: QuantityName::new("custom::forces".into()).unwrap(),
            unit: "eV/A".into(),
            description: None,
            gradients: vec![],
            sample_kind: SampleKind::Atom,
        }];
        let err = check_requested_outputs(&capabilities, &bad_outputs).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: this model can not compute 'custom::forces', \
            the implemented outputs are [energy]"
        );

        // requested gradient not in capabilities
        let bad_grad_outputs = vec![Quantity {
            name: QuantityName::new("energy".into()).unwrap(),
            unit: "eV".into(),
            description: None,
            gradients: vec![Gradients::Strain],
            sample_kind: SampleKind::System,
        }];
        let err = check_requested_outputs(&capabilities, &bad_grad_outputs).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: this model can not compute explicit gradients \
            of 'energy' with respect to 'strain'"
        );

        // sample kind mismatch: requesting atom when model only offers system
        let bad_sample_outputs = vec![Quantity {
            name: QuantityName::new("energy".into()).unwrap(),
            unit: "eV".into(),
            description: None,
            gradients: vec![],
            sample_kind: SampleKind::Atom,
        }];
        let err = check_requested_outputs(&capabilities, &bad_sample_outputs).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: this model can not compute 'energy' with sample \
            kind 'atom', only with sample kind [system]"
        );

        // model with multiple sample_kinds for the same output — requesting
        // one that exists should pass
        let multi_capabilities = ModelCapabilities {
            outputs: vec![
                Quantity {
                    name: QuantityName::new("energy".into()).unwrap(),
                    unit: "eV".into(),
                    description: None,
                    gradients: vec![],
                    sample_kind: SampleKind::System,
                },
                Quantity {
                    name: QuantityName::new("energy".into()).unwrap(),
                    unit: "eV".into(),
                    description: None,
                    gradients: vec![],
                    sample_kind: SampleKind::Atom,
                },
            ],
            atomic_types: vec![1, 6, 8],
            interaction_range: 5.0,
            length_unit: "Angstrom".into(),
            supported_devices: vec![crate::Device::cpu()],
            dtype: DType::Float32,
        };
        let atom_outputs = vec![Quantity {
            name: QuantityName::new("energy".into()).unwrap(),
            unit: "eV".into(),
            description: None,
            gradients: vec![],
            sample_kind: SampleKind::Atom,
        }];
        check_requested_outputs(&multi_capabilities, &atom_outputs).unwrap();

        // requesting a sample kind that doesn't match any of the available ones
        let pair_outputs = vec![Quantity {
            name: QuantityName::new("energy".into()).unwrap(),
            unit: "eV".into(),
            description: None,
            gradients: vec![],
            sample_kind: SampleKind::AtomPair,
        }];
        let err = check_requested_outputs(&multi_capabilities, &pair_outputs).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: this model can not compute 'energy' with \
            sample kind 'atom_pair', only with sample kinds [system, atom]"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_check_inputs() {
        // build capabilities matching the test_system() helper
        let capabilities = ModelCapabilities {
            outputs: vec![Quantity {
                name: QuantityName::new("energy".into()).unwrap(),
                unit: "eV".into(),
                description: None,
                gradients: vec![Gradients::Positions],
                sample_kind: SampleKind::System,
            }],
            atomic_types: vec![1, 6, 8],
            interaction_range: 5.0,
            length_unit: "nm".into(),
            supported_devices: vec![crate::Device::cpu()],
            dtype: DType::Float32,
        };

        let requested_neighbor_lists = [PairListOptions {
            cutoff: 3.5,
            full_list: true,
            strict: false,
            requestors: vec![],
        }];

        let requested_inputs = vec![Quantity {
            name: QuantityName::new("custom::data/name".into()).unwrap(),
            unit: String::new(),
            description: None,
            gradients: vec![],
            sample_kind: SampleKind::System,
        }];

        let systems = vec![test_system("f32")];

        // happy path — everything matches
        check_inputs(
            &capabilities,
            &requested_neighbor_lists,
            &requested_inputs,
            &systems,
            None,
        ).unwrap();

        // wrong dtype
        let f64_capabilities = ModelCapabilities {
            outputs: capabilities.outputs.clone(),
            atomic_types: capabilities.atomic_types.clone(),
            interaction_range: capabilities.interaction_range,
            length_unit: capabilities.length_unit.clone(),
            supported_devices: capabilities.supported_devices.clone(),
            dtype: DType::Float64,
        };
        let err = check_inputs(
            &f64_capabilities,
            &requested_neighbor_lists,
            &requested_inputs,
            &systems,
            None,
        ).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: wrong dtype for the systems: the model wants float64, we got float32"
        );

        // missing neighbor list
        let bad_nl = vec![PairListOptions {
            cutoff: 5.0,
            full_list: false,
            strict: true,
            requestors: vec![],
        }];
        let err = check_inputs(
            &capabilities,
            &bad_nl,
            &requested_inputs,
            &systems,
            None,
        ).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: missing neighbors list in the system: \
            the model requested a list for PairListOptions { cutoff: 5.0, \
            full_list: false, strict: true, requestors: [] }, but it was not \
            provided in the system"
        );

        // missing requested input
        let bad_inputs = vec![Quantity {
            name: QuantityName::new("custom::missing".into()).unwrap(),
            unit: String::new(),
            description: None,
            gradients: vec![],
            sample_kind: SampleKind::System,
        }];
        let err = check_inputs(
            &capabilities,
            &requested_neighbor_lists,
            &bad_inputs,
            &systems,
            None,
        ).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: missing additional input in the system: \
            the model requested 'custom::missing' as an extra input, but it was \
            not provided in the system"
        );

        // invalid selected_atoms names
        let bad_selected = metatensor::Labels::new(["foo", "bar"], [[0i32, 0]]);
        let err = check_inputs(
            &capabilities,
            &requested_neighbor_lists,
            &requested_inputs,
            &systems,
            Some(&bad_selected),
        ).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid names for selected_atoms: expected \
            ['system', 'atom'], got [\"foo\", \"bar\"]"
        );

        // selected_atoms with out-of-range atom
        let bad_selected = metatensor::Labels::new(["system", "atom"], [[0, 99]]);
        let err = check_inputs(
            &capabilities,
            &requested_neighbor_lists,
            &requested_inputs,
            &systems,
            Some(&bad_selected),
        ).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid parameter: invalid selected_atoms: there are entries that \
            are not possible for the current systems"
        );

        // valid selected_atoms
        let good_selected = metatensor::Labels::new(["system", "atom"], [[0, 0], [0, 1]]);
        check_inputs(
            &capabilities,
            &requested_neighbor_lists,
            &requested_inputs,
            &systems,
            Some(&good_selected),
        ).unwrap();
    }
}
