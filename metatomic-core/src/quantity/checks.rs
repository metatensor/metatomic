use std::collections::BTreeSet;
use std::sync::LazyLock;

use metatensor::{Labels, TensorBlockRef, TensorMap};

use super::Quantity;

use crate::{Error, SampleKind, System};
use crate::kernels::{is_equal_i32, ReferenceValue};


pub(super) static XYZ_LABELS_REFERENCE: LazyLock<ReferenceValue<i32>> = LazyLock::new(|| {
    ReferenceValue::new(
        ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3usize, 1]),
            vec![0i32, 1, 2],
        ).unwrap()
    )
});

pub(super) static SINGLE_LABELS_REFERENCE: LazyLock<ReferenceValue<i32>> = LazyLock::new(|| {
    ReferenceValue::new(
        ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1usize, 1]),
            vec![0i32],
        ).unwrap()
    )
});

/// Check that the `sample_kind` is one of the valid kinds for the given quantity.
pub(super) fn it_should_have_valid_sample_kind(
    context: &str,
    sample_kind: SampleKind,
    valid_kinds: &[SampleKind]
) -> Result<(), Error> {
    if !valid_kinds.contains(&sample_kind) {
        return Err(Error::InvalidParameter(format!(
            "invalid sample_kind for {}: expected one of [{}], got '{}'",
            context,
            valid_kinds.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(", "),
            sample_kind
        )));
    }

    return Ok(());
}

/// Ensure the TensorMap has a single block with the expected key
pub(super) fn it_should_have_a_single_block(context: &str, value: &TensorMap) -> Result<(), Error> {
    let keys = value.keys();
    if keys.count() != 1 {
        return Err(Error::InvalidParameter(format!(
            "invalid {}: expected a single block, but found {} blocks",
            context,
            keys.count()
        )));
    }

    if keys.names() != ["_"] {
        return Err(Error::InvalidParameter(format!(
            "invalid {}: expected a single block with key '_', but found key names [{}]",
            context,
            keys.names().join(", ")
        )));
    }

    let values = keys.values().as_dlpack(dlpk::DLDevice::cpu(), None, dlpk::DLPackVersion::current())?;
    if !is_equal_i32(values.as_ref(), &SINGLE_LABELS_REFERENCE)? {
        return Err(Error::InvalidParameter(format!(
            "invalid {}: expected a single block with key value 0",
            context,
        )));
    }

    Ok(())
}

/// Validate the values for "system" samples
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn validate_system_samples(
    context: &str,
    samples: &Labels,
    systems: &[System],
    selected_atoms: Option<&Labels>,
) -> Result<(), Error> {
    let values = if let Some(selected) = selected_atoms {
        // only include the systems that are present in the selected_atoms
        let mut values = BTreeSet::new();
        for [system_i, _] in selected.iter_fixed_size::<2>() {
            values.insert(system_i.i32());
        }
        ndarray::Array2::from_shape_vec(
            (values.len(), 1),
            values.into_iter().collect()
        ).expect("created invalid array for system samples")
    } else {
        ndarray::Array2::from_shape_vec(
            (systems.len(), 1),
            (0..systems.len()).map(|s| s as i32).collect()
        ).expect("created invalid array for system samples")
    };

    let expected = Labels::new_assume_unique(["system"], values);

    if expected.union(samples, None, None)?.count() != expected.count() {
        return Err(Error::InvalidParameter(format!(
            "invalid samples for {}, they do not match the \
            `systems` and `selected_atoms`",
            context,
            // TODO: add Labels::print to metatensor and use it here
        )));
    }

    return Ok(());
}

/// Validate the values for "atom" samples
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn validate_atom_samples(
    context: &str,
    samples: &Labels,
    systems: &[System],
    selected_atoms: Option<&Labels>,
) -> Result<(), Error> {
    let total_atoms: usize = systems.iter().map(|s| s.size()).sum();
    let mut values = ndarray::Array2::from_elem((total_atoms, 2), 0);

    let mut index = 0;
    for (system_i, system) in systems.iter().enumerate() {
        for atom_i in 0..system.size() {
            values[[index, 0]] = system_i as i32;
            values[[index, 1]] = atom_i as i32;
            index += 1;
        }
    }
    let mut expected = Labels::new_assume_unique(["system", "atom"], values);
    if let Some(selected) = selected_atoms {
        expected = expected.intersection(selected, None, None)?;
    }

    if expected.union(samples, None, None)?.count() != expected.count() {
        return Err(Error::InvalidParameter(format!(
            "invalid samples for {}, they do not match the \
            `systems` and `selected_atoms`",
            context,
            // TODO: add Labels::print to metatensor and use it here
        )));
    }

    return Ok(());
}

/// Validate the values for "atom_pair" samples
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn validate_atom_pair_samples(
    context: &str,
    samples: &Labels,
    systems: &[System],
    selected_atoms: Option<&Labels>,
) -> Result<(), Error> {
    for [system, first_atom, second_atom, _, _, _] in samples.iter_fixed_size::<6>() {
        let system = system.i32();
        let first_atom = first_atom.i32();
        let second_atom = second_atom.i32();

        if system < 0 || system >= systems.len() as i32 {
            return Err(Error::InvalidParameter(format!(
                "invalid system index in samples for {}: {} is out of bounds",
                context,
                system
            )));
        }

        let n_atoms = systems[system as usize].size() as i32;
        if first_atom < 0 || first_atom >= n_atoms {
            return Err(Error::InvalidParameter(format!(
                "invalid first_atom index in samples for {}: {} is out of bounds for system {}",
                context,
                first_atom,
                system
            )));
        }
        if second_atom < 0 || second_atom >= n_atoms {
            return Err(Error::InvalidParameter(format!(
                "invalid second_atom index in samples for {}: {} is out of bounds for system {}",
                context,
                second_atom,
                system
            )));
        }
    }

    return Ok(());
}

/// Validates that the sample labels match the expected structure based on the
/// sample_kind and the systems/selected_atoms provided.
pub(super) fn it_should_have_valid_samples(
    context: &str,
    sample_kind: SampleKind,
    block: TensorBlockRef<'_>,
    systems: &[System],
    selected_atoms: Option<&Labels>,
) -> Result<(), Error> {
    let expected_samples_names: &[&str] = match sample_kind {
        SampleKind::System => &["system"],
        SampleKind::Atom => &["system", "atom"],
        SampleKind::AtomPair => &[
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
    };

    let samples = block.samples();
    if samples.names() != expected_samples_names {
        return Err(Error::InvalidParameter(format!(
            "invalid sample names for {}: expected [{}], got [{}]",
            context,
            expected_samples_names.join(", "),
            samples.names().join(", ")
        )));
    }

    // Check if the samples entries match the systems and selected_atoms
    match sample_kind {
        SampleKind::System => validate_system_samples(context, &samples, systems, selected_atoms),
        SampleKind::Atom => validate_atom_samples(context, &samples, systems, selected_atoms),
        SampleKind::AtomPair => validate_atom_pair_samples(context, &samples, systems, selected_atoms),
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ExpectedLabels<'a> {
    /// Expected names of the labels
    pub names: &'a [&'a str],
    /// Expected values of the labels
    pub values: &'a ReferenceValue<i32>,
    /// Message to display if the values do not match, showing the expected values
    pub values_message: &'a str,
}

pub(super) fn it_should_have_expected_labels(
    context: &str,
    labels_kind: &str,
    labels: &Labels,
    expected: ExpectedLabels<'_>,
) -> Result<(), Error> {

    if labels.names() != expected.names {
        return Err(Error::InvalidParameter(format!(
            "invalid {} for {}: expected names [{}], got [{}]",
            labels_kind,
            context,
            expected.names.join(", "),
            labels.names().join(", ")
        )));
    }

    let values = labels.values().as_dlpack(dlpk::DLDevice::cpu(), None, dlpk::DLPackVersion::current())?;
    if !is_equal_i32(values.as_ref(), expected.values)? {
        return Err(Error::InvalidParameter(format!(
            "invalid {} values for {}: expected {}",
            labels_kind,
            context,
            expected.values_message
        )));
    }
    Ok(())
}

pub(super) fn it_should_have_expected_components(
    context: &str,
    block: TensorBlockRef<'_>,
    expected: &[ExpectedLabels<'_>],
) -> Result<(), Error> {
    let components = block.components();
    if components.len() != expected.len() {
        if expected.is_empty() {
            return Err(Error::InvalidParameter(format!(
                "components for {} should be empty",
                context
            )));
        } else {
            return Err(Error::InvalidParameter(format!(
                "invalid components for {}: expected {} component(s), got {}",
                context,
                expected.len(),
                components.len()
            )));
        }
    }

    for (component, &expected) in components.iter().zip(expected) {
        it_should_have_expected_labels(context, "components", component, expected)?;
    }

    return Ok(());
}

pub(super) fn it_should_have_expected_gradients(
    context: &str,
    request: &Quantity,
    block: TensorBlockRef<'_>,
    potential_gradients: &[&str],
) -> Result<(), Error> {
    if potential_gradients.is_empty() && block.gradients().len() > 0 {
        return Err(Error::InvalidParameter(format!(
            "invalid gradients for {}: expected no gradients, but found \
            gradients with respect to [{}]",
            context,
            block.gradient_list().join(", ")
        )));
    }

    for (parameter, gradient) in block.gradients() {
        if !potential_gradients.contains(&parameter) {
            return Err(Error::InvalidParameter(format!(
                "invalid gradient '{}' for {}: expected one of [{}]",
                parameter,
                context,
                potential_gradients.join(", ")
            )));
        }

        match parameter {
            "strain" => {
                if !request.gradients.contains(&super::Gradients::Strain) {
                    return Err(Error::InvalidParameter(format!(
                        "invalid gradient 'strain' for {}: these gradients were not requested",
                        context
                    )));
                }

                let context = format!("strain gradient of {}", context);
                if gradient.samples().names() != ["sample"] {
                    return Err(Error::InvalidParameter(format!(
                        "invalid samples for {}: expected samples names ['sample'], got [{}]",
                        context,
                        gradient.samples().names().join(", ")
                    )));
                }

                it_should_have_expected_components(
                    &context,
                    gradient,
                    &[
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
                    ]
                )?;
            },
            "positions" => {
                if !request.gradients.contains(&super::Gradients::Positions) {
                    return Err(Error::InvalidParameter(format!(
                        "invalid gradient 'positions' for {}: these gradients were not requested",
                        context
                    )));
                }

                let context = format!("positions gradient of {}", context);
                if gradient.samples().names() != ["sample", "system", "atom"] {
                    return Err(Error::InvalidParameter(format!(
                        "invalid samples for {}: expected samples names ['sample', 'system', 'atom'], got [{}]",
                        context,
                        gradient.samples().names().join(", ")
                    )));
                }

                it_should_have_expected_components(
                    &context,
                    gradient,
                    &[
                        ExpectedLabels {
                            names: &["xyz"],
                            values: &XYZ_LABELS_REFERENCE,
                            values_message: "[[0], [1], [2]]",
                        },
                    ]
                )?;
            },
            _ => {
                unreachable!("got unknown gradient parameter for {}: {}", context, parameter);
            }
        }
    }

    return Ok(());
}
