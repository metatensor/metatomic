use metatensor::{Labels, TensorMap};

use crate::{Error, System};


mod quantities;
pub use quantities::{QuantityName, Quantity, SampleKind, Gradients};

mod checks;

mod energy;
mod feature;
mod non_conservative_force;
mod non_conservative_stress;
mod position;
mod momentum;
mod velocity;
mod mass;
mod charge;
mod heat_flux;
mod spin_multiplicity;


/// Check that the provided `TensorMap` matches the expected layout for the
/// given `Quantity`.
///
/// Only standard quantities are checked, custom quantities are only validated
/// for device/dtype compatibility.
///
/// `selected_atoms` can change the expected samples, and should be provided if
/// the `TensorMap` was computed for a subset of atoms.
pub fn check_quantity(
    quantity: &Quantity,
    values: &TensorMap,
    systems: &[System],
    selected_atoms: Option<&Labels>,
) -> Result<(), Error> {
    assert!(!systems.is_empty(), "systems must contain at least one system");
    debug_assert!(systems.iter().all(|s| s.dtype() == systems[0].dtype()), "all systems must have the same dtype");
    debug_assert!(systems.iter().all(|s| s.device() == systems[0].device()), "all systems must have the same device");

    if !values.keys().is_empty() {
        if values.device()? != systems[0].device() {
            return Err(Error::InvalidParameter(format!(
                "invalid device for quantity '{}': expected {}, got {}",
                quantity.name,
                systems[0].device(),
                values.device()?
            )));
        }
        if values.dtype()? != systems[0].dtype() {
            return Err(Error::InvalidParameter(format!(
                "invalid dtype for quantity '{}': expected {}, got {}",
                quantity.name,
                systems[0].dtype(),
                values.dtype()?
            )));
        }
    }

    if quantity.name.is_custom() {
        // nothing to check
        return Ok(());
    }

    match quantity.name.base() {
        "energy" | "energy_ensemble" | "energy_uncertainty" => energy::check(quantity, values, systems, selected_atoms)?,
        "feature" => feature::check(quantity, values, systems, selected_atoms)?,
        "non_conservative_force" => non_conservative_force::check(quantity, values, systems, selected_atoms)?,
        "non_conservative_stress" => non_conservative_stress::check(quantity, values, systems, selected_atoms)?,
        "position" => position::check(quantity, values, systems, selected_atoms)?,
        "momentum" => momentum::check(quantity, values, systems, selected_atoms)?,
        "mass" => mass::check(quantity, values, systems, selected_atoms)?,
        "velocity" => velocity::check(quantity, values, systems, selected_atoms)?,
        "charge" => charge::check(quantity, values, systems, selected_atoms)?,
        "heat_flux" => heat_flux::check(quantity, values, systems, selected_atoms)?,
        "spin_multiplicity" => spin_multiplicity::check(quantity, values, systems, selected_atoms)?,
        _ => {
            return Err(Error::Internal(format!(
                "invalid quantity name '{}': unknown standard quantity",
                quantity.name
            )));
        }
    }

    Ok(())
}
