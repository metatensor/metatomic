#![allow(clippy::doc_markdown)]

#[macro_use]
pub(crate) mod quantities;
pub use self::quantities::{Quantity, SampleKind, Gradients};

mod check;
pub use self::check::check_quantities;


static STANDARD_QUANTITIES: &[&str] = &[
    "charge",
    "energy_ensemble",
    "energy_uncertainty",
    "energy",
    "feature",
    "heat_flux",
    "mass",
    "momentum",
    "non_conservative_force",
    "non_conservative_stress",
    "position",
    "spin_multiplicity",
    "velocity",
];
