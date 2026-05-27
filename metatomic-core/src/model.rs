use metatensor::{Labels, TensorMap};

use crate::{Error, Quantity, System};

use crate::c_api::mta_model_t;

/// TODO
pub struct Model(pub(crate) mta_model_t);


/// TODO
pub fn execute_model(
    model: &Model,
    systems: &[System],
    selected_atoms: Option<Labels>,
    requested_outputs: &[Quantity],
    check_consistency: bool,
) -> Result<Vec<TensorMap>, Error> {
    todo!()
}
