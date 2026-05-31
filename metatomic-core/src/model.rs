use metatensor::{Labels, TensorMap};

use crate::{Error, Quantity, System};

use crate::c_api::mta_model_t;

/// TODO
pub struct Model(pub(crate) mta_model_t);

impl Model {
    /// Create a new `Model` from the corresponding C API struct.
    pub fn new(model: mta_model_t) -> Self {
        return Model(model);
    }

    /// Extract the underlying C API struct.
    pub(crate) fn into_raw(self) -> mta_model_t {
        return self.0;
    }
}

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
