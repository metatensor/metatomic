use metatensor::TensorMap;

use crate::{metadata::DType, quantity::Quantity, Error, System};

pub fn check_quantities(
    systems: &[System],
    requested: &[Quantity],
    selected_atoms: &Option<metatensor::Labels>,
    values: &[TensorMap],
    model_dtype: &DType,
    inputs_or_outputs: &String
) -> Result<(), Error> {
    
    todo!()
}
