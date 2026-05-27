use std::collections::{BTreeMap, HashMap};

use dlpk::DLPackTensor;
use metatensor::{TensorBlock, TensorMap};

use crate::PairListOptions;


/// TODO
pub struct System {
    length_unit: String,
    types: DLPackTensor,
    positions: DLPackTensor,
    cell: DLPackTensor,
    pbc: DLPackTensor,

    pairs: BTreeMap<PairListOptions, TensorBlock>,
    custom_data: HashMap<String, TensorMap>,
}


impl System {
    /// TODO
    pub fn new(
        length_unit: String,
        types: DLPackTensor,
        positions: DLPackTensor,
        cell: DLPackTensor,
        pbc: DLPackTensor
    ) -> Self {
        todo!()
    }

    /// TODO
    pub fn add_pairs(&mut self, options: PairListOptions, pairs: TensorBlock, check_consistency: bool) {
        todo!()
    }

    /// TODO
    pub fn get_pairs(&mut self, options: PairListOptions) -> Option<&TensorBlock> {
        todo!()
    }

    /// TODO
    pub fn set_custom_data(&mut self, name: String, data: TensorMap) {
        todo!()
    }

    /// TODO
    pub fn get_custom_data(&self, name: &str) -> Option<&TensorMap> {
        todo!()
    }
}
