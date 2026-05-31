use std::collections::{BTreeMap, HashMap};
use std::ptr::NonNull;

use dlpk::DLPackTensor;
use metatensor::c_api::{
    mts_block_copy, mts_block_free, mts_block_t, mts_tensormap_copy, mts_tensormap_free,
    mts_tensormap_t,
};

use crate::{Error, PairListOptions};

struct RawBlock(NonNull<mts_block_t>);

impl RawBlock {
    unsafe fn from_ptr(ptr: *mut mts_block_t) -> Result<Self, Error> {
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| Error::InvalidParameter("got invalid NULL pointer for pairs".into()))?;
        return Ok(RawBlock(ptr));
    }

    fn as_ptr(&self) -> *const mts_block_t {
        return self.0.as_ptr();
    }
}

impl Drop for RawBlock {
    fn drop(&mut self) {
        unsafe {
            let _ = mts_block_free(self.0.as_ptr());
        }
    }
}

struct RawTensorMap(NonNull<mts_tensormap_t>);

impl RawTensorMap {
    unsafe fn from_ptr(ptr: *mut mts_tensormap_t) -> Result<Self, Error> {
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            Error::InvalidParameter("got invalid NULL pointer for custom data".into())
        })?;
        return Ok(RawTensorMap(ptr));
    }

    fn as_ptr(&self) -> *const mts_tensormap_t {
        return self.0.as_ptr();
    }
}

impl Drop for RawTensorMap {
    fn drop(&mut self) {
        unsafe {
            let _ = mts_tensormap_free(self.0.as_ptr());
        }
    }
}

/// Storage for an atomistic system.
///
/// This owns the raw DLPack tensors and metatensor objects used at the C API
/// boundary. Validation is intentionally limited here; higher-level wrappers
/// can add richer shape/dtype checks when constructing systems from native
/// arrays.
pub(crate) struct SystemCore {
    length_unit: String,
    types: DLPackTensor,
    positions: DLPackTensor,
    cell: DLPackTensor,
    pbc: DLPackTensor,

    pairs: BTreeMap<PairListOptions, RawBlock>,
    custom_data: HashMap<String, RawTensorMap>,
}

impl SystemCore {
    pub(crate) fn new(
        length_unit: String,
        types: DLPackTensor,
        positions: DLPackTensor,
        cell: DLPackTensor,
        pbc: DLPackTensor,
    ) -> Self {
        return SystemCore {
            length_unit,
            types,
            positions,
            cell,
            pbc,
            pairs: BTreeMap::new(),
            custom_data: HashMap::new(),
        };
    }

    pub(crate) fn length_unit(&self) -> &str {
        return &self.length_unit;
    }

    pub(crate) fn size(&self) -> usize {
        return self.types.shape()[0] as usize;
    }

    pub(crate) fn types(&self) -> &DLPackTensor {
        return &self.types;
    }

    pub(crate) fn positions(&self) -> &DLPackTensor {
        return &self.positions;
    }

    pub(crate) fn cell(&self) -> &DLPackTensor {
        return &self.cell;
    }

    pub(crate) fn pbc(&self) -> &DLPackTensor {
        return &self.pbc;
    }

    pub(crate) unsafe fn add_pairs_raw(
        &mut self,
        options: PairListOptions,
        pairs: *mut mts_block_t,
    ) -> Result<(), Error> {
        if self.pairs.contains_key(&options) {
            return Err(Error::InvalidParameter(
                "the pair list for these options already exists in this system".into(),
            ));
        }

        self.pairs.insert(options, RawBlock::from_ptr(pairs)?);
        return Ok(());
    }

    pub(crate) fn get_pairs(&self, options: &PairListOptions) -> Option<*const mts_block_t> {
        return self.pairs.get(options).map(RawBlock::as_ptr);
    }

    pub(crate) fn known_pairs_json(&self) -> String {
        let mut pairs = json::JsonValue::new_array();
        for options in self.pairs.keys() {
            pairs.push(json::JsonValue::from(options.clone()))
                .expect("pushing to JSON array should not fail");
        }
        return pairs.dump();
    }

    pub(crate) unsafe fn add_custom_data_raw(
        &mut self,
        name: String,
        data: *mut mts_tensormap_t,
    ) -> Result<(), Error> {
        validate_custom_data_name(&name)?;

        if self.custom_data.contains_key(&name) {
            return Err(Error::InvalidParameter(format!(
                "custom data '{}' is already present in this system",
                name
            )));
        }

        self.custom_data.insert(name, RawTensorMap::from_ptr(data)?);
        return Ok(());
    }

    pub(crate) fn get_custom_data(&self, name: &str) -> Option<*const mts_tensormap_t> {
        return self.custom_data.get(name).map(RawTensorMap::as_ptr);
    }

    pub(crate) fn known_custom_data_json(&self) -> String {
        let mut names = json::JsonValue::new_array();
        for name in self.custom_data.keys() {
            names.push(name.as_str())
                .expect("pushing to JSON array should not fail");
        }
        return names.dump();
    }
}

/// Safe Rust wrapper around atomistic system storage.
pub struct System {
    core: SystemCore,
}

impl System {
    /// Create a new system from raw DLPack tensors.
    pub fn new(
        length_unit: String,
        types: DLPackTensor,
        positions: DLPackTensor,
        cell: DLPackTensor,
        pbc: DLPackTensor,
    ) -> Self {
        return System {
            core: SystemCore::new(length_unit, types, positions, cell, pbc),
        };
    }

    pub(crate) fn from_core(core: SystemCore) -> Self {
        return System { core };
    }

    pub(crate) fn as_core(&self) -> &SystemCore {
        return &self.core;
    }

    pub(crate) fn as_core_mut(&mut self) -> &mut SystemCore {
        return &mut self.core;
    }

    /// Get the length unit used by this system.
    pub fn length_unit(&self) -> &str {
        return self.core.length_unit();
    }

    /// Get the number of atoms/particles in this system.
    pub fn size(&self) -> usize {
        return self.core.size();
    }

    /// Get the particle types.
    pub fn types(&self) -> &DLPackTensor {
        return self.core.types();
    }

    /// Get the particle positions.
    pub fn positions(&self) -> &DLPackTensor {
        return self.core.positions();
    }

    /// Get the unit cell.
    pub fn cell(&self) -> &DLPackTensor {
        return self.core.cell();
    }

    /// Get the periodic boundary condition flags.
    pub fn pbc(&self) -> &DLPackTensor {
        return self.core.pbc();
    }
}

impl Drop for SystemCore {
    fn drop(&mut self) {
        let _ = &self.pairs;
        let _ = &self.custom_data;
    }
}

pub(crate) unsafe fn copy_block(ptr: *const mts_block_t) -> Result<*mut mts_block_t, Error> {
    let copy = mts_block_copy(ptr);
    if copy.is_null() {
        return Err(Error::Internal("failed to copy metatensor block".into()));
    }
    return Ok(copy);
}

pub(crate) unsafe fn copy_tensormap(
    ptr: *const mts_tensormap_t,
) -> Result<*mut mts_tensormap_t, Error> {
    let copy = mts_tensormap_copy(ptr);
    if copy.is_null() {
        return Err(Error::Internal("failed to copy metatensor TensorMap".into()));
    }
    return Ok(copy);
}

fn validate_custom_data_name(name: &str) -> Result<(), Error> {
    let lower = name.to_ascii_lowercase();
    if ["types", "positions", "position", "cell", "neighbors", "neighbor"].contains(&lower.as_str())
    {
        return Err(Error::InvalidParameter(format!(
            "custom data can not be named '{}'",
            name
        )));
    }

    validate_quantity_name(name)?;
    return Ok(());
}

fn is_valid_identifier(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    let first = name
        .chars()
        .next()
        .expect("non-empty string should have a first char");
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }

    return name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
}

fn validate_quantity_name(name: &str) -> Result<(), Error> {
    let (main_part, variant) = if let Some(pos) = name.find('/') {
        (&name[..pos], Some(&name[pos + 1..]))
    } else {
        (name, None)
    };

    if main_part.is_empty() {
        return Err(Error::InvalidParameter(format!(
            "quantity name cannot be empty in '{}'",
            name
        )));
    }

    if let Some(variant) = variant {
        if !is_valid_identifier(variant) {
            return Err(Error::InvalidParameter(format!(
                "invalid quantity variant '{}' in '{}': must be a valid identifier (alphanumeric or underscore, not starting with a digit)",
                variant, name
            )));
        }
    }

    for component in main_part.split("::") {
        if !is_valid_identifier(component) {
            return Err(Error::InvalidParameter(format!(
                "invalid quantity name component '{}' in '{}': must be a valid identifier (alphanumeric or underscore, not starting with a digit)",
                component, name
            )));
        }
    }

    return Ok(());
}
