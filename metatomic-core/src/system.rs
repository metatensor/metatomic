use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use dlpk::sys::{DLDataType, DLDevice};
use dlpk::{DLPackTensor, DLPackTensorRef};
use metatensor::{TensorBlock, TensorMap};

use crate::kernels::{self, ReferenceValue};
use crate::quantity::check_quantity;
use crate::{Error, Gradients, PairListOptions, Quantity, QuantityName, SampleKind};

/// Names that can never be used as custom data in a system
static INVALID_DATA_NAMES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from(["types", "type", "positions", "position", "cell", "neighbors", "neighbor", "pair", "pairs"])
});

static XYZ_REFERENCE: LazyLock<ReferenceValue<i32>> = LazyLock::new(|| {
    ReferenceValue::new(
        ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3usize, 1]),
            vec![0i32, 1, 2],
        ).unwrap()
    )
});

static DISTANCE_REFERENCE: LazyLock<ReferenceValue<i32>> = LazyLock::new(|| {
    ReferenceValue::new(
        ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1usize, 1]),
            vec![0i32],
        ).unwrap()
    )
});

/// Storage for an atomistic system.
///
/// This owns the raw DLPack tensors and metatensor objects used at FFI
/// boundaries.
pub struct System {
    length_unit: String,
    types: DLPackTensor,
    positions: DLPackTensor,
    cell: DLPackTensor,
    pbc: DLPackTensor,

    pairs: BTreeMap<PairListOptions, TensorBlock>,
    custom_data: HashMap<String, TensorMap>,
}

unsafe impl Send for System {}
unsafe impl Sync for System {}

impl System {
    /// Clone this system, deep-copying all underlying data (types, positions,
    /// cell, pbc, pair lists, and custom data) to new device allocations.
    ///
    /// The cloned system is fully independent of the original: modifying one
    /// does not affect the other.
    pub fn try_clone(&self) -> Result<Self, Error> {
        let types = kernels::clone_tensor(&self.types.as_ref())?;
        let positions = kernels::clone_tensor(&self.positions.as_ref())?;
        let cell = kernels::clone_tensor(&self.cell.as_ref())?;
        let pbc = kernels::clone_tensor(&self.pbc.as_ref())?;

        let mut pairs = BTreeMap::new();
        for (options, block) in &self.pairs {
            pairs.insert(options.clone(), block.as_ref().try_clone()?);
        }

        let mut custom_data = HashMap::new();
        for (name, data) in &self.custom_data {
            custom_data.insert(name.clone(), data.try_clone()?);
        }

        Ok(System {
            length_unit: self.length_unit.clone(),
            types,
            positions,
            cell,
            pbc,
            pairs,
            custom_data,
        })
    }

    /// Convert this system's data from the system's length unit to
    /// `model_length_unit`, and with custom data matching `requested_inputs`
    /// converted from their stored units to the requested units.
    ///
    /// Returns a new `Arc<System>`. If no conversion is needed, this is just
    /// a refcount bump (the data is shared). Otherwise, the system is
    /// deep-copied and the copy is scaled in place, so the original data is
    /// preserved.
    ///
    /// # Parameters
    /// - `model_length_unit`: the length unit the model expects (e.g.
    ///   "Angstrom"). If empty, no length conversion is performed.
    /// - `requested_inputs`: the list of quantities the model requested as
    ///   extra inputs. Custom data matching one of these will have its values
    ///   converted from the data's stored unit to the requested unit.
    pub fn convert_units(
        self: Arc<Self>,
        model_length_unit: &str,
        requested_inputs: &[Quantity],
    ) -> Result<Arc<Self>, Error> {
        // length conversion factor
        let length_factor = crate::unit_conversion_factor(&self.length_unit, model_length_unit)?;

        // Clone the system before modifying any data, so the engine's original
        // data is preserved.
        #[allow(clippy::float_cmp)]
        let mut system = if length_factor == 1.0 {
            // Check if any custom data needs conversion before returning early
            let mut needs_custom_conversion = false;
            for requested in requested_inputs {
                if let Ok(data) = self.get_custom_data(requested.name.full()) {
                    let data_unit = data.get_info("unit").map(|s| s.to_string()).unwrap_or_default();
                    let factor = crate::unit_conversion_factor(&data_unit, &requested.unit)?;
                    if factor != 1.0 {
                        needs_custom_conversion = true;
                        break;
                    }
                }
            }
            if !needs_custom_conversion {
                return Ok(self);
            }
            self.try_clone()?
        } else {
            let mut system = self.try_clone()?;

            // scale positions and cell
            let positions = system.positions.as_mut();
            kernels::scale_inplace(positions, length_factor)?;

            let cell = system.cell.as_mut();
            kernels::scale_inplace(cell, length_factor)?;

            // scale pair list values (distances). If the underlying DLPack
            // tensor is read-only (e.g. backed by non-writable memory), we copy
            // it, scale the copy, and rebuild the block instead of scaling in
            // place.
            for pairs in system.pairs.values_mut() {
                let mut pairs_ref_mut = pairs.as_ref_mut();
                let values = pairs_ref_mut.values_mut();
                let device = values.device()?;
                let mut dlpack = values.as_dlpack(device, None, dlpk::sys::DLPackVersion::current())?;

                if dlpack.is_read_only() {
                    let copy = pairs.values().copy(device)?;

                    let samples = pairs.samples();
                    let components = pairs.components();
                    let properties = pairs.properties();
                    let new_block = TensorBlock::new(
                        copy,
                        &samples,
                        &components,
                        &properties,
                    )?;
                    *pairs = new_block;

                    let mut pairs_ref_mut = pairs.as_ref_mut();
                    let values = pairs_ref_mut.values_mut();
                    dlpack = values.as_dlpack(device, None, dlpk::sys::DLPackVersion::current())?;
                }

                kernels::scale_inplace(dlpack.as_mut(), length_factor)?;
            }

            system
        };

        // convert custom data units for requested inputs
        let dlpack_version = dlpk::sys::DLPackVersion::current();
        let data_names: Vec<String> = system.custom_data.keys().cloned().collect();
        for name in data_names {
            // find if this data is a requested input
            let requested = requested_inputs.iter().find(|q| q.name.full() == name);

            if let Some(requested) = requested {
                let mut tensor = system.custom_data.remove(&name).unwrap();
                let data_unit = tensor.get_info("unit").map(|s| s.to_string()).unwrap_or_default();

                let factor = crate::unit_conversion_factor(&data_unit, &requested.unit)?;

                tensor = crate::utils::scale_tensormap(tensor, factor)?;

                // update the unit info to the requested unit
                tensor.set_info("unit", &requested.unit);
                system.custom_data.insert(name, tensor);
            }
        }

        // update the system's length unit
        system.length_unit = model_length_unit.to_string();

        Ok(Arc::new(system))
    }
}

impl std::fmt::Debug for System {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("System")
            .field("length_unit", &self.length_unit)
            .field("types", &self.types)
            .field("positions", &self.positions)
            .field("cell", &self.cell)
            .field("pbc", &self.pbc)
            .field("pairs", &self.pairs.keys().collect::<Vec<_>>())
            .field("custom_data", &self.custom_data.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl System {
    /// Create a `System` from raw DLPack tensors
    pub fn new(
        length_unit: String,
        types: DLPackTensor,
        positions: DLPackTensor,
        cell: DLPackTensor,
        pbc: DLPackTensor,
    ) -> Result<Self, Error> {
        validate_system_tensors(&types, &positions, &cell, &pbc)?;

        let system = System {
            length_unit,
            types,
            positions,
            cell,
            pbc,
            pairs: BTreeMap::new(),
            custom_data: HashMap::new(),
        };

        crate::kernels::validate_cell_pbc(system.pbc(), system.cell())?;

        return Ok(system);
    }

    /// Get the length unit used by this system
    pub fn length_unit(&self) -> &str {
        &self.length_unit
    }

    /// Get the number of atoms/particles in this system
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn size(&self) -> usize {
        let size = self.types.shape()[0];
        debug_assert!(usize::try_from(size).is_ok());
        return size as usize;
    }

    /// Get the particle types
    pub fn types(&self) -> DLPackTensorRef<'_> {
        self.types.as_ref()
    }

    /// Get the particle positions
    pub fn positions(&self) -> DLPackTensorRef<'_> {
        self.positions.as_ref()
    }

    /// Get the unit cell
    pub fn cell(&self) -> DLPackTensorRef<'_> {
        self.cell.as_ref()
    }

    /// Get the periodic boundary condition flags
    pub fn pbc(&self) -> DLPackTensorRef<'_> {
        self.pbc.as_ref()
    }

    /// Add a pair list to this system
    pub fn add_pairs(
        self: &mut Arc<Self>,
        options: PairListOptions,
        pairs: TensorBlock,
    ) -> Result<(), Error> {
        let system_mut = Arc::get_mut(self).ok_or_else(|| {
            Error::InvalidParameter(
                "cannot modify system while there are outstanding borrowed views".into(),
            )
        })?;

        if system_mut.pairs.contains_key(&options) {
            return Err(Error::InvalidParameter(
                "the pair list for these options already exists in this system".into(),
            ));
        }

        let samples = pairs.samples();
        let samples_names = samples.names();
        if samples_names != ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"] {
            return Err(Error::InvalidParameter(
                "invalid samples for `pairs`: the samples names must be \
                'first_atom', 'second_atom', 'cell_shift_a', 'cell_shift_b', \
                'cell_shift_c'".into(),
            ));
        }

        let components = pairs.components();
        if components.len() != 1 || components[0].names() != ["xyz"] || components[0].count() != 3 {
            return Err(Error::InvalidParameter(
                "invalid components for `pairs`: there should be a \
                single 'xyz'=[0, 1, 2] component".into()
            ));
        }

        {
            let mts_array = components[0].values();
            let dl_tensor = mts_array.as_dlpack(
                components[0].device(),
                None,
                dlpk::sys::DLPackVersion::current(),
            )?;

            if !crate::kernels::is_equal_i32(dl_tensor.as_ref(), &XYZ_REFERENCE)? {
                return Err(Error::InvalidParameter(
                    "invalid components for `pairs`: the 'xyz' component should \
                    contain [[0], [1], [2]]".into()
                ));
            }
        }

        let properties = pairs.properties();
        if properties.names() != ["distance"] || properties.count() != 1 {
            return Err(Error::InvalidParameter(
                "invalid properties for `pairs`: there should be a single \
                'distance'=0 property".into()
            ));
        }

        {
            let mts_array = properties.values();
            let dl_tensor = mts_array.as_dlpack(
                properties.device(),
                None,
                dlpk::sys::DLPackVersion::current(),
            )?;

            if !crate::kernels::is_equal_i32(dl_tensor.as_ref(), &DISTANCE_REFERENCE)? {
                return Err(Error::InvalidParameter(
                    "invalid properties for `pairs`: the 'distance' property \
                    should contain [0]".into()
                ));
            }
        }

        if !pairs.as_ref().gradient_list().is_empty() {
            return Err(Error::InvalidParameter(
                "`pairs` should not have any gradients".into()
            ));
        }

        if pairs.device()? != system_mut.device() {
            return Err(Error::InvalidParameter(format!(
                "`pairs` device ({}) does not match this system's device ({})",
                pairs.device()?, system_mut.device(),
            )));
        }

        if pairs.dtype()? != system_mut.dtype() {
            return Err(Error::InvalidParameter(format!(
                "`pairs` dtype ({}) does not match this system's dtype ({})",
                pairs.dtype()?, system_mut.dtype(),
            )));
        }

        system_mut.pairs.insert(options, pairs);
        return Ok(());
    }

    /// Get a pair list from this system
    pub fn get_pairs(&self, options: &PairListOptions) -> Option<&TensorBlock> {
        return self.pairs.get(options);
    }

    /// Get all pair list options known by this system
    pub fn known_pairs(&self) -> Vec<&PairListOptions> {
        return self.pairs.keys().collect();
    }

    /// Add custom data to this system
    ///
    /// If `override_` is `true`, existing data with the same name will be
    /// replaced.
    pub fn add_custom_data(self: &mut Arc<Self>, name: impl Into<String>, data: TensorMap, override_: bool) -> Result<(), Error> {
        let name = name.into();
        if INVALID_DATA_NAMES.contains(name.to_lowercase().as_str()) {
            return Err(Error::InvalidParameter(format!(
                "custom data can not be named '{}'", name
            )));
        }

        if data.keys().is_empty() {
            return Err(Error::InvalidParameter(format!(
                "custom data '{}' has no blocks", name
            )));
        }

        // validate the quantity
        let name = QuantityName::new(name)?;
        let quantity = quantity_for_data(name, &data)?;
        check_quantity(&quantity, &data, std::slice::from_ref(self), None)?;

        let system_mut = Arc::get_mut(self).ok_or_else(|| {
            Error::InvalidParameter(
                "cannot modify system while there are outstanding borrowed views".into(),
            )
        })?;

        if !override_ && system_mut.custom_data.contains_key(quantity.name.full()) {
            return Err(Error::InvalidParameter(format!(
                "custom data '{}' is already present in this system",
                quantity.name
            )));
        }

        system_mut.custom_data.insert(quantity.name.full().to_string(), data);
        return Ok(());
    }

    /// Get custom data from this system.
    pub fn get_custom_data(&self, name: &str) -> Result<&TensorMap, Error> {
        let lower = name.to_lowercase();
        if INVALID_DATA_NAMES.contains(lower.as_str()) {
            return Err(Error::InvalidParameter(format!(
                "custom data can not be named '{}'", name
            )));
        }

        return self.custom_data.get(name).ok_or_else(|| Error::InvalidParameter(format!(
            "no custom data for '{}' found in this system", name
        )));
    }

    /// Get all custom data names known by this system.
    pub fn known_custom_data(&self) -> Vec<&str> {
        return self.custom_data.keys().map(String::as_str).collect();
    }

    /// The device used for all tensors in this system
    pub fn device(&self) -> DLDevice {
        self.types.device()
    }

    /// The data type used for the `positions` and `cell` tensors in this
    /// system, as well as any pair lists and custom data added to this system.
    pub fn dtype(&self) -> DLDataType {
        self.positions.dtype()
    }
}

/// Guess the `SampleKind` corresponding to the provided `TensorMap`.
///
/// If `allow_unknown` is `true`, this will return `SampleKind::System` when
/// unable to determine the sample kind. Otherwise, it will return an error.
fn sample_kind_from_sample_names(data: &TensorMap, allow_unknown: bool) -> Result<SampleKind, Error> {
    assert!(!data.keys().is_empty());

    let first_block = data.block_by_id(0);
    let samples = first_block.samples();
    let sample_names = samples.names();

    if sample_names == ["system"] {
        Ok(SampleKind::System)
    } else if sample_names == ["system", "atom"] {
        Ok(SampleKind::Atom)
    } else if sample_names == ["system", "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"] {
        Ok(SampleKind::AtomPair)
    } else if allow_unknown {
        Ok(SampleKind::System)
    } else {
        Err(Error::InvalidParameter(format!(
            "data has unknown sample names: [{}]",
            sample_names.join(", ")
        )))
    }
}

/// Guess the `Quantity` corresponding to the provided custom data name and
/// `TensorMap`.
fn quantity_for_data(name: QuantityName, data: &TensorMap) -> Result<Quantity, Error> {
    assert!(!data.keys().is_empty());

    if name.is_custom() {
        return Ok(Quantity {
            name: name,
            unit: String::new(),
            description: None,
            gradients: vec![],
            sample_kind: sample_kind_from_sample_names(data, true)?,
        });
    }

    let mut gradients = Vec::new();
    let first_block = data.block_by_id(0);
    for parameter in first_block.gradient_list() {
        if parameter == "positions" {
            gradients.push(Gradients::Positions);
        } else if parameter == "cell" {
            gradients.push(Gradients::Strain);
        } else {
            return Err(Error::InvalidParameter(format!(
                "data '{}' has an unknown gradient '{}'",
                name, parameter
            )));
        }
    }

    return Ok(Quantity {
        name: name,
        unit: data.get_info("unit").unwrap_or("").into(),
        description: None,
        gradients: gradients,
        sample_kind: sample_kind_from_sample_names(data, false)?,
    });
}

fn validate_system_tensors(
    types: &DLPackTensor,
    positions: &DLPackTensor,
    cell: &DLPackTensor,
    pbc: &DLPackTensor,
) -> Result<(), Error> {
    let device = types.device();
    if positions.device() != device || cell.device() != device || pbc.device() != device {
        return Err(Error::InvalidParameter(
            "`types`, `positions`, `cell`, and `pbc` must be on the same device".into()
        ));
    }

    let dtype_i32 = <i32 as dlpk::GetDLPackDataType>::get_dlpack_data_type();
    let dtype_f32 = <f32 as dlpk::GetDLPackDataType>::get_dlpack_data_type();
    let dtype_f64 = <f64 as dlpk::GetDLPackDataType>::get_dlpack_data_type();
    let dtype_bool = <bool as dlpk::GetDLPackDataType>::get_dlpack_data_type();

    if types.dtype() != dtype_i32 {
        return Err(Error::InvalidParameter(
            "`types` must be a tensor of 32-bit integers".into()
        ));
    }

    let types_shape = types.shape();
    if types_shape.len() != 1 || types_shape[0] < 0 {
        return Err(Error::InvalidParameter(format!(
            "`types` must be a (n_atoms,) tensor, got a tensor with shape [{}]",
            types_shape.iter().map(|dim| dim.to_string()).collect::<Vec<_>>().join(", ")
        )));
    }

    let n_atoms = types_shape[0];

    let positions_shape = positions.shape();
    if positions_shape.len() != 2  || positions_shape[0] != n_atoms || positions_shape[1] != 3 {
        return Err(Error::InvalidParameter(format!(
            "`positions` must be a (n_atoms x 3) tensor, got a tensor with shape [{}]",
            positions_shape.iter().map(|dim| dim.to_string()).collect::<Vec<_>>().join(", ")
        )));
    }

    if positions.dtype() != dtype_f32 && positions.dtype() != dtype_f64 {
        return Err(Error::InvalidParameter(
            "`positions` must be a tensor of 32 or 64-bit floating point data".into()
        ));
    }

    let cell_shape = cell.shape();
    if cell_shape.len() != 2 || cell_shape[0] != 3 || cell_shape[1] != 3 {
        return Err(Error::InvalidParameter(format!(
            "`cell` must be a (3 x 3) tensor, got a tensor with shape [{}]",
            cell_shape.iter().map(|dim| dim.to_string()).collect::<Vec<_>>().join(", ")
        )));
    }

    if cell.dtype() != positions.dtype() {
        return Err(Error::InvalidParameter(format!(
            "`cell` must have the same dtype as `positions`, got {} and {}",
            cell.dtype(),
            positions.dtype()
        )));
    }

    let pbc_shape = pbc.shape();
    if pbc_shape.len() != 1 || pbc_shape[0] != 3 {
        return Err(Error::InvalidParameter(format!(
            "`pbc` must contain 3 entries, got a tensor with shape [{}]",
            pbc_shape.iter().map(|dim| dim.to_string()).collect::<Vec<_>>().join(", ")
        )));
    }

    if pbc.dtype() != dtype_bool {
        return Err(Error::InvalidParameter(
            "`pbc` must be a tensor of booleans".into()
        ));
    }

    return Ok(());
}

#[cfg(test)]
pub(crate) use tests::test_system;

#[cfg(test)]
mod tests {
    use super::*;
    use metatensor::Labels;
    use ndarray::{Array1, Array2, ArrayViewD};
    use approx::assert_relative_eq;

    // -----------------------------------------------------------------------
    // helpers to create DLPack tensors
    // -----------------------------------------------------------------------
    fn type_tensor(data: &[i32]) -> DLPackTensor {
        Array1::from_vec(data.to_vec()).try_into().unwrap()
    }

    #[allow(clippy::cast_precision_loss)]
    fn positions_tensor(n_atoms: usize, dtype: &str) -> DLPackTensor {
        match dtype {
            "f32" => {
                let mut data = Vec::with_capacity(3 * n_atoms);
                for i in 0..n_atoms {
                    data.extend_from_slice(&[i as f32, 0.0, 0.0]);
                }
                Array2::from_shape_vec((n_atoms, 3), data).unwrap().try_into().unwrap()
            }
            "f64" => {
                let mut data = Vec::with_capacity(3 * n_atoms);
                for i in 0..n_atoms {
                    data.extend_from_slice(&[i as f64, 0.0, 0.0]);
                }
                Array2::from_shape_vec((n_atoms, 3), data).unwrap().try_into().unwrap()
            }
            _ => panic!("unsupported dtype '{}'", dtype),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn cell_tensor(size: f64, dtype: &str) -> DLPackTensor {
        match dtype {
            "f32" => {
                Array2::<f32>::from_shape_vec(
                    (3, 3),
                    vec![
                        size as f32, 0.0, 0.0,
                        0.0, size as f32, 0.0,
                        0.0, 0.0, size as f32,
                    ],
                ).unwrap().try_into().unwrap()
            }
            "f64" => Array2::<f64>::from_shape_vec(
                (3, 3),
                vec![
                    size, 0.0, 0.0,
                    0.0, size, 0.0,
                    0.0, 0.0, size,
                ],
            ).unwrap().try_into().unwrap(),
            _ => panic!("unsupported dtype '{}'", dtype),
        }
    }

    fn pbc_tensor(data: &[bool]) -> DLPackTensor {
        Array1::from_vec(data.to_vec()).try_into().unwrap()
    }

    fn valid_pair_block(dtype: &str) -> TensorBlock {
        let samples = Labels::new(
            ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            [[0i32, 1, 0, 0, 0]],
        );
        let components = vec![Labels::new(["xyz"], [[0i32], [1], [2]])];
        let properties = Labels::new(["distance"], [[0i32]]);

        match dtype {
            "f32" => {
                let values = ndarray::ArrayD::<f32>::from_shape_vec(vec![1, 3, 1], vec![1.5, 2.5, 3.5]).unwrap();
                TensorBlock::new(values, &samples, &components, &properties).unwrap()
            }
            "f64" => {
                let values = ndarray::ArrayD::<f64>::from_shape_vec(vec![1, 3, 1], vec![1.5, 2.5, 3.5]).unwrap();
                TensorBlock::new(values, &samples, &components, &properties).unwrap()
            }
            _ => panic!("unsupported dtype '{}'", dtype),
        }
    }

    fn valid_custom_data(dtype: &str) -> TensorMap {
        let keys = Labels::new(["key"], [[0i32]]);
        let samples = Labels::new(["sample"], [[0i32]]);
        let properties = Labels::new(["property"], [[0i32]]);

        let block = match dtype {
            "f32" => {
                let values = ndarray::ArrayD::<f32>::from_shape_vec(vec![1, 1], vec![42.0]).unwrap();
                TensorBlock::new(values, &samples, &[], &properties).unwrap()
            }
            "f64" => {
                let values = ndarray::ArrayD::<f64>::from_shape_vec(vec![1, 1], vec![42.0]).unwrap();
                TensorBlock::new(values, &samples, &[], &properties).unwrap()
            }
            _ => panic!("unsupported dtype '{}'", dtype),
        };

        let mut tensor = TensorMap::new(keys, vec![block]).unwrap();
        tensor.set_info("unit", "eV");

        return tensor;
    }

    pub(crate) fn test_system(dtype: &str) -> Arc<System> {
        let mut system =  Arc::new(System::new(
            "Angstrom".into(),
            tests::type_tensor(&[1, 6, 8]),
            tests::positions_tensor(3, dtype),
            tests::cell_tensor(10.0, dtype),
            tests::pbc_tensor(&[true, true, true]),
        ).unwrap());

        system.add_custom_data("custom::data/name", valid_custom_data(dtype), true).unwrap();

        let options = PairListOptions {
            cutoff: 3.5,
            full_list: true,
            strict: false,
            requestors: vec![],
        };

        system.add_pairs(options, valid_pair_block(dtype)).unwrap();

        return system;
    }

    #[test]
    fn system() {
        let system =  System::new(
            "Angstrom".into(),
            type_tensor(&[1, 6, 8]),
            positions_tensor(3, "f32"),
            cell_tensor(10.0, "f32"),
            pbc_tensor(&[true, true, true]),
        ).unwrap();

        assert_eq!(system.length_unit(), "Angstrom");
        assert_eq!(system.size(), 3);
        assert_eq!(system.device(), DLDevice::cpu());
        assert_eq!(system.dtype().bits, 32);

        let system = System::new(
            "Angstrom".into(),
            type_tensor(&[1, 6, 8]),
            positions_tensor(3, "f64"),
            cell_tensor(10.0, "f64"),
            pbc_tensor(&[true, true, true]),
        ).unwrap();
        assert_eq!(system.length_unit(), "Angstrom");
        assert_eq!(system.size(), 3);
        assert_eq!(system.device(), DLDevice::cpu());
        assert_eq!(system.dtype().bits, 64);
    }

    #[test]
    fn system_invalid_tensors() {
        let length_unit = "Angstrom".to_string();

        let bad_types: DLPackTensor = Array1::<f32>::from_vec(vec![1.0, 2.0]).try_into().unwrap();
        let positions = positions_tensor(2, "f32");
        let cell = cell_tensor(0.0, "f32");
        let pbc = pbc_tensor(&[true, true, true]);

        let err = System::new(length_unit.clone(), bad_types, positions, cell, pbc).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: `types` must be a tensor of 32-bit integers");

        let bad_types: DLPackTensor = Array2::<i32>::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap().try_into().unwrap();
        let positions = positions_tensor(2, "f32");
        let cell = cell_tensor(0.0, "f32");
        let pbc = pbc_tensor(&[true, true, true]);
        let err = System::new(length_unit.clone(), bad_types, positions, cell, pbc).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: `types` must be a (n_atoms,) tensor, got a tensor with shape [2, 2]");

        let types = type_tensor(&[1]);
        let bad_positions: DLPackTensor = Array2::<i32>::from_shape_vec((1, 3), vec![1, 2, 3]).unwrap().try_into().unwrap();
        let cell = cell_tensor(0.0, "f32");
        let pbc = pbc_tensor(&[true, true, true]);
        let err = System::new(length_unit.clone(), types, bad_positions, cell, pbc).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: `positions` must be a tensor of 32 or 64-bit floating point data");

        let types = type_tensor(&[1, 6]);
        let bad_positions = Array2::<f32>::from_shape_vec((2, 2), vec![0.0; 4]).unwrap().try_into().unwrap();
        let cell = cell_tensor(0.0, "f32");
        let pbc = pbc_tensor(&[true, true, true]);
        let err = System::new("Angstrom".into(), types, bad_positions, cell, pbc).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: `positions` must be a (n_atoms x 3) tensor, got a tensor with shape [2, 2]");

        let types = type_tensor(&[1, 6]);
        let positions = positions_tensor(2, "f32");
        let bad_cell = Array2::<f32>::from_shape_vec((2, 3), vec![0.0; 6]).unwrap().try_into().unwrap();
        let pbc = pbc_tensor(&[true, true, true]);
        let err = System::new(length_unit.clone(), types, positions, bad_cell, pbc).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: `cell` must be a (3 x 3) tensor, got a tensor with shape [2, 3]");

        let types = type_tensor(&[1, 6]);
        let positions = positions_tensor(2, "f32");
        let cell = cell_tensor(0.0, "f64");
        let pbc = pbc_tensor(&[true, true, true]);
        let err = System::new(length_unit.clone(), types, positions, cell, pbc).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: `cell` must have the same dtype as `positions`, got f64 and f32");

        let bad_pbc_dtype: DLPackTensor = Array1::<i32>::from_vec(vec![1, 0, 1]).try_into().unwrap();
        let types = type_tensor(&[1, 6]);
        let positions = positions_tensor(2, "f32");
        let cell = cell_tensor(0.0, "f32");
        let err =  System::new(length_unit.clone(), types, positions, cell, bad_pbc_dtype).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: `pbc` must be a tensor of booleans");

        let types = type_tensor(&[1, 6]);
        let positions = positions_tensor(2, "f32");
        let cell = cell_tensor(0.0, "f32");
        let bad_pbc = pbc_tensor(&[true, true]);
        let err = System::new(length_unit, types, positions, cell, bad_pbc).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: `pbc` must contain 3 entries, got a tensor with shape [2]");
    }

    #[test]
    fn system_periodic() {
        let length_unit = "Angstrom".to_string();

        // valid periodicity combinations: (1) fully periodic
        let types = type_tensor(&[1]);
        let positions = positions_tensor(1, "f32");
        let cell = cell_tensor(10.0, "f32");
        let pbc = pbc_tensor(&[true, true, true]);
        System::new(length_unit.clone(), types, positions, cell, pbc).unwrap();

        // (2) fully non-periodic with zero cell
        let types = type_tensor(&[1]);
        let positions = positions_tensor(1, "f32");
        let cell = cell_tensor(0.0, "f32");
        let pbc = pbc_tensor(&[false, false, false]);
        System::new(length_unit.clone(), types, positions, cell, pbc).unwrap();

        // (3) mixed periodic/non-periodic
        let types = type_tensor(&[1]);
        let positions = positions_tensor(1, "f32");
        let cell: DLPackTensor = Array2::<f32>::from_shape_vec(
            (3, 3),
            vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
        ).unwrap().try_into().unwrap();
        let pbc = pbc_tensor(&[true, false, true]);
        System::new(length_unit.clone(), types, positions, cell, pbc).unwrap();

        // invalid periodicity/cell
        let types = type_tensor(&[1]);
        let positions = positions_tensor(1, "f32");
        let cell = cell_tensor(10.0, "f32");
        let pbc = pbc_tensor(&[true, false, true]);
        let err = System::new(length_unit.clone(), types, positions, cell, pbc).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: invalid cell: for non-periodic dimensions, the corresponding cell vector must be zero, but cell[1] contains non-zero values");
    }

    #[test]
    fn add_pairs() {
        let mut system = Arc::new(System::new(
            "Angstrom".into(),
            type_tensor(&[1, 6, 8]),
            positions_tensor(3, "f32"),
            cell_tensor(10.0, "f32"),
            pbc_tensor(&[true, true, true]),
        ).unwrap());

        let options = PairListOptions { cutoff: 3.5, full_list: true, strict: false, requestors: vec![] };
        let pairs = valid_pair_block("f32");
        let pairs_ptr = pairs.as_ptr();
        system.add_pairs(options.clone(), pairs).unwrap();
        assert_eq!(system.known_pairs().len(), 1);
        assert_eq!(system.get_pairs(&options).unwrap().properties().names(), ["distance"]);

        let options_with_requestor = PairListOptions {
            cutoff: 3.5,
            full_list: true,
            strict: false,
            requestors: vec!["test-requestor".into()],
        };

        let pairs_from_system = system.get_pairs(&options_with_requestor).unwrap();
        assert_eq!(pairs_from_system.as_ptr(), pairs_ptr);

        system.add_pairs(
            PairListOptions { cutoff: 5.0, full_list: false, strict: true, requestors: vec![] },
            valid_pair_block("f32"),
        ).unwrap();
        assert_eq!(system.known_pairs().len(), 2);
    }


    #[test]
    fn custom_data() {
        let mut system = Arc::new(System::new(
            "Angstrom".into(),
            type_tensor(&[1, 6, 8]),
            positions_tensor(3, "f32"),
            cell_tensor(10.0, "f32"),
            pbc_tensor(&[true, true, true]),
        ).unwrap());

        let data = valid_custom_data("f32");
        system.add_custom_data("test::my_data", data, false).unwrap();
        assert_eq!(system.known_custom_data(), vec!["test::my_data"]);
        assert_eq!(system.get_custom_data("test::my_data").unwrap().keys().names(), ["key"]);

        let err = system.add_custom_data("test::my_data", valid_custom_data("f32"), false).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: custom data 'test::my_data' is already present in this system");

        let replacement = valid_custom_data("f32");
        system.add_custom_data("test::my_data", replacement, true).unwrap();
        assert_eq!(system.known_custom_data(), vec!["test::my_data"]);

        let mut system = Arc::new(System::new(
            "Angstrom".into(),
            type_tensor(&[1, 6, 8]),
            positions_tensor(3, "f32"),
            cell_tensor(10.0, "f32"),
            pbc_tensor(&[true, true, true]),
        ).unwrap());

        let test_data_a = valid_custom_data("f32");
        let test_data_a_ptr = test_data_a.as_ptr();
        system.add_custom_data("test::a", test_data_a, false).unwrap();

        let test_data_b = valid_custom_data("f32");
        let test_data_b_ptr = test_data_b.as_ptr();
        system.add_custom_data("test::b", test_data_b, false).unwrap();

        let mut names = system.known_custom_data();
        names.sort_unstable();
        assert_eq!(names, vec!["test::a", "test::b"]);

        let data_a = system.get_custom_data("test::a").unwrap();
        assert_eq!(data_a.as_ptr(), test_data_a_ptr);

        let data_b = system.get_custom_data("test::b").unwrap();
        assert_eq!(data_b.as_ptr(), test_data_b_ptr);

        let err = system.get_custom_data("no_such_data").unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: no custom data for 'no_such_data' found in this system");
    }

    #[test]
    fn custom_data_validation() {
        let mut system = Arc::new(System::new(
            "Angstrom".into(),
            type_tensor(&[1, 6, 8]),
            positions_tensor(3, "f32"),
            cell_tensor(10.0, "f32"),
            pbc_tensor(&[true, true, true]),
        ).unwrap());
        for name in ["types", "type", "Positions", "position", "CELL", "neighbors", "neighbor", "pair", "pairs", "Types", "POSITIONS", "Cell", "Neighbors"] {
            let data = valid_custom_data("f32");
            let err = system.add_custom_data(name.to_string(), data, false).unwrap_err();
            assert_eq!(err.to_string(), format!("invalid parameter: custom data can not be named '{}'", name));
        }

        let err = system.add_custom_data("my_data", valid_custom_data("f32"), false).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: 'my_data' is not a standard quantity name; custom quantity names must use '<namespace>::<name>'");

        let keys = Labels::empty(vec!["key"]);
        let empty = TensorMap::new(keys, vec![]).unwrap();
        let err = system.add_custom_data("test::empty", empty, false).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: custom data 'test::empty' has no blocks");

        let dtype_mismatch = valid_custom_data("f64");
        let err = system.add_custom_data("test::dtype", dtype_mismatch, false).unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter: invalid dtype for quantity 'test::dtype': expected f32, got f64");
    }

    #[test]
    fn system_clone() {
        let system = test_system("f32");
        let cloned = system.try_clone().unwrap();

        // same metadata
        assert_eq!(cloned.length_unit(), system.length_unit());
        assert_eq!(cloned.size(), system.size());
        assert_eq!(cloned.device(), system.device());
        assert_eq!(cloned.dtype(), system.dtype());

        // same pairs
        assert_eq!(cloned.known_pairs().len(), system.known_pairs().len());

        // same custom data
        assert_eq!(cloned.known_custom_data(), system.known_custom_data());

        // the data is independent — modifying one doesn't affect the other.
        assert_ne!(
            system.positions().raw.data as usize,
            cloned.positions().raw.data as usize,
        );
        assert_ne!(
            system.cell().raw.data as usize,
            cloned.cell().raw.data as usize,
        );
        assert_ne!(
            system.types().raw.data as usize,
            cloned.types().raw.data as usize,
        );
    }

    #[test]
    fn system_convert_units() {
        let system = test_system("f64");
        let ref_system = system.try_clone().unwrap();

        let requested_inputs = [Quantity {
            name: QuantityName::new("custom::data/name".into()).unwrap(),
            unit: "kJ/mol".into(),
            description: None,
            gradients: vec![],
            sample_kind: SampleKind::System,
        }];

        // convert from Angstrom to nanometer, and eV to kJ/mol
        let converted = system.convert_units("nanometer", &requested_inputs).unwrap();

        // length unit should be updated
        assert_eq!(converted.length_unit(), "nanometer");

        // positions should be scaled: Angstrom -> nanometer (factor 0.1)
        let positions: ArrayViewD<f64> = converted.positions().try_into().unwrap();
        let ref_positions: ArrayViewD<f64> = ref_system.positions().try_into().unwrap();
        assert_relative_eq!(positions, ref_positions.to_owned() * 0.1, max_relative = 1e-6);

        // cell should be scaled
        let cell: ArrayViewD<f64> = converted.cell().try_into().unwrap();
        let ref_cell: ArrayViewD<f64> = ref_system.cell().try_into().unwrap();
        assert_relative_eq!(cell, ref_cell.to_owned() * 0.1, max_relative = 1e-6);

        // pair list values should be scaled
        let options = PairListOptions {
            cutoff: 3.5,
            full_list: true,
            strict: false,
            requestors: vec![],
        };
        let pairs = converted.get_pairs(&options).unwrap();
        let pairs_values = pairs.values().to_ndarray_lock::<f64>().read().unwrap();
        let pairs_ref = ref_system.get_pairs(&options).unwrap();
        let pairs_values_ref = pairs_ref.values().to_ndarray_lock::<f64>().read().unwrap().clone();
        assert_relative_eq!(*pairs_values, pairs_values_ref * 0.1, max_relative = 1e-6);

        // custom data should have unit info updated
        let converted_data = converted.get_custom_data("custom::data/name").unwrap();
        assert_eq!(converted_data.get_info("unit"), Some("kJ/mol"));
        assert_eq!(converted_data.keys().count(), 1);

        // custom data values should be scaled: eV -> kJ/mol
        let expected_factor = crate::unit_conversion_factor("eV", "kJ/mol").unwrap();

        let ref_data = ref_system.get_custom_data("custom::data/name").unwrap();
        let ref_block = ref_data.block_by_id(0);
        let ref_values = ref_block.values().to_ndarray_lock::<f64>().read().unwrap().clone();

        let block = converted_data.block_by_id(0);
        let values = block.values().to_ndarray_lock::<f64>().read().unwrap();
        assert_relative_eq!(*values, ref_values * expected_factor, max_relative = 1e-6);
    }
}
