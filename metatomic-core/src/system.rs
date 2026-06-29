use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::LazyLock;

use dlpk::sys::{DLDataType, DLDevice};
use dlpk::{DLPackTensor, DLPackTensorRef};
use metatensor::{TensorBlock, TensorMap};

use crate::{Error, PairListOptions};

/// Names that can never be used as custom data in a system
static INVALID_DATA_NAMES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from(["types", "type", "positions", "position", "cell", "neighbors", "neighbor", "pair", "pairs"])
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
        &mut self,
        options: PairListOptions,
        pairs: TensorBlock,
    ) -> Result<(), Error> {
        if self.pairs.contains_key(&options) {
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
            let reference = ndarray::ArrayViewD::<i32>::from_shape(
                ndarray::IxDyn(&[3usize, 1]),
                &[0i32, 1, 2],
            ).unwrap();

            if !crate::kernels::is_equal_i32(dl_tensor.as_ref(), reference)? {
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
            let reference = ndarray::ArrayViewD::<i32>::from_shape(
                ndarray::IxDyn(&[1usize, 1]),
                &[0i32],
            ).unwrap();

            if !crate::kernels::is_equal_i32(dl_tensor.as_ref(), reference)? {
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

        if pairs.device()? != self.device() {
            return Err(Error::InvalidParameter(format!(
                "`pairs` device ({}) does not match this system's device ({})",
                pairs.device()?, self.device(),
            )));
        }

        if pairs.dtype()? != self.dtype() {
            return Err(Error::InvalidParameter(format!(
                "`pairs` dtype ({}) does not match this system's dtype ({})",
                pairs.dtype()?, self.dtype(),
            )));
        }

        self.pairs.insert(options, pairs);
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
    pub fn add_custom_data(&mut self, name: impl Into<String>, data: TensorMap, override_: bool) -> Result<(), Error> {
        let name = name.into();
        if INVALID_DATA_NAMES.contains(name.to_lowercase().as_str()) {
            return Err(Error::InvalidParameter(format!(
                "custom data can not be named '{}'", name
            )));
        }

        crate::quantities::validate_quantity_name(&name)?;

        if !override_ && self.custom_data.contains_key(&name) {
            return Err(Error::InvalidParameter(format!(
                "custom data '{}' is already present in this system",
                name
            )));
        }

        if data.keys().count() == 0 {
            return Err(Error::InvalidParameter(format!(
                "custom data '{}' has no blocks", name
            )));
        }

        // TODO: add TensorMap::device/dtype and use them here
        let block = data.block_by_id(0);
        let values = block.values();
        let data_device = values.device()?;
        if data_device != self.device() {
            return Err(Error::InvalidParameter(format!(
                "device ({}:{}) of the custom data '{}' does not match this system device ({}:{})",
                data_device.device_type, data_device.device_id, name,
                self.device().device_type, self.device().device_id,
            )));
        }

        let values_dtype = values.dtype()?;
        if values_dtype != self.dtype() {
            return Err(Error::InvalidParameter(format!(
                "dtype of custom data '{}' does not match this system dtype",
                name,
            )));
        }

        self.custom_data.insert(name, data);
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
            "no data for '{}' found in this system", name
        )));
    }

    /// Get all custom data names known by this system.
    pub fn known_custom_data(&self) -> Vec<&str> {
        return self.custom_data.keys().map(String::as_str).collect();
    }

    /// The device used for all tensors in this system
    fn device(&self) -> DLDevice {
        self.types.device()
    }

    /// The data type used for the `positions` and `cell` tensors in this
    /// system, as well as any pair lists and custom data added to this system.
    fn dtype(&self) -> DLDataType {
        self.positions.dtype()
    }
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
        return Err(Error::InvalidParameter(
            "`cell` must have the same dtype as `positions`".into()
        ));
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
    use ndarray::{Array1, Array2};

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

        TensorMap::new(keys, vec![block]).unwrap()
    }

    fn assert_error<T>(result: Result<T, Error>, expected: &str) {
        let error = match result {
            Ok(_) => panic!("expected error"),
            Err(error) => error,
        };
        assert_eq!(error.to_string(), expected);
    }

    pub(crate) fn test_system() -> System {
        let mut system =  System::new(
            "Angstrom".into(),
            tests::type_tensor(&[1, 6, 8]),
            tests::positions_tensor(3, "f32"),
            tests::cell_tensor(10.0, "f32"),
            tests::pbc_tensor(&[true, true, true]),
        ).unwrap();

        system.add_custom_data("custom::data/name", valid_custom_data("f32"), true).unwrap();

        let options = PairListOptions {
            cutoff: 3.5,
            full_list: true,
            strict: false,
            requestors: vec![],
        };

        system.add_pairs(options, valid_pair_block("f32")).unwrap();

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

        assert_error(
            System::new(length_unit.clone(), bad_types, positions, cell, pbc),
            "invalid parameter: `types` must be a tensor of 32-bit integers",
        );

        let bad_types: DLPackTensor = Array2::<i32>::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap().try_into().unwrap();
        let positions = positions_tensor(2, "f32");
        let cell = cell_tensor(0.0, "f32");
        let pbc = pbc_tensor(&[true, true, true]);
        assert_error(
            System::new(length_unit.clone(), bad_types, positions, cell, pbc),
            "invalid parameter: `types` must be a (n_atoms,) tensor, got a tensor with shape [2, 2]",
        );

        let types = type_tensor(&[1]);
        let bad_positions: DLPackTensor = Array2::<i32>::from_shape_vec((1, 3), vec![1, 2, 3]).unwrap().try_into().unwrap();
        let cell = cell_tensor(0.0, "f32");
        let pbc = pbc_tensor(&[true, true, true]);
        assert_error(
            System::new(length_unit.clone(), types, bad_positions, cell, pbc),
            "invalid parameter: `positions` must be a tensor of 32 or 64-bit floating point data",
        );

        let types = type_tensor(&[1, 6]);
        let bad_positions = Array2::<f32>::from_shape_vec((2, 2), vec![0.0; 4]).unwrap().try_into().unwrap();
        let cell = cell_tensor(0.0, "f32");
        let pbc = pbc_tensor(&[true, true, true]);
        assert_error(
            System::new("Angstrom".into(), types, bad_positions, cell, pbc),
            "invalid parameter: `positions` must be a (n_atoms x 3) tensor, got a tensor with shape [2, 2]",
        );

        let types = type_tensor(&[1, 6]);
        let positions = positions_tensor(2, "f32");
        let bad_cell = Array2::<f32>::from_shape_vec((2, 3), vec![0.0; 6]).unwrap().try_into().unwrap();
        let pbc = pbc_tensor(&[true, true, true]);
        assert_error(
            System::new(length_unit.clone(), types, positions, bad_cell, pbc),
            "invalid parameter: `cell` must be a (3 x 3) tensor, got a tensor with shape [2, 3]",
        );

        let types = type_tensor(&[1, 6]);
        let positions = positions_tensor(2, "f32");
        let cell = cell_tensor(0.0, "f64");
        let pbc = pbc_tensor(&[true, true, true]);
        assert_error(
            System::new(length_unit.clone(), types, positions, cell, pbc),
            "invalid parameter: `cell` must have the same dtype as `positions`",
        );

        let bad_pbc_dtype: DLPackTensor = Array1::<i32>::from_vec(vec![1, 0, 1]).try_into().unwrap();
        let types = type_tensor(&[1, 6]);
        let positions = positions_tensor(2, "f32");
        let cell = cell_tensor(0.0, "f32");
        assert_error(
            System::new(length_unit.clone(), types, positions, cell, bad_pbc_dtype),
            "invalid parameter: `pbc` must be a tensor of booleans",
        );

        let types = type_tensor(&[1, 6]);
        let positions = positions_tensor(2, "f32");
        let cell = cell_tensor(0.0, "f32");
        let bad_pbc = pbc_tensor(&[true, true]);
        assert_error(
            System::new(length_unit, types, positions, cell, bad_pbc),
            "invalid parameter: `pbc` must contain 3 entries, got a tensor with shape [2]",
        );
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
        assert_error(
            System::new(length_unit.clone(), types, positions, cell, pbc),
            "invalid parameter: invalid cell: for non-periodic dimensions, the corresponding cell vector must be zero, but cell[1] contains non-zero values",
        );
    }

    #[test]
    fn add_pairs() {
        let mut system = System::new(
            "Angstrom".into(),
            type_tensor(&[1, 6, 8]),
            positions_tensor(3, "f32"),
            cell_tensor(10.0, "f32"),
            pbc_tensor(&[true, true, true]),
        ).unwrap();

        let options = PairListOptions { cutoff: 3.5, full_list: true, strict: false, requestors: vec![] };
        let pairs = valid_pair_block("f32");
        system.add_pairs(options.clone(), pairs).unwrap();
        assert_eq!(system.known_pairs().len(), 1);
        assert_eq!(system.get_pairs(&options).unwrap().properties().names(), ["distance"]);

        let options_with_requestor = PairListOptions {
            cutoff: 3.5,
            full_list: true,
            strict: false,
            requestors: vec!["test-requestor".into()],
        };
        // TODO: check that this is the exact same block once we can get the
        // pointer to check for id.
        assert!(system.get_pairs(&options_with_requestor).is_some());

        system.add_pairs(
            PairListOptions { cutoff: 5.0, full_list: false, strict: true, requestors: vec![] },
            valid_pair_block("f32"),
        ).unwrap();
        assert_eq!(system.known_pairs().len(), 2);
    }


    #[test]
    fn custom_data() {
        let mut system = System::new(
            "Angstrom".into(),
            type_tensor(&[1, 6, 8]),
            positions_tensor(3, "f32"),
            cell_tensor(10.0, "f32"),
            pbc_tensor(&[true, true, true]),
        ).unwrap();

        let data = valid_custom_data("f32");
        system.add_custom_data("test::my_data", data, false).unwrap();
        assert_eq!(system.known_custom_data(), vec!["test::my_data"]);
        assert_eq!(system.get_custom_data("test::my_data").unwrap().keys().names(), ["key"]);

        assert_error(
            system.add_custom_data("test::my_data", valid_custom_data("f32"), false),
            "invalid parameter: custom data 'test::my_data' is already present in this system",
        );

        let replacement = valid_custom_data("f32");
        system.add_custom_data("test::my_data", replacement, true).unwrap();
        assert_eq!(system.known_custom_data(), vec!["test::my_data"]);

        let mut system = System::new(
            "Angstrom".into(),
            type_tensor(&[1, 6, 8]),
            positions_tensor(3, "f32"),
            cell_tensor(10.0, "f32"),
            pbc_tensor(&[true, true, true]),
        ).unwrap();
        system.add_custom_data("test::a", valid_custom_data("f32"), false).unwrap();
        system.add_custom_data("test::b", valid_custom_data("f32"), false).unwrap();
        let mut names = system.known_custom_data();
        names.sort_unstable();
        assert_eq!(names, vec!["test::a", "test::b"]);

        // TODO: check we get back the same pointer
        assert!(system.get_custom_data("test::a").is_ok());
        assert!(system.get_custom_data("test::b").is_ok());

        assert_error(
            system.get_custom_data("no_such_data"),
            "invalid parameter: no data for 'no_such_data' found in this system",
        );
    }

    #[test]
    fn custom_data_validation() {
        let mut system = System::new(
            "Angstrom".into(),
            type_tensor(&[1, 6, 8]),
            positions_tensor(3, "f32"),
            cell_tensor(10.0, "f32"),
            pbc_tensor(&[true, true, true]),
        ).unwrap();
        for name in ["types", "type", "Positions", "position", "CELL", "neighbors", "neighbor", "pair", "pairs", "Types", "POSITIONS", "Cell", "Neighbors"] {
            let data = valid_custom_data("f32");
            assert_error(
                system.add_custom_data(name.to_string(), data, false),
                &format!("invalid parameter: custom data can not be named '{}'", name),
            );
        }

        assert_error(
            system.add_custom_data("my_data", valid_custom_data("f32"), false),
            "invalid parameter: 'my_data' is not a standard quantity name; custom quantity names must use '<namespace>::<name>'",
        );

        let keys = Labels::empty(vec!["key"]);
        let empty = TensorMap::new(keys, vec![]).unwrap();
        assert_error(
            system.add_custom_data("test::empty", empty, false),
            "invalid parameter: custom data 'test::empty' has no blocks",
        );

        let dtype_mismatch = valid_custom_data("f64");
        assert_error(
            system.add_custom_data("test::dtype", dtype_mismatch, false),
            "invalid parameter: dtype of custom data 'test::dtype' does not match this system dtype",
        );
    }
}
