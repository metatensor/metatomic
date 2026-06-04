use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Seek, Write};
use std::path::Path;

use once_cell::sync::Lazy;

use dlpk::sys::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType};
use dlpk::{DLPackTensor, DLPackTensorRef};
use metatensor::{TensorBlock, TensorMap};

use crate::{Error, PairListOptions};

/// Names that can never be used as custom data in a system
static INVALID_DATA_NAMES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
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

        if system.device().device_type == DLDeviceType::kDLCPU {
            validate_cpu_system_data(&system)?;
        }

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

        #[allow(clippy::collapsible_if)]
        if components[0].device().device_type == DLDeviceType::kDLCPU {
            if components[0][0] != [0] || components[0][1] != [1] || components[0][2] != [2] {
                return Err(Error::InvalidParameter(
                    "invalid components for `pairs`: the 'xyz' \
                    component should contain [0, 1, 2]".into()
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

        #[allow(clippy::collapsible_if)]
        if properties.device().device_type == DLDeviceType::kDLCPU {
            if properties[0] != [0] {
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

        // TODO: add TensorBlock::device/dtype and use them here
        let values = pairs.values();
        let values_device = values.device()?;
        if values_device != self.device() {
            return Err(Error::InvalidParameter(format!(
                "`pairs` device ({}) does not match this system's device ({})",
                values_device, self.device(),
            )));
        }

        let values_dtype = values.dtype()?;
        if values_dtype != self.dtype() {
            return Err(Error::InvalidParameter(format!(
                "`pairs` dtype ({}) does not match this system's dtype ({})",
                values_dtype, self.dtype(),
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
    pub fn add_custom_data(&mut self, name: String, data: TensorMap, override_: bool) -> Result<(), Error> {
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

    /// Save this `System` to a file.
    ///
    /// The file uses the same ZIP-based format as `metatomic.torch.save`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let file = std::fs::File::create(path)?;
        let mut archive = zip::ZipWriter::new(file);
        write_system_to_zip(&mut archive, self)?;
        archive.finish().map_err(zip_error)?;
        return Ok(());
    }

    /// Load a `System` from a file saved by `metatomic.torch.save`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        return load_torch_system(path.as_ref());
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

fn validate_cpu_system_data(system: &System) -> Result<(), Error> {
    let pbc_array: ndarray::ArrayView1<bool> = system.pbc().try_into()?;

    if system.dtype().bits == 32 {
        let cell_array: ndarray::ArrayView2<f32> = system.cell().try_into()?;
        for i in 0..3 {
            if !pbc_array[i] && !cell_array.row(i).iter().all(|&x| x == 0.0) {
                return Err(Error::InvalidParameter(format!(
                    "invalid cell: for non-periodic dimensions, the corresponding \
                    cell vector must be zero, but cell[{}] contains non-zero values",
                    i
                )));
            }
        }
    } else {
        let cell_array: ndarray::ArrayView2<f64> = system.cell().try_into()?;
        for i in 0..3 {
            if !pbc_array[i] && !cell_array.row(i).iter().all(|&x| x == 0.0) {
                return Err(Error::InvalidParameter(format!(
                    "invalid cell: for non-periodic dimensions, the corresponding \
                    cell vector must be zero, but cell[{}] contains non-zero values",
                    i
                )));
            }
        }
    }

    return Ok(());
}

fn metatensor_serialization_error(error: metatensor::Error) -> Error {
    return Error::Serialization(error.to_string());
}

struct LegacyNpyHeader {
    descr: String,
    fortran_order: bool,
    shape: Vec<i64>,
}

fn zip_error(error: zip::result::ZipError) -> Error {
    return Error::Serialization(error.to_string());
}

fn start_zip_file<W: Write + Seek>(
    archive: &mut zip::ZipWriter<W>,
    name: &str,
) -> Result<(), Error> {
    let options = zip::write::FileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);
    archive.start_file(name, options).map_err(zip_error)?;
    return Ok(());
}

fn write_system_to_zip<W: Write + Seek>(
    archive: &mut zip::ZipWriter<W>,
    system: &System,
) -> Result<(), Error> {
    write_npy_tensor_to_zip(archive, "positions.npy", system.positions())?;
    write_npy_tensor_to_zip(archive, "cell.npy", system.cell())?;
    write_npy_tensor_to_zip(archive, "types.npy", system.types())?;
    write_npy_tensor_to_zip(archive, "pbc.npy", system.pbc())?;

    for (i, (options, pairs_block)) in system.pairs.iter().enumerate() {
        let base = format!("pairs/{}/", i);
        start_zip_file(archive, &(base.clone() + "options.json"))?;
        archive.write_all(options.to_torch_json().dump().as_bytes())?;

        let mut buffer = Vec::new();
        pairs_block.save_buffer(&mut buffer).map_err(metatensor_serialization_error)?;
        start_zip_file(archive, &(base + "data.mts"))?;
        archive.write_all(&buffer)?;
    }

    for (name, tensor) in &system.custom_data {
        let mut buffer = Vec::new();
        tensor.save_buffer(&mut buffer).map_err(metatensor_serialization_error)?;
        start_zip_file(archive, &format!("data/{}.mts", name))?;
        archive.write_all(&buffer)?;
    }

    return Ok(());
}

fn write_npy_tensor_to_zip<W: Write + Seek>(
    archive: &mut zip::ZipWriter<W>,
    name: &str,
    tensor: DLPackTensorRef<'_>,
) -> Result<(), Error> {
    start_zip_file(archive, name)?;
    return write_npy_tensor(archive, name, tensor);
}

fn load_torch_system(path: &Path) -> Result<System, Error> {
    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| Error::Serialization(format!("failed to read System ZIP archive: {}", e)))?;

    let positions = read_legacy_npy_tensor(&read_zip_file(&mut archive, "positions.npy")?, "positions")?;
    let cell = read_legacy_npy_tensor(&read_zip_file(&mut archive, "cell.npy")?, "cell")?;
    let types = read_legacy_npy_tensor(&read_zip_file(&mut archive, "types.npy")?, "types")?;
    let pbc = read_legacy_npy_tensor(&read_zip_file(&mut archive, "pbc.npy")?, "pbc")?;

    let mut system = System::new(String::new(), types, positions, cell, pbc)?;
    let names = archive.file_names().map(str::to_owned).collect::<Vec<_>>();

    let mut pairs = Vec::new();
    for name in &names {
        if let Some(rest) = name.strip_prefix("pairs/").and_then(|name| name.strip_suffix("/options.json")) {
            let index = rest.parse::<usize>().map_err(|_| Error::Serialization(format!(
                "invalid System pair list path '{}'",
                name
            )))?;
            pairs.push((index, name.clone(), format!("pairs/{}/data.mts", index)));
        }
    }
    pairs.sort_by_key(|(index, _, _)| *index);

    for (_, options_path, data_path) in pairs {
        let options = read_zip_file(&mut archive, &options_path)?;
        let options = std::str::from_utf8(&options).map_err(|_| {
            Error::Serialization(format!("pair list options '{}' are not valid UTF-8", options_path))
        })?;
        let options = json::parse(options).map_err(|e| {
            Error::Serialization(format!("invalid pair list options JSON '{}': {}", options_path, e))
        })?;
        let options = PairListOptions::from_torch_json(&options)?;

        let data = read_zip_file(&mut archive, &data_path)?;
        let block = TensorBlock::load_buffer(data.as_slice()).map_err(metatensor_serialization_error)?;
        system.add_pairs(options, block)?;
    }

    for name in names {
        if let Some(data_name) = name.strip_prefix("data/").and_then(|name| name.strip_suffix(".mts")) {
            if data_name.is_empty() {
                return Err(Error::Serialization("invalid empty custom data name in System".into()));
            }

            if system.custom_data.contains_key(data_name) {
                return Err(Error::Serialization(format!(
                    "duplicate custom data '{}' in System",
                    data_name
                )));
            }

            let data = read_zip_file(&mut archive, &name)?;
            let tensor = TensorMap::load_buffer(data.as_slice()).map_err(metatensor_serialization_error)?;
            system.custom_data.insert(data_name.to_string(), tensor);
        }
    }

    return Ok(system);
}

fn read_zip_file<R: Read + Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
) -> Result<Vec<u8>, Error> {
    let mut file = archive.by_name(name).map_err(|e| {
        Error::Serialization(format!("failed to read '{}' from System ZIP archive: {}", name, e))
    })?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    return Ok(buffer);
}

fn read_legacy_npy_tensor(data: &[u8], name: &str) -> Result<DLPackTensor, Error> {
    let (header, data) = parse_legacy_npy(data, name)?;
    if header.fortran_order {
        return Err(Error::Serialization(format!(
            "legacy System tensor '{}' uses Fortran order, which is not supported",
            name
        )));
    }

    let count = product(&header.shape)?;
    return match header.descr.as_str() {
        "<f8" => tensor_from_f64(read_legacy_npy_numeric_data(data, count, name, f64::from_le_bytes)?, &header.shape),
        "<f4" => tensor_from_f32(read_legacy_npy_numeric_data(data, count, name, f32::from_le_bytes)?, &header.shape),
        "<i4" => tensor_from_i32(read_legacy_npy_numeric_data(data, count, name, i32::from_le_bytes)?, &header.shape),
        "<i8" if name == "types" => {
            let data = read_legacy_npy_numeric_data(data, count, name, i64::from_le_bytes)?;
            let data = data.into_iter().map(|value| {
                i32::try_from(value).map_err(|_| Error::Serialization(format!(
                    "legacy System tensor '{}' contains an integer outside the i32 range",
                    name
                )))
            }).collect::<Result<Vec<_>, _>>()?;
            tensor_from_i32(data, &header.shape)
        }
        "|b1" => {
            if data.len() != count {
                return Err(Error::Serialization(format!(
                    "legacy System tensor '{}' has {} bytes of data, expected {}",
                    name,
                    data.len(),
                    count
                )));
            }

            tensor_from_bool(data.iter().map(|&value| value != 0).collect(), &header.shape)
        }
        _ => Err(Error::Serialization(format!(
            "unsupported dtype '{}' for legacy System tensor '{}'",
            header.descr, name
        ))),
    };
}

fn parse_legacy_npy<'a>(
    data: &'a [u8],
    name: &str,
) -> Result<(LegacyNpyHeader, &'a [u8]), Error> {
    if data.len() < 10 || &data[..6] != b"\x93NUMPY" {
        return Err(Error::Serialization(format!(
            "legacy System tensor '{}' is not a valid NPY file",
            name
        )));
    }

    let major = data[6];
    let (header_start, header_len): (usize, usize) = match major {
        1 => (10, u16::from_le_bytes([data[8], data[9]]) as usize),
        2 | 3 => {
            if data.len() < 12 {
                return Err(Error::Serialization(format!(
                    "legacy System tensor '{}' has a truncated NPY header",
                    name
                )));
            }
            (12, u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize)
        }
        _ => {
            return Err(Error::Serialization(format!(
                "unsupported NPY version {} for legacy System tensor '{}'",
                major, name
            )));
        }
    };

    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| Error::Serialization(format!("NPY header length overflows for '{}'", name)))?;
    if header_end > data.len() {
        return Err(Error::Serialization(format!(
            "legacy System tensor '{}' has a truncated NPY header",
            name
        )));
    }

    let header = std::str::from_utf8(&data[header_start..header_end]).map_err(|_| {
        Error::Serialization(format!("NPY header for legacy System tensor '{}' is not UTF-8", name))
    })?;
    let header = parse_legacy_npy_header(header, name)?;
    return Ok((header, &data[header_end..]));
}

fn parse_legacy_npy_header(header: &str, name: &str) -> Result<LegacyNpyHeader, Error> {
    let descr = parse_legacy_npy_header_string(header, "descr", name)?;
    let fortran_order = parse_legacy_npy_header_bool(header, "fortran_order", name)?;
    let shape = parse_legacy_npy_header_shape(header, name)?;
    return Ok(LegacyNpyHeader { descr, fortran_order, shape });
}

fn parse_legacy_npy_header_string(
    header: &str,
    key: &str,
    name: &str,
) -> Result<String, Error> {
    let key = format!("'{}'", key);
    let position = header.find(&key).ok_or_else(|| {
        Error::Serialization(format!("missing '{}' in NPY header for '{}'", key, name))
    })?;
    let after_key = &header[position + key.len()..];
    let colon = after_key.find(':').ok_or_else(|| {
        Error::Serialization(format!("invalid NPY header for '{}'", name))
    })?;
    let value = after_key[colon + 1..].trim_start();
    let quote = value.chars().next().ok_or_else(|| {
        Error::Serialization(format!("invalid NPY header for '{}'", name))
    })?;
    if quote != '\'' && quote != '"' {
        return Err(Error::Serialization(format!(
            "expected string value for '{}' in NPY header for '{}'",
            key, name
        )));
    }

    let value = &value[quote.len_utf8()..];
    let end = value.find(quote).ok_or_else(|| {
        Error::Serialization(format!("unterminated string in NPY header for '{}'", name))
    })?;
    return Ok(value[..end].to_string());
}

fn parse_legacy_npy_header_bool(
    header: &str,
    key: &str,
    name: &str,
) -> Result<bool, Error> {
    let key = format!("'{}'", key);
    let position = header.find(&key).ok_or_else(|| {
        Error::Serialization(format!("missing '{}' in NPY header for '{}'", key, name))
    })?;
    let after_key = &header[position + key.len()..];
    let colon = after_key.find(':').ok_or_else(|| {
        Error::Serialization(format!("invalid NPY header for '{}'", name))
    })?;
    let value = after_key[colon + 1..].trim_start();
    if value.starts_with("True") {
        return Ok(true);
    } else if value.starts_with("False") {
        return Ok(false);
    } else {
        return Err(Error::Serialization(format!(
            "expected boolean value for '{}' in NPY header for '{}'",
            key, name
        )));
    }
}

fn parse_legacy_npy_header_shape(header: &str, name: &str) -> Result<Vec<i64>, Error> {
    let key = "'shape'";
    let position = header.find(key).ok_or_else(|| {
        Error::Serialization(format!("missing '{}' in NPY header for '{}'", key, name))
    })?;
    let after_key = &header[position + key.len()..];
    let colon = after_key.find(':').ok_or_else(|| {
        Error::Serialization(format!("invalid NPY header for '{}'", name))
    })?;
    let value = after_key[colon + 1..].trim_start();
    let start = value.find('(').ok_or_else(|| {
        Error::Serialization(format!("missing shape tuple in NPY header for '{}'", name))
    })?;
    let end = value[start + 1..].find(')').ok_or_else(|| {
        Error::Serialization(format!("unterminated shape tuple in NPY header for '{}'", name))
    })? + start + 1;

    let mut shape = Vec::new();
    for dim in value[start + 1..end].split(',') {
        let dim = dim.trim();
        if dim.is_empty() {
            continue;
        }
        let dim = dim.parse::<i64>().map_err(|_| Error::Serialization(format!(
            "invalid shape dimension '{}' in NPY header for '{}'",
            dim, name
        )))?;
        shape.push(dim);
    }

    return Ok(shape);
}

fn read_legacy_npy_numeric_data<T, const N: usize>(
    data: &[u8],
    count: usize,
    name: &str,
    from_le_bytes: fn([u8; N]) -> T,
) -> Result<Vec<T>, Error> {
    let expected = count.checked_mul(N).ok_or_else(|| {
        Error::Serialization(format!("legacy System tensor '{}' byte count overflows", name))
    })?;
    if data.len() != expected {
        return Err(Error::Serialization(format!(
            "legacy System tensor '{}' has {} bytes of data, expected {}",
            name,
            data.len(),
            expected
        )));
    }

    return Ok(data.chunks_exact(N).map(|chunk| {
        from_le_bytes(chunk.try_into().expect("chunk has the right length"))
    }).collect());
}

fn shape_to_usize(shape: &[i64], name: &str) -> Result<Vec<usize>, Error> {
    return shape.iter().map(|&dim| {
        usize::try_from(dim).map_err(|_| Error::Serialization(format!(
            "legacy System tensor '{}' shape cannot contain negative dimensions",
            name
        )))
    }).collect();
}

fn tensor_from_f64(data: Vec<f64>, shape: &[i64]) -> Result<DLPackTensor, Error> {
    let shape = shape_to_usize(shape, "f64")?;
    let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
        .map_err(|e| Error::Serialization(e.to_string()))?;
    return Ok(array.try_into()?);
}

fn tensor_from_f32(data: Vec<f32>, shape: &[i64]) -> Result<DLPackTensor, Error> {
    let shape = shape_to_usize(shape, "f32")?;
    let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
        .map_err(|e| Error::Serialization(e.to_string()))?;
    return Ok(array.try_into()?);
}

fn tensor_from_i32(data: Vec<i32>, shape: &[i64]) -> Result<DLPackTensor, Error> {
    let shape = shape_to_usize(shape, "i32")?;
    let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
        .map_err(|e| Error::Serialization(e.to_string()))?;
    return Ok(array.try_into()?);
}

fn tensor_from_bool(data: Vec<bool>, shape: &[i64]) -> Result<DLPackTensor, Error> {
    let shape = shape_to_usize(shape, "bool")?;
    let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
        .map_err(|e| Error::Serialization(e.to_string()))?;
    return Ok(array.try_into()?);
}

fn product(shape: &[i64]) -> Result<usize, Error> {
    let mut result = 1usize;
    for &dim in shape {
        if dim < 0 {
            return Err(Error::Serialization("tensor shapes cannot contain negative dimensions".into()));
        }
        result = result.checked_mul(dim as usize)
            .ok_or_else(|| Error::Serialization("tensor shape is too large".into()))?;
    }
    return Ok(result);
}

fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

fn dtype_byte_count(dtype: DLDataType) -> Result<usize, Error> {
    if dtype.bits % 8 != 0 || dtype.lanes != 1 {
        return Err(Error::Serialization(format!(
            "unsupported dtype with {} bits and {} lanes in NPY serialization",
            dtype.bits, dtype.lanes
        )));
    }

    return Ok((dtype.bits / 8).into());
}

fn tensor_byte_count(tensor: &DLPackTensorRef<'_>) -> Result<usize, Error> {
    let dtype_size = dtype_byte_count(tensor.dtype())?;
    let element_count = product(tensor.shape())?;
    return element_count
        .checked_mul(dtype_size)
        .ok_or_else(|| Error::Serialization("tensor byte count overflows".into()));
}

fn ensure_cpu_contiguous(tensor: &DLPackTensorRef<'_>, name: &str) -> Result<(), Error> {
    let device = tensor.device();
    if device.device_type != DLDeviceType::kDLCPU {
        return Err(Error::Serialization(format!(
            "can only serialize CPU tensors, got non-CPU tensor for '{}'",
            name
        )));
    }

    if let Some(strides) = tensor.strides() {
        let expected = contiguous_strides(tensor.shape());
        if strides != expected {
            return Err(Error::Serialization(format!(
                "can only serialize contiguous tensors, tensor '{}' is not contiguous",
                name
            )));
        }
    }

    return Ok(());
}

fn npy_descr(tensor: &DLPackTensorRef<'_>, name: &str) -> Result<&'static str, Error> {
    return match (tensor.dtype().code, tensor.dtype().bits, tensor.dtype().lanes) {
        (DLDataTypeCode::kDLFloat, 64, 1) => Ok("<f8"),
        (DLDataTypeCode::kDLFloat, 32, 1) => Ok("<f4"),
        (DLDataTypeCode::kDLInt, 32, 1) => Ok("<i4"),
        (DLDataTypeCode::kDLBool, 8, 1) => Ok("|b1"),
        _ => Err(Error::Serialization(format!(
            "unsupported dtype for System tensor '{}' in NPY serialization",
            name
        ))),
    };
}

fn npy_header(tensor: &DLPackTensorRef<'_>, name: &str) -> Result<Vec<u8>, Error> {
    let descr = npy_descr(tensor, name)?;
    let mut dict = format!("{{ 'descr': '{}', 'fortran_order': False, 'shape': (", descr);
    for &dim in tensor.shape() {
        if dim < 0 {
            return Err(Error::Serialization(format!(
                "tensor '{}' shape cannot contain negative dimensions",
                name
            )));
        }
        dict.push_str(&format!("{}, ", dim));
    }
    dict.push_str(") }");

    let magic = b"\x93NUMPY";
    let prefix_len = magic.len() + 2 + 2;
    let unpadded_total = prefix_len + dict.len() + 1;
    let padding = (64 - unpadded_total % 64) % 64;
    let header_len = dict.len() + padding + 1;
    let header_len = u16::try_from(header_len)
        .map_err(|_| Error::Serialization("NPY header is too large for version 1.0".into()))?;

    let mut header = Vec::with_capacity(prefix_len + usize::from(header_len));
    header.extend_from_slice(magic);
    header.extend_from_slice(&[1, 0]);
    header.extend_from_slice(&header_len.to_le_bytes());
    header.extend_from_slice(dict.as_bytes());
    header.extend(std::iter::repeat(b' ').take(padding));
    header.push(b'\n');
    return Ok(header);
}

fn write_npy_tensor(
    writer: &mut impl Write,
    name: &str,
    tensor: DLPackTensorRef<'_>,
) -> Result<(), Error> {
    ensure_cpu_contiguous(&tensor, name)?;
    let byte_count = tensor_byte_count(&tensor)?;
    writer.write_all(&npy_header(&tensor, name)?)?;

    if byte_count != 0 {
        let data = tensor.as_dltensor().data;
        if data.is_null() {
            return Err(Error::Serialization(format!(
                "tensor '{}' has a null data pointer",
                name
            )));
        }

        let ptr = unsafe { data.cast::<u8>().add(tensor.byte_offset()) };
        let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_count) };
        writer.write_all(bytes)?;
    }

    return Ok(());
}

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
            &[[0i32, 1, 0, 0, 0]],
        );
        let components = vec![Labels::new(["xyz"], &[[0i32], [1], [2]])];
        let properties = Labels::new(["distance"], &[[0i32]]);

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
        let keys = Labels::new(["key"], &[[0i32]]);
        let samples = Labels::new(["sample"], &[[0i32]]);
        let properties = Labels::new(["property"], &[[0i32]]);

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
        system.add_custom_data("test::my_data".into(), data, false).unwrap();
        assert_eq!(system.known_custom_data(), vec!["test::my_data"]);
        assert_eq!(system.get_custom_data("test::my_data").unwrap().keys().names(), ["key"]);

        assert_error(
            system.add_custom_data("test::my_data".into(), valid_custom_data("f32"), false),
            "invalid parameter: custom data 'test::my_data' is already present in this system",
        );

        let replacement = valid_custom_data("f32");
        system.add_custom_data("test::my_data".into(), replacement, true).unwrap();
        assert_eq!(system.known_custom_data(), vec!["test::my_data"]);

        let mut system = System::new(
            "Angstrom".into(),
            type_tensor(&[1, 6, 8]),
            positions_tensor(3, "f32"),
            cell_tensor(10.0, "f32"),
            pbc_tensor(&[true, true, true]),
        ).unwrap();
        system.add_custom_data("test::a".into(), valid_custom_data("f32"), false).unwrap();
        system.add_custom_data("test::b".into(), valid_custom_data("f32"), false).unwrap();
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
            system.add_custom_data("my_data".into(), valid_custom_data("f32"), false),
            "invalid parameter: 'my_data' is not a standard quantity name; custom quantity names must use '<namespace>::<name>'",
        );

        let keys = Labels::empty(vec!["key"]);
        let empty = TensorMap::new(keys, vec![]).unwrap();
        assert_error(
            system.add_custom_data("test::empty".into(), empty, false),
            "invalid parameter: custom data 'test::empty' has no blocks",
        );

        let dtype_mismatch = valid_custom_data("f64");
        assert_error(
            system.add_custom_data("test::dtype".into(), dtype_mismatch, false),
            "invalid parameter: dtype of custom data 'test::dtype' does not match this system dtype",
        );
    }

    fn tensor_values<T>(tensor: DLPackTensorRef<'_>) -> Vec<T>
    where
        T: Copy,
    {
        let count = product(tensor.shape()).unwrap();
        let data = tensor.as_dltensor().data;
        assert!(!data.is_null());
        let ptr = unsafe { data.cast::<u8>().add(tensor.byte_offset()).cast::<T>() };
        return unsafe { std::slice::from_raw_parts(ptr, count).to_vec() };
    }

    #[test]
    fn load_torch_system_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/legacy-system.mta");
        let system = System::load(path).unwrap();

        assert_eq!(system.length_unit(), "");
        assert_eq!(system.size(), 3);
        assert_eq!(system.types().shape(), [3]);
        assert_eq!(system.positions().shape(), [3, 3]);
        assert_eq!(system.cell().shape(), [3, 3]);
        assert_eq!(system.pbc().shape(), [3]);

        assert_eq!(tensor_values::<i32>(system.types()), vec![1, 6, 8]);
        assert_eq!(
            tensor_values::<f64>(system.positions()),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            tensor_values::<f64>(system.cell()),
            vec![3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0]
        );
        assert_eq!(tensor_values::<bool>(system.pbc()), vec![true, true, true]);

        let options = PairListOptions { cutoff: 3.5, full_list: true, strict: true, requestors: vec![] };
        let pairs = system.get_pairs(&options).unwrap();
        assert_eq!(pairs.samples().names(), ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"]);
        assert_eq!(pairs.samples().count(), 2);
        assert_eq!(pairs.values().shape().unwrap(), [2, 3, 1]);

        assert_eq!(system.known_custom_data(), vec!["custom::data-name"]);
        let custom = system.get_custom_data("custom::data-name").unwrap();
        assert_eq!(custom.block_by_id(0).values().shape().unwrap(), [2, 2]);
    }

    #[test]
    fn save_load_system() {
        let mut system = System::new(
            "Angstrom".into(),
            type_tensor(&[6, 1, 1]),
            positions_tensor(3, "f64"),
            cell_tensor(3.0, "f64"),
            pbc_tensor(&[true, true, true]),
        ).unwrap();

        let options = PairListOptions { cutoff: 3.5, full_list: false, strict: true, requestors: vec![] };
        system.add_pairs(options.clone(), valid_pair_block("f64")).unwrap();
        system.add_custom_data("custom::data".into(), valid_custom_data("f64"), false).unwrap();

        let path = std::env::temp_dir().join(format!(
            "metatomic-core-system-serialization-{}-{}.mta",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        system.save(&path).unwrap();

        let archive_file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(archive_file).unwrap();
        assert!(archive.by_name("types.npy").is_ok());
        assert!(archive.by_name("positions.npy").is_ok());
        assert!(archive.by_name("cell.npy").is_ok());
        assert!(archive.by_name("pbc.npy").is_ok());
        assert!(archive.by_name("pairs/0/data.mts").is_ok());
        assert!(archive.by_name("data/custom::data.mts").is_ok());

        let options_json = read_zip_file(&mut archive, "pairs/0/options.json").unwrap();
        let options_json = std::str::from_utf8(&options_json).unwrap();
        let options_json = json::parse(options_json).unwrap();
        assert_eq!(options_json["class"].as_str(), Some("NeighborListOptions"));
        assert_eq!(options_json["cutoff"].as_i64(), Some(3.5_f64.to_bits() as i64));

        let loaded = System::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded.length_unit(), "");
        assert_eq!(loaded.size(), 3);
        assert_eq!(loaded.types().shape(), [3]);
        assert_eq!(loaded.positions().shape(), [3, 3]);
        assert_eq!(loaded.cell().shape(), [3, 3]);
        assert_eq!(loaded.pbc().shape(), [3]);

        assert_eq!(tensor_values::<i32>(loaded.types()), vec![6, 1, 1]);
        assert_eq!(
            tensor_values::<f64>(loaded.positions()),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0]
        );
        assert_eq!(
            tensor_values::<f64>(loaded.cell()),
            vec![3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0]
        );
        assert_eq!(tensor_values::<bool>(loaded.pbc()), vec![true, true, true]);

        assert!(loaded.get_pairs(&options).is_some());
        assert!(loaded.get_custom_data("custom::data").is_ok());
    }
}
