use std::io::{Read, Write};

use json::JsonValue;
use metatensor::c_api::mts_create_array_callback_t;
use zip::{ZipArchive, ZipWriter};

use crate::{Error, PairListOptions, System};

use super::tensor::{write_tensor, read_tensor};


/// Load the serialized System from the given path.
///
/// Arrays for the data will be created with the given `create_array` callback,
/// and filled by this function with the corresponding data.
///
/// See [`load`] for more details on the serialization format.
pub fn load<R>(reader: R, create_array: mts_create_array_callback_t) -> Result<System, Error>
    where R: std::io::Read + std::io::Seek
{
    let mut archive = ZipArchive::new(reader).map_err(|e| ("<root>", e))?;

    let mut length_unit = String::new();
    if let Some(index) = archive.index_for_name("info.json") {
        let mut info_file = archive.by_index(index).map_err(|e| ("info.json", e))?;
        let mut info_content = String::new();
        info_file.read_to_string(&mut info_content)?;
        let info: JsonValue = json::parse(&info_content)?;


        if info["format"].as_str() != Some("metatomic_system") {
            return Err(Error::Serialization(format!(
                "invalid format in info.json, expected 'metatomic_system', found {:?}",
                info["format"]
            )));
        }

        if info["version"].as_u8() != Some(1) {
            return Err(Error::Serialization(format!(
                "unsupported version in info.json, expected 1, found {:?}",
                info["version"]
            )));
        }

        if !info.has_key("length_unit") || !info["length_unit"].is_string() {
            return Err(Error::Serialization(
                "missing or invalid 'length_unit' field in info.json".into()
            ));
        }
        length_unit = info["length_unit"].as_str().unwrap().to_string();
    } else {
        // this is a legacy file from metatomic-torch
    }

    let data_file = archive.by_name("types.npy").map_err(|e| ("types.npy", e))?;
    let types = read_tensor(data_file, create_array)?;

    let data_file = archive.by_name("positions.npy").map_err(|e| ("positions.npy", e))?;
    let position = read_tensor(data_file, create_array)?;

    let data_file = archive.by_name("cell.npy").map_err(|e| ("cell.npy", e))?;
    let cell = read_tensor(data_file, create_array)?;

    let data_file = archive.by_name("pbc.npy").map_err(|e| ("pbc.npy", e))?;
    let pbc = read_tensor(data_file, create_array)?;

    let mut system = System::new(length_unit, types, position, cell, pbc)?;

    let pairs_paths: Vec<String> = archive.file_names()
        .filter(|path| path.starts_with("pairs/") && path.ends_with("/options.json"))
        .map(|path| path.to_string())
        .collect();

    let mut buffer = Vec::new();
    for path in pairs_paths {
        let options: PairListOptions = {
            let mut options_file = archive.by_name(&path).map_err(|e| (&path, e))?;
            let mut options_content = String::new();
            options_file.read_to_string(&mut options_content)?;
            let options_json: &JsonValue = &json::parse(&options_content)?;

            options_json.try_into()?
        };

        let data_path = path.strip_suffix("/options.json").unwrap().to_string() + "/data.mts";
        let mut data_file = archive.by_name(&data_path).map_err(|e| (data_path, e))?;

        buffer.clear();
        data_file.read_to_end(&mut buffer)?;

        let pairs = metatensor::io::load_block_buffer_custom_array(&buffer, create_array)?;

        system.add_pairs(options, pairs)?;
    }

    let data_paths: Vec<String> = archive.file_names()
        .filter(|path| path.starts_with("data/"))
        .map(|path| path.to_string())
        .collect();

    for path in data_paths {
        let name = path.strip_prefix("data/").expect("data path should start with 'data/'")
            .strip_suffix(".mts").expect("data path should end with '.mts'").to_string();

        let mut data_file = archive.by_name(&path).map_err(|e| (&path, e))?;

        buffer.clear();
        data_file.read_to_end(&mut buffer)?;

        let data = metatensor::io::load_buffer_custom_array(&buffer, create_array)?;

        system.add_custom_data(name, data, /*override*/ true)?;
    }

    return Ok(system);
}

/// Save the given system to a file (or any other writer).
///
/// The format consists of a zip archive containing NPY files for the system's
/// data (types, positions, cell, pbc), a `info.json` file for metadata, and
/// optional sub-directories for pair lists (`pairs/<id>/options.json` and
/// `pairs/<id>/data.mts`) and custom data (`data/<name>.mts`).
///
/// The recommended file extension is `.mta`.
pub fn save<W: std::io::Write + std::io::Seek>(writer: W, system: &System) -> Result<(), Error> {
    let mut archive = ZipWriter::new(writer);

    let options = zip::write::FileOptions::<'_, ()>::default()
        .with_alignment(16)
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(true)
        .last_modified_time(zip::DateTime::from_date_and_time(2000, 1, 1, 0, 0, 0).expect("invalid datetime"));

    archive.start_file("info.json", options).map_err(|e| ("info.json", e))?;
    let info = json::object! {
        "format": "metatomic_system",
        "version": 1,
        "length_unit": system.length_unit(),
    };
    info.write(&mut archive)?;

    archive.start_file("types.npy", options).map_err(|e| ("types.npy", e))?;
    write_tensor(&mut archive, system.types())?;

    archive.start_file("positions.npy", options).map_err(|e| ("positions.npy", e))?;
    write_tensor(&mut archive, system.positions())?;

    archive.start_file("cell.npy", options).map_err(|e| ("cell.npy", e))?;
    write_tensor(&mut archive, system.cell())?;

    archive.start_file("pbc.npy", options).map_err(|e| ("pbc.npy", e))?;
    write_tensor(&mut archive, system.pbc())?;

    let mut buffer = Vec::new();
    for (i, &pairs_options) in system.known_pairs().iter().enumerate() {
        let path = format!("pairs/{}/options.json", i);
        archive.start_file(&path, options).map_err(|e| (path, e))?;
        let json: JsonValue = pairs_options.clone().into();
        json.write(&mut archive)?;


        let pairs_block = system.get_pairs(pairs_options).expect("pairs block should exist");
        buffer.clear();
        pairs_block.save_buffer(&mut buffer)?;

        let path = format!("pairs/{}/data.mts", i);
        archive.start_file(&path, options).map_err(|e| (path, e))?;
        archive.write_all(&buffer)?;
    }

    for name in system.known_custom_data() {
        let tensor = system.get_custom_data(name).expect("custom data should exist");
        buffer.clear();
        tensor.save_buffer(&mut buffer)?;
        let path = format!("data/{}.mts", name);
        archive.start_file(&path, options).map_err(|e| (path, e))?;
        archive.write_all(&buffer)?;
    }

    archive.finish().map_err(|e| ("<root>", e))?;

    return Ok(());
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_legacy() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/legacy.mta");

        let file = std::fs::File::open(&path).unwrap();
        let system = load(file, Some(metatensor::io::create_ndarray)).unwrap();

        assert_eq!(system.length_unit(), "");

        let types: ndarray::ArrayView1<i32> = system.types().try_into().unwrap();
        let positions: ndarray::ArrayView2<f64> = system.positions().try_into().unwrap();
        let cell: ndarray::ArrayView2<f64> = system.cell().try_into().unwrap();
        let pbc: ndarray::ArrayView1<bool> = system.pbc().try_into().unwrap();

        assert_eq!(types, ndarray::arr1(&[1, 6, 7, 8]));
        assert_eq!(
            positions,
            ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        );
        assert_eq!(
            cell,
            ndarray::arr2(&[[6.0, 0.0, 0.0], [0.0, 4.3, 0.0], [0.0, 0.0, 0.0]])
        );
        assert_eq!(pbc, ndarray::arr1(&[true, true, false]));

        let options = PairListOptions { cutoff: 5.5, full_list: true, strict: true, requestors: vec![] };
        let pairs = system.get_pairs(&options).unwrap();
        assert_eq!(pairs.samples().names(), ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"]);
        assert_eq!(pairs.samples().count(), 28);
        assert_eq!(pairs.values().shape().unwrap(), [28, 3, 1]);

        let options = system.known_pairs();
        assert_eq!(options.len(), 1);
        // requestors are not used when looking up pairs, but are stored in the file
        assert_eq!(options[0].requestors, ["some requestor", "another one with UTF8 Θµ"]);

        assert_eq!(system.known_custom_data(), vec!["custom::data"]);
        let custom = system.get_custom_data("custom::data").unwrap();
        assert_eq!(custom.keys().count(), 2);
    }

    #[test]
    fn save_load_system() {
        let system = crate::system::test_system();

        let path = std::env::temp_dir().join(format!("system-{}.mta", std::process::id()));
        {
            let file = std::fs::File::create(&path).unwrap();
            save(file, &system).unwrap();
        }

        {
            let file = std::fs::File::open(&path).unwrap();
            let mut archive = zip::ZipArchive::new(file).unwrap();
            assert!(archive.by_name("types.npy").is_ok());
            assert!(archive.by_name("positions.npy").is_ok());
            assert!(archive.by_name("cell.npy").is_ok());
            assert!(archive.by_name("pbc.npy").is_ok());
            assert!(archive.by_name("pairs/0/data.mts").is_ok());
            assert!(archive.by_name("data/custom::data/name.mts").is_ok());

            let options_file = archive.by_name("pairs/0/options.json").unwrap();
            let options_json = std::io::read_to_string(options_file).unwrap();
            let options_json: JsonValue = json::parse(&options_json).unwrap();

            assert_eq!(options_json["type"].as_str(), Some("metatomic_pair_options"));
            assert_eq!(options_json["cutoff"].as_str(), Some(&*format!("0x{:x}", 3.5_f64.to_bits())));
            assert_eq!(options_json["full_list"].as_bool(), Some(true));
            assert_eq!(options_json["strict"].as_bool(), Some(false));
        }

        let loaded = {
            let file = std::fs::File::open(&path).unwrap();
            let loaded = load(file, Some(metatensor::io::create_ndarray)).unwrap();
            std::fs::remove_file(&path).unwrap();
            loaded
        };

        assert_eq!(loaded.length_unit(), "Angstrom");

        let types: ndarray::ArrayView1<i32> = loaded.types().try_into().unwrap();
        let positions: ndarray::ArrayView2<f32> = loaded.positions().try_into().unwrap();
        let cell: ndarray::ArrayView2<f32> = loaded.cell().try_into().unwrap();
        let pbc: ndarray::ArrayView1<bool> = loaded.pbc().try_into().unwrap();

        assert_eq!(types, ndarray::arr1(&[1, 6, 8]));
        assert_eq!(
            positions,
            ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        );
        assert_eq!(
            cell,
            ndarray::arr2(&[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        );
        assert_eq!(pbc, ndarray::arr1(&[true, true, true]));

        let options = PairListOptions {
            cutoff: 3.5,
            full_list: true,
            strict: false,
            requestors: vec![],
        };
        assert!(loaded.get_pairs(&options).is_some());
        assert!(loaded.get_custom_data("custom::data/name").is_ok());
    }
}
