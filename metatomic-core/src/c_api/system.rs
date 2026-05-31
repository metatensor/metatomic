use std::ffi::c_char;

use dlpk::sys::{DLManagedTensorVersioned, DLTensor};
use dlpk::{DLPackTensor, DLPackVersion};
use metatensor::c_api::{mts_block_t, mts_tensormap_t};

use crate::system::SystemCore;
use crate::{Error, PairListOptions, System};

use super::{mta_status_t, mta_string_t};

/// An atomistic system.
#[allow(non_camel_case_types)]
pub struct mta_system_t(pub(crate) System);

struct BorrowedTensorStorage {
    shape: Vec<i64>,
    strides: Option<Vec<i64>>,
}

unsafe extern "C" fn borrowed_tensor_deleter(tensor: *mut DLManagedTensorVersioned) {
    let _ = Box::from_raw((*tensor).manager_ctx.cast::<BorrowedTensorStorage>());
    let _ = Box::from_raw(tensor);
}

fn borrowed_tensor(tensor: &DLPackTensor) -> *mut DLManagedTensorVersioned {
    let mut storage = Box::new(BorrowedTensorStorage {
        shape: tensor.shape().to_vec(),
        strides: tensor.strides().map(<[i64]>::to_vec),
    });

    let shape = if storage.shape.is_empty() {
        std::ptr::null_mut()
    } else {
        storage.shape.as_mut_ptr()
    };

    let strides = storage
        .strides
        .as_mut()
        .map_or(std::ptr::null_mut(), |strides| strides.as_mut_ptr());

    let source = tensor.as_dltensor();
    let dl_tensor = DLTensor {
        data: source.data,
        device: source.device,
        ndim: source.ndim,
        dtype: source.dtype,
        shape,
        strides,
        byte_offset: source.byte_offset,
    };

    let borrowed = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion::current(),
        manager_ctx: Box::into_raw(storage).cast(),
        deleter: Some(borrowed_tensor_deleter),
        flags: tensor.flags(),
        dl_tensor,
    });

    return Box::into_raw(borrowed);
}

fn parse_pair_options(options: *const c_char) -> Result<PairListOptions, Error> {
    check_pointers_non_null!(options);

    let options = unsafe { std::ffi::CStr::from_ptr(options) }
        .to_str()
        .map_err(|_| Error::InvalidParameter("invalid UTF-8 in pair list options".into()))?;
    let options = json::parse(options)
        .map_err(|e| Error::Serialization(format!("invalid pair list options JSON: {}", e)))?;

    return PairListOptions::try_from(options);
}

/// Create a new system.
///
/// Ownership of all DLPack tensors is transferred to the system.
#[no_mangle]
pub unsafe extern "C" fn mta_system_create(
    length_unit: *const c_char,
    types: *mut DLManagedTensorVersioned,
    positions: *mut DLManagedTensorVersioned,
    cell: *mut DLManagedTensorVersioned,
    pbc: *mut DLManagedTensorVersioned,
    system: *mut *mut mta_system_t,
) -> mta_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(system);

    super::catch_unwind(move || {
        check_pointers_non_null!(length_unit, types, positions, cell, pbc, system);

        let length_unit = std::ffi::CStr::from_ptr(length_unit)
            .to_str()
            .map_err(|_| Error::InvalidParameter("invalid UTF-8 in length unit".into()))?
            .to_string();

        let core = SystemCore::new(
            length_unit,
            DLPackTensor::from_ptr(types),
            DLPackTensor::from_ptr(positions),
            DLPackTensor::from_ptr(cell),
            DLPackTensor::from_ptr(pbc),
        );

        unsafe {
            *unwind_wrapper.0 = Box::into_raw(Box::new(mta_system_t(System::from_core(core))));
        }
        return Ok(());
    })
}

/// Free a system previously created with `mta_system_create`.
#[no_mangle]
pub unsafe extern "C" fn mta_system_free(system: *mut mta_system_t) -> mta_status_t {
    super::catch_unwind(move || {
        if !system.is_null() {
            let _ = Box::from_raw(system);
        }
        return Ok(());
    })
}

/// Get the number of atoms/particles in this system.
#[no_mangle]
pub unsafe extern "C" fn mta_system_size(
    system: *const mta_system_t,
    size: *mut usize,
) -> mta_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(size);

    super::catch_unwind(move || {
        check_pointers_non_null!(system, size);

        unsafe {
            *unwind_wrapper.0 = (*system).0.size();
        }
        return Ok(());
    })
}

/// Built-in data stored in a system.
#[allow(non_camel_case_types)]
#[repr(C)]
#[non_exhaustive]
pub enum mta_system_data_kind {
    MTA_SYSTEM_DATA_TYPES = 0,
    MTA_SYSTEM_DATA_POSITIONS = 1,
    MTA_SYSTEM_DATA_CELL = 2,
    MTA_SYSTEM_DATA_PBC = 3,
}

/// Get a copy of the DLPack tensor wrapper for built-in system data.
///
/// The tensor data is borrowed from the system and remains valid while the
/// system is alive.
#[no_mangle]
pub unsafe extern "C" fn mta_system_get_data(
    system: *const mta_system_t,
    request: mta_system_data_kind,
    data: *mut *mut DLManagedTensorVersioned,
) -> mta_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(data);

    super::catch_unwind(move || {
        check_pointers_non_null!(system, data);

        let core = (*system).0.as_core();
        let tensor = match request {
            mta_system_data_kind::MTA_SYSTEM_DATA_TYPES => core.types(),
            mta_system_data_kind::MTA_SYSTEM_DATA_POSITIONS => core.positions(),
            mta_system_data_kind::MTA_SYSTEM_DATA_CELL => core.cell(),
            mta_system_data_kind::MTA_SYSTEM_DATA_PBC => core.pbc(),
        };

        unsafe {
            *unwind_wrapper.0 = borrowed_tensor(tensor);
        }
        return Ok(());
    })
}

/// Get the length unit for positions and cell.
#[no_mangle]
pub unsafe extern "C" fn mta_system_get_length_unit(
    system: *const mta_system_t,
    length_unit: *mut mta_string_t,
) -> mta_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(length_unit);

    super::catch_unwind(move || {
        check_pointers_non_null!(system, length_unit);

        unsafe {
            *unwind_wrapper.0 = mta_string_t::new((*system).0.length_unit());
        }
        return Ok(());
    })
}

/// Add a pair list to this system.
///
/// Ownership of `pairs` is transferred to the system.
#[no_mangle]
pub unsafe extern "C" fn mta_system_add_pairs(
    system: *mut mta_system_t,
    options: *const c_char,
    pairs: *mut mts_block_t,
) -> mta_status_t {
    super::catch_unwind(move || {
        check_pointers_non_null!(system, pairs);

        let options = parse_pair_options(options)?;
        (*system).0.as_core_mut().add_pairs_raw(options, pairs)?;
        return Ok(());
    })
}

/// Get a pair list from this system.
///
/// The returned pointer is borrowed from the system.
#[no_mangle]
pub unsafe extern "C" fn mta_system_get_pairs(
    system: *const mta_system_t,
    options: *const c_char,
    pairs: *mut *const mts_block_t,
) -> mta_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(pairs);

    super::catch_unwind(move || {
        check_pointers_non_null!(system, pairs);

        let options = parse_pair_options(options)?;
        let pair_list = (*system)
            .0
            .as_core()
            .get_pairs(&options)
            .ok_or_else(|| Error::InvalidParameter("unknown pair list options".into()))?;

        unsafe {
            *unwind_wrapper.0 = pair_list;
        }
        return Ok(());
    })
}

/// Get all known pair list options as a JSON array.
#[no_mangle]
pub unsafe extern "C" fn mta_system_known_pairs(
    system: *const mta_system_t,
    pairs_options: *mut mta_string_t,
) -> mta_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(pairs_options);

    super::catch_unwind(move || {
        check_pointers_non_null!(system, pairs_options);

        unsafe {
            *unwind_wrapper.0 = mta_string_t::new((*system).0.as_core().known_pairs_json());
        }
        return Ok(());
    })
}

/// Add custom data to this system.
///
/// Ownership of `data` is transferred to the system.
#[no_mangle]
pub unsafe extern "C" fn mta_system_add_custom_data(
    system: *mut mta_system_t,
    name: *const c_char,
    data: *mut mts_tensormap_t,
) -> mta_status_t {
    super::catch_unwind(move || {
        check_pointers_non_null!(system, name, data);

        let name = std::ffi::CStr::from_ptr(name)
            .to_str()
            .map_err(|_| Error::InvalidParameter("invalid UTF-8 in custom data name".into()))?
            .to_string();

        (*system).0.as_core_mut().add_custom_data_raw(name, data)?;
        return Ok(());
    })
}

/// Get custom data from this system.
///
/// The returned pointer is borrowed from the system.
#[no_mangle]
pub unsafe extern "C" fn mta_system_get_custom_data(
    system: *const mta_system_t,
    name: *const c_char,
    data: *mut *const mts_tensormap_t,
) -> mta_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(data);

    super::catch_unwind(move || {
        check_pointers_non_null!(system, name, data);

        let name = std::ffi::CStr::from_ptr(name)
            .to_str()
            .map_err(|_| Error::InvalidParameter("invalid UTF-8 in custom data name".into()))?;
        let tensor = (*system)
            .0
            .as_core()
            .get_custom_data(name)
            .ok_or_else(|| Error::InvalidParameter(format!("unknown custom data '{}'", name)))?;

        unsafe {
            *unwind_wrapper.0 = tensor;
        }
        return Ok(());
    })
}

/// Get all known custom data names as a JSON array.
#[no_mangle]
pub unsafe extern "C" fn mta_system_known_custom_data(
    system: *const mta_system_t,
    names: *mut mta_string_t,
) -> mta_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(names);

    super::catch_unwind(move || {
        check_pointers_non_null!(system, names);

        unsafe {
            *unwind_wrapper.0 = mta_string_t::new((*system).0.as_core().known_custom_data_json());
        }
        return Ok(());
    })
}
