use std::ffi::c_char;

use dlpk::sys::DLManagedTensorVersioned;
use metatensor::c_api::{mts_block_t, mts_tensormap_t};

use crate::System;
use super::{mta_status_t, mta_string_t};

/// TODO
#[allow(non_camel_case_types)]
pub struct mta_system_t(pub(crate) System);


/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_system_create(
    length_unit: *const c_char,
    types: *mut DLManagedTensorVersioned,
    positions: *mut DLManagedTensorVersioned,
    cell: *mut DLManagedTensorVersioned,
    pbc: *mut DLManagedTensorVersioned,
    system: *mut *mut mta_system_t,
) -> mta_status_t {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_system_free(system: *mut mta_system_t) -> mta_status_t {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_system_size(
    system: *const mta_system_t,
    size: *mut usize,
) -> mta_status_t {
    todo!()
}

/// TODO
#[allow(non_camel_case_types)]
#[repr(C)]
#[non_exhaustive]
pub enum mta_system_data_kind {
    MTA_SYSTEM_DATA_TYPES = 0,
    MTA_SYSTEM_DATA_POSITIONS = 1,
    MTA_SYSTEM_DATA_CELL = 2,
    MTA_SYSTEM_DATA_PBC = 3,
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_system_get_data(
    system: *const mta_system_t,
    request: mta_system_data_kind,
    data: *mut *mut DLManagedTensorVersioned,
) -> mta_status_t {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_system_get_length_unit(
    system: *const mta_system_t,
    length_unit: *mut mta_string_t,
) -> mta_status_t {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_system_add_pairs(
    system: *mut mta_system_t,
    options: *const c_char,
    pairs: *mut mts_block_t,
) -> mta_status_t {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_system_get_pairs(
    system: *const mta_system_t,
    options: *const c_char,
    pairs: *mut *const mts_block_t,
) -> mta_status_t {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_system_known_pairs(
    system: *const mta_system_t,
    pairs_options: *mut mta_string_t,
) -> mta_status_t {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_system_add_custom_data(
    system: *mut mta_system_t,
    name: *const c_char,
    data: *mut mts_tensormap_t,
) -> mta_status_t {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_system_get_custom_data(
    system: *const mta_system_t,
    name: *const c_char,
    data: *mut *const mts_tensormap_t,
) -> mta_status_t {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_system_known_custom_data(
    system: *const mta_system_t,
    names: *mut mta_string_t,
) -> mta_status_t {
    todo!()
}


// TODO: mta_system_to(device, dtype)
