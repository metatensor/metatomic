use std::ffi::{c_void, c_char};
use metatensor::c_api::{mts_labels_t, mts_tensormap_t};

use super::{mta_status_t, mta_string_t, mta_system_t};

/// TODO
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct mta_model_t {
    /// TODO
    pub data: *mut c_void,

    /// TODO
    pub unload: Option<unsafe extern "C" fn(model_data: *mut c_void) -> mta_status_t>,

    /// TODO
    pub metadata: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        metadata_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// TODO
    pub supported_outputs: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        outputs_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// TODO
    pub requested_pair_lists_count: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        pair_options_count: *mut usize,
    ) -> mta_status_t>,

    /// TODO
    pub requested_pair_list: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        index: usize,
        pair_options_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// TODO
    pub requested_inputs_count: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        inputs_count: *mut usize,
    ) -> mta_status_t>,

    /// TODO
    pub requested_input: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        index: usize,
        inputs_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// TODO
    pub execute_inner: Option<unsafe extern "C" fn(
        model_data: *mut c_void,
        systems: *const *const mta_system_t,
        systems_count: usize,
        selected_atoms: *const mts_labels_t,
        requested_outputs_json: *const *const c_char,
        requested_outputs_count: usize,
        outputs: *mut *mut mts_tensormap_t,
        outputs_count: usize,
    ) -> mta_status_t>,
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_execute_model(
    model: mta_model_t,
    systems: *const *const mta_system_t,
    systems_count: usize,
    selected_atoms: *const mts_labels_t,
    requested_outputs_json: *const *const c_char,
    requested_outputs_count: usize,
    outputs: *mut *mut mts_tensormap_t,
    outputs_count: usize,
) -> mta_status_t {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_format_metadata(
    metadata: *const c_char,
    printed: *mut mta_string_t,
) -> mta_status_t {
    todo!()
}
