use std::ffi::c_char;

use super::{mta_model_t, mta_status_t};

/// TODO
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct mta_plugin_t {
    /// TODO
    pub name: *const c_char,

    /// TODO
    pub load_model: Option<unsafe extern "C" fn(
        load_from: *const c_char,
        options_json: *const c_char,
        model: *mut mta_model_t,
    ) -> mta_status_t>,
}

/// TODO
#[no_mangle]
pub extern "C" fn mta_register_plugin(plugin: mta_plugin_t) {
    todo!()
}

/// TODO
#[no_mangle]
pub extern "C" fn mta_load_plugin(path: *const c_char) -> mta_status_t {
    todo!()
}

/// TODO
#[no_mangle]
pub extern "C" fn mta_load_model(
    plugin_name: *const c_char,
    load_from: *const c_char,
    options_json: *const c_char,
    model: *mut mta_model_t,
) -> mta_status_t {
    todo!()
}
