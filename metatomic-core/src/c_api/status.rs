use std::ffi::{c_char, c_void};

use crate::Error;


// TODO
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(PartialEq, Eq, Debug)]
pub enum mta_status_t {
    MTA_SUCCESS = 0,
    // ...
    MTA_ERROR_OTHER = 255,
}


impl From<Error> for mta_status_t {
    fn from(err: Error) -> Self {
        todo!()
    }
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_last_error(
    message: *mut *const c_char,
    origin: *mut *const c_char,
    data: *mut *mut c_void,
) -> mta_status_t {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_set_last_error(
    message: *const c_char,
    origin: *const c_char,
    data: *mut c_void,
    data_deleter: Option<unsafe extern "C" fn(*mut c_void)>,
) -> mta_status_t {
    todo!()
}
