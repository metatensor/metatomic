use std::ffi::{CString, c_char};

use once_cell::sync::Lazy;

use super::mta_status_t;


static VERSION: Lazy<CString> = Lazy::new(|| {
    CString::new(env!("METATOMIC_FULL_VERSION")).expect("version contains NULL byte")
});


/// Get the runtime version of the metatomic library as a string.
///
/// This version follows the `<major>.<minor>.<patch>[-<dev>]` format.
#[no_mangle]
pub extern "C" fn mta_version() -> *const c_char {
    return VERSION.as_ptr();
}

/// TODO
#[allow(non_camel_case_types)]
pub struct mta_opaque_string_t(CString);

/// TODO
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct mta_string_t(*mut mta_opaque_string_t);

impl std::fmt::Debug for mta_string_t {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut builder = f.debug_tuple("mta_string_t");

        if self.0.is_null() {
            builder.field(&"NULL");
        } else {
            builder.field(&self.as_str());
        }
        builder.finish()
    }
}

impl mta_string_t {
    /// TODO
    pub fn new(value: impl Into<String>) -> Self {
        let cstring = CString::new(value.into()).unwrap();
        let boxed = Box::new(mta_opaque_string_t(cstring));
        mta_string_t(Box::into_raw(boxed))
    }

    /// TODO
    pub fn null() -> Self {
        mta_string_t(std::ptr::null_mut())
    }

    /// TODO
    pub fn as_str(&self) -> &str {
        if self.0.is_null() {
            return "";
        }
        unsafe {
            return (*(self.0)).0.to_str().expect("mta_string_t is not valid UTF8")
        }
    }
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_string_create(
    raw: *const c_char,
) -> mta_string_t {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_string_free(string: mta_string_t) {
    todo!()
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_string_view(
    string: mta_string_t,
) -> *const c_char {
    todo!()
}


/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_unit_conversion_factor(
    from_unit: *const c_char,
    to_unit: *const c_char,
    conversion: *mut f64,
) -> mta_status_t {
    todo!()
}



// TODO: logging & warnings?
