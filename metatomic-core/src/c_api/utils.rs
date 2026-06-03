use std::ffi::{CString, c_char};

use once_cell::sync::Lazy;

use super::{mta_status_t, catch_unwind};
use crate::Error;

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

/// Heap-allocated backing storage for `mta_string_t`, opaque to C users.
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct mta_opaque_string_t(c_char);

/// An heap-allocated UTF-8 string passed across the C API boundary.
///
/// This is used whenever a C API function or callback needs to return a string.
///
/// A null pointer represents an absent or empty string. Use `mta_string_create`
/// to allocate, `mta_string_free` to release, and `mta_string_view` to get a
/// pointer to the inner C string.
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
    /// Create a new `mta_string_t` from a Rust string.
    pub fn new(value: impl Into<String>) -> Self {
        let cstring = CString::new(value.into()).expect("string contains NULL byte");
        let ptr = CString::into_raw(cstring);
        return mta_string_t(ptr.cast());
    }

    /// Create a null `mta_string_t`, representing an absent string.
    pub fn null() -> Self {
        mta_string_t(std::ptr::null_mut())
    }

    /// View the string as a `&str`. Returns `""` for a null string.
    pub fn as_str(&self) -> &str {
        if self.0.is_null() {
            return "";
        }
        unsafe {
            let cstr = std::ffi::CStr::from_ptr(self.0.cast());
            return cstr.to_str().expect("invalid UTF-8 in mta_string_t");
        }
    }
}

/// Allocate a new `mta_string_t` by copying the null-terminated C string
/// `string`.
///
/// The returned string must be freed with `mta_string_free`.
///
/// @param string A pointer to a null-terminated C string. Must not be null.
/// @return A new `mta_string_t` containing a copy of `string`, or null if an
///     error occurred. You can check the error with `mta_last_error`.
#[no_mangle]
pub unsafe extern "C" fn mta_string_create(
    string: *const c_char,
) -> mta_string_t {
    let mut result = mta_string_t::null();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    catch_unwind(move || {
        check_pointers_non_null!(string);

        let cstr = std::ffi::CStr::from_ptr(string);
        let string = CString::from(cstr);

        let ptr = CString::into_raw(string);

        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mta_string_t(ptr.cast());
        Ok(())
    });

    return result;
}

/// Free a `mta_string_t` previously created by `mta_string_create`.
///
/// @param string A `mta_string_t` to free. Can be null, in which case this function is a no-op.
#[no_mangle]
pub unsafe extern "C" fn mta_string_free(string: mta_string_t) {
    catch_unwind(|| {
        if string.0.is_null() {
            return Ok(());
        }

        let ptr = string.0.cast::<c_char>();
        let cstring = CString::from_raw(ptr);
        std::mem::drop(cstring);

        Ok(())
    });
}

/// Return a pointer to the null-terminated string data inside `string`.
///
/// The pointer is valid only for the lifetime of `string`.
///
/// @param string A `mta_string_t` containing the string to view. Must not be null.
/// @return A pointer to the null-terminated C string inside `string`
#[no_mangle]
pub unsafe extern "C" fn mta_string_view(
    string: mta_string_t,
) -> *const c_char {
    let mut result = std::ptr::null();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    catch_unwind(move || {
        let string = string.0;
        check_pointers_non_null!(string);

        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = string.cast();

        Ok(())
    });

    return result;
}

/// Get the multiplicative conversion factor to use to convert from `from_unit`
/// to `to_unit`. Both units are parsed as expressions (e.g. `kJ / mol / A^2`,
/// `(eV * u)^(1/2)`) and their dimensions must match.
///
/// @verbatim embed:rst:leading-asterisk
///
/// .. seealso::
///
///     The general documentation for :ref:`units`, with the expression
///     syntax and list of supported base units.
///
/// @endverbatim
///
/// @param from_unit A null-terminated C string containing the unit to convert from.
/// @param to_unit A null-terminated C string containing the unit to convert to.
/// @param conversion A pointer to a `double` where the conversion factor will be stored.
/// @return The status code of the operation. If this code is not `MTA_SUCCESS`,
///     you can get more details about the error with `mta_last_error`.
#[no_mangle]
pub unsafe extern "C" fn mta_unit_conversion_factor(
    from_unit: *const c_char,
    to_unit: *const c_char,
    conversion: *mut f64,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(from_unit, to_unit, conversion);

        let from_cstr = std::ffi::CStr::from_ptr(from_unit);
        let to_cstr = std::ffi::CStr::from_ptr(to_unit);

        let from_str = from_cstr.to_str().map_err(|_| {
            Error::InvalidParameter("from_unit is not valid UTF-8".into())
        })?;
        let to_str = to_cstr.to_str().map_err(|_| {
            Error::InvalidParameter("to_unit is not valid UTF-8".into())
        })?;

        *conversion = crate::unit_conversion_factor(from_str, to_str)?;

        Ok(())
    })
}


// TODO: logging & warnings?
