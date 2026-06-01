use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::UnwindSafe;

use crate::Error;

#[derive(Debug)]
struct LastError {
    message: CString,
    origin: CString,
    custom_data: *mut c_void,
    custom_data_deleter: Option<unsafe extern "C" fn(*mut c_void)>,
}

// Save the last error message in thread local storage.
thread_local! {
    pub static LAST_ERROR: RefCell<LastError> = RefCell::new(LastError {
        message: CString::new("").expect("invalid C string"),
        origin: CString::new("").expect("invalid C string"),
        custom_data: std::ptr::null_mut(),
        custom_data_deleter: None,
    });
}

/// Status type returned by all functions in the C API.
///
/// The value 0 (`MTA_SUCCESS`) indicates success, while any non-zero value indicates an error.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum mta_status_t {
    /// Status code indicating success
    MTA_SUCCESS = 0,
    /// Status code indicating invalid function parameters
    MTA_INVALID_PARAMETER_ERROR = 1,
    /// Status code indicating I/O errors
    MTA_IO_ERROR = 2,
    /// Status code indicating serialization/deserialization errors
    MTA_SERIALIZATION_ERROR = 3,
    /// Status code indicating dlpack errors
    MTA_DLPACK_ERROR = 4,
    /// Status code indicating metatensor errors
    MTA_METATENSOR_ERROR = 5,
    /// Status code used by plugins when a model is not supported by the
    /// current plugin
    MTA_MODEL_NOT_SUPPORTED_ERROR = 6,
    /// Status code used when there is an internal error
    MTA_INTERNAL_ERROR = 255,
}

/// `std::panic::catch_unwind` that automatically transform
/// the error into `mta_status_t`.
pub fn catch_unwind<F>(function: F) -> mta_status_t
where
    F: FnOnce() -> Result<(), Error> + UnwindSafe,
{
    match std::panic::catch_unwind(function) {
        Ok(Ok(())) => mta_status_t::MTA_SUCCESS,
        Ok(Err(error)) => error.into(),
        Err(error) => Error::from(error).into(),
    }
}

/// Check that pointers (used as C API function parameters) are not null.
#[macro_export]
#[doc(hidden)]
macro_rules! check_pointers_non_null {
    ($pointer: ident) => {
        if $pointer.is_null() {
            return Err($crate::Error::InvalidParameter(
                format!(
                    "got invalid NULL pointer for {} at {}:{}",
                    stringify!($pointer), file!(), line!()
                )
            ));
        }
    };
    ($($pointer: ident),* $(,)?) => {
        $(check_pointers_non_null!($pointer);)*
    }
}

impl From<Error> for mta_status_t {
    fn from(error: Error) -> mta_status_t {
        if let Error::CallbackError(status) = error {
            // If the error is already a CallbackError, we can directly return the corresponding status code.
            return status;
        }

        LAST_ERROR.with(|last_error| {
            let mut last_error = last_error.borrow_mut();

            // If there is a custom data deleter,
            // use it to free the custom data before overwriting it with the new error.
            if let Some(deleter) = last_error.custom_data_deleter {
                unsafe {
                    deleter(last_error.custom_data);
                }
            }

            *last_error = LastError {
                message: CString::new(format!("{}", error))
                    .expect("error message contains a null byte"),
                origin: CString::new("metatomic-core").expect("invalid C string"),
                custom_data: std::ptr::null_mut(),
                custom_data_deleter: None,
            };
        });

        match error {
            Error::InvalidParameter(_) => mta_status_t::MTA_INVALID_PARAMETER_ERROR,
            Error::Io(_) => mta_status_t::MTA_IO_ERROR,
            Error::Serialization(_) => mta_status_t::MTA_SERIALIZATION_ERROR,
            Error::Dlpack(_) => mta_status_t::MTA_DLPACK_ERROR,
            Error::Metatensor(_) => mta_status_t::MTA_METATENSOR_ERROR,
            Error::CallbackError(_) => unreachable!("already handled above"),
            Error::Internal(_) => mta_status_t::MTA_INTERNAL_ERROR,

        }
    }
}

/// Get last error message that was created on the current thread.
#[no_mangle]
pub unsafe extern "C" fn mta_last_error(
    message: *mut *const c_char,
    origin: *mut *const c_char,
    data: *mut *mut c_void,
) -> mta_status_t {
    let status = std::panic::catch_unwind(|| {
        LAST_ERROR.with(|last_error| {
            let last_error = last_error.borrow();
            if !message.is_null() {
                *message = last_error.message.as_ptr();
            }
            if !origin.is_null() {
                *origin = last_error.origin.as_ptr();
            }
            if !data.is_null() {
                *data = last_error.custom_data;
            }
        });
    });

    match status {
        Ok(()) => mta_status_t::MTA_SUCCESS,
        Err(error) => {
            let last_error_debug =
                LAST_ERROR.with(|last_error| format!("{:?}", last_error.borrow()));
            if error.is::<String>() {
                eprintln!(
                    "panic in mta_last_error: {:?}, last_error: {:?}",
                    error.downcast_ref::<String>(),
                    last_error_debug
                );
            } else if error.is::<&str>() {
                eprintln!(
                    "panic in mta_last_error: {:?}, last_error: {:?}",
                    error.downcast_ref::<&str>(),
                    last_error_debug
                );
            } else {
                eprintln!(
                    "panic in mta_last_error: unknown panic error type. last_error: {:?}",
                    last_error_debug
                );
            }
            mta_status_t::MTA_INTERNAL_ERROR
        }
    }
}

/// Set last error message for the current thread.
#[no_mangle]
pub unsafe extern "C" fn mta_set_last_error(
    message: *const c_char,
    origin: *const c_char,
    data: *mut c_void,
    data_deleter: Option<unsafe extern "C" fn(*mut c_void)>,
) -> mta_status_t {
    catch_unwind(move || {
        let message = if message.is_null() {
            CString::new("<no message provided>").expect("invalid C string")
        } else {
            CString::from(CStr::from_ptr(message))
        };

        let origin = if origin.is_null() {
            CString::new("<no origin provided>").expect("invalid C string")
        } else {
            CString::from(CStr::from_ptr(origin))
        };

        LAST_ERROR.with(|last_error| {
            let mut last_error = last_error.borrow_mut();

            // Call custom data deleter before overwriting the custom data with the new one, to avoid memory leaks.
            if let Some(deleter) = last_error.custom_data_deleter {
                unsafe {
                    deleter(last_error.custom_data);
                }
            }

            *last_error = LastError {
                message: message,
                origin: origin,
                custom_data: data,
                custom_data_deleter: data_deleter,
            };
        });
        Ok(())
    })
}
