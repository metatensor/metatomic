use std::ffi::CStr;
use std::ptr::NonNull;
use std::cell::RefCell;

use crate::c_api::{mts_status_t, MTS_SUCCESS, mts_last_error};

/// Error code used to indicate failure of a Rust function
const RUST_FUNCTION_FAILED_ERROR_CODE: i32 = -4242;

thread_local! {
    /// Storage for the last error coming from a Rust function
    pub static LAST_RUST_ERROR: RefCell<Error> = RefCell::new(Error {code: None, message: String::new()});
}

/// Error type used in metatensor
#[derive(Debug, Clone)]
pub struct Error {
    /// Error code from the metatensor-core C API
    pub code: Option<mts_status_t>,
    /// Error message associated with the code
    pub message: String
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(code) = self.code {
            write!(f, " (code {})", code)?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

/// Check an `mts_status_t`, returning an error if is it not `MTS_SUCCESS`
pub fn check_status(status: mts_status_t) -> Result<(), Error> {
    if status == MTS_SUCCESS {
        return Ok(())
    } else if status > 0 {
        let message = unsafe {
            CStr::from_ptr(mts_last_error())
        };
        let message = message.to_str().expect("invalid UTF8");

        return Err(Error { code: Some(status), message: message.to_owned() });
    } else if status == RUST_FUNCTION_FAILED_ERROR_CODE {
        return Err(LAST_RUST_ERROR.with(|e| e.borrow().clone()));
    } else {
        return Err(Error { code: Some(status), message: "external function call failed".into() });
    }
}

/// Check a pointer allocated by metatensor-core, returning an error if is null
pub fn check_ptr<T>(ptr: *mut T) -> Result<NonNull<T>, Error> {
    if let Some(ptr) = NonNull::new(ptr) {
        return Ok(ptr);
    } else {
        let message = unsafe {
            CStr::from_ptr(mts_last_error())
        };
        let message = message.to_str().expect("invalid UTF8");

        return Err(Error { code: None, message: message.to_owned() });
    }
}


// Box<dyn Any + Send + 'static> is the error type in std::panic::catch_unwind
impl From<Box<dyn std::any::Any + Send + 'static>> for Error {
    fn from(error: Box<dyn std::any::Any + Send + 'static>) -> Error {
        let message = if let Some(message) = error.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = error.downcast_ref::<&str>() {
            (*message).to_owned()
        } else {
            panic!("panic message is not a string, something is very wrong")
        };

        return Error {
            code: None, message,
        }
    }
}


/// An alternative to `std::panic::catch_unwind` that automatically transform
/// the error into `mts_status_t`.
pub(crate) fn catch_unwind<F>(function: F) -> mts_status_t where F: FnOnce() + std::panic::UnwindSafe {
    match std::panic::catch_unwind(function) {
        Ok(()) => MTS_SUCCESS,
        Err(e) => {
            // Store the error in LAST_RUST_ERROR, we will extract it later
            // in `check_status`
            LAST_RUST_ERROR.with(|last_error| {
                let mut last_error = last_error.borrow_mut();
                *last_error = e.into();
            });

            RUST_FUNCTION_FAILED_ERROR_CODE
        }
    }
}
