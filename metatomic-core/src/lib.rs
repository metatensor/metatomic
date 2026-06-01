#![warn(clippy::all, clippy::pedantic)]

// disable some style lints
#![allow(clippy::needless_return, clippy::must_use_candidate, clippy::comparison_chain)]
#![allow(clippy::redundant_field_names, clippy::redundant_closure_for_method_calls, clippy::redundant_else)]
#![allow(clippy::unreadable_literal, clippy::option_if_let_else, clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc, clippy::missing_safety_doc)]
#![allow(clippy::similar_names, clippy::borrow_as_ptr, clippy::uninlined_format_args)]
#![allow(clippy::doc_markdown, clippy::needless_continue)]
#![allow(clippy::let_underscore_untyped, clippy::manual_let_else, clippy::empty_line_after_doc_comments)]

// To be removed later
#![allow(unused_variables, dead_code, clippy::needless_pass_by_value)]

use std::sync::Arc;

#[doc(hidden)]
pub mod c_api;

mod metadata;
use crate::c_api::mta_status_t;

pub use self::metadata::{ModelMetadata, PairListOptions};

mod quantities;
pub use self::quantities::{Quantity, SampleKind, Gradients};

mod system;
pub use self::system::System;

mod model;
pub use self::model::Model;

mod plugin;
pub use self::plugin::Plugin;

mod units;
pub use self::units::unit_conversion_factor;

/// The possible sources of error in metatomic
#[derive(Debug, Clone)]
pub enum Error {
    /// Error while serializing data to or deserializing data from JSON
    Serialization(String),
    /// Invalid parameters passed to a function
    InvalidParameter(String),
    /// I/O error
    Io(Arc<std::io::Error>),
    /// Error related to dlpack tensors, such as invalid tensor shapes or types
    Dlpack(Arc<dlpk::ndarray::DLPackNDarrayError>),
    /// Error coming from metatensor
    Metatensor(metatensor::Error),
    /// Error coming from an external function used as a callback
    CallbackError(mta_status_t),
    /// Any other internal error, usually these are internal bugs.
    Internal(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Serialization(e) => write!(f, "serialization error: {}", e),
            Error::InvalidParameter(e) => write!(f, "invalid parameter: {}", e),
            Error::Io(e) => write!(f, "io error: {}", e),
            Error::Dlpack(e) => write!(f, "dlpack error: {}", e),
            Error::Metatensor(e) => write!(f, "metatensor error: {}", e),
            Error::CallbackError(e) => write!(f, "callback error, status code: {:?}", e),
            Error::Internal(e) => write!(f,
                "internal metatomic error (this is likely a bug, please report it): {}", e
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::InvalidParameter(_)
            | Error::Serialization(_)
            | Error::Internal(_)
            | Error::CallbackError(_) => None,
            Error::Io(e) => Some(e),
            Error::Dlpack(e) => Some(e),
            Error::Metatensor(e) => Some(e),
        }
    }

    fn cause(&self) -> Option<&dyn std::error::Error> {
        self.source()
    }
}

// Box<dyn Any + Send + 'static> is the error type in std::panic::catch_unwind
impl From<Box<dyn std::any::Any + Send + 'static>> for Error {
    fn from(error: Box<dyn std::any::Any + Send + 'static>) -> Error {
        if error.is::<String>() {
            Error::Internal(*error.downcast::<String>().expect("should be a String"))
        } else if error.is::<&str>() {
            Error::Internal((*error.downcast::<&str>().expect("should be an &str")).to_owned())
        } else if error.is::<Error>() {
            return *error.downcast::<Error>().expect("it should be an Error");
        } else {
            panic!("panic message is not a string, something is very wrong")
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Io(Arc::new(error))
    }
}

impl From<dlpk::ndarray::DLPackNDarrayError> for Error {
    fn from(error: dlpk::ndarray::DLPackNDarrayError) -> Self {
        Error::Dlpack(Arc::new(error))
    }
}

impl From<metatensor::Error> for Error {
    fn from(error: metatensor::Error) -> Self {
        Error::Metatensor(error)
    }
}
