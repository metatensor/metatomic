#![warn(clippy::all, clippy::pedantic)]

// disable some style lints
#![allow(clippy::needless_return, clippy::must_use_candidate, clippy::comparison_chain)]
#![allow(clippy::redundant_field_names, clippy::redundant_closure_for_method_calls, clippy::redundant_else)]
#![allow(clippy::unreadable_literal, clippy::option_if_let_else, clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc, clippy::missing_safety_doc)]
#![allow(clippy::similar_names, clippy::borrow_as_ptr, clippy::uninlined_format_args)]
#![allow(clippy::let_underscore_untyped, clippy::manual_let_else, clippy::empty_line_after_doc_comments)]


// To be removed lated
#![allow(unused_variables, dead_code, clippy::needless_pass_by_value)]


#[doc(hidden)]
pub mod c_api;

mod metadata;
pub use self::metadata::{ModelMetadata, PairListOptions};

mod quantities;
pub use self::quantities::Quantity;

mod system;
pub use self::system::System;

mod model;
pub use self::model::Model;

mod plugin;
pub use self::plugin::{Plugin, load_plugin, load_model};

mod units;
pub use self::units::unit_conversion_factor;

/// Error type used throughout `metatomic-core`.
#[derive(Debug)]
pub enum Error {
    /// Error while serializing data to or deserializing data from JSON
    Serialization(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Serialization(message) => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Serialization(_) => None,
        }
    }

    fn cause(&self) -> Option<&dyn std::error::Error> {
        self.source()
    }
}
