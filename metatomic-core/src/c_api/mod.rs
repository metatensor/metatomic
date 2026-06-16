#![allow(clippy::doc_markdown)]

#[macro_use]
mod status;
pub use self::status::{mta_status_t, catch_unwind};

mod utils;
pub use self::utils::mta_string_t;
pub use self::utils::{mta_string_create, mta_string_free, mta_string_view};

mod metadata;
pub use self::metadata::{
    mta_dtype_t,
    mta_device_t,
    mta_references_section_t,
    mta_pair_list_options_t,
    mta_model_metadata_t,
    mta_model_capabilities_t,
};

mod system;
pub use self::system::mta_system_t;

mod model;
pub use self::model::mta_model_t;

mod plugin;
pub use self::plugin::{mta_plugin_t, mta_register_plugin, mta_load_plugin, mta_load_model};
