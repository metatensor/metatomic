use crate::c_api::{mta_kv_pair_t, mta_plugin_t};
use crate::{Error, Model};

/// TODO
pub const MTA_ABI_VERSION: i32 = 1;

/// TODO
pub struct Plugin(mta_plugin_t);

impl Plugin {
    /// TODO
    pub fn new(c_plugin: mta_plugin_t) -> Self {
        Self(c_plugin)
    }

    /// TODO
    pub fn name(&self) -> &str {
       todo!()
    }

    /// TODO
    pub fn load_model(&self, load_from: &str, options: &[mta_kv_pair_t]) -> Result<Model, Error> {
        todo!()
    }
}

/// TODO
pub fn load_plugin(path: &str) -> Result<(), Error> {
    todo!()
}

/// TODO
pub fn load_model(plugin: Option<&str>, load_from: &str, options: &[mta_kv_pair_t]) -> Result<Model, Error> {
    todo!()
}
