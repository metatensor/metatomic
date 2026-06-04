use std::ffi::c_void;

use metatensor::{Labels, TensorMap};

use crate::{Error, ModelCapabilities, ModelMetadata, PairListOptions, Quantity, System};
use crate::c_api::{mta_model_t, mta_status_t, mta_string_t, mta_string_free};

/// TODO
pub struct Model(pub(crate) mta_model_t);

impl Drop for Model {
    fn drop(&mut self) {
        if let Some(unload) = self.0.unload {
            unsafe { unload(self.0.data) };
        }
    }
}

impl Model {
    /// Create a new `Model` from the corresponding C API struct.
    pub fn new(model: mta_model_t) -> Self {
        return Model(model);
    }

    /// Extract the underlying C API struct.
    pub fn into_raw(self) -> mta_model_t {
        let model = std::mem::ManuallyDrop::new(self);
        return unsafe { std::ptr::read(&model.0) };
    }

    fn call_string_callback(
        callback: unsafe extern "C" fn(*const c_void, *mut mta_string_t) -> mta_status_t,
        data: *const c_void,
    ) -> Result<String, Error> {
        let mut output = mta_string_t::null();
        let status = unsafe { callback(data, &mut output) };
        if status != mta_status_t::MTA_SUCCESS {
            unsafe { mta_string_free(output) };
            return Err(Error::CallbackError(status));
        }
        let json_str = output.as_str().to_owned();
        unsafe { mta_string_free(output) };
        return Ok(json_str);
    }

    pub fn metadata(&self) -> Result<ModelMetadata, Error> {
        let callback = self.0.metadata.ok_or_else(|| {
            Error::Internal("model is missing a 'metadata' callback".into())
        })?;
        let json_str = Self::call_string_callback(callback, self.0.data)?;
        let json = json::parse(&json_str).map_err(|e| {
            Error::Serialization(format!("model returned invalid JSON for metadata: {}", e))
        })?;
        return ModelMetadata::try_from(&json);
    }

    pub fn capabilities(&self) -> Result<ModelCapabilities, Error> {
        let callback = self.0.capabilities.ok_or_else(|| {
            Error::Internal("model is missing a 'capabilities' callback".into())
        })?;
        let json_str = Self::call_string_callback(callback, self.0.data)?;
        let json = json::parse(&json_str).map_err(|e| {
            Error::Serialization(format!("model returned invalid JSON for capabilities: {}", e))
        })?;
        return ModelCapabilities::try_from(&json);
    }

    pub fn requested_pair_lists(&self) -> Result<Vec<PairListOptions>, Error> {
        let callback = self.0.requested_pair_lists.ok_or_else(|| {
            Error::Internal("model is missing a 'requested_pair_lists' callback".into())
        })?;
        let json_str = Self::call_string_callback(callback, self.0.data)?;
        let json = json::parse(&json_str).map_err(|e| {
            Error::Serialization(format!("model returned invalid JSON for requested_pair_lists: {}", e))
        })?;
        if !json.is_array() {
            return Err(Error::Serialization(
                "model returned invalid JSON for requested_pair_lists, expected an array".into()
            ));
        }
        let mut result = Vec::new();
        for item in json.members() {
            result.push(PairListOptions::try_from(item)?);
        }
        return Ok(result);
    }

    pub fn requested_inputs(&self) -> Result<Vec<Quantity>, Error> {
        let callback = self.0.requested_inputs.ok_or_else(|| {
            Error::Internal("model is missing a 'requested_inputs' callback".into())
        })?;
        let json_str = Self::call_string_callback(callback, self.0.data)?;
        let json = json::parse(&json_str).map_err(|e| {
            Error::Serialization(format!("model returned invalid JSON for requested_inputs: {}", e))
        })?;
        if !json.is_array() {
            return Err(Error::Serialization(
                "model returned invalid JSON for requested_inputs, expected an array".into()
            ));
        }
        let mut result = Vec::new();
        for item in json.members() {
            result.push(Quantity::try_from(item)?);
        }
        return Ok(result);
    }
}

/// TODO
pub fn execute_model(
    model: &Model,
    systems: &[System],
    selected_atoms: Option<Labels>,
    requested_outputs: &[Quantity],
    check_consistency: bool,
) -> Result<Vec<TensorMap>, Error> {
    todo!()
}
