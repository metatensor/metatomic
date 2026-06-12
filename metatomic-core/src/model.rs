use std::ffi::c_void;

use metatensor::{Labels, TensorMap};

use crate::{Error, ModelCapabilities, ModelMetadata, PairListOptions, Quantity, System};
use crate::c_api::{mta_model_t, mta_status_t, mta_string_t, mta_string_free};

/// A loaded atomistic model, ready to be executed on a set of systems.
///
/// `Model` wraps a [`mta_model_t`] vtable provided by a plugin. It gives
/// access to the model's metadata and capabilities, and can be run with
/// [`execute_model`].
pub struct Model(pub(crate) mta_model_t);

impl Drop for Model {
    fn drop(&mut self) {
        if let Some(unload) = self.0.unload {
            unsafe { unload(self.0.data) };
        }
    }
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

impl Model {
    /// Create a new `Model` from the corresponding C API struct.
    ///
    /// The `Model` takes ownership of `model` and will call its `unload`
    /// callback when dropped.
    pub fn new(model: mta_model_t) -> Self {
        return Model(model);
    }

    /// Extract the underlying C API struct, transferring ownership to the caller.
    ///
    /// The caller is responsible for eventually calling the `unload` callback
    /// on the returned [`mta_model_t`] to free its resources. The `Model`'s
    /// own `Drop` implementation is skipped.
    pub fn into_raw(self) -> mta_model_t {
        let model = std::mem::ManuallyDrop::new(self);
        return unsafe { std::ptr::read(&model.0) };
    }

    /// Get the metadata describing this model (name, authors, description,
    /// references, ...).
    pub fn metadata(&self) -> Result<ModelMetadata, Error> {
        let callback = self.0.metadata.ok_or_else(|| {
            Error::Internal("model is missing a 'metadata' callback".into())
        })?;
        let json_str = call_string_callback(callback, self.0.data)?;
        let json = json::parse(&json_str).map_err(|e| {
            Error::Serialization(format!("model returned invalid JSON for metadata: {}", e))
        })?;
        return ModelMetadata::try_from(&json);
    }

    /// Get the capabilities of this model: which outputs it can compute, which
    /// atomic types it supports, its interaction range, length unit, supported
    /// devices, and data type.
    pub fn capabilities(&self) -> Result<ModelCapabilities, Error> {
        let callback = self.0.capabilities.ok_or_else(|| {
            Error::Internal("model is missing a 'capabilities' callback".into())
        })?;
        let json_str = call_string_callback(callback, self.0.data)?;
        let json = json::parse(&json_str).map_err(|e| {
            Error::Serialization(format!("model returned invalid JSON for capabilities: {}", e))
        })?;
        return ModelCapabilities::try_from(&json);
    }

    /// Get the pair lists (neighbor lists) this model needs as input.
    ///
    /// The engine must compute these and attach them to every system with
    /// `mta_system_add_pairs` before calling [`execute_model`].
    pub fn requested_pair_lists(&self) -> Result<Vec<PairListOptions>, Error> {
        let callback = self.0.requested_pair_lists.ok_or_else(|| {
            Error::Internal("model is missing a 'requested_pair_lists' callback".into())
        })?;
        let json_str = call_string_callback(callback, self.0.data)?;
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

    /// Get the additional per-system inputs this model needs.
    ///
    /// The engine must attach these to every system with
    /// `mta_system_add_custom_data` before calling [`execute_model`].
    pub fn requested_inputs(&self) -> Result<Vec<Quantity>, Error> {
        let callback = self.0.requested_inputs.ok_or_else(|| {
            Error::Internal("model is missing a 'requested_inputs' callback".into())
        })?;
        let json_str = call_string_callback(callback, self.0.data)?;
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

    /// Get the outputs this model can compute.
    pub fn supported_outputs(&self) -> Result<Vec<Quantity>, Error> {
        let callback = self.0.supported_outputs.ok_or_else(|| {
            Error::Internal("model is missing a 'supported_outputs' callback".into())
        })?;
        let json_str = call_string_callback(callback, self.0.data)?;
        let json = json::parse(&json_str).map_err(|e| {
            Error::Serialization(format!("model returned invalid JSON for supported_outputs: {}", e))
        })?;
        if !json.is_array() {
            return Err(Error::Serialization(
                "model returned invalid JSON for supported_outputs, expected an array".into()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_api::{mta_model_t, mta_status_t, mta_string_t};


    // Each function below is a stand-in for what a real plugin would implement.
    // They simply write a hard-coded JSON string into the output mta_string_t
    // and return MTA_SUCCESS.
    unsafe extern "C" fn metadata_impl(
        _data: *const c_void,
        out: *mut mta_string_t,
    ) -> mta_status_t {
        *out = mta_string_t::new(r#"{
            "type": "metatomic_model_metadata",
            "name": "test-model",
            "authors": ["Alice"],
            "description": "A test model",
            "references": {"model": [], "architecture": [], "implementation": []},
            "extra": {}
        }"#);
        return mta_status_t::MTA_SUCCESS;
    }

    unsafe extern "C" fn capabilities_impl(
        _data: *const c_void,
        out: *mut mta_string_t,
    ) -> mta_status_t {
        *out = mta_string_t::new(r#"{
            "type": "metatomic_model_capabilities",
            "outputs": [{"type": "metatomic_quantity", "name": "energy", "unit": "eV", "gradients": [], "sample_kind": "system"}],
            "atomic_types": [1, 6],
            "interaction_range": 5.0,
            "length_unit": "Angstrom",
            "supported_devices": ["cpu"],
            "dtype": "float32"
        }"#);
        return mta_status_t::MTA_SUCCESS;
    }

    unsafe extern "C" fn requested_pair_lists_impl(
        _data: *const c_void,
        out: *mut mta_string_t,
    ) -> mta_status_t {
        *out = mta_string_t::new(format!(r#"[{{
            "type": "metatomic_pair_options",
            "cutoff": "{:#x}",
            "full_list": true,
            "strict": true
        }}]"#, 3.5_f64.to_bits()));
        return mta_status_t::MTA_SUCCESS;
    }

    unsafe extern "C" fn requested_inputs_impl(
        _data: *const c_void,
        out: *mut mta_string_t,
    ) -> mta_status_t {
        *out = mta_string_t::new(r#"[{
            "type": "metatomic_quantity",
            "name": "charge",
            "unit": "e",
            "gradients": [],
            "sample_kind": "atom"
        }]"#);
        return mta_status_t::MTA_SUCCESS;
    }

    unsafe extern "C" fn supported_outputs_impl(
        _data: *const c_void,
        out: *mut mta_string_t,
    ) -> mta_status_t {
        *out = mta_string_t::new(r#"[
            {
                "type": "metatomic_quantity",
                "name": "energy",
                "unit": "eV",
                "gradients": ["positions"],
                "sample_kind": "system"
            },
            {
                "type": "metatomic_quantity",
                "name": "custom::output",
                "unit": "",
                "gradients": [],
                "sample_kind": "atom_pair"
            }]"#);
        return mta_status_t::MTA_SUCCESS;
    }


    fn test_model() -> Model {
        Model(mta_model_t {
            metadata: Some(metadata_impl),
            capabilities: Some(capabilities_impl),
            requested_pair_lists: Some(requested_pair_lists_impl),
            requested_inputs: Some(requested_inputs_impl),
            supported_outputs:Some(supported_outputs_impl),
            ..mta_model_t::null()
        })
    }

    #[test]
    fn metadata() {
        let metadata = test_model().metadata().unwrap();
        assert_eq!(metadata.name, "test-model");
        assert_eq!(metadata.authors, vec!["Alice"]);
        assert_eq!(metadata.description, "A test model");
    }


    #[test]
    fn capabilities() {
        let capabilities = test_model().capabilities().unwrap();
        assert_eq!(capabilities.outputs.len(), 1);
        assert_eq!(capabilities.outputs[0].name, "energy");
        assert_eq!(capabilities.atomic_types, vec![1, 6]);
        assert_eq!(capabilities.interaction_range.to_bits(), 5.0_f64.to_bits());
        assert_eq!(capabilities.length_unit, "Angstrom");
    }

    #[test]
    fn requested_pair_lists() {
        let options = test_model().requested_pair_lists().unwrap();
        assert_eq!(options.len(), 1);
        assert_eq!(options[0].cutoff.to_bits(), 3.5_f64.to_bits());
        assert!(options[0].full_list);
        assert!(options[0].strict);
    }

    #[test]
    fn requested_inputs() {
        let inputs = test_model().requested_inputs().unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].name, "charge");
        assert_eq!(inputs[0].unit, "e");
    }

    #[test]
    fn supported_outputs() {
        let outputs = test_model().supported_outputs().unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].name, "energy");
        assert_eq!(outputs[0].unit, "eV");

        assert_eq!(outputs[1].name, "custom::output");
        assert_eq!(outputs[1].unit, "");
    }
}
