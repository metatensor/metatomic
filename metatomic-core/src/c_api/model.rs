use std::ffi::{c_void, c_char};
use metatensor::c_api::{mts_labels_t, mts_tensormap_t};

use super::{mta_status_t, mta_string_t, mta_system_t};
use crate::{Error, Quantity};

/// TODO
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct mta_model_t {
    /// TODO
    pub data: *mut c_void,

    /// TODO
    pub unload: Option<unsafe extern "C" fn(model_data: *mut c_void) -> mta_status_t>,

    /// TODO
    pub metadata: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        metadata_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// TODO
    pub supported_outputs: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        outputs_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// TODO
    pub requested_pair_lists: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        pair_options_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// TODO
    pub requested_inputs: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        inputs_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// TODO
    pub execute_inner: Option<unsafe extern "C" fn(
        model_data: *mut c_void,
        systems: *const *const mta_system_t,
        systems_count: usize,
        selected_atoms: *const mts_labels_t,
        requested_outputs_json: *const *const c_char,
        requested_outputs_count: usize,
        outputs: *mut *mut mts_tensormap_t,
        outputs_count: usize,
    ) -> mta_status_t>,
}

impl mta_model_t {
    pub(crate) fn null() -> Self {
        return mta_model_t {
            data: std::ptr::null_mut(),
            unload: None,
            metadata: None,
            supported_outputs: None,
            requested_pair_lists: None,
            requested_inputs: None,
            execute_inner: None,
        };
    }
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_execute_model(
    model: mta_model_t,
    systems: *const *const mta_system_t,
    systems_count: usize,
    selected_atoms: *const mts_labels_t,
    requested_outputs_json: *const *const c_char,
    requested_outputs_count: usize,
    check_consistency: bool,
    outputs: *mut *mut mts_tensormap_t,
    outputs_count: usize,
) -> mta_status_t {
    let outputs_wrapper = std::panic::AssertUnwindSafe(outputs);

    super::catch_unwind(move || {
        if model.data.is_null() {
            return Err(Error::InvalidParameter(
                "can not execute a model with NULL data".into(),
            ));
        }

        let execute_inner = model.execute_inner.ok_or_else(|| {
            Error::InvalidParameter("model does not define an execute_inner callback".into())
        })?;

        if systems_count != 0 {
            check_pointers_non_null!(systems);
            let systems = std::slice::from_raw_parts(systems, systems_count);
            for (i, system) in systems.iter().enumerate() {
                if system.is_null() {
                    return Err(Error::InvalidParameter(format!(
                        "got invalid NULL pointer for systems[{}]",
                        i
                    )));
                }
            }
        }

        if requested_outputs_count != outputs_count {
            return Err(Error::InvalidParameter(format!(
                "expected one output storage slot for each requested output, got {} requested outputs and {} output slots",
                requested_outputs_count, outputs_count
            )));
        }

        if requested_outputs_count != 0 {
            check_pointers_non_null!(requested_outputs_json);
            let requested_outputs =
                std::slice::from_raw_parts(requested_outputs_json, requested_outputs_count);
            for (i, output) in requested_outputs.iter().enumerate() {
                if output.is_null() {
                    return Err(Error::InvalidParameter(format!(
                        "got invalid NULL pointer for requested_outputs_json[{}]",
                        i
                    )));
                }
                let output = std::ffi::CStr::from_ptr(*output).to_str().map_err(|_| {
                    Error::InvalidParameter(format!(
                        "invalid UTF-8 in requested_outputs_json[{}]",
                        i
                    ))
                })?;
                let output = json::parse(output).map_err(|e| {
                    Error::Serialization(format!(
                        "invalid JSON in requested_outputs_json[{}]: {}",
                        i, e
                    ))
                })?;
                let _ = Quantity::try_from(output)?;
            }
        }

        if outputs_count != 0 {
            check_pointers_non_null!(outputs);
            let outputs = std::slice::from_raw_parts_mut(outputs_wrapper.0, outputs_count);
            for output in outputs.iter_mut() {
                *output = std::ptr::null_mut();
            }
        }

        let status = execute_inner(
            model.data,
            systems,
            systems_count,
            selected_atoms,
            requested_outputs_json,
            requested_outputs_count,
            outputs_wrapper.0,
            outputs_count,
        );
        if status != mta_status_t::MTA_SUCCESS {
            return Err(Error::CallbackError);
        }

        if check_consistency {
            let outputs = std::slice::from_raw_parts(outputs_wrapper.0, outputs_count);
            for (i, output) in outputs.iter().enumerate() {
                if output.is_null() {
                    return Err(Error::InvalidParameter(format!(
                        "model returned a NULL pointer for outputs[{}]",
                        i
                    )));
                }
            }
        }

        return Ok(());
    })
}

/// TODO
#[no_mangle]
pub unsafe extern "C" fn mta_format_metadata(
    metadata: *const c_char,
    printed: *mut mta_string_t,
) -> mta_status_t {
    let printed_wrapper = std::panic::AssertUnwindSafe(printed);

    super::catch_unwind(move || {
        check_pointers_non_null!(metadata, printed);

        let metadata = std::ffi::CStr::from_ptr(metadata)
            .to_str()
            .map_err(|_| Error::InvalidParameter("invalid UTF-8 in model metadata".into()))?;

        unsafe {
            *printed_wrapper.0 = mta_string_t::new(metadata);
        }
        return Ok(());
    })
}
