use std::ffi::{c_void, c_char};
use metatensor::c_api::{mts_labels_t, mts_tensormap_t};

use super::{mta_status_t, mta_string_t, mta_system_t};

/// A model that computes physical properties of atomistic systems.
///
/// `mta_model_t` is a small virtual table: `data` holds the model's own state,
/// and the function pointers describe what the model can do. A model is usually
/// produced by a plugin's `load_model` callback (see `mta_load_model`) and then
/// executed with `mta_execute_model`.
///
/// Every callback receives `data` as its first argument. metatomic treats
/// `data` as opaque and only hands it back to the callbacks. Callbacks should
/// report any error by saving it with `mta_set_last_error` and returning a
/// non-success `mta_status_t`.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct mta_model_t {
    /// Opaque pointer to the model's internal state
    ///
    /// Its layout and meaning are private to the model implementation. It is
    /// initialized by whoever creates the model (e.g. a plugin's `load_model`)
    /// and released by `unload`.
    pub data: *mut c_void,

    /// Release the resources owned by `model_data`
    ///
    /// Called exactly once when the model is no longer needed. May be `NULL` if
    /// the model owns no resources.
    ///
    /// @param model_data the model's `data` pointer
    /// @return `MTA_SUCCESS` on success, another status code on error
    pub unload: Option<unsafe extern "C" fn(model_data: *mut c_void) -> mta_status_t>,

    /// Get the capabilities of the model as a JSON string.
    ///
    /// @verbatim embed:rst:leading-asterisk
    /// The expected JSON structure is documented in :ref:`core-json-model-capabilities`.
    /// @endverbatim
    ///
    /// @param model_data the model's `data` pointer
    /// @param capabilities_json output string, set to a JSON-serialized
    ///     `ModelCapabilities` object. The caller takes ownership and must
    ///     free it with `mta_string_free`.
    /// @return `MTA_SUCCESS` on success, another status code on error
    pub capabilities: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        capabilities_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// Get metadata describing the model (name, authors, references, ...) as a
    /// JSON string.
    ///
    /// @verbatim embed:rst:leading-asterisk
    /// The expected JSON structure is documented in :ref:`core-json-model-metadata`.
    /// @endverbatim
    ///
    /// @param model_data the model's `data` pointer
    /// @param metadata_json output string, set to a JSON-serialized
    ///     `ModelMetadata` object. The caller takes ownership and must
    ///     free it with `mta_string_free`.
    /// @return `MTA_SUCCESS` on success, another status code on error
    pub metadata: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        metadata_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// List the outputs this model is able to compute as a JSON string.
    ///
    /// @verbatim embed:rst:leading-asterisk
    /// The expected JSON structure for each output is documented in :ref:`core-json-quantity`.
    /// @endverbatim
    ///
    /// @param model_data the model's `data` pointer
    /// @param outputs_json output string, set to a JSON array of `Quantity`
    ///     objects, one per supported output. The caller takes ownership and
    ///     must free it with `mta_string_free`.
    /// @return `MTA_SUCCESS` on success, another status code on error
    pub supported_outputs: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        outputs_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// List the pair lists (neighbor lists) the model needs as input as a JSON
    /// string.
    ///
    /// @verbatim embed:rst:leading-asterisk
    ///
    /// The engine is expected to compute these and attach them to every system
    /// with :c:func:`mta_system_add_pairs` before calling
    /// :c:func:`mta_execute_model`.
    ///
    /// The expected JSON structure for each pair list is documented in :ref:`core-json-pair-options`.
    ///
    /// @endverbatim
    ///
    /// @param model_data the model's `data` pointer
    /// @param pair_options_json output string, set to a JSON array of
    ///     `PairListOptions` objects. The caller takes ownership and must
    ///     free it with `mta_string_free`.
    /// @return `MTA_SUCCESS` on success, another status code on error
    pub requested_pair_lists: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        pair_options_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// List the additional per-system inputs the model needs as a JSON string.
    ///
    /// @verbatim embed:rst:leading-asterisk
    ///
    /// These correspond to custom data the engine should attach to every system
    /// with :c:func:`mta_system_add_custom_data` before execution.
    ///
    /// The expected JSON structure for each input is documented in :ref:`core-json-quantity`.
    ///
    /// @endverbatim
    ///
    /// @param model_data the model's `data` pointer
    /// @param inputs_json output string, set to a JSON array of `Quantity`
    ///     objects, one per requested input. The caller takes ownership and
    ///     must free it with `mta_string_free`.
    /// @return `MTA_SUCCESS` on success, another status code on error
    pub requested_inputs: Option<unsafe extern "C" fn(
        model_data: *const c_void,
        inputs_json: *mut mta_string_t,
    ) -> mta_status_t>,

    /// Run the model and compute the requested outputs
    ///
    /// @verbatim embed:rst:leading-asterisk
    ///
    /// This performs the model's actual computation. This should not be called
    /// directly, but rather through :c:func:`mta_execute_model`, which handles
    /// unit conversion and can check inputs and output data for consistency.
    ///
    /// @endverbatim
    ///
    /// @param model_data the model's `data` pointer
    /// @param systems array of `systems_count` systems to run the model on
    /// @param systems_count number of entries in `systems`
    /// @param selected_atoms optional labels selecting the subset of atoms to
    ///     compute outputs for, or `NULL` to use all atoms. When set, it has the
    ///     dimensions `"system"` and `"atom"` holding 0-based indices.
    /// @param requested_outputs_json JSON string containing an array of
    ///     `Quantity`, one for each output the model should produce
    /// @param outputs array of `outputs_count` tensor maps to fill, one per
    ///     requested output and in the same order
    /// @param outputs_count number of entries in `outputs`, must equal
    ///     `requested_outputs_count`
    /// @return `MTA_SUCCESS` on success, another status code on error
    pub execute_inner: Option<unsafe extern "C" fn(
        model_data: *mut c_void,
        systems: *const *const mta_system_t,
        systems_count: usize,
        selected_atoms: *const mts_labels_t,
        requested_outputs_json: *const c_char,
        outputs: *mut *mut mts_tensormap_t,
        outputs_count: usize,
    ) -> mta_status_t>,
}

impl mta_model_t {
    pub(crate) fn null() -> Self {
        return mta_model_t {
            data: std::ptr::null_mut(),
            unload: None,
            capabilities: None,
            metadata: None,
            supported_outputs: None,
            requested_pair_lists: None,
            requested_inputs: None,
            execute_inner: None,
        };
    }
}

/// Execute a model to compute the requested outputs for a set of systems
///
/// This is the main entry point to run a model loaded through the C API. It
/// validates the arguments and delegates the computation to the model's
/// `execute_inner` callback.
///
/// @param model the model to execute
/// @param systems array of `systems_count` systems to run the model on
/// @param systems_count number of entries in `systems`
/// @param selected_atoms optional labels selecting the subset of atoms to
///     compute outputs for, or `NULL` to use all atoms
/// @param requested_outputs_json JSON string containing an array of
///     `Quantity`, one for each output the model should produce
/// @param check_consistency if `true`, run additional checks on the
///     inputs and on the data produced by the model
/// @param outputs array of `outputs_count` tensor maps to fill, one per
///     requested output and in the same order. The caller takes ownership of
///     the returned tensor maps.
/// @param outputs_count number of entries in `outputs`, must equal
///     `requested_outputs_count`
/// @return `MTA_SUCCESS` on success, another status code on error (the message
///     is available through `mta_last_error`)
#[no_mangle]
pub unsafe extern "C" fn mta_execute_model(
    model: mta_model_t,
    systems: *const *const mta_system_t,
    systems_count: usize,
    selected_atoms: *const mts_labels_t,
    requested_outputs_json: *const c_char,
    check_consistency: bool,
    outputs: *mut *mut mts_tensormap_t,
    outputs_count: usize,
) -> mta_status_t {
    todo!()
}

/// Render model metadata as a human-readable string
///
/// @param metadata a JSON-serialized `ModelMetadata` object as produced by a
///     model's `metadata` callback. Must not be null.
/// @param printed output string, set to a human-readable rendering of the
///     metadata. The caller takes ownership and must free it with
///     `mta_string_free`.
/// @return `MTA_SUCCESS` on success, another status code on error
#[no_mangle]
pub unsafe extern "C" fn mta_format_metadata(
    metadata: *const c_char,
    printed: *mut mta_string_t,
) -> mta_status_t {
    todo!()
}
