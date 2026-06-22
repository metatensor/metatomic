use std::ffi::{c_char, CStr, CString};

use crate::{Error, ModelCapabilities, ModelMetadata, PairListOptions};
use crate::metadata::References;
use super::{catch_unwind, mta_status_t, mta_string_t};

// /// Data type used by a model for all inputs and outputs.
// ///
// /// The model can still internally use a different data type for its calculations,
// ///  but it will get inputs in this type and must produce outputs in this type.
// #[allow(non_camel_case_types)]
// #[repr(C)]
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum mta_dtype_t {
//     /// 32-bit floating point, following the IEEE 754 standard
//     MTA_DTYPE_FLOAT32 = 0,
//     /// 64-bit floating point, following the IEEE 754 standard
//     MTA_DTYPE_FLOAT64 = 1,
// }

// impl From<DType> for mta_dtype_t {
//     fn from(dtype: DType) -> Self {
//         match dtype {
//             DType::Float32 => mta_dtype_t::MTA_DTYPE_FLOAT32,
//             DType::Float64 => mta_dtype_t::MTA_DTYPE_FLOAT64,
//         }
//     }
// }

// impl From<mta_dtype_t> for DType {
//     fn from(dtype: mta_dtype_t) -> Self {
//         match dtype {
//             mta_dtype_t::MTA_DTYPE_FLOAT32 => DType::Float32,
//             mta_dtype_t::MTA_DTYPE_FLOAT64 => DType::Float64,
//         }
//     }
// }

// /// Device on which a model can run.
// #[allow(non_camel_case_types)]
// #[repr(C)]
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum mta_device_t {
//     /// CPU device
//     MTA_DEVICE_CPU = 0,
//     /// CUDA-capable NVIDIA GPU
//     MTA_DEVICE_CUDA = 1,
//     /// ROCm-capable AMD GPU
//     MTA_DEVICE_ROCM = 2,
//     /// Apple Metal GPU
//     MTA_DEVICE_METAL = 3,
// }

/// Section of the references stored in a `mta_model_metadata_t`.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum mta_references_section_t {
    /// References describing the model as a whole (e.g. the paper that
    /// introduced the model)
    MTA_REFERENCES_MODEL = 0,
    /// References describing the model architecture (e.g. papers that
    /// describe the mathematical form)
    MTA_REFERENCES_ARCHITECTURE = 1,
    /// References describing the model implementation (e.g. a link to the
    /// source-code repository)
    MTA_REFERENCES_IMPLEMENTATION = 2,
}

/// Opaque handle for a `PairListOptionsOptions` pair list (neighbor list) requested by a model.
#[allow(non_camel_case_types)]
pub struct mta_pair_list_options_t(PairListOptions);

/// Create a new `mta_pair_list_options_t` with an empty requestors list.
#[no_mangle]
pub unsafe extern "C" fn mta_pair_list_options_create(
    cutoff: f64,
    full_list: bool,
    strict: bool,
    options: *mut *mut mta_pair_list_options_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(options);

        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(Error::InvalidParameter(
                "cutoff must be a finite positive number".into(),
            ));
        }

        let inner = PairListOptions {
            cutoff,
            full_list,
            strict,
            requestors: Vec::new(),
        };

        *options = Box::into_raw(Box::new(mta_pair_list_options_t(inner)));
        Ok(())
    })
}


/// Deserialize a `mta_pair_list_options_t` from a JSON string.
#[no_mangle]
pub unsafe extern "C" fn mta_pair_list_options_from_json(
    json: *const c_char,
    options: *mut *mut mta_pair_list_options_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(json);

        let s = CStr::from_ptr(json).to_str().map_err(|_| {
            Error::InvalidParameter("json is not valid UTF-8".into())
        })?;

        let json_val = json::parse(s).map_err(|e| {
            Error::Serialization(format!("invalid JSON: {e}"))
        })?;

        let inner = PairListOptions::try_from(&json_val)?;

        *options = Box::into_raw(Box::new(mta_pair_list_options_t(inner)));
        Ok(())
    })
}

/// Free a `mta_pair_list_options_t` object
#[no_mangle]
pub unsafe extern "C" fn mta_pair_list_options_free(
    options: *mut mta_pair_list_options_t,
) -> mta_status_t {
    catch_unwind(|| {
        if !options.is_null() {
            let _ = Box::from_raw(options);
        }
        Ok(())
    })
}

/// Serialize a `mta_pair_list_options_t` to a JSON string.
#[no_mangle]
pub unsafe extern "C" fn mta_pair_list_options_to_json(
    options: *const mta_pair_list_options_t,
    json: *mut *mut c_char,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(options, json);

        let json_val = json::JsonValue::from((*options).0.clone());
        *json = CString::new(json_val.dump())
            .map_err(|_| Error::Serialization("failed to create JSON string".into()))?
            .into_raw();
        Ok(())
    })
}

/// Get the type discriminator of a `mta_pair_list_options_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_pair_list_options_get_type(
    options: *const mta_pair_list_options_t,
    type_: *mut *mut c_char,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(options, type_);

        *type_ = CString::new("metatomic_pair_options")
            .map_err(|_| Error::Serialization("failed to create type string".into()))?
            .into_raw();
        Ok(())
    })
}

/// Get the cutoff radius from a `mta_pair_list_options_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_pair_list_options_get_cutoff(
    options: *const mta_pair_list_options_t,
    cutoff: *mut f64,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(options, cutoff);
        *cutoff = (*options).0.cutoff;
        Ok(())
    })
}

/// Get the `full_list` flag from a `mta_pair_list_options_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_pair_list_options_get_full_list(
    options: *const mta_pair_list_options_t,
    full_list: *mut bool,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(options, full_list);
        *full_list = (*options).0.full_list;
        Ok(())
    })
}

/// Get the `strict` flag from a `mta_pair_list_options_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_pair_list_options_get_strict(
    options: *const mta_pair_list_options_t,
    strict: *mut bool,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(options, strict);
        *strict = (*options).0.strict;
        Ok(())
    })
}

/// Get the number of requestors stored in a `mta_pair_list_options_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_pair_list_options_requestors_count(
    options: *const mta_pair_list_options_t,
    count: *mut usize,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(options, count);
        *count = (*options).0.requestors.len();
        Ok(())
    })
}

/// Get a requestor string by index from a `mta_pair_list_options_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_pair_list_options_get_requestor(
    options: *const mta_pair_list_options_t,
    index: usize,
    requestor: *mut *mut c_char,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(options, requestor);

        let requestors = &(*options).0.requestors;
        if index >= requestors.len() {
            return Err(Error::InvalidParameter(format!(
                "requestor index {} is out of bounds, there are {} requestors",
                index,
                requestors.len()
            )));
        }

        *requestor = CString::new(requestors[index].clone())
            .map_err(|_| Error::Serialization("failed to create requestor string".into()))?
            .into_raw();
        Ok(())
    })
}

/// Add a requestor string to a `mta_pair_list_options_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_pair_list_options_add_requestor(
    options: *mut mta_pair_list_options_t,
    requestor: *const c_char,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(options, requestor);

        let s = CStr::from_ptr(requestor).to_str().map_err(|_| {
            Error::InvalidParameter("requestor is not valid UTF-8".into())
        })?;

        if !s.is_empty() {
            let requestors = &mut (*options).0.requestors;
            if !requestors.iter().any(|r| r == s) {
                requestors.push(s.to_string());
            }
        }

        Ok(())
    })
}

/// Opaque handle to metadata describing a model: name, authors, description, references, and
/// arbitrary extra key-value pairs.
#[allow(non_camel_case_types)]
pub struct mta_model_metadata_t(ModelMetadata);

/// Get `mta_model_metadata_t` from a JSON string.
#[no_mangle]
pub unsafe extern "C" fn mta_model_metadata_from_json(
    json: *const c_char,
    metadata: *mut *mut mta_model_metadata_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(json, metadata);

        let s = CStr::from_ptr(json).to_str().map_err(|_| {
            Error::InvalidParameter("json is not valid UTF-8".into())
        })?;

        let json_val = json::parse(s).map_err(|e| {
            Error::Serialization(format!("invalid JSON for ModelMetadata: {e}"))
        })?;

        let inner = ModelMetadata::try_from(&json_val)?;

        *metadata = Box::into_raw(Box::new(mta_model_metadata_t(inner)));
        Ok(())
    })
}

/// Free a `mta_model_metadata_t` previously created by any
/// `mta_model_metadata_*` function.
#[no_mangle]
pub unsafe extern "C" fn mta_model_metadata_free(
    metadata: *mut mta_model_metadata_t,
) -> mta_status_t {
    catch_unwind(|| {
        if !metadata.is_null() {
            let _ = Box::from_raw(metadata);
        }
        Ok(())
    })
}

/// Serialize `mta_model_metadata_t` to a JSON string.
#[no_mangle]
pub unsafe extern "C" fn mta_model_metadata_to_json(
    metadata: *const mta_model_metadata_t,
    json: *mut mta_string_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(metadata, json);

        let json_val = json::JsonValue::from((*metadata).0.clone());
        *json = mta_string_t::new(json_val.dump());
        Ok(())
    })
}

/// Get the name of a model from a `mta_model_metadata_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_model_metadata_get_name(
    metadata: *const mta_model_metadata_t,
    name: *mut mta_string_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(metadata, name);
        *name = mta_string_t::new((*metadata).0.name.clone());
        Ok(())
    })
}

/// Get the description of a model from a `mta_model_metadata_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_model_metadata_get_description(
    metadata: *const mta_model_metadata_t,
    description: *mut mta_string_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(metadata, description);
        *description = mta_string_t::new((*metadata).0.description.clone());
        Ok(())
    })
}

/// Get the number of authors of a model from a `mta_model_metadata_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_model_metadata_authors_count(
    metadata: *const mta_model_metadata_t,
    count: *mut usize,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(metadata, count);
        *count = (*metadata).0.authors.len();
        Ok(())
    })
}

/// Get an author string by index from a `mta_model_metadata_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_model_metadata_get_author(
    metadata: *const mta_model_metadata_t,
    index: usize,
    author: *mut mta_string_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(metadata, author);

        let authors = &(*metadata).0.authors;
        if index >= authors.len() {
            return Err(Error::InvalidParameter(format!(
                "author index {} is out of bounds, there are {} authors",
                index,
                authors.len()
            )));
        }

        *author = mta_string_t::new(authors[index].clone());
        Ok(())
    })
}

/// Get the number of references in a section of a `mta_model_metadata_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_model_metadata_references_count(
    metadata: *const mta_model_metadata_t,
    section: mta_references_section_t,
    count: *mut usize,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(metadata, count);
        *count = references_section(&(*metadata).0.references, section).len();
        Ok(())
    })
}

/// Get a reference string by index from a section of a `mta_model_metadata_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_model_metadata_get_reference(
    metadata: *const mta_model_metadata_t,
    section: mta_references_section_t,
    index: usize,
    reference: *mut mta_string_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(metadata, reference);

        let refs = references_section(&(*metadata).0.references, section);
        if index >= refs.len() {
            return Err(Error::InvalidParameter(format!(
                "reference index {} is out of bounds, there are {} references in this section",
                index,
                refs.len()
            )));
        }

        *reference = mta_string_t::new(refs[index].clone());
        Ok(())
    })
}

/// Get the number of entries in the `extra` key-value map of a
/// `mta_model_metadata_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_model_metadata_extra_count(
    metadata: *const mta_model_metadata_t,
    count: *mut usize,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(metadata, count);
        *count = (*metadata).0.extra.len();
        Ok(())
    })
}

/// Get an extra metadata key by position from a `mta_model_metadata_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_model_metadata_get_extra_key(
    metadata: *const mta_model_metadata_t,
    index: usize,
    key: *mut mta_string_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(metadata, key);

        let extra = &(*metadata).0.extra;
        let k = extra.keys().nth(index).ok_or_else(|| {
            Error::InvalidParameter(format!(
                "extra key index {} is out of bounds, there are {} extra entries",
                index,
                extra.len()
            ))
        })?;

        *key = mta_string_t::new(k.clone());
        Ok(())
    })
}

/// Get an extra metadata value by key from a `mta_model_metadata_t`.
#[no_mangle]
pub unsafe extern "C" fn mta_model_metadata_get_extra_value(
    metadata: *const mta_model_metadata_t,
    key: *const c_char,
    value: *mut mta_string_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(metadata, key, value);

        let key_str = CStr::from_ptr(key).to_str().map_err(|_| {
            Error::InvalidParameter("key is not valid UTF-8".into())
        })?;

        let v = (*metadata).0.extra.get(key_str).ok_or_else(|| {
            Error::InvalidParameter(format!(
                "key '{}' not found in extra metadata",
                key_str
            ))
        })?;

        *value = mta_string_t::new(v.clone());
        Ok(())
    })
}

/// Opaque handle to model capabilities of a model: which outputs it can compute, which atomic types
/// it supports, its interaction range, supported devices, and data type.
#[allow(non_camel_case_types)]
pub struct mta_model_capabilities_t(ModelCapabilities);

// /// Deserialize a `mta_model_capabilities_t` from a JSON string.
// ///
// /// @param json         Null-terminated UTF-8 JSON string. Must not be `NULL`.
// /// @param capabilities Output pointer. On success, `*capabilities` is set to a
// ///     newly-allocated `mta_model_capabilities_t`. The caller takes ownership
// ///     and must free it with `mta_model_capabilities_free`.
// /// @return The status code of the operation. If this code is not
// ///     `MTA_SUCCESS`, you can get more details with `mta_last_error`.
// #[no_mangle]
// pub unsafe extern "C" fn mta_model_capabilities_from_json(
//     json: *const c_char,
//     capabilities: *mut *mut mta_model_capabilities_t,
// ) -> mta_status_t {
//     catch_unwind(|| {
//         check_pointers_non_null!(json, capabilities);

//         let s = CStr::from_ptr(json).to_str().map_err(|_| {
//             Error::InvalidParameter("json is not valid UTF-8".into())
//         })?;

//         let json_val = json::parse(s).map_err(|e| {
//             Error::Serialization(format!("invalid JSON for ModelCapabilities: {e}"))
//         })?;

//         let inner = ModelCapabilities::try_from(&json_val)?;

//         *capabilities = Box::into_raw(Box::new(mta_model_capabilities_t(inner)));
//         Ok(())
//     })
// }

// /// Free a `mta_model_capabilities_t` previously created by any
// /// `mta_model_capabilities_*` function.
// ///
// /// @param capabilities The capabilities to free. If `NULL`, this is a no-op.
// /// @return The status code of the operation. If this code is not
// ///     `MTA_SUCCESS`, you can get more details with `mta_last_error`.
// #[no_mangle]
// pub unsafe extern "C" fn mta_model_capabilities_free(
//     capabilities: *mut mta_model_capabilities_t,
// ) -> mta_status_t {
//     catch_unwind(|| {
//         if !capabilities.is_null() {
//             let _ = Box::from_raw(capabilities);
//         }
//         Ok(())
//     })
// }

// /// Serialize a `mta_model_capabilities_t` to a JSON string.
// ///
// /// @param capabilities The capabilities to serialize. Must not be `NULL`.
// /// @param json         Output string. On success, `*json` is set to a
// ///     newly-allocated string containing the JSON representation. The caller
// ///     takes ownership and must free it with `mta_string_free`.
// /// @return The status code of the operation. If this code is not
// ///     `MTA_SUCCESS`, you can get more details with `mta_last_error`.
// #[no_mangle]
// pub unsafe extern "C" fn mta_model_capabilities_to_json(
//     capabilities: *const mta_model_capabilities_t,
//     json: *mut mta_string_t,
// ) -> mta_status_t {
//     catch_unwind(|| {
//         check_pointers_non_null!(capabilities, json);

//         let json_val = json::JsonValue::from((*capabilities).0.clone());
//         *json = mta_string_t::new(json_val.dump());
//         Ok(())
//     })
// }

// /// Get the interaction range of a model from a `mta_model_capabilities_t`.
// ///
// /// @param capabilities      The capabilities to read. Must not be `NULL`.
// /// @param interaction_range Output pointer. On success, `*interaction_range`
// ///     is set to the interaction range in the model's length unit.
// /// @return The status code of the operation. If this code is not
// ///     `MTA_SUCCESS`, you can get more details with `mta_last_error`.
// #[no_mangle]
// pub unsafe extern "C" fn mta_model_capabilities_get_interaction_range(
//     capabilities: *const mta_model_capabilities_t,
//     interaction_range: *mut f64,
// ) -> mta_status_t {
//     catch_unwind(|| {
//         check_pointers_non_null!(capabilities, interaction_range);
//         *interaction_range = (*capabilities).0.interaction_range;
//         Ok(())
//     })
// }

// /// Get the length unit of a model from a `mta_model_capabilities_t`.
// ///
// /// @param capabilities The capabilities to read. Must not be `NULL`.
// /// @param length_unit  Output string. On success, `*length_unit` is set to a
// ///     newly-allocated copy of the length unit string (e.g. `"angstrom"`).
// ///     The caller takes ownership and must free it with `mta_string_free`.
// /// @return The status code of the operation. If this code is not
// ///     `MTA_SUCCESS`, you can get more details with `mta_last_error`.
// #[no_mangle]
// pub unsafe extern "C" fn mta_model_capabilities_get_length_unit(
//     capabilities: *const mta_model_capabilities_t,
//     length_unit: *mut mta_string_t,
// ) -> mta_status_t {
//     catch_unwind(|| {
//         check_pointers_non_null!(capabilities, length_unit);
//         *length_unit = mta_string_t::new((*capabilities).0.length_unit.clone());
//         Ok(())
//     })
// }

// /// Get the data type of a model from a `mta_model_capabilities_t`.
// ///
// /// @param capabilities The capabilities to read. Must not be `NULL`.
// /// @param dtype        Output pointer. On success, `*dtype` is set to the
// ///     data type used for all inputs and outputs.
// /// @return The status code of the operation. If this code is not
// ///     `MTA_SUCCESS`, you can get more details with `mta_last_error`.
// #[no_mangle]
// pub unsafe extern "C" fn mta_model_capabilities_get_dtype(
//     capabilities: *const mta_model_capabilities_t,
//     dtype: *mut mta_dtype_t,
// ) -> mta_status_t {
//     catch_unwind(|| {
//         check_pointers_non_null!(capabilities, dtype);
//         *dtype = mta_dtype_t::from((*capabilities).0.dtype);
//         Ok(())
//     })
// }

// /// Get the number of outputs a model can compute from a
// /// `mta_model_capabilities_t`.
// ///
// /// @param capabilities The capabilities to read. Must not be `NULL`.
// /// @param count        Output pointer. On success, `*count` is set to the
// ///     number of outputs.
// /// @return The status code of the operation. If this code is not
// ///     `MTA_SUCCESS`, you can get more details with `mta_last_error`.
// #[no_mangle]
// pub unsafe extern "C" fn mta_model_capabilities_outputs_count(
//     capabilities: *const mta_model_capabilities_t,
//     count: *mut usize,
// ) -> mta_status_t {
//     catch_unwind(|| {
//         check_pointers_non_null!(capabilities, count);
//         *count = (*capabilities).0.outputs.len();
//         Ok(())
//     })
// }

// /// Get a JSON-serialized `Quantity` by index from a `mta_model_capabilities_t`.
// ///
// /// @param capabilities The capabilities to read. Must not be `NULL`.
// /// @param index        Zero-based index of the output to retrieve.
// /// @param output_json  Output string. On success, `*output_json` is set to a
// ///     newly-allocated JSON string describing the `Quantity` at `index`. The
// ///     caller takes ownership and must free it with `mta_string_free`.
// /// @return The status code of the operation. If this code is not
// ///     `MTA_SUCCESS`, you can get more details with `mta_last_error`.
// #[no_mangle]
// pub unsafe extern "C" fn mta_model_capabilities_get_output_json(
//     capabilities: *const mta_model_capabilities_t,
//     index: usize,
//     output_json: *mut mta_string_t,
// ) -> mta_status_t {
//     catch_unwind(|| {
//         check_pointers_non_null!(capabilities, output_json);

//         let outputs = &(*capabilities).0.outputs;
//         if index >= outputs.len() {
//             return Err(Error::InvalidParameter(format!(
//                 "output index {} is out of bounds, there are {} outputs",
//                 index,
//                 outputs.len()
//             )));
//         }

//         let json_val = json::JsonValue::from(outputs[index].clone());
//         *output_json = mta_string_t::new(json_val.dump());
//         Ok(())
//     })
// }

// /// Get the number of supported atomic types from a `mta_model_capabilities_t`.
// ///
// /// @param capabilities The capabilities to read. Must not be `NULL`.
// /// @param count        Output pointer. On success, `*count` is set to the
// ///     number of supported atomic types.
// /// @return The status code of the operation. If this code is not
// ///     `MTA_SUCCESS`, you can get more details with `mta_last_error`.
// #[no_mangle]
// pub unsafe extern "C" fn mta_model_capabilities_atomic_types_count(
//     capabilities: *const mta_model_capabilities_t,
//     count: *mut usize,
// ) -> mta_status_t {
//     catch_unwind(|| {
//         check_pointers_non_null!(capabilities, count);
//         *count = (*capabilities).0.atomic_types.len();
//         Ok(())
//     })
// }

// /// Get an atomic type by index from a `mta_model_capabilities_t`.
// ///
// /// @param capabilities The capabilities to read. Must not be `NULL`.
// /// @param index        Zero-based index of the atomic type to retrieve.
// /// @param atomic_type  Output pointer. On success, `*atomic_type` is set to
// ///     the atomic type integer at `index`.
// /// @return The status code of the operation. If this code is not
// ///     `MTA_SUCCESS`, you can get more details with `mta_last_error`.
// #[no_mangle]
// pub unsafe extern "C" fn mta_model_capabilities_get_atomic_type(
//     capabilities: *const mta_model_capabilities_t,
//     index: usize,
//     atomic_type: *mut i64,
// ) -> mta_status_t {
//     catch_unwind(|| {
//         check_pointers_non_null!(capabilities, atomic_type);

//         let atomic_types = &(*capabilities).0.atomic_types;
//         if index >= atomic_types.len() {
//             return Err(Error::InvalidParameter(format!(
//                 "atomic type index {} is out of bounds, there are {} atomic types",
//                 index,
//                 atomic_types.len()
//             )));
//         }

//         *atomic_type = atomic_types[index];
//         Ok(())
//     })
// }

// /// Get the number of supported devices from a `mta_model_capabilities_t`.
// ///
// /// @param capabilities The capabilities to read. Must not be `NULL`.
// /// @param count        Output pointer. On success, `*count` is set to the
// ///     number of supported devices.
// /// @return The status code of the operation. If this code is not
// ///     `MTA_SUCCESS`, you can get more details with `mta_last_error`.
// #[no_mangle]
// pub unsafe extern "C" fn mta_model_capabilities_supported_devices_count(
//     capabilities: *const mta_model_capabilities_t,
//     count: *mut usize,
// ) -> mta_status_t {
//     catch_unwind(|| {
//         check_pointers_non_null!(capabilities, count);
//         *count = (*capabilities).0.supported_devices.len();
//         Ok(())
//     })
// }

// /// Get a supported device by index from a `mta_model_capabilities_t`.
// ///
// /// The `Device` is converted to a `mta_device_t` via a JSON round-trip
// /// (`Device` -> JSON string -> `mta_device_t`).
// ///
// /// @param capabilities The capabilities to read. Must not be `NULL`.
// /// @param index        Zero-based index of the device to retrieve.
// /// @param device       Output pointer. On success, `*device` is set to the
// ///     `mta_device_t` corresponding to the device at `index`.
// /// @return The status code of the operation. If this code is not
// ///     `MTA_SUCCESS`, you can get more details with `mta_last_error`.
// #[no_mangle]
// pub unsafe extern "C" fn mta_model_capabilities_get_supported_device(
//     capabilities: *const mta_model_capabilities_t,
//     index: usize,
//     device: *mut mta_device_t,
// ) -> mta_status_t {
//     catch_unwind(|| {
//         check_pointers_non_null!(capabilities, device);

//         let devices = &(*capabilities).0.supported_devices;
//         if index >= devices.len() {
//             return Err(Error::InvalidParameter(format!(
//                 "device index {} is out of bounds, there are {} supported devices",
//                 index,
//                 devices.len()
//             )));
//         }

//         let json_val = json::JsonValue::from(devices[index]);
//         let s = json_val.as_str().ok_or_else(|| {
//             Error::Internal("Device JSON serialization did not produce a string".into())
//         })?;

//         let mta_dev = match s {
//             "cpu"   => mta_device_t::MTA_DEVICE_CPU,
//             "cuda"  => mta_device_t::MTA_DEVICE_CUDA,
//             "rocm"  => mta_device_t::MTA_DEVICE_ROCM,
//             "metal" => mta_device_t::MTA_DEVICE_METAL,
//             _ => return Err(Error::Internal(format!(
//                 "unknown device type '{}' from Device JSON serialization", s
//             ))),
//         };

//         *device = mta_dev;
//         Ok(())
//     })
// }

/// Return reference section within a `References` struct based on the `mta_references_section_t` enum.
fn references_section(refs: &References, section: mta_references_section_t) -> &[String] {
    match section {
        mta_references_section_t::MTA_REFERENCES_MODEL => &refs.model,
        mta_references_section_t::MTA_REFERENCES_ARCHITECTURE => &refs.architecture,
        mta_references_section_t::MTA_REFERENCES_IMPLEMENTATION => &refs.implementation,
    }
}
