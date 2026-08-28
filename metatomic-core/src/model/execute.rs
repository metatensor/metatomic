use std::ffi::CString;
use std::sync::{Arc, LazyLock, Mutex};

use lru::LruCache;
use metatensor::TensorMap;
use metatensor::c_api::{mts_labels_t, mts_tensormap_t};

use crate::{Error, Quantity, System};
use crate::c_api::{mta_system_t, mta_status_t};
use crate::kernels::{check_atomic_types, ReferenceValue};
use crate::quantity::check_quantity;
use crate::unit_conversion_factor;
use crate::utils::scale_tensormap;

use super::Model;
use super::inputs::{check_inputs, check_requested_outputs};

/// LRU cache of `ReferenceValue<i32>` for atomic types, keyed by the sorted
/// vector of valid types. This avoids re-uploading the same set of valid types
/// to the device on every `execute_model` call.
static ATOMIC_TYPES_CACHE: LazyLock<Mutex<LruCache<Vec<i32>, ReferenceValue<i32>>>> = LazyLock::new(
    || Mutex::new(LruCache::new(std::num::NonZero::new(8).unwrap()))
);

/// Run a model on a set of systems, computing the requested outputs.
///
/// This is the main entry point for executing a model. It validates the
/// arguments (optionally), converts units, delegates the computation to the
/// model's `execute_inner` callback, and converts the outputs to the requested
/// units.
#[allow(clippy::too_many_lines)]
pub fn execute_model(
    model: &Model,
    systems: &[Arc<System>],
    selected_atoms: *const mts_labels_t,
    requested_outputs_json: &str,
    check_consistency: bool,
    outputs: *mut *mut mts_tensormap_t,
    outputs_count: usize,
) -> Result<(), Error> {
    let capabilities = model.capabilities()?;
    let requested_inputs = model.requested_inputs()?;

    // parse requested outputs for validation and unit conversion
    let requested_outputs: Vec<Quantity> = {
        let json = json::parse(requested_outputs_json).map_err(|e| {
            Error::Serialization(format!("invalid JSON for requested_outputs: {e}"))
        })?;
        if !json.is_array() {
            return Err(Error::InvalidParameter(
                "requested_outputs_json must contain a JSON array".into()
            ));
        }
        let mut result = Vec::new();
        for item in json.members() {
            result.push(Quantity::try_from(item)?);
        }
        result
    };

    if requested_outputs.len() != outputs_count {
        return Err(Error::InvalidParameter(format!(
            "the number of requested outputs ({}) does not match the outputs buffer length ({})",
            requested_outputs.len(), outputs_count
        )));
    }

    let mut selected_atoms_labels = None;
    if check_consistency {
        selected_atoms_labels = if selected_atoms.is_null() {
            None
        } else {
            // increase the refcount on Labels, and only keep one for ourself
            // (we do not own the data passed by the caller)
            let labels = unsafe { metatensor::Labels::from_raw(selected_atoms) };
            let clone = labels.clone();
            std::mem::forget(labels);
            Some(clone)
        };

        check_requested_outputs(&capabilities, &requested_outputs)?;

        let requested_pair_lists = model.requested_pair_lists()?;
        check_inputs(
            &capabilities,
            &requested_pair_lists,
            &requested_inputs,
            systems,
            selected_atoms_labels.as_ref(),
        )?;
    }

    // always check atomic types (even when check_consistency is false)
    {
        let mut cache = ATOMIC_TYPES_CACHE.lock().expect("ATOMIC_TYPES_CACHE lock poisoned");
        let atomic_types = &capabilities.atomic_types;
        let valid_types = cache.get_or_insert_ref(atomic_types.as_slice(), || {
            ReferenceValue::new(
                ndarray::ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[atomic_types.len()]),
                    atomic_types.clone(),
                ).expect("atomic_types should be contiguous")
            )
        });
        for system in systems {
            check_atomic_types(system.types(), valid_types)?;
        }
    }

    // Convert systems from engine to model units. This returns a new
    // Arc<System> for each system — either a refcount bump (no conversion
    // needed) or a deep copy that has been scaled.
    let model_length_unit = &capabilities.length_unit;
    let converted_systems: Vec<Arc<System>> = systems.iter()
        .map(|s| s.clone().convert_units(model_length_unit, &requested_inputs))
        .collect::<Result<Vec<_>, _>>()?;

    // Marshal systems as *const mta_system_t for execute_inner.
    // We borrow from the Arcs (Arc::as_ptr) instead of consuming them, so the
    // converted_systems Vec keeps the Arcs alive for the duration of the call.
    let system_ptrs: Vec<*const mta_system_t> = converted_systems.iter()
        .map(|s| Arc::as_ptr(s).cast::<mta_system_t>())
        .collect();

    // call execute_inner
    let execute_inner = model.0.execute_inner.ok_or_else(|| {
        Error::Internal("model is missing an 'execute_inner' callback".into())
    })?;

    let requested_outputs_cstr = CString::new(requested_outputs_json).map_err(|e| {
        Error::InvalidParameter(format!("requested_outputs_json contains a null byte: {e}"))
    })?;

    let status = unsafe {
        execute_inner(
            model.0.data,
            system_ptrs.as_ptr(),
            system_ptrs.len(),
            selected_atoms,
            requested_outputs_cstr.as_ptr(),
            outputs,
            outputs_count,
        )
    };

    if status != mta_status_t::MTA_SUCCESS {
        // the model reported an error; free any outputs it may have partially
        // filled to avoid leaking them, then propagate the error
        for i in 0..outputs_count {
            unsafe {
                // Set the pointer to null so that the caller doesn't double-free
                let ptr = std::mem::replace(&mut *outputs.add(i), std::ptr::null_mut());

                if !ptr.is_null() {
                    std::mem::drop(TensorMap::from_raw(ptr));
                }
            }
        }
        return Err(Error::CallbackError(status));
    }

    // verify all outputs were filled
    for i in 0..outputs_count {
        if unsafe { *outputs.add(i) }.is_null() {
            // free any outputs that were filled so far
            for j in 0..outputs_count {
                unsafe {
                    let ptr = std::mem::replace(&mut *outputs.add(j), std::ptr::null_mut());

                    if !ptr.is_null() {
                        std::mem::drop(TensorMap::from_raw(ptr));
                    }
                }
            }
            return Err(Error::InvalidParameter(
                "model's execute_inner did not fill all requested outputs".into()
            ));
        }
    }

    // determine whether we need to take ownership of the outputs at all:
    // only if check_consistency is true or any unit conversion factor != 1.0
    let mut needs_ownership = check_consistency;
    for requested in &requested_outputs {
        #[allow(clippy::float_cmp)]
        if let Some(declared) = capabilities.find_output(requested) {
            let factor = unit_conversion_factor(&declared.unit, &requested.unit)?;
            needs_ownership |= factor != 1.0;
        }
    }

    if needs_ownership {
        // take ownership of returned tensor maps for checking and unit conversion
        let mut result: Vec<TensorMap> = Vec::with_capacity(outputs_count);
        for i in 0..outputs_count {
            unsafe {
                let ptr = std::mem::replace(&mut *outputs.add(i), std::ptr::null_mut());
                result.push(TensorMap::from_raw(ptr));
            }
        }

        // check outputs
        if check_consistency {
            for (output, requested) in result.iter().zip(&requested_outputs) {
                check_quantity(
                    requested,
                    output,
                    systems,
                    selected_atoms_labels.as_ref(),
                )?;
            }
        }

        // convert output units: declared.unit → requested.unit
        for (i, tensor) in result.into_iter().enumerate() {
            let requested = &requested_outputs[i];

            // find the declared quantity in capabilities
            let tensor = if let Some(declared) = capabilities.find_output(requested) {
                let factor = unit_conversion_factor(&declared.unit, &requested.unit)?;
                scale_tensormap(tensor, factor)?
            } else {
                tensor
            };

            // write back into the outputs buffer
            unsafe {
                *outputs.add(i) = TensorMap::into_raw(tensor);
            }
        }
    }

    Ok(())
}
