use std::ffi::{c_char, CStr};
use std::sync::Arc;

use dlpk::sys::DLManagedTensorVersioned;
use dlpk::{DLPackTensor, DLPackVersion};
use metatensor::c_api::{mts_block_t, mts_tensormap_t};
use metatensor::{TensorBlock, TensorMap};

use crate::{Error, PairListOptions, System};
use super::{catch_unwind, mta_status_t, mta_string_t};

/// Opaque handle to an atomistic system.
///
/// The system owns DLPack tensors for types, positions, cell, and PBC, as well
/// as metatensor blocks for pair lists and tensor maps for custom data.
#[allow(non_camel_case_types)]
pub struct mta_system_t(pub(crate) System);

impl mta_system_t {
    /// Convert an mta_system_t into a pointer inside an Arc<mta_system_t>, to be
    /// passed through the C API
    pub(crate) fn into_raw(self) -> *mut mta_system_t {
        Arc::into_raw(Arc::new(self)).cast_mut()
    }

    /// Recover the Arc<mta_system_t> from a pointer created with
    /// [`mta_system_t::into_raw`]
    pub(crate) unsafe fn from_raw(ptr: *const mta_system_t) -> Arc<mta_system_t> {
        unsafe { Arc::from_raw(ptr) }
    }
}

/// Create a new system from raw DLPack tensors.
///
/// This function **takes ownership** of `types`, `positions`, `cell`, and
/// `pbc`. The caller must not use these tensors after calling this function.
///
/// @param length_unit A null-terminated C string containing the length unit
///     (e.g. "Angstrom", "nanometer"). Must not be null.
/// @param types A DLPack managed tensor with shape `(n_atoms,)` and dtype
///     `int32`. Ownership is transferred.
/// @param positions A DLPack managed tensor with shape `(n_atoms, 3)` and
///     dtype `float32` or `float64`. Ownership is transferred.
/// @param cell A DLPack managed tensor with shape `(3, 3)` and the same dtype
///     as `positions`. Ownership is transferred.
/// @param pbc A DLPack managed tensor with shape `(3,)` and dtype `bool`.
///     Ownership is transferred.
/// @param system Output parameter, set to the newly created system handle.
///     The caller takes ownership and must free it with `mta_system_free`.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_system_create(
    length_unit: *const c_char,
    types: *mut DLManagedTensorVersioned,
    positions: *mut DLManagedTensorVersioned,
    cell: *mut DLManagedTensorVersioned,
    pbc: *mut DLManagedTensorVersioned,
    system: *mut *mut mta_system_t,
) -> mta_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(system);
    catch_unwind(move || {
        check_pointers_non_null!(length_unit, types, positions, cell, pbc, system);

        unsafe {
            let length_unit = CStr::from_ptr(length_unit)
                .to_str()
                .map_err(|_| Error::InvalidParameter("length_unit is not valid UTF-8".into()))?
                .to_string();


            let types = DLPackTensor::from_ptr(types);
            let positions = DLPackTensor::from_ptr(positions);
            let cell = DLPackTensor::from_ptr(cell);
            let pbc = DLPackTensor::from_ptr(pbc);

            let system = mta_system_t(System::new(length_unit, types, positions, cell, pbc)?);

            let _ = &unwind_wrapper;
            *unwind_wrapper.0 = system.into_raw();
        }
        Ok(())
    })
}

/// Free a system previously created by `mta_system_create`.
///
/// If there are outstanding borrowed views (from `mta_system_get_data`), the
/// system's data will remain alive until all views are released.
///
/// @param system The system handle to free. Can be null, in which case this
///     function is a no-op.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_system_free(system: *mut mta_system_t) -> mta_status_t {
    catch_unwind(|| {
        if system.is_null() {
            return Ok(());
        }

        let system = unsafe { mta_system_t::from_raw(system.cast_const()) };
        std::mem::drop(system);
        Ok(())
    })
}

/// Get the number of atoms in a system.
///
/// @param system The system handle. Must not be null.
/// @param size Output parameter, set to the number of atoms.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_system_size(
    system: *const mta_system_t,
    size: *mut usize,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(system, size);

        unsafe {
            let system = &*system;
            *size = system.0.size();
        }
        Ok(())
    })
}

/// Kind of data always stored in a system.
///
/// Other kinds of data can be stored with `mta_system_add_custom_data` and
/// retrieved with `mta_system_get_custom_data`.
#[allow(non_camel_case_types)]
#[repr(C)]
#[non_exhaustive]
pub enum mta_system_data_kind {
    MTA_SYSTEM_DATA_TYPES = 0,
    MTA_SYSTEM_DATA_POSITIONS = 1,
    MTA_SYSTEM_DATA_CELL = 2,
    MTA_SYSTEM_DATA_PBC = 3,
}

/// Custom deleter for borrowed DLPack tensors returned by `mta_system_get_data`.
///
/// Releases the `Arc<mta_system_t>` reference stored in `manager_ctx` and
/// frees the heap-allocated `DLManagedTensorVersioned`.
unsafe extern "C" fn borrowed_tensor_deleter(
    tensor: *mut DLManagedTensorVersioned,
) {
    let system = unsafe {
        mta_system_t::from_raw((*tensor).manager_ctx.cast())
    };
    std::mem::drop(system);
    unsafe {
        std::mem::drop(Box::from_raw(tensor));
    }
}

/// Get a DLPack tensor from a system for the requested data.
///
/// This function **returns a borrowed view** of the system's internal data.
/// The returned `DLManagedTensorVersioned` has a custom deleter that decrements
/// the system's reference count, keeping the system alive as long as the
/// borrowed view exists.
///
/// The caller is responsible for calling the deleter on the returned tensor
/// when it is no longer needed. The tensor shares the data pointer with the
/// system; do **not** modify it.
///
/// @param system The system handle. Must not be null.
/// @param request Which data to retrieve (types, positions, cell, or PBC).
/// @param data Output parameter, set to a pointer to a newly allocated
///     `DLManagedTensorVersioned` containing the requested data. The caller
///     takes ownership and must call the deleter when done.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_system_get_data(
    system: *const mta_system_t,
    request: mta_system_data_kind,
    data: *mut *mut DLManagedTensorVersioned,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(system, data);
        unsafe {
            *data = std::ptr::null_mut();
        }

        // increase the reference count of the system so that it stays alive as
        // long as the returned tensor is alive. We do this by creating a
        // temporary Arc from the raw pointer, cloning it and storing the clone
        // in the manager_ctx.
        let system = unsafe { mta_system_t::from_raw(system) };
        let arc_clone = system.clone();

        let tensor_ref = match request {
            mta_system_data_kind::MTA_SYSTEM_DATA_TYPES => system.0.types(),
            mta_system_data_kind::MTA_SYSTEM_DATA_POSITIONS => system.0.positions(),
            mta_system_data_kind::MTA_SYSTEM_DATA_CELL => system.0.cell(),
            mta_system_data_kind::MTA_SYSTEM_DATA_PBC => system.0.pbc(),
        };

        let packed = Box::new(DLManagedTensorVersioned {
            version: DLPackVersion::current(),
            manager_ctx: Arc::into_raw(arc_clone) as *mut std::ffi::c_void,
            deleter: Some(borrowed_tensor_deleter),
            flags: dlpk::sys::DLPACK_FLAG_BITMASK_READ_ONLY,
            dl_tensor: tensor_ref.raw.clone(),
        });

        // do not drop the system, it is still owned by the caller.
        std::mem::forget(system);

        unsafe {
            *data = Box::into_raw(packed);
        }
        Ok(())
    })
}

/// Get the length unit of a system.
///
/// This function returns a new `mta_string_t` that the caller must free with
/// `mta_string_free`.
///
/// @param system The system handle. Must not be null.
/// @param length_unit Output parameter, set to the length unit string.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_system_get_length_unit(
    system: *const mta_system_t,
    length_unit: *mut mta_string_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(system, length_unit);

        unsafe {
            let system = &*system;
            *length_unit = mta_string_t::new(system.0.length_unit());
        }
        Ok(())
    })
}

/// Add a pair list (neighbor list) to a system.
///
/// This function **takes ownership** of `pairs`. The caller must not use the
/// block after calling this function.
///
/// @param system The system handle. Must not be null.
/// @param options A JSON-serialized `PairListOptions` object. Must not be null.
/// @param pairs A `mts_block_t` containing the pair data. Ownership is
///     transferred.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_system_add_pairs(
    system: *mut mta_system_t,
    options: *const c_char,
    pairs: *mut mts_block_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(system, options, pairs);

        let options_str = unsafe { CStr::from_ptr(options) }
            .to_str()
            .map_err(|_| Error::InvalidParameter("options is not valid UTF-8".into()))?;

        let options_json = json::parse(options_str)
            .map_err(|e| Error::Serialization(format!("invalid JSON for PairListOptions: {e}")))?;

        let options = PairListOptions::try_from(&options_json)?;

        let pairs = unsafe { TensorBlock::from_raw(pairs) };

        let mut system = unsafe { mta_system_t::from_raw(system.cast_const()) };
        let system_mut = Arc::get_mut(&mut system).ok_or_else(|| {
            Error::InvalidParameter(
                "cannot modify system while there are outstanding borrowed views".into(),
            )
        })?;
        system_mut.0.add_pairs(options, pairs)?;

        // do not drop the system, it is still owned by the caller.
        std::mem::forget(system);

        Ok(())
    })
}

/// Get a pair list from a system.
///
/// **Returns a borrowed view** of the pair list. The system must outlive the
/// returned pointer. Do **not** free the returned block.
///
/// @param system The system handle. Must not be null.
/// @param options A JSON-serialized `PairListOptions` object identifying which
///     pair list to retrieve. Must not be null.
/// @param pairs Output parameter, set to a pointer to the pair list block, or
///     NULL if no pair list matches the options.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_system_get_pairs(
    system: *const mta_system_t,
    options: *const c_char,
    pairs: *mut *const mts_block_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(system, options, pairs);

        let options_str = unsafe { CStr::from_ptr(options) }
            .to_str()
            .map_err(|_| Error::InvalidParameter("options is not valid UTF-8".into()))?;

        let options_json = json::parse(options_str)
            .map_err(|e| Error::Serialization(format!("invalid JSON for PairListOptions: {e}")))?;

        let options = PairListOptions::try_from(&options_json)?;

        let system = unsafe { &*system };
        match system.0.get_pairs(&options) {
            Some(block) => {
                unsafe {
                    *pairs = block.as_ptr();
                }
            }
            None => {
                return Err(Error::InvalidParameter(
                    "no pair list found for the given options".into(),
                ));
            }
        }

        Ok(())
    })
}

/// Get all pair list options known by a system.
///
/// This function returns a new `mta_string_t` containing a JSON array of
/// `PairListOptions` objects. The caller must free it with `mta_string_free`.
///
/// @param system The system handle. Must not be null.
/// @param pairs_options Output parameter, set to a JSON string containing an
///     array of `PairListOptions` objects.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_system_known_pairs(
    system: *const mta_system_t,
    pairs_options: *mut mta_string_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(system, pairs_options);

        let system = unsafe { &*system };
        let known = system.0.known_pairs();
        let mut json_array = json::JsonValue::new_array();
        for options in known {
            json_array.push(json::JsonValue::from(options.clone())).map_err(|_| {
                Error::Internal("failed to build JSON array".into())
            })?;
        }

        unsafe {
            *pairs_options = mta_string_t::new(json::stringify(json_array));
        }
        Ok(())
    })
}

/// Add custom data to a system.
///
/// This function **takes ownership** of `data`. The caller must not use the
/// tensor map after calling this function.
///
/// @param system The system handle. Must not be null.
/// @param name A null-terminated C string containing the name of the custom
///     data. Must not be null.
/// @param data A `mts_tensormap_t` containing the custom data. Ownership is
///     transferred.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_system_add_custom_data(
    system: *mut mta_system_t,
    name: *const c_char,
    data: *mut mts_tensormap_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(system, name, data);

        let name = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| Error::InvalidParameter("name is not valid UTF-8".into()))?
            .to_string();

        let data = unsafe { TensorMap::from_raw(data) };

        let mut system = unsafe { mta_system_t::from_raw(system.cast_const()) };
        let system_mut = Arc::get_mut(&mut system).ok_or_else(|| {
            Error::InvalidParameter(
                "cannot modify system while there are outstanding borrowed views".into(),
            )
        })?;
        system_mut.0.add_custom_data(name, data, false)?;

        // do not drop the system, it is still owned by the caller.
        std::mem::forget(system);

        Ok(())
    })
}

/// Get custom data from a system by name.
///
/// **Returns a borrowed view** of the custom data. The system must outlive the
/// returned pointer. Do **not** free the returned tensor map.
///
/// @param system The system handle. Must not be null.
/// @param name A null-terminated C string containing the name of the custom
///     data. Must not be null.
/// @param data Output parameter, set to a pointer to the custom data tensor
///     map, or an error if no data with the given name exists.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_system_get_custom_data(
    system: *const mta_system_t,
    name: *const c_char,
    data: *mut *const mts_tensormap_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(system, name, data);

        let name = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| Error::InvalidParameter("name is not valid UTF-8".into()))?;

        let system = unsafe { &*system };
        let result = system.0.get_custom_data(name)?;

        unsafe {
            *data = result.as_ptr();
        }

        Ok(())
    })
}

/// Get all custom data names known by a system.
///
/// **Returns a new** `mta_string_t` containing a JSON array of strings. The
/// caller must free it with `mta_string_free`.
///
/// @param system The system handle. Must not be null.
/// @param names Output parameter, set to a JSON string containing an array of
///     custom data names.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_system_known_custom_data(
    system: *const mta_system_t,
    names: *mut mta_string_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(system, names);

        let system = unsafe { &*system };
        let known = system.0.known_custom_data();
        let mut json_array = json::JsonValue::new_array();
        for name in known {
            json_array.push(name).map_err(|_| {
                Error::Internal("failed to build JSON array".into())
            })?;
        }

        unsafe {
            *names = mta_string_t::new(json::stringify(json_array));
        }
        Ok(())
    })
}

// TODO: mta_system_to(device, dtype)
