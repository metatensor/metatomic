use std::ffi::{c_char, c_void, CStr};
use std::fs::File;
use std::io::{BufReader, Cursor};

use metatensor::c_api::{mts_create_array_callback_t, mts_realloc_buffer_t};

use super::{catch_unwind, mta_status_t, mta_system_t};
use crate::Error;

/// Wrapper for an externally managed buffer, that can be grown to fit more data
struct ExternalBuffer {
    data: *mut *mut u8,
    writen: u64,
    allocated: u64,

    realloc_user_data: *mut c_void,
    realloc: unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> *mut u8,

    current: u64,
}

impl std::io::Write for ExternalBuffer {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let remaining_space = self.allocated.saturating_sub(self.current);

        if (remaining_space as usize) < buf.len() {
            let required_size = self.current.saturating_add(buf.len() as u64);
            let mut new_size = if self.allocated == 0 { 1024 } else { self.allocated };
            while new_size < required_size {
                new_size = new_size.saturating_mul(2);
                if new_size == 0 {
                     return Err(std::io::Error::new(
                         std::io::ErrorKind::OutOfMemory,
                         "requested allocation size overflow",
                     ));
                }
            }

            let new_ptr = unsafe {
                (self.realloc)(self.realloc_user_data, *self.data, new_size as usize)
            };

            if new_ptr.is_null() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "failed to allocate memory with the realloc callback"
                ));
            }

            unsafe {
                *self.data = new_ptr;
            }

            self.allocated = new_size;
        }

        let mut output = unsafe {
            let start = (*self.data).offset(self.current as isize);
            // allocated >= current + buf.len()
            std::slice::from_raw_parts_mut(start, buf.len())
        };

        let count = output.write(buf).expect("failed to write to pre-allocated slice");
        assert_eq!(count, buf.len());
        self.current += count as u64;

        if self.current > self.writen {
            self.writen = self.current;
        }
        return Ok(count);
    }

    fn flush(&mut self) -> std::io::Result<()> {
        return Ok(());
    }
}


#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
impl std::io::Seek for ExternalBuffer {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        match pos {
            std::io::SeekFrom::Start(offset) => {
                if offset > self.writen {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof, "tried to seek past the end of the buffer")
                    );
                }

                self.current = offset;
            },

            std::io::SeekFrom::End(offset) => {
                if offset > 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof, "tried to seek past the end of the buffer")
                    );
                }

                if -offset > self.writen as i64 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof, "tried to seek past the beginning of the buffer")
                    );
                }

                self.current = (self.writen as i64 + offset) as u64;
            },

            std::io::SeekFrom::Current(offset) => {
                let result = self.current as i64 + offset;
                if result > self.writen as i64 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof, "tried to seek past the end of the buffer")
                    );
                }

                if result < 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof, "tried to seek past the beginning of the buffer")
                    );
                }

                self.current = result as u64;
            },
        }

        return Ok(self.current);
    }

    fn rewind(&mut self) -> std::io::Result<()> {
        self.current = 0;
        return Ok(());
    }

    fn stream_position(&mut self) -> std::io::Result<u64> {
        return Ok(self.current);
     }
}


/// Save a system to a file.
///
/// The format consists of a zip archive containing NPY files for the system's
/// data (types, positions, cell, pbc), a `info.json` file for metadata, and
/// optional sub-directories for pair lists (`pairs/<id>/options.json` and
/// `pairs/<id>/data.mts`) and custom data (`data/<name>.mts`).
///
/// @param path A null-terminated C string containing the file path. Must not be
///     null.
/// @param system The system to save. Must not be null.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_save(
    path: *const c_char,
    system: *const mta_system_t
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(path, system);

        let path = unsafe { CStr::from_ptr(path) }.to_str()
            .map_err(|_| Error::InvalidParameter("path is not valid UTF-8".into()))?;

        let file = File::create(path)?;
        let system = unsafe { &*system };
        crate::io::save(file, &system.0)?;

        Ok(())
    })
}

/// Save a system to an in-memory buffer.
///
/// The buffer is grown as needed using the provided `realloc` callback. On
/// success, `*buffer` points to the serialized data and `*buffer_count`
/// contains the number of bytes written.
///
/// @param buffer Pointer to the buffer pointer. On input, `*buffer` may be NULL
///     (in which case `*buffer_count` must be 0). On output, `*buffer` is
///     updated to point to the serialized data.
/// @param buffer_count Pointer to the buffer size. On input, `*buffer_count`
///     must contain the current allocation size. On output, it is set to the
///     number of bytes written.
/// @param realloc_user_data User data passed as the first argument to
///     `realloc`.
/// @param realloc Callback to grow the buffer. Must not be NULL.
/// @param system The system to save. Must not be null.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern "C" fn mta_save_buffer(
    buffer: *mut *mut u8,
    buffer_count: *mut usize,
    realloc_user_data: *mut c_void,
    realloc: mts_realloc_buffer_t,
    system: *const mta_system_t,
) -> mta_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(buffer, buffer_count, system);

        let realloc = if let Some(realloc) = realloc {
            realloc
        } else {
            return Err(Error::InvalidParameter(
                "realloc callback can not be NULL in mta_save_buffer".into()
            ));
        };

        if unsafe { (*buffer).is_null() } {
            // `ExternalBuffer.write` calls realloc with the current `*buffer`
            // (which may be null) for the initial allocation.
            unsafe { *buffer = std::ptr::null_mut(); }
        }

        let system = unsafe { &*system };
        let mut external_buffer = ExternalBuffer {
            data: buffer,
            allocated: unsafe { *buffer_count } as u64,
            writen: 0,
            realloc_user_data,
            realloc,
            current: 0,
        };

        crate::io::save(&mut external_buffer, &system.0)?;

        unsafe {
            *buffer_count = external_buffer.current as usize;
        }

        Ok(())
    })
}

/// Load a system from a file.
///
/// The file must have been written by `mta_save` and contain a valid metatomic
/// system.
///
/// @param path A null-terminated C string containing the file path. Must not be
///     null.
/// @param create_array Callback to allocate arrays for the system's data. Must
///     not be NULL.
/// @param system Output parameter, set to the newly created system handle.
///     The caller takes ownership and must free it with `mta_system_free`.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_load(
    path: *const c_char,
    create_array: mts_create_array_callback_t,
    system: *mut *mut mta_system_t,
) -> mta_status_t {
    catch_unwind(move || {
        check_pointers_non_null!(path);

        let path = unsafe { CStr::from_ptr(path) }.to_str()
            .map_err(|_| Error::InvalidParameter("path is not valid UTF-8".into()))?;

        let file = BufReader::new(File::open(path)?);
        let new_system = mta_system_t(crate::io::load(file, create_array)?);

        unsafe {
            *system = mta_system_t::into_raw(new_system);
        }

        Ok(())
    })
}

/// Load a system from an in-memory buffer.
///
/// The buffer must contain data serialized by `mta_save_buffer` (or the
/// equivalent Rust function).
///
/// @param buffer Pointer to the serialized data. Must not be NULL.
/// @param buffer_size Number of bytes in `buffer`.
/// @param create_array Callback to allocate arrays for the system's data. Must
///     not be NULL.
/// @param system Output parameter, set to the newly created system handle.
///     The caller takes ownership and must free it with `mta_system_free`.
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_load_buffer(
    buffer: *const u8,
    buffer_size: usize,
    create_array: mts_create_array_callback_t,
    system: *mut *mut mta_system_t,
) -> mta_status_t {
    catch_unwind(move || {
        check_pointers_non_null!(buffer);

        let slice = unsafe {
            std::slice::from_raw_parts(buffer, buffer_size)
        };
        let cursor = Cursor::new(slice);
        let new_system = mta_system_t(crate::io::load(cursor, create_array)?);

        unsafe {
            *system = mta_system_t::into_raw(new_system);
        }

        Ok(())
    })
}
