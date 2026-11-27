use dlpk::sys::DLDevice;

use super::{catch_unwind, mta_status_t};

/// Block the calling thread until all the work submitted to a stream is
/// complete.
///
/// @verbatim embed:rst:leading-asterisk
///
/// This is useful to fulfill the contract of :c:func:`mta_execute_model` when
/// a model can not submit its work directly to the caller's stream (for
/// example, because the library used by the model manages its own streams).
/// See :c:func:`mta_execute_model` for the possible values of ``stream``.
///
/// This function does nothing if ``stream`` is ``NULL``, since there is no
/// outstanding work in this case.
///
/// @endverbatim
///
/// @param device the device on which the stream lives
/// @param stream pointer to the stream to synchronize, or `NULL`
/// @return `MTA_SUCCESS` on success, or another status code if an error occurs.
///     You can get more details about the error with `mta_last_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mta_stream_synchronize(
    device: DLDevice,
    stream: *const i64,
) -> mta_status_t {
    catch_unwind(|| {
        let stream = unsafe { stream.as_ref().copied() };
        crate::kernels::synchronize(device, stream)
    })
}
