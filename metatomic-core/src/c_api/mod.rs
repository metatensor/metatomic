use std::ffi::CString;
use std::os::raw::c_char;

use once_cell::sync::Lazy;


static VERSION: Lazy<CString> = Lazy::new(|| {
    CString::new(env!("METATOMIC_FULL_VERSION")).expect("version contains NULL byte")
});


/// Get the runtime version of the metatomic library as a string.
///
/// This version follows the `<major>.<minor>.<patch>[-<dev>]` format.
#[no_mangle]
pub extern "C" fn mta_version() -> *const c_char {
    return VERSION.as_ptr();
}
