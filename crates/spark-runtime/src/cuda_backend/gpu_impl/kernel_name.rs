// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::c_void;

/// The kernel's name for a launch-failure message; `?` where the driver
/// cannot say (SCALE / HIP shim builds, or an old driver).
pub(super) fn kernel_name(func: *mut c_void) -> String {
    #[cfg(not(atlas_scale))]
    {
        let mut name: *const std::ffi::c_char = std::ptr::null();
        // SAFETY: `func` is a live CUfunction handle from the registry, and the
        // driver writes a NUL-terminated string it owns (valid for the module's
        // lifetime) into `name`, which is only read here.
        let rc = unsafe { super::super::cuFuncGetName(&mut name, func) };
        if rc == 0 && !name.is_null() {
            return unsafe { std::ffi::CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned();
        }
    }
    let _ = func;
    "?".to_string()
}
