// SPDX-License-Identifier: AGPL-3.0-only
// The two libc items nvml.rs uses, so the harness needs no crates.io access.
use std::ffi::{c_char, c_int, c_void};
pub const RTLD_NOW: c_int = 2;
unsafe extern "C" {
    pub fn dlopen(file: *const c_char, flag: c_int) -> *mut c_void;
    pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
