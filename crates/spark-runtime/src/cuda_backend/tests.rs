// SPDX-License-Identifier: AGPL-3.0-only

//! cuda_backend unit tests. Pure CPU — no `cuInit`, no GPU touch — so
//! they run on every CI host.

use std::ffi::c_void;

use atlas_core::registry::RawCudaFunc;

use crate::gpu::{DevicePtr, KernelHandle};

#[test]
fn kernel_handle_roundtrip() {
    // Verify KernelHandle <-> RawCudaFunc pointer conversion is lossless.
    let fake_ptr = 0xDEAD_BEEF_CAFE_u64;
    let handle = KernelHandle(fake_ptr);
    let raw = RawCudaFunc(handle.0 as *mut c_void);
    let back = raw.0 as u64;
    assert_eq!(back, fake_ptr);
}

#[test]
fn null_free_is_noop() {
    // AtlasCudaBackend::free should handle null pointers gracefully.
    // Can't call without GPU, but verify the DevicePtr::is_null logic.
    assert!(DevicePtr::NULL.is_null());
    assert!(!DevicePtr(0x1000).is_null());
}

/// Exercises the production terminal dispatch, not a CUDA memory-pressure event.
#[cfg(unix)]
#[test]
fn oom_terminal_dispatch_preserves_legacy_and_selected_exit() {
    const MODE: &str = "ATLAS_TEST_OOM_TERMINAL_MODE";
    const TEST: &str =
        "cuda_backend::tests::oom_terminal_dispatch_preserves_legacy_and_selected_exit";
    extern "C" fn at_exit() {
        let bytes = b"oom-c-atexit\n";
        unsafe { libc::write(libc::STDOUT_FILENO, bytes.as_ptr().cast(), bytes.len()) };
    }
    struct DropWitness;
    impl Drop for DropWitness {
        fn drop(&mut self) {
            let bytes = b"oom-rust-drop\n";
            unsafe { libc::write(libc::STDOUT_FILENO, bytes.as_ptr().cast(), bytes.len()) };
        }
    }
    fn selected_exit() -> ! {
        unsafe { libc::_exit(74) }
    }
    if let Ok(mode) = std::env::var(MODE) {
        assert_eq!(unsafe { libc::atexit(at_exit) }, 0);
        let _witness = DropWitness;
        match mode.as_str() {
            "legacy" => super::dispatch_oom_exit(super::legacy_oom_exit),
            "selected" => super::dispatch_oom_exit(selected_exit),
            _ => panic!("unknown terminal dispatch case"),
        }
    }
    for (mode, code, c_exit) in [("legacy", 1, true), ("selected", 74, false)] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST, "--nocapture"])
            .env(MODE, mode)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(output.status.code(), Some(code), "{mode}: {output:?}");
        assert_eq!(stdout.contains("oom-c-atexit"), c_exit, "{mode}: {stdout}");
        assert!(!stdout.contains("oom-rust-drop"), "{mode}: {stdout}");
    }
}
