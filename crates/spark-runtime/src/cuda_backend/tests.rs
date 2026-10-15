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

/// Slab bookkeeping against a real backend: interior frees are no-ops and the
/// region stays registered until the BASE is freed — the invariant the
/// slab-aliased FP8 weight path relies on.
#[test]
#[ignore = "requires a GPU device (runs on strix via --ignored)"]
fn slab_interior_frees_keep_region_until_base() {
    use crate::cuda_backend::AtlasCudaBackend;
    use crate::gpu::GpuBackend;
    let gpu = AtlasCudaBackend::new(0, &[]).expect("GPU backend");
    let slab = gpu.alloc(64 * 1024).unwrap();
    gpu.register_slab(slab, 64 * 1024);
    let interior = crate::gpu::DevicePtr(slab.0 + 4096);
    assert!(GpuBackend::is_slab_interior(&gpu, interior));
    assert!(!GpuBackend::is_slab_interior(&gpu, slab)); // base itself is not interior
    // Freeing interior pointers must be a no-op that leaves the region live.
    gpu.free(interior).unwrap();
    assert!(GpuBackend::is_slab_interior(&gpu, interior));
    gpu.free(crate::gpu::DevicePtr(slab.0 + 60 * 1024)).unwrap();
    assert!(GpuBackend::is_slab_interior(&gpu, interior));
    // Freeing the base drops the region; interior no longer matches.
    gpu.free(slab).unwrap();
    assert!(!GpuBackend::is_slab_interior(&gpu, interior));
}
