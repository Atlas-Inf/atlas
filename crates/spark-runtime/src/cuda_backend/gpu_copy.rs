// SPDX-License-Identifier: AGPL-3.0-only

//! The synchronous host/device copy family, as inherent methods.
//!
//! `impl GpuBackend for AtlasCudaBackend` cannot be split across files —
//! Rust requires one block per impl — so the trait body in `gpu_impl.rs`
//! delegates these four to the implementations here. They carry an `_impl`
//! suffix so an inherent method can never silently shadow the trait method
//! at a concrete-typed call site. They are the natural
//! group to move: same shape, same error handling, and each is one driver
//! call plus one stream synchronize.
//!
//! **There is no bounds check here, on either end.** This module doc used to
//! claim "a bounds check plus one driver call"; nothing in this file has ever
//! checked a bound, and a reader who trusted that sentence would have been
//! looking for a guard that does not exist. What is actually true:
//!
//! * The HOST end is bounded by construction — `src: &[u8]` / `dst: &mut [u8]`
//!   supply their own `len()`, so a host over-run is not expressible.
//! * The DEVICE end is UNCHECKED. `copy_d2d_impl` in particular takes a bare
//!   `bytes` that is validated against neither allocation. It cannot be checked
//!   here: `AtlasCudaBackend::live_allocs` is keyed by BASE pointer (its
//!   sizes exist for footprint accounting), and callers legitimately pass
//!   interior pointers from `DevicePtr::offset`, so there is no lookup to
//!   compare against. Sizing a device copy correctly is the CALLER's obligation.
//!
//! The Metal backend does check its device end (`metal_backend.rs`), because a
//! `metal::Buffer` knows its own `length()`. The divergence is real, not an
//! oversight to paper over in prose.
//!
//! The safety contract is the one documented in `gpu_impl.rs`: a primary
//! context is current on the calling thread, every `DevicePtr` came from a
//! live allocation, and byte counts are exact.

use std::ffi::c_void;

use anyhow::{Result, bail};
use atlas_core::registry::cuda_error_text;

use super::{
    AtlasCudaBackend, cuMemcpyDtoDAsync_v2, cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2,
    cuStreamQuery, cuStreamSynchronize,
};
use crate::gpu::{DevicePtr, HostPitched, Pitched};

/// One pitched copy between PAGE-LOCKED host memory and the device, enqueued
/// on `stream` with no synchronisation (`to_device` picks the direction).
///
/// `cudaMemcpy2DAsync` is a runtime (cudart) entry point, as in
/// `copy_d2d_2d_async`; kinds 1 / 2 are `cudaMemcpyHostToDevice` /
/// `cudaMemcpyDeviceToHost`. The caller guarantees `host` spans
/// `shape.host_span()` bytes that stay valid until the next sync on `stream`.
fn host_pitched_async(
    host: *mut c_void,
    dev: DevicePtr,
    shape: Pitched,
    to_device: bool,
    stream: u64,
) -> Result<()> {
    #[cfg(not(atlas_scale))]
    {
        unsafe extern "C" {
            fn cudaMemcpy2DAsync(
                dst: *mut c_void,
                dpitch: usize,
                src: *const c_void,
                spitch: usize,
                width: usize,
                height: usize,
                kind: i32,
                stream: u64,
            ) -> i32;
        }
        let d = dev.0 as *mut c_void;
        let (dst, dpitch, src, spitch, kind) = if to_device {
            (d, shape.dev_pitch, host, shape.host_pitch, 1)
        } else {
            (host, shape.host_pitch, d, shape.dev_pitch, 2)
        };
        let status = unsafe {
            cudaMemcpy2DAsync(
                dst,
                dpitch,
                src,
                spitch,
                shape.width,
                shape.height,
                kind,
                stream,
            )
        };
        if status != 0 {
            bail!(
                "cudaMemcpy2DAsync (host pitched, to_device={to_device}) failed: status {status}"
            );
        }
        Ok(())
    }
    // strix/SCALE: no cudart runtime linked — one driver-API copy per row.
    #[cfg(atlas_scale)]
    {
        for r in 0..shape.height {
            let h = unsafe { (host as *mut u8).add(r * shape.host_pitch) } as *mut c_void;
            let d = dev.0 + (r * shape.dev_pitch) as u64;
            let status = unsafe {
                if to_device {
                    cuMemcpyHtoDAsync_v2(d, h, shape.width, stream)
                } else {
                    cuMemcpyDtoHAsync_v2(h, d, shape.width, stream)
                }
            };
            if status != 0 {
                bail!("host pitched copy row {r} failed: status {status}");
            }
        }
        Ok(())
    }
}

impl HostPitched for AtlasCudaBackend {
    fn h2d_retained(&self, src: &[u8], dst: DevicePtr, shape: Pitched, stream: u64) -> Result<()> {
        anyhow::ensure!(src.len() >= shape.host_span(), "pitched H2D: short source");
        host_pitched_async(src.as_ptr() as *mut c_void, dst, shape, true, stream)
    }

    fn d2h(&self, src: DevicePtr, dst: &mut [u8], shape: Pitched, stream: u64) -> Result<()> {
        anyhow::ensure!(dst.len() >= shape.host_span(), "pitched D2H: short dest");
        host_pitched_async(dst.as_mut_ptr() as *mut c_void, src, shape, false, stream)
    }
}

impl AtlasCudaBackend {
    pub(crate) fn copy_h2d_impl(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        let status = unsafe {
            cuMemcpyHtoDAsync_v2(
                dst.0,
                src.as_ptr() as *const c_void,
                src.len(),
                self.default_stream,
            )
        };
        if status != 0 {
            bail!("cuMemcpyHtoDAsync_v2 failed: status {status}");
        }
        // Synchronize to ensure the copy completes before host buffer is freed.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after H2D failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    pub(crate) fn copy_d2h_impl(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr() as *mut c_void,
                src.0,
                dst.len(),
                self.default_stream,
            )
        };
        if status != 0 {
            bail!("cuMemcpyDtoHAsync_v2 failed: status {status}");
        }
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2H failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    pub(crate) fn copy_d2h_on_stream_impl(
        &self,
        src: DevicePtr,
        dst: &mut [u8],
        stream: u64,
    ) -> Result<()> {
        // Enqueue the copy on the caller's stream so CUDA orders it after
        // any prior kernel launches on the same stream. Without this, the
        // copy may run on the default stream concurrently with kernels on
        // `stream` and read torn bytes (HSS Turbo8 race, 2026-04-28).
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, src.0, dst.len(), stream)
        };
        if status != 0 {
            bail!("cuMemcpyDtoHAsync_v2 (on_stream) failed: status {status}");
        }
        // ATLAS_D2H_SPIN_SYNC=1: poll cuStreamQuery instead of parking in the
        // blocking sync. Diagnostic for the 130 ms verify stall (PROGRESS_LOG
        // 6.15/6.16): a blocked thread appears to leave the submission ring
        // unflushed until a ~130 ms driver housekeeping tick doorbells the
        // GPU; each query forces a flush, so if the theory holds the wait
        // collapses to the real GPU time.
        let spin = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("ATLAS_D2H_SPIN_SYNC").as_deref() == Ok("1"))
        };
        let sync = if spin {
            const CUDA_ERROR_NOT_READY: i32 = 600;
            loop {
                let q = unsafe { cuStreamQuery(stream) };
                if q != CUDA_ERROR_NOT_READY {
                    break q;
                }
                std::hint::spin_loop();
            }
        } else {
            unsafe { cuStreamSynchronize(stream) }
        };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2H on_stream failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    pub(crate) fn copy_d2d_impl(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        let status = unsafe { cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, self.default_stream) };
        if status != 0 {
            // 901 = STREAM_CAPTURE_INVALIDATED: some EARLIER op poisoned the
            // recording and this copy is merely the reporter. The backtrace
            // still names the captured segment the poison sits in.
            tracing::error!(
                "sync copy_d2d failed (status {status}) at:\n{}",
                std::backtrace::Backtrace::force_capture()
            );
            bail!("cuMemcpyDtoDAsync_v2 (sync copy_d2d) failed: status {status}");
        }
        // Synchronize to ensure copy completes before kernels on other streams read it.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2D failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }
}
