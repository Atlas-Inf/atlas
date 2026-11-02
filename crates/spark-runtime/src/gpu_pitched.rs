// SPDX-License-Identifier: AGPL-3.0-only

//! Host staging shared by every [`GpuBackend`]: the alignment of page-locked
//! memory, and pitched (2-D) copies between it and the device. Kept out of
//! `gpu.rs` (500-LoC cap); re-exported from there.
//!
//! On GB10 a small async copy costs several microseconds of copy-engine time
//! whatever its size, so moving N blocks × M regions as M pitched copies
//! instead of N × M plain ones is what the NVMe prefix tier's fast path is
//! built on. A backend with a native pitched copy exposes it through
//! [`GpuBackend::host_pitched`]; every other backend gets one plain copy per
//! row from the free functions below.

use anyhow::Result;

use crate::gpu::{DevicePtr, GpuBackend};

/// Alignment of the heap stand-in for page-locked memory (mock / backends
/// without pinning). Real pinned memory is page-aligned, and callers rely on
/// it: O_DIRECT I/O straight into a staging buffer needs a 4 KiB boundary.
pub const HOST_PINNED_ALIGN: usize = 4096;

/// Geometry of a pitched (2-D) copy between host staging and a device pool:
/// `height` rows of `width` bytes, rows `host_pitch` apart on the host and
/// `dev_pitch` apart on the device (each pitch ≥ `width`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pitched {
    pub host_pitch: usize,
    pub dev_pitch: usize,
    pub width: usize,
    pub height: usize,
}

impl Pitched {
    /// Host bytes the copy spans: the last row ends `width` into its pitch.
    pub fn host_span(self) -> usize {
        match self.height {
            0 => 0,
            h => (h - 1) * self.host_pitch + self.width,
        }
    }
}

/// A backend's native pitched copies (one enqueue for the whole run).
pub trait HostPitched {
    /// See [`copy_h2d_pitched_async_retained`].
    fn h2d_retained(&self, src: &[u8], dst: DevicePtr, shape: Pitched, stream: u64) -> Result<()>;
    /// See [`copy_d2h_pitched_async`].
    fn d2h(&self, src: DevicePtr, dst: &mut [u8], shape: Pitched, stream: u64) -> Result<()>;
}

/// Pitched host-to-device copy from a source the CALLER keeps alive (as
/// [`GpuBackend::copy_h2d_async_retained`]): row `r` is `shape.width` bytes
/// from `src[r * host_pitch..]` to `dst + r * dev_pitch`.
pub fn copy_h2d_pitched_async_retained(
    gpu: &dyn GpuBackend,
    src: &[u8],
    dst: DevicePtr,
    shape: Pitched,
    stream: u64,
) -> Result<()> {
    if let Some(native) = gpu.host_pitched() {
        return native.h2d_retained(src, dst, shape, stream);
    }
    for r in 0..shape.height {
        let row = &src[r * shape.host_pitch..][..shape.width];
        gpu.copy_h2d_async_retained(row, dst.offset(r * shape.dev_pitch), stream)?;
    }
    Ok(())
}

/// Pitched device-to-host copy, the mirror of
/// [`copy_h2d_pitched_async_retained`]; same lifetime rule as
/// [`GpuBackend::copy_d2h_async`] (no read of `dst` before the next sync).
pub fn copy_d2h_pitched_async(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    dst: &mut [u8],
    shape: Pitched,
    stream: u64,
) -> Result<()> {
    if let Some(native) = gpu.host_pitched() {
        return native.d2h(src, dst, shape, stream);
    }
    for r in 0..shape.height {
        let row = &mut dst[r * shape.host_pitch..][..shape.width];
        gpu.copy_d2h_async(src.offset(r * shape.dev_pitch), row, stream)?;
    }
    Ok(())
}
