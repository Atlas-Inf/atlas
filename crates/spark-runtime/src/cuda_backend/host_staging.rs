// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side copy staging: the H2D enqueue shared by both async entry points,
//! the D2H trace tick, and the page-locked-source tripwire.
//!
//! Split out of `gpu_impl.rs` for the file-size cap. The seam is a real one —
//! none of these is a `GpuBackend` method, and none touches `AtlasCudaBackend`
//! state; they are the free functions the impl's copy methods call. They also
//! had to move for a second reason: `gpu_impl.rs` is one `impl GpuBackend`
//! block, which Rust forbids splitting, so the only lines that CAN leave it are
//! the ones outside the block — and in the PR's merge with main it was at 514
//! LoC with a 500 cap while neither parent was over (main +27, the branch +21,
//! a plain sum).

use std::ffi::c_void;

use anyhow::{Result, bail};

use crate::gpu::DevicePtr;

use super::cuMemcpyHtoDAsync_v2;

/// D2H call counter + one-shot caller identification
/// (`ATLAS_D2H_TRACE=<N>`: log a backtrace on the Nth call, and the running
/// count on every 10000th).
///
/// Every `copy_d2h*` below pairs its async copy with a `cuStreamSynchronize`,
/// so each call BLOCKS the host until the GPU drains. An nsys trace of a 1K
/// Laguna prefill counted 32,343 D2H + 32,533 syncs inside the prefill span,
/// accounting for 212.8 ms of 306 ms of GPU starvation (58% idle). This exists
/// to name whoever is issuing them.
pub(super) static D2H_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(super) fn d2h_trace_tick() {
    use std::sync::atomic::Ordering;
    let n = D2H_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let Ok(target) = std::env::var("ATLAS_D2H_TRACE") else {
        return;
    };
    let target: u64 = target.parse().unwrap_or(0);
    if target != 0 && n == target {
        tracing::warn!(
            "ATLAS_D2H_TRACE: call #{n} backtrace:\n{}",
            std::backtrace::Backtrace::force_capture()
        );
    }
    if n.is_multiple_of(10_000) {
        tracing::warn!("ATLAS_D2H_TRACE: {n} D2H copies so far (each forces a stream sync)");
    }
}

/// Enqueue an H2D copy on `stream` and return without waiting. Shared by both
/// async H2D entry points so the two differ ONLY in the ordering they add
/// afterwards, never in the copy itself.
pub(super) fn h2d_enqueue(src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
    let status =
        unsafe { cuMemcpyHtoDAsync_v2(dst.0, src.as_ptr() as *const c_void, src.len(), stream) };
    if status != 0 {
        bail!("cuMemcpyHtoDAsync_v2 failed: status {status}");
    }
    Ok(())
}

/// Say once, loudly, that a page-locked buffer reached the transient H2D path.
///
/// This is the tripwire the whole `pinned_hosts` registry exists to arm. It is a
/// warning and not a `bail!` because the copy is still CORRECT — the sync above
/// restores the guarantee — but it is a real, silent latency regression, and the
/// call site almost certainly wants `copy_h2d_async_retained` instead.
pub(super) fn warn_pinned_transient_source() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "copy_h2d_async was handed a PAGE-LOCKED source. That copy is genuinely \
             asynchronous, so the promise that the caller may drop the buffer on return \
             is now being paid for with a cuStreamSynchronize on every such call. If the \
             source outlives the next sync, switch the call site to \
             copy_h2d_async_retained; if it does not, this sync is what keeps it from \
             being a use-after-free."
        );
    });
}
