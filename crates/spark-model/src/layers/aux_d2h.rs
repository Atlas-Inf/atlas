// SPDX-License-Identifier: AGPL-3.0-only

//! Aux-snapshot readback through a reused page-locked stage
//! (`ATLAS_QWEN4EXP_AUX_PINNED=1`, default off).
//!
//! A prefix-cache checkpoint of a qwen4_exp prompt reads every QSA layer's
//! pooled indexer keys and the PLE conv carry back to the host -- 12 x ~1 MB
//! and ~0.4 MB at 16K. Into pageable memory those copies ran at 0.23-0.40 GB/s
//! on the pair (`cuMemcpyDtoHAsync`, 3.2-4.4 ms each; nsys `sqpf-p3s7-r0`,
//! two saves on the TTFT path: ~45 ms each), against ~0.03 ms for a 1 MB copy
//! into page-locked memory. Here the copy lands in one page-locked stage per
//! thread (grown, never shrunk) and is copied on into the caller's buffer.
//! Same bytes, same ordering (the copy is still enqueued on `stream` and
//! synchronised before the host reads it).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::cell::Cell;

/// `ATLAS_QWEN4EXP_AUX_PINNED=1`.
pub fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_AUX_PINNED").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Copies below this stay pageable: the stage's extra host copy would cost
/// more than it saves.
const MIN_BYTES: usize = 64 << 10;

thread_local! {
    /// `(ptr, bytes)` of this thread's page-locked stage.
    static STAGE: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

/// `src` -> `dst`, ordered after the work already on `stream`, as
/// `GpuBackend::copy_d2h_on_stream` (which it is when the switch is off).
pub fn copy(gpu: &dyn GpuBackend, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
    copy_via(gpu, src, dst, stream, requested())
}

fn copy_via(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    dst: &mut [u8],
    stream: u64,
    pinned: bool,
) -> Result<()> {
    if !pinned || dst.len() < MIN_BYTES {
        return gpu.copy_d2h_on_stream(src, dst, stream);
    }
    let ptr = stage(gpu, dst.len())?;
    // SAFETY: `stage` returned a live page-locked allocation of at least
    // `dst.len()` bytes (zeroed at allocation, so initialised), owned by this
    // thread and used by nothing else while this borrow lives.
    let staged = unsafe { std::slice::from_raw_parts_mut(ptr, dst.len()) };
    gpu.copy_d2h_on_stream(src, staged, stream)?;
    dst.copy_from_slice(staged);
    Ok(())
}

/// This thread's stage, at least `bytes`. Every copy through it is
/// synchronous, so the old stage is idle when it is replaced.
fn stage(gpu: &dyn GpuBackend, bytes: usize) -> Result<*mut u8> {
    let (ptr, size) = STAGE.with(Cell::get);
    if size >= bytes {
        return Ok(ptr as *mut u8);
    }
    if ptr != 0 {
        gpu.free_host_pinned(ptr as *mut u8, size)?;
        STAGE.with(|s| s.set((0, 0)));
    }
    let want = bytes.next_power_of_two();
    let p = gpu.alloc_host_pinned(want)?;
    STAGE.with(|s| s.set((p as usize, want)));
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    /// The staged copy returns the device bytes, below and above the
    /// threshold, and reuses the stage once it is big enough.
    #[test]
    fn staged_copy_returns_the_device_bytes() {
        let gpu = MockGpuBackend::new();
        for n in [100usize, MIN_BYTES, 3 * MIN_BYTES + 7, 2 * MIN_BYTES] {
            let want: Vec<u8> = (0..n).map(|i| (i * 31 % 251) as u8).collect();
            let src = gpu.alloc(n).unwrap();
            gpu.copy_h2d(&want, src).unwrap();
            let mut got = vec![0u8; n];
            copy_via(&gpu, src, &mut got, 0, true).unwrap();
            assert_eq!(got, want, "{n} bytes");
        }
        assert_eq!(
            STAGE.with(Cell::get).1,
            (3 * MIN_BYTES + 7).next_power_of_two()
        );
    }

    /// On CUDA: the staged copy returns the same bytes as the pageable one,
    /// and the time of each for a checkpoint's worth of QSA blobs (12 x
    /// 1.025 MB) -- `--ignored --nocapture` on a GB10.
    #[test]
    #[ignore]
    fn staged_copy_matches_pageable_on_cuda() {
        let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
            .expect("build with ATLAS_TARGET_MODEL='*'");
        let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules)
            .expect("CUDA backend");
        let g: &dyn GpuBackend = &gpu;
        let stream = g.default_stream();
        let n = 1_025 * 1024;
        let want: Vec<u8> = (0..n).map(|i| (i * 7 % 253) as u8).collect();
        let src = g.alloc(n).unwrap();
        g.copy_h2d(&want, src).unwrap();
        for pinned in [false, true, false, true] {
            let t = std::time::Instant::now();
            for _ in 0..12 {
                let mut got = vec![0u8; n];
                copy_via(g, src, &mut got, stream, pinned).unwrap();
                assert!(got == want, "pinned={pinned}: bytes differ");
            }
            println!(
                "12 x {n} B aux readback, pinned={pinned}: {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}
