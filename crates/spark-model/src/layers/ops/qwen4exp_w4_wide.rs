// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_W4_ROWS_WIDE=1`: the NVFP4 GEMVs of a 9+-row batched
//! verify (C8 x K=4: 24..36 rows) -- MoE router, attention K/V and o_proj --
//! in ONE `qwen4exp_w4_rows_wide` launch (`qwen4exp_w4_wide.cu`) instead of
//! 16-row `w4a16_gemv_batch16` chunks. Row `t` is `w4a16_gemv`'s bytes on
//! that row (as batch16's are), so nothing downstream moves and the switch
//! is rank-local. Rides `Qwen4ExpWideRows` (`ATLAS_QWEN4EXP_BATCH_FAST`).
//!
//! GB10, weights streamed from DRAM, us per call
//! (`scripts/dev/qwen4exp_w4_wide_bench.cu`):
//!
//! | shape (a rank, TP2) | M | batch16 chunks | wide |
//! |---|---|---|---|
//! | router 512 x 2560 | 24 / 32 / 36 | 34.6 / 42.8 / 51.1 | 13.7 / 18.4 / 18.4 |
//! | K/V 256 x 2560 | 24 / 32 / 36 | 32.8 / 39.0 / 47.2 | 12.3 / 12.3 / 12.3 |
//! | o_proj 2560 x 3072 | 24 / 32 / 36 | 121.6 / 160.5 / 183.8 | 43.2 / 57.8 / 71.9 |
//!
//! Up to 8 rows the narrow tiers stay (they are as fast or faster there).
//!
//! A launch error (it should have none: all shared memory is dynamic and the
//! runtime opts in past 48 KB) is logged once, turns the tier off for the
//! process, and the call falls back to the narrow chunking.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::try_kernel;
use crate::weight_map::QuantizedWeight;

/// Rows a CTA stages (the kernel's MT) and outputs a CTA tile (NT x W).
const CTA_ROWS: u32 = 8;
const CTA_OUTPUTS: u32 = 32;
const THREADS: u32 = 256;
/// Below this many rows the narrow tiers keep the call.
pub const W4_WIDE_MIN_ROWS: u32 = 9;
/// Rows a launch (more chunk; any count is exact, this bounds the grid).
pub const W4_WIDE_MAX_ROWS: u32 = 64;
/// The staged rows' shared memory cap (sm_121: 99 KB a block): K <= 3072.
const MAX_SMEM: u32 = 96 * 1024;

/// Set on the first launch error: the tier is off for the process.
static FAILED: AtomicBool = AtomicBool::new(false);

/// `ATLAS_QWEN4EXP_W4_ROWS_WIDE=1`, read once.
pub fn w4_rows_wide_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_W4_ROWS_WIDE").as_deref() == Ok("1"))
}

/// `qwen4exp_w4_rows_wide`, 0 when off.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpW4Wide {
    kernel: KernelHandle,
    sms: u32,
}

/// Dynamic shared bytes for `k`: 8 rows x ceil(K/512) x 2 KiB (FP32).
fn smem(k: u32) -> u32 {
    CTA_ROWS * div_ceil(k / 16, 32) * 2048
}

impl Qwen4ExpW4Wide {
    pub const OFF: Self = Self {
        kernel: KernelHandle(0),
        sms: 0,
    };

    /// Resolved when the switch is on and the target ships the module.
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        if !w4_rows_wide_requested() {
            return Self::OFF;
        }
        Self {
            kernel: try_kernel(gpu, "qwen4exp_w4_wide", "qwen4exp_w4_rows_wide"),
            sms: gpu.sm_count().unwrap_or(0),
        }
    }

    /// Whether `m` rows of a `[n, k]` weight take this tier.
    pub fn serves(&self, m: u32, k: u32) -> bool {
        self.kernel.0 != 0
            && self.sms > 0
            && m >= W4_WIDE_MIN_ROWS
            && k.is_multiple_of(16)
            && smem(k) <= MAX_SMEM
            && !FAILED.load(Ordering::Relaxed)
    }

    /// `m` (<= [`W4_WIDE_MAX_ROWS`]) rows of `input` (`[m, k]`) through the
    /// NVFP4 `w` into `output` (`[m, n]`). `Ok(false)`: not served, or the
    /// launch failed (logged, tier off) -- the caller runs its narrow path.
    #[allow(clippy::too_many_arguments)]
    pub fn try_launch(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        w: &QuantizedWeight,
        output: DevicePtr,
        (m, n, k): (u32, u32, u32),
        stream: u64,
    ) -> bool {
        if !self.serves(m, k) || m > W4_WIDE_MAX_ROWS {
            return false;
        }
        // Persistent: every CTA resident (one an SM), shared out over the
        // row groups, at most one a tile.
        let groups = div_ceil(m, CTA_ROWS);
        let grid_x = div_ceil(n, CTA_OUTPUTS).min(self.sms / groups).max(1);
        let launched: Result<()> = KernelLaunch::new(gpu, self.kernel)
            .grid([grid_x, groups, 1])
            .block([THREADS, 1, 1])
            .shared_mem(smem(k))
            .arg_ptr(input)
            .arg_ptr(w.weight)
            .arg_ptr(w.weight_scale)
            .arg_f32(w.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream);
        match launched {
            Ok(()) => true,
            Err(e) => {
                if !FAILED.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        "qwen4exp_w4_rows_wide: launch failed ({m} rows, n={n}, k={k}, \
                         grid=[{grid_x},{groups},1], smem={}): {e:#}; tier off, narrow GEMVs",
                        smem(k)
                    );
                }
                false
            }
        }
    }
}
