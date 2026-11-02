// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_W4_ROWS=1`: the qwen4_exp MTP drafter's NVFP4 draft head
//! (`layers/qwen4exp_draft_head.rs`) over 4..8 rows with the persistent
//! `qwen4exp_w4_rows{M}` (`qwen4exp_w4_rows.cu`) instead of the scalar
//! `w4a16_gemv_batch{M}` tiers. Row `t` is `w4a16_gemv`'s bytes on that row
//! (as the tiers' are), so drafts, confidences and acceptance do not move;
//! the switch is rank-local.
//!
//! GB10, the draft head half a rank under `ATLAS_QWEN4EXP_MTP_DRAFT_TP`
//! (50,000 x 2560, 72 MB), us per call (`scripts/dev/qwen4exp_w4_rows_bench.cu`):
//!
//! | M | 4 | 5 | 6 | 7 | 8 |
//! |---|---|---|---|---|---|
//! | `w4a16_gemv_batch{M}` | 349 | 390 | 535 | 563 | 653 |
//! | `qwen4exp_w4_rows{M}` | 343 | 344 | 342 | 340 | 378 |
//!
//! At 1..3 rows the narrow kernels stay (as fast there).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::try_kernel;
use crate::weight_map::QuantizedWeight;

/// Rows the persistent tier serves.
const MIN_ROWS: u32 = 4;
const MAX_ROWS: u32 = 8;
/// Threads a CTA (4 output groups of 64 lanes).
const THREADS: u32 = 256;
/// Outputs a tile (4 a thread x 4 output groups).
const NPB: u32 = 16;
/// The kernel's lane steps: K/16 <= 2 x 128 chunks.
const LANE_STEPS: u32 = 2;
/// GB10's largest dynamic shared memory a CTA may opt into.
const MAX_SMEM: u32 = 99 * 1024;

/// Dynamic shared memory of `qwen4exp_w4_rows{m}` (all of its shared memory;
/// the launcher opts in past 48 KB from this size): the m rows staged for
/// the whole K, the two reduction buffers, the E2M1 LUT.
pub(crate) fn w4_rows_smem(m: u32) -> u32 {
    m * LANE_STEPS * 4096 + (2 * m * NPB * 2 + 16) * 4
}

/// `ATLAS_QWEN4EXP_W4_ROWS=1`, read once.
pub fn w4_rows_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_W4_ROWS").as_deref() == Ok("1"))
}

/// The `qwen4exp_w4_rows{M}` handles (index M), all 0 when off.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpW4Rows {
    rows: [KernelHandle; MAX_ROWS as usize + 1],
    sms: u32,
}

impl Qwen4ExpW4Rows {
    pub const OFF: Self = Self {
        rows: [KernelHandle(0); MAX_ROWS as usize + 1],
        sms: 0,
    };

    /// Resolved when the switch is on and the target ships the module.
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        if !w4_rows_requested() {
            return Self::OFF;
        }
        // Literal names (the kernel-name check reads them).
        let mut rows = Self::OFF.rows;
        rows[4] = try_kernel(gpu, "qwen4exp_w4_rows", "qwen4exp_w4_rows4");
        rows[5] = try_kernel(gpu, "qwen4exp_w4_rows", "qwen4exp_w4_rows5");
        rows[6] = try_kernel(gpu, "qwen4exp_w4_rows", "qwen4exp_w4_rows6");
        rows[7] = try_kernel(gpu, "qwen4exp_w4_rows", "qwen4exp_w4_rows7");
        rows[8] = try_kernel(gpu, "qwen4exp_w4_rows", "qwen4exp_w4_rows8");
        Self {
            rows,
            sms: gpu.sm_count().unwrap_or(0),
        }
    }

    /// Whether `m` rows of a `[n, k]` weight take the persistent tier: the
    /// kernel resolved, `k / 16` even and at most 256 (two lane steps).
    pub fn serves(&self, m: u32, k: u32) -> bool {
        (MIN_ROWS..=MAX_ROWS).contains(&m)
            && self.rows[m as usize].0 != 0
            && self.sms > 0
            && k.is_multiple_of(32)
            && k <= 4096
            && !self.failed()
    }

    /// A launch of the tier failed once: every later call takes the narrow
    /// kernels (the caller's fallback), so a bad launch never fails a step.
    fn failed(&self) -> bool {
        FAILED.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `m` rows of `input` (`[m, k]`) through the NVFP4 `w` into `output`
    /// (`[m, n]`), row `t` `w4a16_gemv`'s bytes. Gate on [`Self::serves`].
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        w: &QuantizedWeight,
        output: DevicePtr,
        (m, n, k): (u32, u32, u32),
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(self.serves(m, k), "qwen4exp_w4_rows: {m} rows, k={k}");
        let smem = w4_rows_smem(m);
        anyhow::ensure!(
            smem <= MAX_SMEM,
            "qwen4exp_w4_rows{m}: {smem} B of shared memory"
        );
        KernelLaunch::new(gpu, self.rows[m as usize])
            .grid([self.sms.min(div_ceil(n, 16)), 1, 1])
            .block([THREADS, 1, 1])
            .shared_mem(smem)
            .arg_ptr(input)
            .arg_ptr(w.weight)
            .arg_ptr(w.weight_scale)
            .arg_f32(w.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)
    }

    /// [`Self::launch`] when [`Self::serves`], else `fallback`; a failed
    /// launch logs once, retires the tier for the process and runs
    /// `fallback` (the narrow kernels: the same bytes).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_or(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        w: &QuantizedWeight,
        output: DevicePtr,
        (m, n, k): (u32, u32, u32),
        stream: u64,
        fallback: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        if !self.serves(m, k) {
            return fallback();
        }
        match self.launch(gpu, input, w, output, (m, n, k), stream) {
            Ok(()) => Ok(()),
            Err(e) => {
                FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!(
                    "qwen4exp_w4_rows{m} launch failed ({e:#}); ATLAS_QWEN4EXP_W4_ROWS retired, \
                     the draft head takes w4a16_gemv_batch{{M}} from here (same bytes)"
                );
                fallback()
            }
        }
    }
}

/// Set by the first failed launch ([`Qwen4ExpW4Rows::launch_or`]).
static FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
mod tests {
    use super::{MAX_ROWS, MAX_SMEM, MIN_ROWS, w4_rows_smem};

    /// The 2026-10-08 pair failure: at M = 6 the rows were exactly 48 KB of
    /// dynamic shared memory with static memory on top, so the launcher (which
    /// opts in past 48 KB of DYNAMIC memory) did not opt in and the launch
    /// overflowed the default limit. All of it is dynamic now; every tier must
    /// fit the opt-in ceiling, and any size the launcher does not opt in for
    /// is under the default limit by itself.
    #[test]
    fn every_tier_fits_shared_memory() {
        for m in MIN_ROWS..=MAX_ROWS {
            let smem = w4_rows_smem(m);
            assert!(smem <= MAX_SMEM, "M={m}: {smem} B");
            assert!(smem != 48 * 1024, "M={m}: exactly at the opt-in threshold");
        }
        assert_eq!(w4_rows_smem(6), 6 * 2 * 4096 + (2 * 6 * 16 * 2 + 16) * 4);
        assert!(w4_rows_smem(6) > 48 * 1024, "M=6 takes the opt-in");
        assert!(w4_rows_smem(5) < 48 * 1024);
    }
}
