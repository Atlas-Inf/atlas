// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_BATCH_FAST=1`: the exact GEMVs of a 9..32-row batched step
//! (C4..C8 sequences x a K=4 verify) in one weight pass instead of 8-row
//! (Q+gate: 4-row) chunks (`model/qwen4exp_batch_fast.rs`).
//!
//! | projection | single-row kernel | rows a launch: before -> now |
//! |---|---|---|
//! | GDN qkvz / out_proj, LM head (BF16) | `dense_gemv_bf16` | 8 -> `qwen4exp_bf16_rows16/32` |
//! | attention Q+gate (NVFP4) | `w4a16_gemv_qg` | 4 -> `qwen4exp_qg_rows16/32` |
//! | MoE router, attention K/V (NVFP4) | `w4a16_gemv` | 8 -> `w4a16_gemv_batch16` |
//!
//! Every row of every launch here is byte-identical to the single-row kernel
//! on that row (the kernels' notes; `scripts/dev/qwen4exp_wide_rows_bench.cu`
//! checks M = 1..32), so the chunking below is launch shape only and
//! changes no collective: the ranks need not agree on it.
//!
//! Chunking, from the bench (GB10, weights streamed from DRAM): at most
//! [`WIDE_MIN_ROWS`] rows stay on the narrow kernels, which are DRAM-bound
//! there; past it a chunk takes the narrowest wide tier that covers it, up
//! to 32 rows. 16 rows of GDN qkvz cost what 8 did (177 us), 32 rows 278 us
//! against 764 for four 8-row passes; the LM head 3.85 ms against 10.2.
//! `w4a16_gemv_batch32` is slower than two `batch16` passes at these shapes
//! and is not used.
//!
//! `ATLAS_QWEN4EXP_ROWS32_TILE=1` serves the 32-row BF16 tier with the
//! register-tiled `qwen4exp_bf16_rows32t` (`qwen4exp_rows32_tile.cu`): same
//! bytes, 16 rows x 4 outputs a thread and fused multiply-adds where they
//! are provably exact. GB10, checkpoint weights, M = 32: LM head half 4.0 ->
//! 2.69 ms, GDN qkvz 278 -> 190 us, out_proj 120 -> 83 us
//! (`scripts/dev/qwen4exp_rows32_tile_bench.cu`). Rank-local: the bytes do
//! not change.
//!
//! `ATLAS_QWEN4EXP_W4_ROWS_WIDE=1` serves the NVFP4 rows past 8 with one
//! `qwen4exp_w4_rows_wide` launch (`ops::Qwen4ExpW4Wide`) instead of 16-row
//! `w4a16_gemv_batch16` chunks: same bytes, 2.7-3.2x at 24..36 rows.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::try_kernel;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Row capacities of the wide tiers, narrowest first (the kernels' MAX_M).
const WIDE_ROWS: [u32; 2] = [16, 32];
/// `(outputs, threads)` a CTA of each tier: the kernels' NPB, and 64 threads
/// an output (the 16-row tiers) or a pair of outputs (the 32-row pair tiers).
const BF16_CTA: [(u32, u32); 2] = [(4, 256), (16, 512)];
const QG_CTA: [(u32, u32); 2] = [(4, 256), (8, 256)];
/// Chunks of at most this many rows keep the narrow kernels.
const WIDE_MIN_ROWS: u32 = 8;
/// `qwen4exp_bf16_rows32t`: outputs and threads a CTA, dynamic shared bytes
/// (a double-buffered K-step of 32 rows).
const TILE_CTA: (u32, u32, u32) = (16, 512, 2 * 32 * 64 * 16);

/// `ATLAS_QWEN4EXP_ROWS32_TILE=1`, read once: the 32-row BF16 tier runs the
/// register-tiled kernel (module docs).
pub fn rows32_tile_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_ROWS32_TILE").as_deref() == Ok("1"))
}

/// The kernel handles, all 0 unless the lane is on for a qwen4_exp model and
/// the target ships them (every caller then chunks exactly as before).
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpWideRows {
    bf16: [KernelHandle; 2],
    /// `qwen4exp_bf16_rows32t`, serving the 32-row BF16 tier when resolved.
    bf16_tile: KernelHandle,
    qg: [KernelHandle; 2],
    w4a16_16: KernelHandle,
    /// `ATLAS_QWEN4EXP_W4_ROWS_WIDE`: the one-launch NVFP4 tier past 8 rows.
    w4_wide: super::Qwen4ExpW4Wide,
}

/// The next launch over `remaining` rows: its row count and the wide tier
/// serving it (`present[i]`: tier `WIDE_ROWS[i]` resolved), or `None` for
/// the narrow kernel at up to `narrow_cap` rows.
fn next_chunk(remaining: u32, present: [bool; 2], narrow_cap: u32) -> (u32, Option<usize>) {
    if remaining > WIDE_MIN_ROWS
        && let Some(widest) = (0..WIDE_ROWS.len()).rev().find(|&i| present[i])
    {
        let take = remaining.min(WIDE_ROWS[widest]);
        let tier = (0..=widest).find(|&i| present[i] && WIDE_ROWS[i] >= take);
        return (take, tier);
    }
    (remaining.min(narrow_cap), None)
}

impl Qwen4ExpWideRows {
    pub const OFF: Self = Self {
        bf16: [KernelHandle(0); 2],
        bf16_tile: KernelHandle(0),
        qg: [KernelHandle(0); 2],
        w4a16_16: KernelHandle(0),
        w4_wide: super::Qwen4ExpW4Wide::OFF,
    };

    pub fn resolve(gpu: &dyn GpuBackend, model_type: &str) -> Self {
        // Only looked up when wanted, so no other model or default boot asks.
        if !(crate::model::qwen4exp_batch_fast::requested() && model_type == "qwen4_exp") {
            return Self::OFF;
        }
        // Literal names, parallel to `WIDE_ROWS` (the kernel-name check reads them).
        Self {
            bf16: [
                try_kernel(gpu, "qwen4exp_wide_rows", "qwen4exp_bf16_rows16"),
                try_kernel(gpu, "qwen4exp_wide_rows", "qwen4exp_bf16_rows32"),
            ],
            bf16_tile: if rows32_tile_requested() {
                try_kernel(gpu, "qwen4exp_rows32_tile", "qwen4exp_bf16_rows32t")
            } else {
                KernelHandle(0)
            },
            qg: [
                try_kernel(gpu, "qwen4exp_wide_rows", "qwen4exp_qg_rows16"),
                try_kernel(gpu, "qwen4exp_wide_rows", "qwen4exp_qg_rows32"),
            ],
            w4a16_16: try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch16"),
            w4_wide: super::Qwen4ExpW4Wide::resolve(gpu),
        }
    }

    fn present(handles: &[KernelHandle; 2]) -> [bool; 2] {
        handles.map(|h| h.0 != 0)
    }

    /// `m` rows of `input` (`[m, k]` BF16) through the BF16 `weight` into
    /// `output` rows `out_stride` elements apart, each row `dense_gemv_bf16`'s
    /// bytes: the wide tiers past [`WIDE_MIN_ROWS`] rows, `narrow`
    /// (`dense_gemv_bf16_batchm`) in 8-row chunks otherwise.
    #[allow(clippy::too_many_arguments)]
    pub fn dense_rows(
        &self,
        gpu: &dyn GpuBackend,
        narrow: KernelHandle,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        (m, n, k, out_stride): (u32, u32, u32, u32),
        stream: u64,
    ) -> Result<()> {
        let mut present = Self::present(&self.bf16);
        present[1] |= self.bf16_tile.0 != 0;
        let mut first = 0u32;
        while first < m {
            let (rows, tier) = next_chunk(m - first, present, super::DENSE_GEMV_BATCHM_MAX_M);
            let (a, c) = (
                input.offset(first as usize * k as usize * 2),
                output.offset(first as usize * out_stride as usize * 2),
            );
            let Some(t) = tier else {
                super::dense_gemv_batchm(
                    gpu, narrow, a, weight, c, rows, n, k, out_stride, stream,
                )?;
                first += rows;
                continue;
            };
            ensure!(
                k.is_multiple_of(8),
                "qwen4exp_bf16_rows: k={k} not a multiple of 8"
            );
            let tile = t == 1 && self.bf16_tile.0 != 0;
            let ((npb, threads), smem, kernel) = if tile {
                let (npb, threads, smem) = TILE_CTA;
                ((npb, threads), smem, self.bf16_tile)
            } else {
                (BF16_CTA[t], 0, self.bf16[t])
            };
            KernelLaunch::new(gpu, kernel)
                .grid([div_ceil(n, npb), 1, 1])
                .block([threads, 1, 1])
                .shared_mem(smem)
                .arg_ptr(a)
                .arg_ptr(weight.weight)
                .arg_ptr(c)
                .arg_u32(rows)
                .arg_u32(n)
                .arg_u32(k)
                .arg_u32(out_stride)
                .launch(stream)?;
            first += rows;
        }
        Ok(())
    }

    /// The next Q+gate launch over `remaining` rows: its row count and, past
    /// [`WIDE_MIN_ROWS`] rows, the wide tier; `None` for the caller's
    /// `w4a16_gemv_qg_batch4/3/2` / `w4a16_gemv_qg` chunk.
    pub fn qg_chunk(&self, remaining: u32) -> (u32, Option<usize>) {
        next_chunk(remaining, Self::present(&self.qg), 4)
    }

    /// One wide Q+gate launch ([`Self::qg_chunk`]'s tier `t`): `rows` rows of
    /// `input` (`[rows, k]`) through the interleaved NVFP4 Q+gate weight
    /// (`n` = heads x head_dim x 2), each output row deinterleaved `[Q | G]`
    /// at `output + r * n`, row `r` `w4a16_gemv_qg`'s bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn qg_launch(
        &self,
        gpu: &dyn GpuBackend,
        t: usize,
        input: DevicePtr,
        w: &QuantizedWeight,
        output: DevicePtr,
        (rows, n, k): (u32, u32, u32),
        (heads, head_dim): (u32, u32),
        stream: u64,
    ) -> Result<()> {
        ensure!(
            k.is_multiple_of(16) && rows <= WIDE_ROWS[t],
            "qwen4exp_qg_rows{}: {rows} rows, k={k}",
            WIDE_ROWS[t]
        );
        let (npb, threads) = QG_CTA[t];
        KernelLaunch::new(gpu, self.qg[t])
            .grid([div_ceil(n, npb), 1, 1])
            .block([threads, 1, 1])
            .arg_ptr(input)
            .arg_ptr(w.weight)
            .arg_ptr(w.weight_scale)
            .arg_f32(w.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(rows)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(n)
            .arg_u32(heads)
            .arg_u32(head_dim)
            .launch(stream)
    }

    /// Whether [`Self::w4a16_rows`] takes `m` rows of a `k`-wide NVFP4
    /// weight in one `qwen4exp_w4_rows_wide` launch.
    pub fn w4_wide_serves(&self, m: u32, k: u32) -> bool {
        self.w4_wide.serves(m, k)
    }

    /// `rows` rows of `input` (`[rows, k]`) through the NVFP4 `w` into
    /// `output` (`[rows, n]`), each row `w4a16_gemv`'s bytes: one
    /// `qwen4exp_w4_rows_wide` launch (up to 64 rows) past [`WIDE_MIN_ROWS`]
    /// rows when `ATLAS_QWEN4EXP_W4_ROWS_WIDE` is on, else 16-row
    /// `w4a16_gemv_batch16` launches there; the narrow scalar tiers (8,
    /// else 4 rows a launch) otherwise.
    #[allow(clippy::too_many_arguments)]
    pub fn w4a16_rows(
        &self,
        gpu: &dyn GpuBackend,
        narrow: &W4a16BatchmTiers,
        input: DevicePtr,
        w: &QuantizedWeight,
        output: DevicePtr,
        (rows, n, k): (u32, u32, u32),
        stream: u64,
    ) -> Result<()> {
        let narrow_cap = if narrow.scalar_kernel(8).0 != 0 { 8 } else { 4 };
        // `batch16` stands in the 16-row tier's slot; there is no 32.
        let present = [self.w4a16_16.0 != 0, false];
        let mut first = 0u32;
        while first < rows {
            let (a, c) = (
                input.offset(first as usize * k as usize * 2),
                output.offset(first as usize * n as usize * 2),
            );
            let wide = (rows - first).min(super::W4_WIDE_MAX_ROWS);
            if wide > WIDE_MIN_ROWS && self.w4_wide.try_launch(gpu, a, w, c, (wide, n, k), stream) {
                first += wide;
                continue;
            }
            let (m, tier) = next_chunk(rows - first, present, narrow_cap);
            let kernel = match tier {
                Some(_) => self.w4a16_16,
                None => narrow.scalar_kernel(m),
            };
            super::w4a16_gemv_batchm(gpu, kernel, a, w, c, m, n, k, stream)?;
            first += m;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{WIDE_MIN_ROWS, next_chunk};

    fn plan(rows: u32, present: [bool; 2], narrow: u32) -> Vec<(u32, Option<usize>)> {
        let mut out = Vec::new();
        let mut left = rows;
        while left > 0 {
            let c = next_chunk(left, present, narrow);
            assert!(c.0 >= 1 && c.0 <= left);
            out.push(c);
            left -= c.0;
        }
        out
    }

    #[test]
    fn narrow_rows_keep_the_narrow_kernel() {
        for rows in 1..=WIDE_MIN_ROWS {
            assert_eq!(plan(rows, [true, true], 8), vec![(rows, None)]);
        }
        assert_eq!(plan(8, [true, true], 4), vec![(4, None), (4, None)]);
    }

    #[test]
    fn wide_rows_take_the_narrowest_covering_tier() {
        assert_eq!(plan(9, [true, true], 8), vec![(9, Some(0))]);
        assert_eq!(plan(16, [true, true], 8), vec![(16, Some(0))]);
        assert_eq!(plan(17, [true, true], 8), vec![(17, Some(1))]);
        assert_eq!(plan(32, [true, true], 8), vec![(32, Some(1))]);
        assert_eq!(plan(40, [true, true], 8), vec![(32, Some(1)), (8, None)]);
        assert_eq!(
            plan(50, [true, true], 8),
            vec![(32, Some(1)), (18, Some(1))]
        );
    }

    #[test]
    fn missing_tiers_fall_back() {
        assert_eq!(
            plan(32, [true, false], 8),
            vec![(16, Some(0)), (16, Some(0))]
        );
        assert_eq!(plan(12, [false, true], 8), vec![(12, Some(1))]);
        assert_eq!(
            plan(20, [false, false], 8),
            vec![(8, None), (8, None), (4, None)]
        );
        assert_eq!(
            plan(10, [false, false], 4),
            vec![(4, None), (4, None), (2, None)]
        );
    }
}
