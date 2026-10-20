// SPDX-License-Identifier: AGPL-3.0-only
//! Large-M BF16 projections through CUTLASS tiles with an L2-aware CTA raster.
//!
//! cuBLASLt's heuristic picks an sm80 kernel on GB10 that plateaus near
//! 62 TFLOPS; at K >= 4096 an unswizzled raster re-streams every weight strip
//! from DRAM per row of tiles. `examples/bf16_gemm_bench` measured the configs
//! below at 80-92 TFLOPS on the GLM prefill shapes (M = 4096).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::sync::OnceLock;

/// Rows below this keep cuBLASLt (decode/verify widths).
const MIN_ROWS: u32 = 256;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_BF16_GEMM_CUTLASS").as_deref() == Ok("1"))
}

/// `spark_runtime::cutlass::bf16_gemm_tuned` config for an `[n, k]` weight:
/// N <= 512 or K <= 128 take 128x128x64 (swizzle 8: one launch reads A once
/// for up to four N tiles), other narrow N 128x256 tiles (swizzle 8), K >= 8192
/// 128x256 (swizzle 4), short K the unswizzled 128x128x32, everything else
/// 128x128x64. Every config keeps warp K == CTA K (sequential k16 MMAs, no
/// split-K), so for [`bf16_gemm_matches_pipelined`] K all are bit-identical
/// to each other and to the pipelined dense GEMM (`examples/bf16_gemm_bench`
/// counts differing outputs); any other K keeps the picks it always had.
fn config(n: u32, k: u32) -> u32 {
    if bf16_gemm_matches_pipelined(k) && (n <= 512 || k <= 128) {
        9
    } else if n <= 1536 {
        5
    } else if k >= 8192 {
        4
    } else if k <= 2048 {
        0
    } else {
        9
    }
}

/// Row-major `out[m, n] = act[m, k] @ weight[n, k]^T`, BF16 in/out with FP32
/// accumulation: CUTLASS for large M when `ATLAS_BF16_GEMM_CUTLASS=1`, else
/// (or if CUTLASS rejects the operands before launching anything) cuBLASLt.
pub fn bf16_gemm(
    act: DevicePtr,
    weight: u64,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // ATLAS_QWEN4EXP_PREFILL_ROWINV: an in-order k-chain inside the pass.
    if super::qwen4exp_rowinv::try_bf16_gemm(act, weight, out, [m, n, k], stream)? {
        return Ok(());
    }
    if enabled() && m >= MIN_ROWS {
        let cfg = config(n, k);
        let launched = run_blocks(
            row_blocks(m, k, cfg),
            |row, rows| {
                spark_runtime::cutlass::bf16_gemm_tuned(
                    act.0 + u64::from(row) * u64::from(k) * 2,
                    weight,
                    out.0 + u64::from(row) * u64::from(n) * 2,
                    rows,
                    n,
                    k,
                    k,
                    n,
                    cfg,
                    stream,
                )
            },
            spark_runtime::cutlass::rejected_before_launch,
        )?;
        if launched {
            return Ok(());
        }
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            tracing::warn!(
                "bf16_gemm: CUTLASS rejected {m}x{n}x{k} config {cfg}; such shapes run in \
                 cuBLASLt (reported once)"
            )
        });
    }
    spark_runtime::cublaslt::bf16_gemm_act_weight_t(act.0, weight, out.0, m, n, k, stream)
}

/// Launches every row block. `Ok(false)` means the backend `rejected` the
/// operands before launching anything. That depends only on the shape and the
/// operands' alignment, which every row block shares, so the first block
/// decides and the caller may run the whole GEMM in another backend. Any other failure is
/// returned: the cuBLASLt rerun rounds differently from the CUTLASS tiles, so
/// recomputing after a failed launch made some calls of one shape differ from
/// the rest.
fn run_blocks(
    blocks: impl Iterator<Item = (u32, u32)>,
    mut launch: impl FnMut(u32, u32) -> Result<()>,
    rejected: impl Fn(&anyhow::Error) -> bool,
) -> Result<bool> {
    for (row, rows) in blocks {
        match launch(row, rows) {
            Ok(()) => {}
            Err(error) if rejected(&error) => return Ok(false),
            Err(error) => return Err(error),
        }
    }
    Ok(true)
}

/// Activation bytes the unswizzled raster (config 0) keeps L2-resident while
/// every column of tiles re-reads them: 4096 rows at K 2048. Past this (an
/// 8K prefill chunk) each re-read goes to DRAM, ~7x slower, so wider calls
/// run as consecutive row blocks (rows are independent: identical output).
const UNSWIZZLED_ACT_BYTES: u64 = 16 << 20;

fn row_blocks(m: u32, k: u32, cfg: u32) -> impl Iterator<Item = (u32, u32)> {
    let limit = ((UNSWIZZLED_ACT_BYTES / (u64::from(k) * 2)) as u32 / 128 * 128).max(128);
    // A few scheduling rows past the limit (a 4100-row chunk) stay whole
    // rather than pay a tiny tail launch that re-reads the weight.
    let block = if cfg == 0 && m > limit + 64 {
        m.div_ceil(m.div_ceil(limit)).next_multiple_of(128)
    } else {
        m
    };
    (0..m.div_ceil(block)).map(move |i| (i * block, block.min(m - i * block)))
}

/// Whether [`bf16_gemm`] routes `m` rows through CUTLASS.
pub fn bf16_gemm_cutlass_rows(m: u32) -> bool {
    enabled() && m >= MIN_ROWS
}

/// Whether the CUTLASS configs and `dense_gemm_bf16_pipelined` accumulate a
/// `k`-wide row in the same k16 MMA steps. CUTLASS tiles a K residue first
/// (per config K tile, 32 or 64) and the pipelined kernel zero-fills it last,
/// so only whole k16 steps group alike: measured 0 differing outputs at K
/// 16..4096 with K % 16 == 0, and differences at K 40, 72, 104, 120, 4040.
pub fn bf16_gemm_matches_pipelined(k: u32) -> bool {
    k.is_multiple_of(16)
}

/// Exact E4M3 -> BF16 widening of `count` values (`count % 8 == 0`).
pub fn fp8_e4m3_to_bf16(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    dst: DevicePtr,
    count: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        count.is_multiple_of(8),
        "E4M3 widening needs count % 8 == 0 ({count})"
    );
    let kernel = gpu
        .op_cache()
        .kernel(gpu, "fp8_e4m3_to_bf16", "fp8_e4m3_to_bf16")?;
    let count8 = (count / 8) as u64;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(u32::try_from(count8)?, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u64(count8)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::{bf16_gemm_matches_pipelined as matches_pipelined, config, row_blocks, run_blocks};
    use spark_runtime::cutlass::{RejectedBeforeLaunch, rejected_before_launch};

    /// Launches `blocks` with the n-th call answering `outcome(n)`; returns
    /// the verdict and how many blocks were launched.
    fn launch_all(
        blocks: u32,
        outcome: impl Fn(u32) -> anyhow::Result<()>,
    ) -> (anyhow::Result<bool>, u32) {
        let mut calls = 0;
        let verdict = run_blocks(
            (0..blocks).map(|i| (i * 128, 128)),
            |row, _| {
                calls += 1;
                outcome(row / 128)
            },
            rejected_before_launch,
        );
        (verdict, calls)
    }

    #[test]
    fn only_a_rejection_before_launch_allows_another_backend() {
        let rejected = || Err(anyhow::Error::new(RejectedBeforeLaunch).context("status 1"));
        // Every block launched.
        let (verdict, calls) = launch_all(3, |_| Ok(()));
        assert!(verdict.unwrap());
        assert_eq!(calls, 3);
        // Rejected operands: nothing more is launched and the caller may fall back.
        let (verdict, calls) = launch_all(3, |_| rejected());
        assert!(!verdict.unwrap());
        assert_eq!(calls, 1);
        // A failed launch is an error on any block, never a silent rerun
        // (a rerun in cuBLASLt rounds differently from the CUTLASS tiles).
        for failing in 0..3 {
            let (verdict, calls) = launch_all(3, |block| {
                if block == failing {
                    anyhow::bail!("launch failed, CUTLASS status 7")
                }
                Ok(())
            });
            assert!(verdict.unwrap_err().to_string().contains("launch failed"));
            assert_eq!(calls, failing + 1);
        }
    }

    #[test]
    fn unswizzled_calls_split_into_l2_resident_row_blocks() {
        let blocks: Vec<_> = row_blocks(8188, 2048, 0).collect();
        assert_eq!(blocks, [(0, 4096), (4096, 4092)]);
        assert_eq!(row_blocks(4100, 2048, 0).collect::<Vec<_>>(), [(0, 4100)]);
        assert_eq!(row_blocks(16388, 2048, 0).count(), 5);
        // Swizzled configs keep one launch.
        assert_eq!(row_blocks(8188, 4096, 9).collect::<Vec<_>>(), [(0, 8188)]);
    }

    #[test]
    fn config_follows_measured_glm_shapes() {
        assert_eq!(config(1536, 4096), 5); // q_a
        assert_eq!(config(576, 4096), 5); // kv_a with RoPE
        assert_eq!(config(512, 4096), 9); // GLM kv_a
        assert_eq!(config(288, 4096), 9); // GLM router
        assert_eq!(config(128, 4096), 9); // index wk / kpool_gate
        assert_eq!(config(32, 4096), 9); // index weights_proj
        assert_eq!(config(4096, 8192), 4); // o
        assert_eq!(config(8192, 1536), 0); // q_b
        assert_eq!(config(4096, 2048), 0); // shared down
        assert_eq!(config(4096, 128), 9); // KDA f_b / g_b
        assert_eq!(config(2048, 4096), 9); // shared gate/up
    }

    #[test]
    fn narrow_picks_need_whole_k16_steps_to_stay_bit_identical() {
        // K % 16 == 0 (K % 64 != 0 included): 128x128x64 where measured faster.
        assert_eq!(config(128, 4000), 9);
        assert_eq!(config(4096, 112), 9);
        // A K residue is tiled first, per config K tile: keep the prior picks.
        assert_eq!(config(128, 4040), 5);
        assert_eq!(config(4096, 104), 0);
        assert!(matches_pipelined(128) && matches_pipelined(4096));
        assert!(!matches_pipelined(104) && !matches_pipelined(120));
    }
}
