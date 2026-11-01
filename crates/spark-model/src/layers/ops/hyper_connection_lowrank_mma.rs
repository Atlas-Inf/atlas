// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_HC_MMA=1`: the decode-shaped mHC collapse's down and
//! finish on tensor cores (`hc_mma_down` + `hc_mma_finish`,
//! kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_hc_mma.cu) for every
//! 1..=32-row collapse the vectorized split path serves, the serial T = 1
//! decode included.
//!
//! NOT bit-identical to the FP32 `_vec*` kernels: this is exactness
//! contract (b), a new numerics baseline that is ROW-INVARIANT -- a row's
//! `low`, `inj` and `y` bytes are the same whatever rows share its launch and
//! wherever it sits among them (scripts/dev/qwen4exp_hc_mma_bench.cu `check`
//! byte-compares every row of T = 1..32 batches at three offsets against the
//! row alone; `hc_mma_tests` repeats it from Rust). So speculative verify
//! still equals serial decode bitwise when both run under the switch, and the
//! model-quality question is answered by greedy + quality probes, not here.
//! Error against FP64 is ~10x the FP32 kernels' (activations enter as
//! hi + lo bf16 pairs, ~16 bits) and well inside one bf16 ulp of `y`.
//!
//! Applies only through `hc_pre_vec` (so only with `ATLAS_QWEN4EXP_HC_FAST`),
//! after its stage; whether it applies depends on the model shape alone,
//! never on the row count, so every width of one model takes the same
//! arithmetic.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::qwen3_attention::HcLowRank;

// Must match the HCM_* defines of qwen4exp_hc_mma.cu.
const DN_WM: u32 = 3; // m16 weight tiles a CTA
const DN_WK: u32 = 1; // warps splitting a CTA's K slice
const DN_CL: u32 = 8; // cluster CTAs splitting K
const DN_U: u32 = 4; // weight ring (32-k chunks)
const FN_DW: u32 = 32; // output dims a warp
const FN_WK: u32 = 1; // warps splitting rank
const FN_U: u32 = 5; // up_w ring (16-row k-steps)
/// Rows (tokens) one launch takes: four n8 token tiles.
pub(super) const HC_MMA_MAX_T: u32 = 32;

/// `ATLAS_QWEN4EXP_HC_MMA=1` (read once, default off). Never under HIP: the
/// strix twin of the kernel directory has no `qwen4exp_hc_mma`.
pub(crate) fn hc_mma() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !cfg!(atlas_hip) && std::env::var("ATLAS_QWEN4EXP_HC_MMA").as_deref() == Ok("1")
    })
}

/// `ATLAS_QWEN4EXP_HC_STAGE_FIT=1` (read once, default off): pick the
/// `hc_pre_stage_vec` blocks-per-token so the (T, split) grid of 1024-thread
/// blocks -- one a GB10 SM holds -- is ONE wave: the largest power of two
/// <= 8 with T x split <= SMs. Bit-identical (every block of a token computes
/// the same 1024-thread RMS; scripts/dev/qwen4exp_hc_mma_bench.cu `check`
/// byte-compares `normed` across splits 1/2/4/8). The fixed 8 (2 at 25+
/// rows under HC_WIDE) spills into a second wave past 6 rows: stage us at
/// T = 8/16/24/32, 8.2/12.3/14.6/10.2 -> 6.2/6.2/6.2/8.1 (bench, GB10).
pub(crate) fn hc_stage_fit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_HC_STAGE_FIT").as_deref() == Ok("1"))
}

/// The stage's blocks per token under [`hc_stage_fit`], else `default`.
/// `max_split` (a power of two) is the largest the caller's shape check
/// admits.
pub(super) fn hc_stage_split(
    gpu: &dyn GpuBackend,
    num_tokens: u32,
    max_split: u32,
    default: u32,
) -> u32 {
    if !hc_stage_fit() {
        return default;
    }
    static SMS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let sms = *SMS.get_or_init(|| gpu.sm_count().unwrap_or(48).max(1));
    stage_split_for(sms, num_tokens, max_split)
}

fn stage_split_for(sms: u32, num_tokens: u32, max_split: u32) -> u32 {
    let mut split = max_split.max(1);
    while split > 1 && num_tokens * split > sms {
        split /= 2;
    }
    split
}

/// Whether the kernels' geometry tiles this model's site: hc == 4, the K
/// split (cluster x warps x ring) divides hc*H, the rank split divides rank,
/// warps of FN_DW dims cover H. Row count is deliberately NOT an input.
pub(super) fn hc_mma_shape_fits(hidden_size: u32, hc_mult: u32, rank: u32) -> bool {
    let k = hc_mult * hidden_size;
    hc_mult == 4
        && k.is_multiple_of(32 * DN_CL * DN_WK * DN_U)
        && rank.is_multiple_of(16 * FN_WK * FN_U)
        && hidden_size.is_multiple_of(FN_DW)
}

/// Dynamic shared bytes of `hc_mma_finish` at `num_tokens`: the staged `low`
/// fragments, then (reused) the K-split partials and the stream-mean tile.
pub(super) fn hc_mma_finish_smem(rank: u32, num_tokens: u32) -> u32 {
    let nt = num_tokens.div_ceil(8);
    let frags = rank / 16 * nt * 32 * 16;
    let mix = 4 * nt * 8 * FN_DW * 4;
    let partials = (FN_WK - 1) * 4 * (FN_DW / 16) * nt * 4 * 32 * 4;
    frags.max(mix).max(partials)
}

/// Launch down (+ injection rows unless `inj_out` is NULL: the model-level
/// head) and finish over `normed` (FP32 `[>= T, hc*H]`, already staged), the
/// FP32 `low` scratch `[T, rank]` in between. `Ok(false)`, nothing launched,
/// where the switch is off or the shape or the build does not fit.
/// `ptrs` = [normed, low, y_out, inj_out], `shape` = [num_tokens,
/// hidden_size, hc_mult].
pub(super) fn hc_mma_down_finish(
    gpu: &dyn GpuBackend,
    w: &HcLowRank,
    ptrs: [DevicePtr; 4],
    shape: [u32; 3],
    stream: u64,
) -> Result<bool> {
    if !hc_mma() {
        return Ok(false);
    }
    let ([normed, low, y_out, inj_out], [t, h, hc]) = (ptrs, shape);
    hc_mma_launch(gpu, w, normed, low, y_out, inj_out, t, h, hc, stream)
}

/// [`hc_mma_down_finish`] without the switch (the GPU test's entry).
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_mma_launch(
    gpu: &dyn GpuBackend,
    w: &HcLowRank,
    normed: DevicePtr,
    low: DevicePtr,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<bool> {
    let rank = w.rank as u32;
    if !hc_mma_shape_fits(hidden_size, hc_mult, rank) {
        return Ok(false);
    }
    anyhow::ensure!(
        (1..=HC_MMA_MAX_T).contains(&num_tokens),
        "hc_mma_down_finish: {num_tokens} rows, the kernels take 1..={HC_MMA_MAX_T}"
    );
    let k_down = crate::layers::try_kernel(gpu, "qwen4exp_hc_mma", "hc_mma_down");
    let k_fin = crate::layers::try_kernel(gpu, "qwen4exp_hc_mma", "hc_mma_finish");
    if k_down.0 == 0 || k_fin.0 == 0 {
        return Ok(false);
    }
    let rows = rank + if inj_out.is_null() { 0 } else { hc_mult };
    KernelLaunch::new(gpu, k_down)
        .grid([rows.div_ceil(16).div_ceil(DN_WM), DN_CL, 1])
        .block([32 * DN_WM * DN_WK, 1, 1])
        .arg_ptr(normed)
        .arg_ptr(w.down_w)
        .arg_ptr(if inj_out.is_null() {
            DevicePtr::NULL
        } else {
            w.inject_w
        })
        .arg_ptr(low)
        .arg_ptr(inj_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(rank)
        .arg_u32(num_tokens)
        .launch(stream)?;
    KernelLaunch::new(gpu, k_fin)
        .grid([hidden_size / FN_DW, 1, 1])
        .block([128 * FN_WK, 1, 1])
        .shared_mem(hc_mma_finish_smem(rank, num_tokens))
        .arg_ptr(normed)
        .arg_ptr(low)
        .arg_ptr(w.up_w)
        .arg_ptr(y_out)
        .arg_u32(hidden_size)
        .arg_u32(rank)
        .arg_u32(num_tokens)
        .launch(stream)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_site_shape_fits_and_others_do_not() {
        // Qwen3.8-Flash-Next: hidden 2560, hc 4, rank 320.
        assert!(hc_mma_shape_fits(2560, 4, 320));
        assert!(!hc_mma_shape_fits(2560, 2, 320));
        assert!(!hc_mma_shape_fits(2560, 4, 312));
        assert!(!hc_mma_shape_fits(2576, 4, 320));
    }

    #[test]
    fn stage_split_fits_one_wave_of_48_sms() {
        let got: Vec<u32> = [1, 4, 6, 7, 8, 12, 16, 24, 25, 32, 48, 64]
            .iter()
            .map(|&t| stage_split_for(48, t, 8))
            .collect();
        assert_eq!(got, [8, 8, 8, 4, 4, 4, 2, 2, 1, 1, 1, 1]);
        assert_eq!(stage_split_for(48, 1, 2), 2);
    }

    #[test]
    fn finish_shared_memory_stays_under_the_default_block_limit() {
        for t in 1..=HC_MMA_MAX_T {
            let b = hc_mma_finish_smem(320, t);
            assert!(b <= 48 * 1024, "T={t}: {b} B");
            assert!(b >= 320 / 16 * t.div_ceil(8) * 512);
        }
    }
}
