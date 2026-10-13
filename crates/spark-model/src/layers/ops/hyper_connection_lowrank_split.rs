// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-shaped (small-T) mHC collapse: the three-launch split path.
//! Split out of `hyper_connection_lowrank.rs` for the 500-LoC cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::hyper_connection_lowrank_gemm::{
    hc_fast, hc_finish_block, hc_finish_x4, hc_token_fused,
};
use super::qwen4exp_decode_fuse::{HcPostFold, hc_post_stage};
use crate::layers::qwen3_attention::HcLowRank;

// Shape of the vectorized kernels; each must match its HC_V_* define in
// kernels/gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu. T <= 4 runs
// `hc_pre_{down,finish}_vec`, T = 5..HC_V_MAX their `_vec8` twins (the same
// per-output operations; shorter rings for the larger token count).
const HC_V_MAX: u32 = 8;
const HC_V_TWIN_MIN: u32 = 5;
const HC_V_DOWN_CPT: u32 = 2; // chains per thread
const HC_V_DOWN_UNROLL: u32 = 32;
const HC_V_FIN_DPT: u32 = 4; // output dims per thread
const HC_V_FIN_UNROLL: u32 = 32;
// Launch geometry: pure scheduling, swept in the bench (2026-10-05, GB10).
const HC_V_DOWN_BLOCK: u32 = 128;
const HC_V_FIN_BLOCK: u32 = 128;
const HC_V_STAGE_SPLIT: u32 = 8;
pub(super) const HC_V_POST_BLOCK: u32 = 64;

/// Whether every pointer the vectorized kernels read with a vector load is
/// 16-byte aligned (device allocations and the arena are; this guards a
/// sub-allocation that is not).
pub(super) fn hc_vec_aligned(ptrs: &[DevicePtr]) -> bool {
    ptrs.iter().all(|p| p.0.is_multiple_of(16))
}

/// `ATLAS_QWEN4EXP_HC_FAST` arm of [`hc_pre_split`]: the same collapse as
/// stage + down + finish, with 16-byte-class vector loads kept in flight by a
/// rolling register ring, the injection rows folded into the down launch and
/// the stage spread over `HC_V_STAGE_SPLIT` blocks per token. Bit-identical
/// by construction (see the kernel note). Returns `false`, having launched
/// nothing, when the shape or the build does not fit, so the caller falls
/// through to the default kernels.
#[allow(clippy::too_many_arguments)]
fn hc_pre_vec(
    gpu: &dyn GpuBackend,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    normed: DevicePtr,
    low: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    fold: Option<HcPostFold>,
    stream: u64,
) -> Result<bool> {
    let hc_dim = hc_mult * hidden_size;
    let rank = w.rank as u32;
    let fits = num_tokens <= HC_V_MAX
        && hc_mult == 4
        && hidden_size.is_multiple_of(HC_V_FIN_DPT)
        && hc_dim.is_multiple_of(4 * HC_V_STAGE_SPLIT)
        && hc_dim.is_multiple_of(32)
        && (hc_dim / 32).is_multiple_of(HC_V_DOWN_UNROLL)
        && rank.is_multiple_of(HC_V_FIN_UNROLL)
        && hc_vec_aligned(&[
            streams, w.norm_w, w.down_w, w.up_w, w.inject_w, y_out, normed, low,
        ]);
    if !fits {
        return Ok(false);
    }
    let twin = num_tokens >= HC_V_TWIN_MIN;
    let k_stage = crate::layers::try_kernel(gpu, "hyper_connection", "hc_pre_stage_vec");
    let k_down = crate::layers::try_kernel(
        gpu,
        "hyper_connection",
        if twin {
            "hc_pre_down_vec8"
        } else {
            "hc_pre_down_vec"
        },
    );
    let k_fin = crate::layers::try_kernel(
        gpu,
        "hyper_connection",
        if twin {
            "hc_pre_finish_vec8"
        } else {
            "hc_pre_finish_vec"
        },
    );
    if k_stage.0 == 0 || k_down.0 == 0 || k_fin.0 == 0 {
        return Ok(false);
    }

    // The stage kernel's RMS is `hc_pre_stage`'s 1024-thread reduction; it is
    // only bit-identical at that width.
    if let Some(fold) = fold {
        hc_post_stage(
            gpu,
            fold,
            streams,
            w.norm_w,
            normed,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            stream,
        )?;
    } else {
        KernelLaunch::new(gpu, k_stage)
            .grid([num_tokens, HC_V_STAGE_SPLIT, 1])
            .block([1024, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(w.norm_w)
            .arg_ptr(normed)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(norm_eps)
            .launch(stream)?;
    }

    // `inj_out` NULL is the model-level head: no injection rows.
    let rows = rank + if inj_out.is_null() { 0 } else { hc_mult };
    let down_threads = rows * (32 / HC_V_DOWN_CPT);
    KernelLaunch::new(gpu, k_down)
        .grid([down_threads.div_ceil(HC_V_DOWN_BLOCK), 1, 1])
        .block([HC_V_DOWN_BLOCK, 1, 1])
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

    let fin_threads = hc_dim / HC_V_FIN_DPT;
    KernelLaunch::new(gpu, k_fin)
        .grid([fin_threads.div_ceil(HC_V_FIN_BLOCK), 1, 1])
        .block([HC_V_FIN_BLOCK, 1, 1])
        .shared_mem(num_tokens * rank * 4)
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

/// The three-launch collapse for small T. Same math as the fused kernel;
/// the parity probe's T=8 fixture runs THIS path.
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_pre_split(
    gpu: &dyn GpuBackend,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    inject: bool,
    fold: Option<HcPostFold>,
    stream: u64,
) -> Result<()> {
    let hc_dim = hc_mult * hidden_size;
    // Scratch layout: normed [T<=64, hc_dim] then low [T<=64, rank], F32.
    let normed = scratch;
    let low = scratch.offset(64 * hc_dim as usize * 4);

    if hc_fast()
        && hc_pre_vec(
            gpu,
            streams,
            w,
            y_out,
            if inject { inj_out } else { DevicePtr::NULL },
            normed,
            low,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            fold,
            stream,
        )?
    {
        return Ok(());
    }

    let k_stage = gpu.kernel("hyper_connection", "hc_pre_stage")?;
    // Two shapes of the same math. `hc_pre_down` stages the whole 40 KB
    // `normed` row and makes one pass -- right for decode (T=1), where a single
    // pass costs no barriers. `hc_pre_down_tiled` tiles tokens and chunks
    // hc_dim -- right for prefill widths, where the single-pass version is
    // L2-bandwidth-bound re-reading the 6.55 MB `down_w` once per token.
    //
    // Measured (nsys, 87-token prefill chunked 64+23):
    //   prefill hc_pre_down  89.2 ms -> 33.0 ms   (prefill window 491.5 -> 438.3)
    //   decode  hc_pre_down  28.8 ms -> 65.0 ms   (679 calls; 2.3x WORSE)
    // so neither kernel wins everywhere and the dispatch is on T.
    let k_down = gpu.kernel("hyper_connection", "hc_pre_down")?;
    let k_down_tiled = gpu.kernel("hyper_connection", "hc_pre_down_tiled")?;
    let k_fin = gpu.kernel("hyper_connection", "hc_pre_finish")?;

    if let Some(fold) = fold {
        // `normed` is bit-identical to `hc_pre_stage`'s (as `hc_pre_stage_vec`'s is).
        hc_post_stage(
            gpu,
            fold,
            streams,
            w.norm_w,
            normed,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            stream,
        )?;
    } else {
        KernelLaunch::new(gpu, k_stage)
            .grid([num_tokens, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(w.norm_w)
            .arg_ptr(normed)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(norm_eps)
            .launch(stream)?;
    }

    // Token-fused stages 2+3 for decode and the K=2 MTP verify. The per-token
    // kernels below stream each 6.55 MB weight once PER TOKEN with ~8 rows of
    // loads in flight per warp: at T=2 on gfx1151 that pair measured 136 + 143
    // us per call, 97 calls per verify step, 27 ms of a 111 ms step
    // (ATLAS_TRACE_LAUNCH_SYNC, winbox 2026-10-02). The `_mt` kernels read each
    // weight once for all tokens and keep 16 loads in flight per lane. Bit-
    // identical by construction (same per-output accumulation order; see the
    // kernel note), so this picks launch shape only.
    const HC_MT_MAX: u32 = 4; // must match HC_MT_MAX in hyper_connection.cu
    const HC_MT_DOWN_BLOCK: u32 = 128;
    // The finish kernel computes token t's injection in block t, so its grid
    // (H / 32 blocks) must cover every token.
    if num_tokens <= HC_MT_MAX
        && hc_mult == 4
        && hidden_size.div_ceil(32) >= num_tokens
        && hc_token_fused()
    {
        let k_down_mt = crate::layers::try_kernel(gpu, "hyper_connection", "hc_pre_down_mt");
        let k_fin_mt = crate::layers::try_kernel(gpu, "hyper_connection", "hc_pre_finish_x4_mt");
        if k_down_mt.0 != 0 && k_fin_mt.0 != 0 {
            let rank = w.rank as u32;
            KernelLaunch::new(gpu, k_down_mt)
                .grid([rank.div_ceil(HC_MT_DOWN_BLOCK / 32), 1, 1])
                .block([HC_MT_DOWN_BLOCK, 1, 1])
                .arg_ptr(normed)
                .arg_ptr(w.down_w)
                .arg_ptr(low)
                .arg_u32(hidden_size)
                .arg_u32(hc_mult)
                .arg_u32(rank)
                .arg_u32(num_tokens)
                .launch(stream)?;
            return KernelLaunch::new(gpu, k_fin_mt)
                .grid([hidden_size.div_ceil(32), 1, 1])
                .block([128, 1, 1])
                .shared_mem(num_tokens * rank * 4)
                .arg_ptr(normed)
                .arg_ptr(low)
                .arg_ptr(w.up_w)
                .arg_ptr(if inject { w.inject_w } else { DevicePtr::NULL })
                .arg_ptr(y_out)
                .arg_ptr(inj_out)
                .arg_u32(hidden_size)
                .arg_u32(hc_mult)
                .arg_u32(rank)
                .arg_u32(num_tokens)
                .launch(stream);
        }
    }

    // `hc_pre_down` stages the token's `normed` row in SHARED memory, so the
    // 40 KB vector is read once per block instead of once per `rank` row. That
    // was the dominant traffic term (T x rank x 40 KB = 786 MB at T=60, against
    // 393 MB for the weight); see the kernel note.
    //
    // Shared budget: hc_dim floats. At hc_dim=10240 that is 40 KB, inside the
    // 48 KB default. If a model ever exceeds it the launch would fail, so fall
    // back to the un-staged path rather than trusting the geometry.
    // HC_TT / HC_CH must match hyper_connection.cu.
    const HC_TT: u32 = 8;
    const HC_CH: usize = 512;
    const HC_SMEM_MAX: usize = 48 * 1024;
    // Tiled pays off once there are enough tokens to amortise a weight row over
    // and to fill the part after grid.x shrinks by HC_TT. At or below HC_TT the
    // tile degenerates to one token per block while still paying
    // hc_dim/HC_CH rounds of barriers, which is the decode regression measured
    // above -- so the single-pass kernel keeps those shapes.
    if num_tokens > HC_TT {
        let smem = HC_TT as usize * HC_CH * 4;
        anyhow::ensure!(
            smem <= HC_SMEM_MAX,
            "hc_pre_down_tiled: nx tile is {} B of shared, over the {} B limit; \
             lower HC_TT or HC_CH in BOTH files together.",
            smem,
            HC_SMEM_MAX,
        );
        // One row per warp keeps the kernel's accumulator array at HC_TT
        // registers rather than HC_TT x rows_per_warp.
        let dsplit = (w.rank as u32).div_ceil(32).clamp(1, 16);
        KernelLaunch::new(gpu, k_down_tiled)
            .grid([num_tokens.div_ceil(HC_TT), dsplit, 1])
            .block([1024, 1, 1])
            .shared_mem(0)
            .arg_ptr(normed)
            .arg_ptr(w.down_w)
            .arg_ptr(low)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_u32(w.rank as u32)
            .arg_u32(num_tokens)
            .launch(stream)?;
    } else {
        let hc_smem = hc_dim as usize * 4;
        anyhow::ensure!(
            hc_smem <= HC_SMEM_MAX,
            "hc_pre_down: normed row is {} B of shared, over the {} B block \
             limit (hc_dim={}).",
            hc_smem,
            HC_SMEM_MAX,
            hc_dim,
        );
        let dsplit = (48 / num_tokens.max(1)).clamp(1, 10);
        KernelLaunch::new(gpu, k_down)
            .grid([num_tokens, dsplit, 1])
            .block([1024, 1, 1])
            .shared_mem(hc_smem as u32)
            .arg_ptr(normed)
            .arg_ptr(w.down_w)
            .arg_ptr(low)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_u32(w.rank as u32)
            .arg_u32(num_tokens)
            .launch(stream)?;
    }

    // Stage 3 was the largest kernel in the decode profile: 23% of all GPU
    // time (11.86 s of 51.47 s, nsys 2026-08-28), a flat ~173 us regardless of
    // T, ~38 GB/s against the part's ~273. Each output dim gets its own THREAD
    // and that thread contracts over `rank` sequentially — which, in the
    // checkpoint's `[hc*H, rank]` layout, means walking a contiguous row, so
    // consecutive threads touched rows 640 B apart and every warp load
    // scattered over 32 sectors.
    //
    // `up_w` is now stored TRANSPOSED as `[rank, hc*H]` (see the kernel's
    // "WHY `up_w` IS STORED TRANSPOSED" note and `weight_loader::qwen4_exp::
    // hc::transpose_up_w`), so thread `d` reads `up_w[r*hc_dim + i]`:
    // consecutive threads read consecutive bf16. The loop body is otherwise
    // untouched, so the FP32 accumulation order is IDENTICAL and the output is
    // bitwise unchanged — which is the whole point. Two kernel-side fixes were
    // measured first and both failed one half of that: warp-per-dim with a
    // shfl reduction was +17.8% but reassociates, and shared-memory staging
    // was bit-exact but 44% slower. The layout was the only thing that could
    // give both.
    // Block width, swept 32/64/128/256 with `ATLAS_HC_FIN_BLOCK` (agg tok/s,
    // C=1 / C=2, every arm bitwise identical since this is pure geometry):
    //
    //   256 -> 21.65 / 25.20    128 -> 21.68 / 25.24  <- default
    //    64 -> 20.57 / 23.84     32 -> 18.73 / 21.65
    //
    // NARROWER IS WORSE, which is the opposite of the guess. Thread-per-`d`
    // caps the kernel at H threads per token, so a narrower block spreads the
    // same 2560 threads over more SMs — but every block re-stages the whole
    // rank-320 `low` vector into its own shared memory first, and at block 32
    // that is 80 blocks each paying the same staging cost for 32 threads of
    // work. The extra SMs do not pay for the extra staging. rsafier's original
    // `S = clamp(48/T, 1, 10)` was already at the useful end of this curve;
    // 128 is a hair better and 256 is inside the noise.
    // Stream-per-warp layout (`hc_pre_finish_x4`, hc == 4 only): 4x the
    // threads of the thread-per-`d` kernel, identical accumulation order.
    let x4 = hc_mult == 4 && hc_finish_x4();
    let (k_fin, grid_y, fblock) = if x4 {
        (
            gpu.kernel("hyper_connection", "hc_pre_finish_x4")?,
            hidden_size.div_ceil(32),
            128,
        )
    } else {
        let fblock = hc_finish_block();
        let fsplit = hidden_size
            .div_ceil(fblock)
            .max((48 / num_tokens.max(1)).clamp(1, 10));
        (k_fin, fsplit, fblock)
    };
    KernelLaunch::new(gpu, k_fin)
        .grid([num_tokens, grid_y, 1])
        .block([fblock, 1, 1])
        .shared_mem(w.rank as u32 * 4)
        .arg_ptr(normed)
        .arg_ptr(low)
        .arg_ptr(w.up_w)
        .arg_ptr(if inject { w.inject_w } else { DevicePtr::NULL })
        .arg_ptr(y_out)
        .arg_ptr(inj_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .launch(stream)
}
