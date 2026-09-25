// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-shaped (small-T) mHC collapse: the three-launch split path.
//! Split out of `hyper_connection_lowrank.rs` for the 500-LoC cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::qwen3_attention::HcLowRank;
use super::hyper_connection_lowrank_gemm::{hc_finish_block, hc_finish_x4};

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
    stream: u64,
) -> Result<()> {
    let hc_dim = hc_mult * hidden_size;
    // Scratch layout: normed [T<=64, hc_dim] then low [T<=64, rank], F32.
    let normed = scratch;
    let low = scratch.offset(64 * hc_dim as usize * 4);

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
