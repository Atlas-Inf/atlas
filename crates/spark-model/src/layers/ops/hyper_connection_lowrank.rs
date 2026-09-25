// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.8-Flash-Next low-rank mHC dispatch.
//!
//! Companion to `hyper_connection.rs`, which drives DeepSeek-V4's Sinkhorn
//! mixer. Both families share the `[T, hc_mult, H]` FP32 highway and the same
//! four kernel NAMES — a model shadow overrides the whole
//! `hyper_connection.cu` file, so `qwen3.8-flash-next` resolves
//! `hyper_connection::hc_pre` to the low-rank kernel while
//! `deepseek-v4-flash` resolves it to the Sinkhorn one. The two take
//! DIFFERENT argument lists, which is why the launches live apart.
//!
//! `hc_expand` is byte-identical across both and is not duplicated here.
//!
//! Selection is by WEIGHTS, not by model name: `HcSiteWeights::lowrank`
//! being `Some` is what routes here. A model that somehow carried both would
//! be a load-time bug, not a silent dispatch coin-flip.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::qwen3_attention::HcLowRank;

/// Above this many tokens the collapse goes to the GEMM formulation; at or
/// below it, to the hand-rolled split kernels.
///
/// 8 covers every genuinely decode-shaped call -- T=1 decode, and T=2/3/4 MTP
/// verify (`forward_k2`/`k3`, `forward_atomic_c4`) -- and nothing else. Those
/// are the shapes the split path was written for: its premise is that
/// `grid=[T]` means `grid=[1]` on one SM, which stops being true the moment T
/// reaches the tens.
///
/// This WAS 64, and lowering it to 8 was measured as a 13% REGRESSION
/// (2026-08-30, TTFT_GAP.md 6b). That measurement was real and its conclusion
/// -- "the hand-rolled path beats the tensor-core GEMM at these shapes" -- was
/// wrong. Two of `hc_pre_gemm`'s three projections were launching on THREE and
/// ONE CTA of a 48-SM part (`gemm_raw` emits a 128x128 output tile, and N is
/// 320 and 4), so lowering the gate moved short prefills onto two starved
/// kernels. With those routed by machine-fill in `hc_gemm` the collapse is
/// 9-60x faster on exactly those projections and the premise holds again.
///
/// A prefill is chunked (96 + tail here), so the TAIL chunk is what this gate
/// decides: at 64 a 118-token prompt ran its 22-token tail on the split path
/// for 27.7 ms of a 422 ms window.
const HC_DECODE_MAX_T: u32 = 8;

/// `ATLAS_QWEN4EXP_NO_HC_GEMM=1`: revert the large-T collapse to the fused
/// FP32 kernel (deploy-time kill switch; the GEMM path rounds `normed` to
/// BF16 before the projections).
use super::hyper_connection_lowrank_gemm::{gemm_raw, hc_gemm};
use super::hyper_connection_lowrank_split::hc_pre_split;

fn hc_gemm_disabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_NO_HC_GEMM").as_deref() == Ok("1"))
}

/// Collapse the `hc_mult` streams to one, and emit the per-stream injection
/// weights the matching [`hc_post_lowrank`] needs.
///
/// `streams [T, hc, H] -> y_out [T, H]`, `inj_out [T, hc]`.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre_lowrank(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        !w.inject_w.is_null(),
        "hc_pre_lowrank needs block_inject_weight; a site loaded without one \
         is the model-level mixer and must use hc_head_lowrank"
    );
    // SMALL T (decode): three multi-block launches instead of the fused
    // kernel, whose grid=[T] means grid=[1] at decode — one block, one SM,
    // ~13 MB of weights per call (measured 2.0 ms; the whole token was
    // 96 x that). The fused kernel stays for prefill, where grid=[T]
    // already fills the machine and skips the global round trip.
    //
    // THRESHOLD. This was `<= 64`, chosen for decode shapes — but a short
    // PREFILL also has num_tokens <= 64, so a 60-token prompt took the
    // decode path while a 65-token one took the tensor-core GEMM. That split
    // is where the short-prefill cost lived: `hc_pre_down` + `hc_pre_finish`
    // measured 143 ms, 31% of a 60-token prefill (nsys 2026-08-30), both
    // running FP32 warp loops with a single dependent accumulator chain per
    // lane — ~19x off the roofline for what is a 393 MFLOP GEMM.
    //
    // The split path's own premise says when it stops applying: it exists
    // because "grid=[T] means grid=[1] at decode - one block, one SM". At
    // T=60, grid=[60] already fills a 48-SM part, so the premise is false and
    // the GEMM formulation — which the comment below notes was written
    // precisely because "47% of prefill was this collapse running as FP32
    // warp loops" — is the right one.
    //
    // 8 covers every genuinely decode-shaped call: T=1 decode, and T=2/3/4
    // MTP verify (forward_k2/k3, forward_atomic_c4). Above that we are in
    // prefill and want the GEMM.
    //
    // NOTE this makes short prefills numerically CONSISTENT with long ones
    // rather than introducing a new regime: the GEMM path rounds `normed` to
    // BF16, and every prefill over 64 tokens already took it. A 60- and a
    // 65-token prompt previously ran different arithmetic.
    if num_tokens <= HC_DECODE_MAX_T && !scratch.is_null() {
        return hc_pre_split(
            gpu,
            streams,
            w,
            y_out,
            inj_out,
            scratch,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            /* inject */ true,
            stream,
        );
    }
    // LARGE T (prefill): tensor-core GEMM formulation — 47% of prefill was
    // this collapse running as FP32 warp loops. Kill switch reverts to the
    // fused kernel below.
    if !scratch.is_null() && !hc_gemm_disabled() {
        return hc_pre_gemm(
            gpu,
            streams,
            w,
            y_out,
            inj_out,
            scratch,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            /* inject */ true,
            stream,
        );
    }
    // Block 1024 + dynamic shared for the staged normed vector [hc*H] and
    // the rank vector — the warp-cooperative core. This launch WAS the whole
    // decode budget at block 256 with per-thread serial rows (4.5 ms/call,
    // x96 calls/token); see the kernel's PERFORMANCE SHAPE note.
    let smem = (hc_mult * hidden_size + w.rank as u32) * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([1024, 1, 1])
        .shared_mem(smem)
        .arg_ptr(streams)
        .arg_ptr(w.norm_w)
        .arg_ptr(w.down_w)
        .arg_ptr(w.up_w)
        .arg_ptr(w.inject_w)
        .arg_ptr(y_out)
        .arg_ptr(inj_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .arg_f32(norm_eps)
        .launch(stream)
}

/// The model-level mixer (`use_combine=False`): the same collapse with no
/// injection vector.
///
/// This is also the model's FINAL NORMALIZATION — the checkpoint ships no
/// `model.norm.weight` because `hc_norm` here plays that role.
#[allow(clippy::too_many_arguments)]
pub fn hc_head_lowrank(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    if num_tokens <= HC_DECODE_MAX_T && !scratch.is_null() {
        return hc_pre_split(
            gpu,
            streams,
            w,
            y_out,
            DevicePtr::NULL,
            scratch,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            /* inject */ false,
            stream,
        );
    }
    // Same GEMM formulation as hc_pre — the head is the identical collapse
    // minus the injection GEMM (hc_pre_mix skips inj on a null inj_pre).
    if !scratch.is_null() && !hc_gemm_disabled() {
        return hc_pre_gemm(
            gpu,
            streams,
            w,
            y_out,
            DevicePtr::NULL,
            scratch,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            /* inject */ false,
            stream,
        );
    }
    let smem = (hc_mult * hidden_size + w.rank as u32) * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([1024, 1, 1])
        .shared_mem(smem)
        .arg_ptr(streams)
        .arg_ptr(w.norm_w)
        .arg_ptr(w.down_w)
        .arg_ptr(w.up_w)
        .arg_ptr(y_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .arg_f32(norm_eps)
        .launch(stream)
}

/// Inject the block output back into every stream:
/// `out[t, s*H + d] = residual[t, s*H + d] + block_out[t, d] * inj[t, s]`.
///
/// Note there is no `comb` argument: DeepSeek mixes streams with a full
/// `[hc, hc]` combine matrix on the way back, Qwen scales by one scalar per
/// stream. Passing a combine matrix here would not type-check, which is the
/// point of keeping the two launches separate.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_lowrank(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    residual: DevicePtr,
    inj: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(inj)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// LARGE T (prefill): the down/up projections are GEMM-shaped and the fused
/// kernel ran them as hand-rolled FP32 warp loops at ~4% of the machine —
/// measured 45 ms/call, 47% of the whole prefill. Stage `normed` in BF16 and
/// hand both projections (and the tiny injection one) to the tensor-core
/// `dense_gemm_bf16_pipelined`, keeping only the elementwise seams custom.
/// Slabbed at <= 2048 tokens to bound the scratch region.
///
/// `ATLAS_QWEN4EXP_NO_HC_GEMM=1` falls back to the fused kernel (kill switch,
/// same convention as ATLAS_NO_GDN_FLA).
#[allow(clippy::too_many_arguments)]
fn hc_pre_gemm(
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
    const SLAB: u32 = 2048;
    let hc_dim = (hc_mult * hidden_size) as usize;
    let rank = w.rank as u32;
    // Scratch layout (BF16): normed [L, hc_dim], up_pre [L, hc_dim],
    // low [L, rank], inj_pre [L, hc], up_wt [hc_dim, rank], where
    // L = min(T, 2048). sizes.rs sizes the region with m.min(2048) and
    // T <= m always, so L-based offsets fit even when the arena was sized for
    // fewer than 2048 tokens; `up_wt` is L-independent and sits last.
    let lay = num_tokens.min(SLAB) as usize;
    // Aligned placement (odd slabs used to put `up_wt` 8 bytes off a 16-byte
    // boundary and fault the GEMM); sizes.rs reserves from the same layout.
    let l = spark_runtime::buffers::hc_pre_scratch_layout(lay, hc_dim, w.rank, hc_mult as usize);
    let normed = scratch;
    let up_pre = scratch.offset(l.up_pre);
    let low = scratch.offset(l.low);
    let inj_pre = scratch.offset(l.inj_pre);
    // Still computed, and the region still sized for it, even though only the
    // no-cuBLASLt arm reads it: shrinking `hc_lowrank_scratch` would shift every
    // later buffer in the shared arena, which measured 6% SLOWER when tried for
    // alignment slack (TTFT_GAP.md 6b). Layout stability beats 6.55 MB.
    let up_wt = scratch.offset(l.up_wt);

    let k_stage = gpu.kernel("hyper_connection", "hc_pre_stage_bf16")?;
    let k_silu = gpu.kernel("hyper_connection", "hc_silu_scale")?;
    let k_mix = gpu.kernel("hyper_connection", "hc_pre_mix")?;
    let k_gemm = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    let k_tr = gpu.kernel("hyper_connection", "hc_transpose_bf16")?;
    // Read once per call, not once per projection. On failure the machine-fill
    // rule can never fire, so the path keeps exactly today's behaviour.
    let sm_count = gpu.sm_count().unwrap_or(0);
    let inv_hc = 1.0f32 / hc_mult as f32;

    // `up_w` is stored `[rank, hc_dim]` — the layout the decode stage-3 kernel
    // needs to coalesce (see `hc_pre_split`). This GEMM's tensor-core kernel is
    // NT (`C[m,n] = A[m,k] . B[n,k]`), so it wants the checkpoint's
    // `[hc_dim, rank]`. Stage a transposed copy rather than keeping a second
    // resident buffer: 6.55 MB x 97 sites is 635 MB on a box that already
    // loads at 113 of 119.6 GB, against ~50 us per call here on a collapse
    // that measured ~45 ms. Once per call, not once per slab — `up_wt` does
    // not depend on `t0`.
    // `up_w` is `[rank, hc_dim]`; `gemm_raw` is NT and wants `[hc_dim, rank]`,
    // so it needs the staging transpose. cuBLASLt does not -- `op_a` selects the
    // layout -- and the transpose was 9.0 ms of a 422 ms prefill window (97
    // launches at ~93 us, grid 320x10 of 32-thread blocks). So decide the
    // layout ONCE, up front, from whether cuBLASLt is usable at all; a per-GEMM
    // `Result` is too late, because by then the transpose has been skipped.
    let lt = spark_runtime::cublaslt::available();
    if !lt {
        KernelLaunch::new(gpu, k_tr)
            .grid([(hc_dim as u32).div_ceil(32), rank.div_ceil(32), 1])
            .block([32, 32, 1])
            .arg_ptr(w.up_w)
            .arg_ptr(up_wt)
            .arg_u32(rank)
            .arg_u32(hc_dim as u32)
            .launch(stream)?;
    }

    let mut t0 = 0u32;
    while t0 < num_tokens {
        let ts = SLAB.min(num_tokens - t0);
        let streams_s = streams.offset(t0 as usize * hc_dim * 4);

        KernelLaunch::new(gpu, k_stage)
            .grid([ts, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(streams_s)
            .arg_ptr(w.norm_w)
            .arg_ptr(normed)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(norm_eps)
            .launch(stream)?;

        // low_pre = normed x down_w^T   [ts, rank]   (N=320: skinny, split-K)
        hc_gemm(
            gpu,
            k_gemm,
            normed,
            w.down_w,
            low,
            ts,
            rank,
            hc_dim as u32,
            sm_count,
            stream,
        )?;
        let n_low = ts * rank;
        KernelLaunch::new(gpu, k_silu)
            .grid([n_low.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(low)
            .arg_u32(n_low)
            .arg_f32(inv_hc)
            .launch(stream)?;

        // up_pre = low x up_w   [ts, hc_dim]. N=10240 is 80 CTAs, so the tile
        // kernel's grid is not the problem here -- the staging transpose it
        // would need is. Off the checkpoint layout when cuBLASLt is there.
        if lt {
            spark_runtime::cublaslt::bf16_gemm_act_weight_n(
                low.0,
                w.up_w.0,
                up_pre.0,
                ts,
                hc_dim as u32,
                rank,
                stream,
            )?;
        } else {
            gemm_raw(gpu, k_gemm, low, up_wt, up_pre, ts, hc_dim as u32, rank, stream)?;
        }
        if inject {
            // inj_pre = normed x inject_w^T   [ts, hc]   (N=4: one CTA)
            hc_gemm(
                gpu,
                k_gemm,
                normed,
                w.inject_w,
                inj_pre,
                ts,
                hc_mult,
                hc_dim as u32,
                sm_count,
                stream,
            )?;
        }

        KernelLaunch::new(gpu, k_mix)
            .grid([ts, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(normed)
            .arg_ptr(up_pre)
            .arg_ptr(if inject { inj_pre } else { DevicePtr::NULL })
            .arg_ptr(y_out.offset(t0 as usize * hidden_size as usize * 2))
            .arg_ptr(inj_out.offset(t0 as usize * hc_mult as usize * 4))
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(inv_hc)
            .launch(stream)?;

        t0 += ts;
    }
    Ok(())
}
