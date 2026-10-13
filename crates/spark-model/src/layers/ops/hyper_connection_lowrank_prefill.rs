// SPDX-License-Identifier: AGPL-3.0-only

//! The large-T (prefill) GEMM formulation of the Qwen3.8-Flash-Next low-rank
//! mHC collapse, split out of `hyper_connection_lowrank.rs` for the 500-LoC
//! cap. One 2048-token slab at a time: stage `normed` (BF16), the three
//! projections on tensor cores, the elementwise mix.
//!
//! The slab body is a function of its own so that the opt-in fused arm
//! (`qwen4exp_prefill_hc::hc_slab_fast`, `ATLAS_QWEN4EXP_PREFILL_HC=1`) can run
//! this one beside itself and compare bytes under `_HC_CHECK`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::hyper_connection_lowrank_gemm::{gemm_raw, hc_gemm};
use super::qwen4exp_decode_fuse::HcPostFold;
use super::qwen4exp_prefill_hc;
use crate::layers::qwen3_attention::HcLowRank;

/// Rows per slab: bounds the scratch region (`sizes.rs` sizes it with
/// `m.min(2048)`).
pub(crate) const HC_PREFILL_SLAB: u32 = 2048;

/// Everything a slab needs that does not change across the slabs of a call.
pub(super) struct HcPrefillCall<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub w: &'a HcLowRank,
    /// Scratch, see `hc_pre_scratch_layout`: normed, up_pre, low, inj_pre,
    /// up_wt (the `[hc*H, rank]` copy of `up_w`).
    pub normed: DevicePtr,
    pub up_pre: DevicePtr,
    pub low: DevicePtr,
    pub inj_pre: DevicePtr,
    pub up_wt: DevicePtr,
    pub hidden: u32,
    pub hc_mult: u32,
    pub eps: f32,
    pub inject: bool,
    /// cuBLASLt usable (decided once per call; picks the up GEMM's layout).
    pub lt: bool,
    pub sm_count: u32,
}

/// One slab's rows: highway in (and out, under a fold), `y`/`inj` out.
#[derive(Clone, Copy)]
pub(super) struct HcSlabIo {
    pub streams: DevicePtr,
    pub y: DevicePtr,
    pub inj: DevicePtr,
    /// This slab's rows of the previous sublayer's `hc_post`, when the caller
    /// fused it into this collapse (the fast arm only).
    pub fold: Option<HcPostFold>,
    pub ts: u32,
}

impl HcPrefillCall<'_> {
    pub(super) fn hc_dim(&self) -> u32 {
        self.hc_mult * self.hidden
    }
    pub(super) fn rank(&self) -> u32 {
        self.w.rank as u32
    }
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
///
/// `fold`: the previous sublayer's `hc_post` to run inside this collapse
/// (`hc_post_stage_bf16`). Only [`qwen4exp_prefill_hc::hc_post_pre_seam`] passes
/// one, and only when the fast arm serves the call.
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_pre_gemm(
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
    let hc_dim = (hc_mult * hidden_size) as usize;
    let rank = w.rank as u32;
    // Scratch layout (BF16): normed [L, hc_dim], up_pre [L, hc_dim],
    // low [L, rank], inj_pre [L, hc], up_wt [hc_dim, rank], where
    // L = min(T, 2048). sizes.rs sizes the region with m.min(2048) and
    // T <= m always, so L-based offsets fit even when the arena was sized for
    // fewer than 2048 tokens; `up_wt` is L-independent and sits last.
    let lay = num_tokens.min(HC_PREFILL_SLAB) as usize;
    // Aligned placement (odd slabs used to put `up_wt` 8 bytes off a 16-byte
    // boundary and fault the GEMM); sizes.rs reserves from the same layout.
    let l = spark_runtime::buffers::hc_pre_scratch_layout(lay, hc_dim, w.rank, hc_mult as usize);
    // `up_wt` is still sized even though only the no-cuBLASLt arm and the fast
    // arm read it: shrinking `hc_lowrank_scratch` would shift every later
    // buffer in the shared arena, which measured 6% SLOWER when tried for
    // alignment slack (TTFT_GAP.md 6b). Layout stability beats 6.55 MB.
    // `up_w` is `[rank, hc_dim]`; `gemm_raw` is NT and wants `[hc_dim, rank]`,
    // so it needs the staging transpose. cuBLASLt does not -- `op_a` selects the
    // layout -- and the transpose was 9.0 ms of a 422 ms prefill window (97
    // launches at ~93 us, grid 320x10 of 32-thread blocks). So decide the
    // layout ONCE, up front, from whether cuBLASLt is usable at all; a per-GEMM
    // `Result` is too late, because by then the transpose has been skipped.
    let call = HcPrefillCall {
        gpu,
        w,
        normed: scratch,
        up_pre: scratch.offset(l.up_pre),
        low: scratch.offset(l.low),
        inj_pre: scratch.offset(l.inj_pre),
        up_wt: scratch.offset(l.up_wt),
        hidden: hidden_size,
        hc_mult,
        eps: norm_eps,
        inject,
        lt: spark_runtime::cublaslt::available(),
        // Read once per call, not once per projection. On failure the
        // machine-fill rule can never fire, so the path keeps exactly
        // today's behaviour.
        sm_count: gpu.sm_count().unwrap_or(0),
    };
    let fast = qwen4exp_prefill_hc::prefill_arm(gpu, w, num_tokens, hidden_size, hc_mult);
    anyhow::ensure!(
        fold.is_none() || fast.is_some(),
        "hc_pre_gemm: a folded hc_post needs the fast prefill arm"
    );
    // The fast arm's fused up+mix reads the `[hc_dim, rank]` copy too. Once
    // per call, not once per slab -- `up_wt` does not depend on `t0`.
    if !call.lt || fast.as_ref().is_some_and(|k| k.reads_up_wt()) {
        let k_tr = gpu.kernel("hyper_connection", "hc_transpose_bf16")?;
        KernelLaunch::new(gpu, k_tr)
            .grid([(hc_dim as u32).div_ceil(32), rank.div_ceil(32), 1])
            .block([32, 32, 1])
            .arg_ptr(w.up_w)
            .arg_ptr(call.up_wt)
            .arg_u32(rank)
            .arg_u32(hc_dim as u32)
            .launch(stream)?;
    }

    let mut t0 = 0u32;
    while t0 < num_tokens {
        let ts = HC_PREFILL_SLAB.min(num_tokens - t0);
        let io = HcSlabIo {
            streams: streams.offset(t0 as usize * hc_dim * 4),
            y: y_out.offset(t0 as usize * hidden_size as usize * 2),
            inj: inj_out.offset(t0 as usize * hc_mult as usize * 4),
            fold: fold.map(|f| HcPostFold {
                block_out: f.block_out.offset(t0 as usize * hidden_size as usize * 2),
                inj: f.inj.offset(t0 as usize * hc_mult as usize * 4),
            }),
            ts,
        };
        match &fast {
            Some(k) => qwen4exp_prefill_hc::hc_slab_fast(&call, k, &io, stream)?,
            None => hc_slab_default(&call, &io, stream)?,
        }
        t0 += ts;
    }
    Ok(())
}

/// The default slab: stage, down (+silu), up, injection, mix. `io.fold` must
/// be `None` (the checker runs the fold's `hc_post` itself first).
pub(super) fn hc_slab_default(c: &HcPrefillCall, io: &HcSlabIo, stream: u64) -> Result<()> {
    debug_assert!(io.fold.is_none());
    hc_stage(c, io, stream)?;
    hc_low(c, io.ts, stream)?;
    hc_slab_tail(c, io, stream)
}

/// normed = rmsnorm(highway) * norm_w, BF16, over the slab's rows.
pub(super) fn hc_stage(c: &HcPrefillCall, io: &HcSlabIo, stream: u64) -> Result<()> {
    KernelLaunch::new(
        c.gpu,
        c.gpu.kernel("hyper_connection", "hc_pre_stage_bf16")?,
    )
    .grid([io.ts, 1, 1])
    .block([1024, 1, 1])
    .arg_ptr(io.streams)
    .arg_ptr(c.w.norm_w)
    .arg_ptr(c.normed)
    .arg_u32(c.hidden)
    .arg_u32(c.hc_mult)
    .arg_f32(c.eps)
    .launch(stream)
}

/// low = silu(normed x down_w^T / hc)   [ts, rank]: the down projection on
/// whichever GEMM fills the machine (`hc_gemm`; N=320 is skinny), then silu.
pub(super) fn hc_low(c: &HcPrefillCall, ts: u32, stream: u64) -> Result<()> {
    let k_gemm = c.gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    hc_gemm(
        c.gpu,
        k_gemm,
        c.normed,
        c.w.down_w,
        c.low,
        ts,
        c.rank(),
        c.hc_dim(),
        c.sm_count,
        stream,
    )?;
    hc_silu(c, ts, stream)
}

/// The default slab after `low`: up projection, injection, mix.
pub(super) fn hc_slab_tail(c: &HcPrefillCall, io: &HcSlabIo, stream: u64) -> Result<()> {
    let gpu = c.gpu;
    let (ts, hc_dim, rank) = (io.ts, c.hc_dim(), c.rank());
    let k_mix = gpu.kernel("hyper_connection", "hc_pre_mix")?;
    let k_gemm = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    let inv_hc = 1.0f32 / c.hc_mult as f32;

    // up_pre = low x up_w   [ts, hc_dim]. N=10240 is 80 CTAs, so the tile
    // kernel's grid is not the problem here -- the staging transpose it
    // would need is. Off the checkpoint layout when cuBLASLt is there.
    if c.lt {
        spark_runtime::cublaslt::bf16_gemm_act_weight_n(
            c.low.0, c.w.up_w.0, c.up_pre.0, ts, hc_dim, rank, stream,
        )?;
    } else {
        gemm_raw(
            gpu, k_gemm, c.low, c.up_wt, c.up_pre, ts, hc_dim, rank, stream,
        )?;
    }
    hc_inject(c, k_gemm, ts, stream)?;

    KernelLaunch::new(gpu, k_mix)
        .grid([ts, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(c.normed)
        .arg_ptr(c.up_pre)
        .arg_ptr(if c.inject { c.inj_pre } else { DevicePtr::NULL })
        .arg_ptr(io.y)
        .arg_ptr(io.inj)
        .arg_u32(c.hidden)
        .arg_u32(c.hc_mult)
        .arg_f32(inv_hc)
        .launch(stream)
}

/// low = silu(low_pre / hc), in place over the slab's `[ts, rank]`.
pub(super) fn hc_silu(c: &HcPrefillCall, ts: u32, stream: u64) -> Result<()> {
    let k_silu = c.gpu.kernel("hyper_connection", "hc_silu_scale")?;
    let n_low = ts * c.rank();
    KernelLaunch::new(c.gpu, k_silu)
        .grid([n_low.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(c.low)
        .arg_u32(n_low)
        .arg_f32(1.0f32 / c.hc_mult as f32)
        .launch(stream)
}

/// inj_pre = normed x inject_w^T   [ts, hc]   (N=4: one CTA), when injecting.
pub(super) fn hc_inject(
    c: &HcPrefillCall,
    k_gemm: KernelHandle,
    ts: u32,
    stream: u64,
) -> Result<()> {
    if !c.inject {
        return Ok(());
    }
    hc_gemm(
        c.gpu,
        k_gemm,
        c.normed,
        c.w.inject_w,
        c.inj_pre,
        ts,
        c.hc_mult,
        c.hc_dim(),
        c.sm_count,
        stream,
    )
}
