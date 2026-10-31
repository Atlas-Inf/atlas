// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill mHC pre-mix dispatch for full-attention GLM-5 layers.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{HcSiteWeights, HcWeights, Qwen3AttentionLayer};
use crate::layer::ForwardContext;
use crate::layers::ops;

pub(super) fn fast_prefill(tokens: u32) -> bool {
    tokens >= 128
        && matches!(
            std::env::var("ATLAS_HC_CUBLAS_PREFILL").ok().as_deref(),
            Some("1" | "true" | "yes")
        )
}

pub(super) fn fused_prefill(model: &str, hidden: u32, hc_mult: u32, tokens: u32) -> bool {
    model == "glm5_next"
        && hidden == 4096
        && hc_mult == 4
        && tokens > 8
        && matches!(
            std::env::var("ATLAS_GLM_HC_FUSED_PREFILL").ok().as_deref(),
            Some("1" | "true" | "yes")
        )
}

/// `ATLAS_GLM_HC_PREFILL_FUSE2=1`: a prefill pre site's finalizer also writes
/// the block's input norm ([`HcNorm`]), so the separate `rms_norm` launch
/// and its re-read of `hidden` go away; the same bytes.
pub(crate) fn hc_prefill_fuse2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_HC_PREFILL_FUSE2").as_deref() == Ok("1"))
}

/// A site's block input norm for the finalizer to fuse:
/// `out = rms_norm_vanilla(hidden, weight)` (BF16 `[4096]` weight, the
/// config's `rms_norm_eps`), the separate launch's bytes.
#[derive(Clone, Copy)]
pub(crate) struct HcNorm {
    pub(crate) weight: DevicePtr,
    pub(crate) out: DevicePtr,
}

/// What [`hc_post_pre_prefill_fused`] ran.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HcSeam {
    /// Nothing: run the post and the next pre separately.
    Skipped,
    /// The post and the next site's pre.
    Ran,
    /// Those and the [`HcNorm`] it was given.
    RanNormed,
}

impl HcSeam {
    pub(crate) fn ran(self) -> bool {
        self != Self::Skipped
    }

    pub(crate) fn normed(self) -> bool {
        self == Self::RanNormed
    }
}

/// Batched prefill mHC pre shared by the GLM KDA and MLA sites. The default
/// path is a TF32 cuBLASLt pre-mix GEMM plus a finalizer. The opt-in GLM HC4
/// path computes the mix and RMS in one FP32 pass over the highway, then
/// finalizes without re-reading it for the RMS; that finalizer also writes
/// `norm` when given one. Returns whether it did (the TF32 path never does).
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_pre_prefill_mix(
    site: &HcSiteWeights,
    hidden: DevicePtr,
    tokens: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    hc_eps: f32,
    fallback_finalizer: KernelHandle,
    norm: Option<HcNorm>,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<bool> {
    let streams = ctx.buffers.hc_streams();
    let h = ctx.config.hidden_size as u32;
    let mix = (2 + hc_mult) * hc_mult;
    let raw_mix = ctx.buffers.gate_logits_f32();
    let norm_eps = ctx.config.rms_norm_eps as f32;
    let bf16 = ops::hc_bf16_for(&ctx.config.model_type);
    // A BF16 highway always takes the fused mixer (the TF32 GEMM reads FP32).
    if bf16 || fused_prefill(&ctx.config.model_type, h, hc_mult, tokens) {
        ensure!(
            ctx.buffers.sizes().gate_logits_f32 >= tokens as usize * (mix as usize + 1) * 4,
            "mHC fused pre-mix scratch is too small"
        );
        mix_ss(site, None, 0, tokens, ctx, stream)?;
        finalize_ss(
            site,
            hidden,
            0,
            tokens,
            sinkhorn_iters,
            hc_eps,
            norm,
            ctx,
            stream,
        )?;
        return Ok(norm.is_some());
    }
    ensure!(
        ctx.buffers.sizes().gate_logits_f32 >= tokens as usize * mix as usize * 4,
        "mHC TF32 pre-mix scratch is too small"
    );
    spark_runtime::cublaslt::tf32_gemm_act_weight_t(
        streams.0,
        site.hc_fn.0,
        raw_mix.0,
        tokens,
        mix,
        hc_mult * h,
        stream,
    )?;
    ops::hc_pre_from_raw_mix(
        ctx.gpu,
        ops::glm_hc_prefill_finalize_kernel(
            ctx.gpu,
            &ctx.config.model_type,
            fallback_finalizer,
            tokens,
            h,
            hc_mult,
        )?,
        streams,
        raw_mix,
        site.hc_scale,
        site.hc_base,
        hidden,
        ctx.buffers.hc_post(),
        ctx.buffers.hc_comb(),
        tokens,
        h,
        hc_mult,
        sinkhorn_iters,
        norm_eps,
        hc_eps,
        stream,
    )?;
    Ok(false)
}

/// The highway, post and comb rows from this rank's row `row0` on.
fn hc_rows(ctx: &ForwardContext, row0: usize) -> (DevicePtr, DevicePtr, DevicePtr) {
    let b = ctx.buffers;
    let highway_row = 4 * ctx.config.hidden_size * ops::hc_elem_bytes(&ctx.config.model_type);
    (
        b.hc_streams().offset(row0 * highway_row),
        b.hc_post().offset(row0 * 4 * 4),
        b.hc_comb().offset(row0 * 16 * 4),
    )
}

/// Tokens a warp, kernel-name suffix and block size of the
/// `glm_hc_mix_ss` / `glm_hc_post_mix_ss` tile a call runs: a row's bytes
/// are the same at any tile (a token is one warp's, or with `m2` one warp
/// pair's, its columns and order fixed), so the tile is chosen for speed
/// alone. Cold-L2 medians on one GB10 (BF16 highway, register-prefetching
/// kernels) at 512 / 2048 / 8192 rows: the post-mix at two tokens a warp
/// 234 / 819 / 3115 us (one: 258 / 835 / 3092; the kernels before the
/// prefetch 294 / 930 / 3473 at their best tile); the mix at two with the
/// 24 dots split over a warp pair 170 / 491 / 1560 us (one, unsplit: 184 /
/// 520 / 1774; before: 234 / 571 / 1652). At 1024 rows every tile is within
/// 7%.
fn mix_tile(post: bool) -> (u32, &'static str, u32) {
    match post {
        true => (2, "_w2", 256),
        false => (2, "_w2m2", 512),
    }
}

/// `glm_hc_mix_ss` (or, with the finishing site's `block_out`,
/// `glm_hc_post_mix_ss`) over highway rows `[row0, row0 + tokens)`: the raw
/// mix and sum of squares of each row into `gate_logits_f32`. Both kernels
/// compute each row on its own, so a row block gets the whole call's bytes.
pub(super) fn mix_ss(
    next: &HcSiteWeights,
    block_out: Option<DevicePtr>,
    row0: usize,
    tokens: u32,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let raw_mix = ctx.buffers.gate_logits_f32();
    let ss = raw_mix.offset(tokens as usize * 24 * 4);
    let (streams, post, comb) = hc_rows(ctx, row0);
    let (per_warp, tile, threads) = mix_tile(block_out.is_some());
    let launch = |kernel: &str| -> Result<KernelLaunch<'_>> {
        let name = ops::hc_kernel_name(&ctx.config.model_type, &format!("{kernel}{tile}"));
        Ok(
            KernelLaunch::new(ctx.gpu, ctx.gpu.kernel("glm_hc_prefill_vec", &name)?)
                .grid([tokens.div_ceil(8 * per_warp), 1, 1])
                .block([threads, 1, 1]),
        )
    };
    match block_out {
        Some(block_out) => launch("glm_hc_post_mix_ss")?
            .arg_ptr(block_out)
            .arg_ptr(streams)
            .arg_ptr(post)
            .arg_ptr(comb),
        None => launch("glm_hc_mix_ss")?.arg_ptr(streams),
    }
    .arg_ptr(next.hc_fn)
    .arg_ptr(raw_mix)
    .arg_ptr(ss)
    .arg_u32(tokens)
    .launch(stream)
}

/// Split/Sinkhorn/collapse from the raw mix and sum of squares that
/// `glm_hc_mix_ss` or `glm_hc_post_mix_ss` left in `gate_logits_f32`, plus
/// `norm` of the collapsed rows when given (`glm_hc_pre_finalize_ss_norm`),
/// over highway rows `[row0, row0 + tokens)` (`hidden` and `norm` at their
/// first row).
#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_ss(
    site: &HcSiteWeights,
    hidden: DevicePtr,
    row0: usize,
    tokens: u32,
    sinkhorn_iters: u32,
    hc_eps: f32,
    norm: Option<HcNorm>,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let raw_mix = ctx.buffers.gate_logits_f32();
    let (streams, post, comb) = hc_rows(ctx, row0);
    let ss = raw_mix.offset(tokens as usize * 24 * 4);
    let base = match norm {
        Some(_) => "glm_hc_pre_finalize_ss_norm",
        None => "glm_hc_pre_finalize_ss_vec",
    };
    let name = ops::hc_kernel_name(&ctx.config.model_type, base);
    let mut launch = KernelLaunch::new(ctx.gpu, ctx.gpu.kernel("glm_hc_prefill_vec", &name)?)
        .grid([tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(raw_mix)
        .arg_ptr(ss)
        .arg_ptr(site.hc_scale)
        .arg_ptr(site.hc_base)
        .arg_ptr(hidden)
        .arg_ptr(post)
        .arg_ptr(comb);
    if let Some(norm) = norm {
        launch = launch.arg_ptr(norm.weight).arg_ptr(norm.out);
    }
    launch
        .arg_u32(sinkhorn_iters)
        .arg_f32(ctx.config.rms_norm_eps as f32)
        .arg_f32(hc_eps)
        .launch(stream)
}

/// Which form of [`hc_post_pre_prefill_fused`] runs at `tokens` rows:
/// `(prefill, decode)`, neither when it is skipped.
pub(super) fn seam_forms(
    with_post: bool,
    tokens: u32,
    hc_mult: u32,
    ctx: &ForwardContext,
) -> (bool, bool) {
    static ON: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    let (prefill_on, decode_on) = *ON.get_or_init(|| {
        let flag = |k| std::env::var(k).as_deref() == Ok("1");
        (
            flag("ATLAS_GLM_HC_POST_MIX"),
            flag("ATLAS_GLM_HC_DECODE_SEAM"),
        )
    });
    let prefill = prefill_on && with_post && tokens >= 512;
    let decode = decode_on && (1..=32).contains(&tokens);
    let shaped = hc_mult == 4
        && ctx.config.hidden_size == 4096
        && ctx.config.model_type == "glm5_next"
        && (ops::hc_bf16_for(&ctx.config.model_type)
            || fused_prefill("glm5_next", 4096, 4, tokens))
        && ctx.buffers.sizes().gate_logits_f32 >= (tokens as usize * 25 * 4).max(64 * 32 * 25 * 4);
    (prefill && shaped, decode && shaped)
}

/// GLM HC4 seam: the finishing site's `hc_post` of `block_out` fused with
/// the next site's pre-mix, then the finalizer writes `hidden` and the next
/// post/comb. `block_out = None` runs the next site's pre alone.
///
/// * At >= 512 rows with a post (`ATLAS_GLM_HC_POST_MIX=1`): one highway pass
///   over 32-token CTAs (measured 1.78 vs 2.50 ms at 4096 rows).
/// * At <= 32 rows (`ATLAS_GLM_HC_DECODE_SEAM=1`): the highway split across 64
///   CTAs x 4-row groups, then one finalize block per row (20.6 vs ~57 us per
///   site at 8 verify rows).
///
/// Skipped (nothing launched) when neither applies; callers then run
/// `hc_post` + `hc_pre` separately. The prefill form also writes `norm`
/// when given one; the decode form never does.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_post_pre_prefill_fused(
    next: &HcSiteWeights,
    block_out: Option<DevicePtr>,
    hidden: DevicePtr,
    tokens: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    hc_eps: f32,
    norm: Option<HcNorm>,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<HcSeam> {
    let (prefill, decode) = seam_forms(block_out.is_some(), tokens, hc_mult, ctx);
    if !(prefill || decode) {
        return Ok(HcSeam::Skipped);
    }
    let raw_mix = ctx.buffers.gate_logits_f32();
    let (streams, post, comb) = (
        ctx.buffers.hc_streams(),
        ctx.buffers.hc_post(),
        ctx.buffers.hc_comb(),
    );
    if decode {
        let (partial, hc_fn) = ops::glm_decode_fuse::hc_decode_partial(
            ctx.gpu,
            &ctx.config.model_type,
            block_out.is_some(),
            next.hc_fn,
            next.hc_fn_bf16,
        )?;
        KernelLaunch::new(ctx.gpu, partial)
            .grid([64, tokens.div_ceil(4), 1])
            .block([128, 1, 1])
            .arg_ptr(block_out.unwrap_or(DevicePtr::NULL))
            .arg_ptr(streams)
            .arg_ptr(post)
            .arg_ptr(comb)
            .arg_ptr(hc_fn)
            .arg_ptr(raw_mix)
            .arg_u32(tokens)
            .launch(stream)?;
        KernelLaunch::new(
            ctx.gpu,
            ops::glm_decode_fuse::hc_decode_finalize(ctx.gpu, &ctx.config.model_type)?,
        )
        .grid([tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(raw_mix)
        .arg_ptr(next.hc_scale)
        .arg_ptr(next.hc_base)
        .arg_ptr(hidden)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_u32(tokens)
        .arg_u32(sinkhorn_iters)
        .arg_f32(ctx.config.rms_norm_eps as f32)
        .arg_f32(hc_eps)
        .launch(stream)?;
        return Ok(HcSeam::Ran);
    }
    mix_ss(next, block_out, 0, tokens, ctx, stream)?;
    finalize_ss(
        next,
        hidden,
        0,
        tokens,
        sinkhorn_iters,
        hc_eps,
        norm,
        ctx,
        stream,
    )?;
    Ok(if norm.is_some() {
        HcSeam::RanNormed
    } else {
        HcSeam::Ran
    })
}

impl Qwen3AttentionLayer {
    /// The site's pre into `hidden`; with `norm`, also its block input norm
    /// where the fused prefill mixer runs. Returns whether `norm` was written.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_pre_prefill(
        &self,
        site: &HcSiteWeights,
        hc: &HcWeights,
        hidden: DevicePtr,
        tokens: u32,
        norm: Option<HcNorm>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let streams = ctx.buffers.hc_streams();
        let h = ctx.config.hidden_size as u32;
        let hc_mult = hc.hc_mult as u32;
        if tokens <= 32
            && hc_post_pre_prefill_fused(
                site,
                None,
                hidden,
                tokens,
                hc_mult,
                hc.sinkhorn_iters as u32,
                hc.hc_eps,
                None,
                ctx,
                stream,
            )?
            .ran()
        {
            return Ok(false);
        }
        if fast_prefill(tokens) {
            return hc_pre_prefill_mix(
                site,
                hidden,
                tokens,
                hc_mult,
                hc.sinkhorn_iters as u32,
                hc.hc_eps,
                self.hc_pre_from_raw_mix_k,
                norm,
                ctx,
                stream,
            );
        }
        // Short batches (verify blocks): the exact split mix + finalize
        // spreads the mix over the SMs instead of one CTA per token.
        if ops::try_hc_pre_split(
            ctx.gpu,
            self.hc_pre_k,
            self.hc_pre_mix_k,
            self.hc_pre_from_raw_mix_k,
            ctx.buffers.gate_logits_f32(),
            ctx.buffers.sizes().gate_logits_f32,
            streams,
            site.hc_fn,
            site.hc_scale,
            site.hc_base,
            hidden,
            ctx.buffers.hc_post(),
            ctx.buffers.hc_comb(),
            tokens,
            h,
            hc_mult,
            hc.sinkhorn_iters as u32,
            ctx.config.rms_norm_eps as f32,
            hc.hc_eps,
            stream,
        )? {
            return Ok(false);
        }
        ops::hc_pre(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            site.hc_fn,
            site.hc_scale,
            site.hc_base,
            hidden,
            ctx.buffers.hc_post(),
            ctx.buffers.hc_comb(),
            tokens,
            h,
            hc_mult,
            hc.sinkhorn_iters as u32,
            ctx.config.rms_norm_eps as f32,
            hc.hc_eps,
            stream,
        )?;
        Ok(false)
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "hc_prefill_test_gpu.rs"]
mod gpu_tests;
