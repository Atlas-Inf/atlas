// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill mHC pre-mix dispatch for full-attention GLM-5 layers.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{HcSiteWeights, HcWeights, Qwen3AttentionLayer};
use crate::layer::ForwardContext;
use crate::layers::ops;

fn fast_prefill(tokens: u32) -> bool {
    tokens >= 128
        && matches!(
            std::env::var("ATLAS_HC_CUBLAS_PREFILL").ok().as_deref(),
            Some("1" | "true" | "yes")
        )
}

fn fused_prefill(model: &str, hidden: u32, hc_mult: u32, tokens: u32) -> bool {
    model == "glm5_next"
        && hidden == 4096
        && hc_mult == 4
        && tokens > 8
        && matches!(
            std::env::var("ATLAS_GLM_HC_FUSED_PREFILL").ok().as_deref(),
            Some("1" | "true" | "yes")
        )
}

/// Batched prefill mHC pre shared by the GLM KDA and MLA sites. The default
/// path is a TF32 cuBLASLt pre-mix GEMM plus a finalizer. The opt-in GLM HC4
/// path computes the mix and RMS in one FP32 pass over the highway, then
/// finalizes without re-reading it for the RMS.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_pre_prefill_mix(
    site: &HcSiteWeights,
    hidden: DevicePtr,
    tokens: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    hc_eps: f32,
    fallback_finalizer: KernelHandle,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
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
        let ss = raw_mix.offset(tokens as usize * mix as usize * 4);
        let name = |base| ops::hc_kernel_name(&ctx.config.model_type, base);
        KernelLaunch::new(
            ctx.gpu,
            ctx.gpu
                .kernel("glm_hc_prefill_vec", &name("glm_hc_mix_ss"))?,
        )
        .grid([tokens.div_ceil(32), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(site.hc_fn)
        .arg_ptr(raw_mix)
        .arg_ptr(ss)
        .arg_u32(tokens)
        .launch(stream)?;
        return finalize_ss(site, hidden, tokens, sinkhorn_iters, hc_eps, ctx, stream);
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
    )
}

/// Split/Sinkhorn/collapse from the raw mix and sum of squares that
/// `glm_hc_mix_ss` or `glm_hc_post_mix_ss` left in `gate_logits_f32`.
fn finalize_ss(
    site: &HcSiteWeights,
    hidden: DevicePtr,
    tokens: u32,
    sinkhorn_iters: u32,
    hc_eps: f32,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let raw_mix = ctx.buffers.gate_logits_f32();
    let ss = raw_mix.offset(tokens as usize * 24 * 4);
    let name = ops::hc_kernel_name(&ctx.config.model_type, "glm_hc_pre_finalize_ss_vec");
    KernelLaunch::new(ctx.gpu, ctx.gpu.kernel("glm_hc_prefill_vec", &name)?)
        .grid([tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ctx.buffers.hc_streams())
        .arg_ptr(raw_mix)
        .arg_ptr(ss)
        .arg_ptr(site.hc_scale)
        .arg_ptr(site.hc_base)
        .arg_ptr(hidden)
        .arg_ptr(ctx.buffers.hc_post())
        .arg_ptr(ctx.buffers.hc_comb())
        .arg_u32(sinkhorn_iters)
        .arg_f32(ctx.config.rms_norm_eps as f32)
        .arg_f32(hc_eps)
        .launch(stream)
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
/// Returns false (nothing launched) when neither applies; callers then run
/// `hc_post` + `hc_pre` separately.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_post_pre_prefill_fused(
    next: &HcSiteWeights,
    block_out: Option<DevicePtr>,
    hidden: DevicePtr,
    tokens: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    hc_eps: f32,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<bool> {
    static ON: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    let (prefill_on, decode_on) = *ON.get_or_init(|| {
        let flag = |k| std::env::var(k).as_deref() == Ok("1");
        (
            flag("ATLAS_GLM_HC_POST_MIX"),
            flag("ATLAS_GLM_HC_DECODE_SEAM"),
        )
    });
    let prefill = prefill_on && block_out.is_some() && tokens >= 512;
    let decode = decode_on && (1..=32).contains(&tokens);
    if !(prefill || decode)
        || hc_mult != 4
        || ctx.config.hidden_size != 4096
        || ctx.config.model_type != "glm5_next"
        || !(ops::hc_bf16_for(&ctx.config.model_type)
            || fused_prefill("glm5_next", 4096, 4, tokens))
        || ctx.buffers.sizes().gate_logits_f32 < (tokens as usize * 25 * 4).max(64 * 32 * 25 * 4)
    {
        return Ok(false);
    }
    let raw_mix = ctx.buffers.gate_logits_f32();
    let name = |base| ops::hc_kernel_name(&ctx.config.model_type, base);
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
        return Ok(true);
    }
    KernelLaunch::new(
        ctx.gpu,
        ctx.gpu
            .kernel("glm_hc_prefill_vec", &name("glm_hc_post_mix_ss"))?,
    )
    .grid([tokens.div_ceil(32), 1, 1])
    .block([256, 1, 1])
    .arg_ptr(block_out.expect("prefill seam requires a post"))
    .arg_ptr(streams)
    .arg_ptr(post)
    .arg_ptr(comb)
    .arg_ptr(next.hc_fn)
    .arg_ptr(raw_mix)
    .arg_ptr(raw_mix.offset(tokens as usize * 24 * 4))
    .arg_u32(tokens)
    .launch(stream)?;
    finalize_ss(next, hidden, tokens, sinkhorn_iters, hc_eps, ctx, stream)?;
    Ok(true)
}

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_pre_prefill(
        &self,
        site: &HcSiteWeights,
        hc: &HcWeights,
        hidden: DevicePtr,
        tokens: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
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
                ctx,
                stream,
            )?
        {
            return Ok(());
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
            return Ok(());
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
        )
    }
}
