// SPDX-License-Identifier: AGPL-3.0-only

//! One call site per mHC entry point, for both variants.
//!
//! Atlas now runs two different hyper-connection families over the same
//! `[T, hc_mult, H]` FP32 highway:
//!
//! * DeepSeek-V4's — a Sinkhorn-normalized mix over `hc_fn`/`hc_scale`/
//!   `hc_base`, emitting a `[hc, hc]` combine matrix that `hc_post` mixes the
//!   saved streams through.
//! * Qwen3.8-Flash-Next's — a low-rank pair (rank 320) with a grouped
//!   RMSNorm, emitting one scalar per stream that `hc_post` scales by.
//!
//! They share the four kernel NAMES (a model shadow overrides the whole
//! `hyper_connection.cu` file) but take different argument lists. Rather than
//! branch at each of the 23 call sites across `prefill_inner`, `decode_inner`
//! and `multi_seq`, the branch lives here once, behind a signature that is
//! the union of both.
//!
//! **`post_out` doubles as the low-rank injection vector.** Both are `[T, hc]`
//! f32 and both are consumed by the matching `hc_post`, so the existing
//! `ctx.buffers.hc_post()` serves each variant without a second allocation.
//! `comb_out` is untouched on the low-rank path — Qwen scales each stream by
//! one scalar where DeepSeek mixes them through a full matrix.
//!
//! Selection is by WEIGHTS (`lowrank.is_some()`), never by model name. A site
//! carrying both would be a load-time bug rather than a silent coin-flip.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::hyper_connection as sinkhorn;
use super::hyper_connection_lowrank as lowrank;
use super::hyper_connection_lowrank_split::hc_pre_split;
use super::qwen4exp_decode_fuse::{HcPostFold, hc_post_stage_fits};
use crate::layers::qwen3_attention::{HcHeadWeights, HcSiteWeights, HcWeights};

/// Which family a site's weights select.
///
/// Exposed so a caller can answer questions the launch itself cannot — most
/// importantly whether to apply the layer's `input_norm` after `hc_pre`.
/// Qwen has no per-layer `input_layernorm`; `hc_norm` inside `hc_pre` plays
/// that role, and the loader's ones-filled placeholder does NOT make a second
/// RMS pass an identity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HcVariant {
    /// DeepSeek-V4: Sinkhorn mix, `[hc, hc]` combine matrix.
    Sinkhorn,
    /// Qwen3.8-Flash-Next: low-rank pair, per-stream scalar injection.
    LowRank,
}

impl HcVariant {
    pub fn of_site(site: &HcSiteWeights) -> Self {
        if site.lowrank.is_some() {
            Self::LowRank
        } else {
            Self::Sinkhorn
        }
    }

    pub fn of(hc: &HcWeights) -> Self {
        Self::of_site(&hc.attn)
    }

    /// Whether the block's own `input_norm` should run on `hc_pre`'s output.
    /// False for Qwen — see the type's doc comment.
    pub fn applies_block_input_norm(self) -> bool {
        self == Self::Sinkhorn
    }
}

/// Collapse the streams to one and emit whatever the matching `hc_post`
/// needs — `post`+`comb` for Sinkhorn, the injection vector in `post_out` for
/// low-rank.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre_site(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    site: &HcSiteWeights,
    hc: &HcWeights,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    match &site.lowrank {
        Some(w) => lowrank::hc_pre_lowrank(
            gpu,
            kernel,
            streams,
            w,
            y_out,
            post_out,
            scratch,
            num_tokens,
            hidden_size,
            hc.hc_mult as u32,
            norm_eps,
            stream,
        ),
        None => sinkhorn::hc_pre(
            gpu,
            kernel,
            streams,
            site.hc_fn,
            site.hc_scale,
            site.hc_base,
            y_out,
            post_out,
            comb_out,
            num_tokens,
            hidden_size,
            hc.hc_mult as u32,
            hc.sinkhorn_iters as u32,
            norm_eps,
            hc.hc_eps,
            stream,
        ),
    }
}

/// Rows the decode-shaped (split) low-rank collapse serves in one launch
/// (`hyper_connection_lowrank::HC_DECODE_MAX_T`); wider batches take the
/// GEMM formulation, which rounds `normed` to BF16 — another function of the
/// row than the decode's.
pub const HC_SPLIT_MAX_ROWS: u32 = 8;

/// [`hc_pre_site`] for a multi-row decode or verify step. With `max_rows > 0`
/// (`ModelLevers::qwen4exp_hc_rows`: [`HC_SPLIT_MAX_ROWS`] under
/// `ATLAS_QWEN4EXP_BATCH_FAST`) a low-rank collapse of more rows runs as
/// split-path chunks of at most that many rows, each row then the single-row
/// decode collapse's bytes (the split kernels' per-row arithmetic does not
/// depend on the row count). Anything else is `hc_pre_site` unchanged.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre_site_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    site: &HcSiteWeights,
    hc: &HcWeights,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    norm_eps: f32,
    max_rows: u32,
    stream: u64,
) -> Result<()> {
    if !(max_rows > 0 && site.lowrank.is_some() && num_tokens > max_rows && !scratch.is_null()) {
        return hc_pre_site(
            gpu,
            kernel,
            streams,
            site,
            hc,
            y_out,
            post_out,
            comb_out,
            scratch,
            num_tokens,
            hidden_size,
            norm_eps,
            stream,
        );
    }
    let (m, h) = (hc.hc_mult, hidden_size as usize);
    for t0 in (0..num_tokens).step_by(max_rows as usize) {
        let t = t0 as usize;
        hc_pre_site(
            gpu,
            kernel,
            streams.offset(t * m * h * 4),
            site,
            hc,
            y_out.offset(t * h * 2),
            post_out.offset(t * m * 4),
            comb_out.offset(t * m * m * 4),
            scratch,
            (num_tokens - t0).min(max_rows),
            hidden_size,
            norm_eps,
            stream,
        )?;
    }
    Ok(())
}

/// Inject the block output back into every stream. `out` may alias
/// `residual`.
///
/// Takes the whole [`HcWeights`] rather than a site, because NEITHER variant's
/// `hc_post` reads site weights — Sinkhorn consumes the `comb` its `hc_pre`
/// emitted, low-rank the injection vector — so the layer's variant is the
/// only thing being selected on.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_site(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hc: &HcWeights,
    block_out: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    match HcVariant::of(hc) {
        // `post` IS the injection vector here; `comb` is not read.
        HcVariant::LowRank => lowrank::hc_post_lowrank(
            gpu,
            kernel,
            block_out,
            residual,
            post,
            out,
            num_tokens,
            hidden_size,
            hc.hc_mult as u32,
            stream,
        ),
        HcVariant::Sinkhorn => sinkhorn::hc_post(
            gpu,
            kernel,
            block_out,
            residual,
            post,
            comb,
            out,
            num_tokens,
            hidden_size,
            hc.hc_mult as u32,
            stream,
        ),
    }
}

/// A decode layer's seam: [`hc_post_site`] of the sublayer that produced
/// `block_out` (in place on `streams`), then [`hc_pre_site`] of `site`.
///
/// Under `ATLAS_QWEN4EXP_DECODE_FUSE` (mHC seam group) a low-rank site at a
/// decode width runs the post inside its stage (`hc_post_stage_vec`): the
/// same highway and `normed` bytes, one launch fewer. Otherwise the two
/// calls run as they always have.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_pre_site(
    gpu: &dyn GpuBackend,
    post_kernel: KernelHandle,
    pre_kernel: KernelHandle,
    hc: &HcWeights,
    block_out: DevicePtr,
    streams: DevicePtr,
    site: &HcSiteWeights,
    y_out: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    if let Some(w) = &site.lowrank
        && HcVariant::of(hc) == HcVariant::LowRank
        && num_tokens <= lowrank::HC_DECODE_MAX_T
        && !scratch.is_null()
        && !w.inject_w.is_null()
    {
        let fold = HcPostFold {
            block_out,
            inj: post,
        };
        // `normed` sits at the start of the split path's scratch.
        let hc_mult = hc.hc_mult as u32;
        if hc_post_stage_fits(gpu, fold, streams, w.norm_w, scratch, hidden_size, hc_mult)? {
            return hc_pre_split(
                gpu,
                streams,
                w,
                y_out,
                post,
                scratch,
                num_tokens,
                hidden_size,
                hc_mult,
                norm_eps,
                /* inject */ true,
                Some(fold),
                stream,
            );
        }
    }
    hc_post_site(
        gpu,
        post_kernel,
        hc,
        block_out,
        streams,
        post,
        comb,
        streams,
        num_tokens,
        hidden_size,
        stream,
    )?;
    hc_pre_site(
        gpu,
        pre_kernel,
        streams,
        site,
        hc,
        y_out,
        post,
        comb,
        scratch,
        num_tokens,
        hidden_size,
        norm_eps,
        stream,
    )
}

/// [`hc_head_site`] for a multi-row step, chunked like [`hc_pre_site_rows`].
#[allow(clippy::too_many_arguments)]
pub fn hc_head_site_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    head: &HcHeadWeights,
    hc: &HcWeights,
    y_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    norm_eps: f32,
    max_rows: u32,
    stream: u64,
) -> Result<()> {
    let chunk = if max_rows > 0 && head.lowrank.is_some() && !scratch.is_null() {
        max_rows
    } else {
        num_tokens.max(1)
    };
    let (m, h) = (hc.hc_mult, hidden_size as usize);
    for t0 in (0..num_tokens).step_by(chunk as usize) {
        let t = t0 as usize;
        hc_head_site(
            gpu,
            kernel,
            streams.offset(t * m * h * 4),
            head,
            hc,
            y_out.offset(t * h * 2),
            scratch,
            (num_tokens - t0).min(chunk),
            hidden_size,
            norm_eps,
            stream,
        )?;
    }
    Ok(())
}

/// The model-level final collapse before the LM head.
///
/// On Qwen this is ALSO the model's final normalization — the checkpoint
/// ships no `model.norm.weight` because `hyper_connection_mixer`'s `hc_norm`
/// plays that role.
#[allow(clippy::too_many_arguments)]
pub fn hc_head_site(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    head: &HcHeadWeights,
    hc: &HcWeights,
    y_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    match &head.lowrank {
        Some(w) => lowrank::hc_head_lowrank(
            gpu,
            kernel,
            streams,
            w,
            y_out,
            scratch,
            num_tokens,
            hidden_size,
            hc.hc_mult as u32,
            norm_eps,
            stream,
        ),
        None => sinkhorn::hc_head(
            gpu,
            kernel,
            streams,
            head.hc_fn,
            head.hc_scale,
            head.hc_base,
            y_out,
            num_tokens,
            hidden_size,
            hc.hc_mult as u32,
            norm_eps,
            hc.hc_eps,
            stream,
        ),
    }
}
