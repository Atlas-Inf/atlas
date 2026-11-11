// SPDX-License-Identifier: AGPL-3.0-only

//! One GLM prefill pre site over a block of this rank's highway rows, for
//! the overlapped sequence-parallel seams (`layers::glm_sp_overlap`): the
//! kernels the whole-site call runs, on rows `[row0, row0 + rows)`. Each of
//! them computes every row on its own, so a row gets the whole call's bytes.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::hc_prefill::{HcNorm, fast_prefill, finalize_ss, fused_prefill, mix_ss, seam_forms};
use super::{HcSiteWeights, HcWeights};
use crate::layer::ForwardContext;
use crate::layers::glm_sp::SpRows;
use crate::layers::{glm_sp_moe_seam, glm_sp_overlap, ops};

/// Whether a site of `total` rows, cut into blocks of at least `min_block`
/// rows, runs per block exactly the kernels its whole call runs: the fused
/// prefill mixer and finalizer for a pre (`hc_pre` / `hc_pre_prefill` past
/// 32 rows), or the seam's prefill form with the finishing site's post.
/// Every threshold is monotone, so the smallest block and the whole decide.
fn hc_site_blockable(
    with_post: bool,
    total: u32,
    min_block: u32,
    hc_mult: u32,
    ctx: &ForwardContext,
) -> bool {
    let fused = |tokens: u32| {
        if with_post {
            return seam_forms(true, tokens, hc_mult, ctx).0;
        }
        let model = &ctx.config.model_type;
        tokens > 32
            && fast_prefill(tokens)
            && (ops::hc_bf16_for(model)
                || fused_prefill(model, ctx.config.hidden_size as u32, hc_mult, tokens))
    };
    min_block <= total && fused(min_block) && fused(total)
}

/// The pre site `next` over highway rows `[row0, row0 + rows)`, after the
/// finishing site's post of `block_out` (those rows of it) when given;
/// `hidden` and `norm.out` point at the block's first row. Returns whether
/// `norm` was written. Callers check [`hc_site_blockable`] first.
#[allow(clippy::too_many_arguments)]
fn hc_site_block(
    next: &HcSiteWeights,
    block_out: Option<DevicePtr>,
    hidden: DevicePtr,
    norm: Option<HcNorm>,
    (row0, rows): (usize, usize),
    sinkhorn_iters: u32,
    hc_eps: f32,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<bool> {
    let rows = rows as u32;
    mix_ss(next, block_out, row0, rows, ctx, stream)?;
    finalize_ss(
        next,
        hidden,
        row0,
        rows,
        sinkhorn_iters,
        hc_eps,
        norm,
        ctx,
        stream,
    )?;
    Ok(norm.is_some())
}

/// A GLM prefill seam of the pre site `site` (of `hc`) as an overlapped
/// sequence-parallel seam over this rank's rows (`ATLAS_GLM_SP_GATHER_OVERLAP`):
/// with `post` (an FFN site) it first reduce-scatters the attention output
/// in `normed` and fuses the attention site's post of those rows; it writes
/// the block input norm into `normed` (`norm_for(out)` in the finalizer, else
/// `rms_norm(hidden, out, rows)`); with `gather` it all-gathers the normed
/// rows. `false` when the seam does not overlap; the caller then runs it as
/// before. An attention seam (no `post`) also runs, block by block, the
/// previous MoE layer's pending output seam (`layers::glm_sp_moe_seam`):
/// its reduce-scatter, blend and post; when it does not overlap it runs that
/// pending seam whole first. `reduced`: the post's input rows are already
/// whole on this rank (`glm5_kda::sp_oproj_a2a`), so nothing is reduced.
/// `extra` (with no `gather`): another tensor gathered per block, its rows
/// written by `extra.rows` once the block is normed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_seam_overlapped(
    hc: &HcWeights,
    site: &HcSiteWeights,
    (post, gather): (bool, bool),
    reduced: bool,
    extra: Option<SeamGather<'_>>,
    hidden: DevicePtr,
    normed: DevicePtr,
    sp: Option<SpRows>,
    ctx: &ForwardContext,
    stream: u64,
    norm_for: impl Fn(DevicePtr) -> Option<HcNorm>,
    rms_norm: impl Fn(DevicePtr, DevicePtr, u32) -> Result<()>,
) -> Result<bool> {
    let pending = glm_sp_moe_seam::take();
    let not_overlapped = |pending: Option<glm_sp_moe_seam::Pending>| {
        pending.map_or(Ok(()), |p| p.run_whole(ctx, stream))?;
        Ok(false)
    };
    let Some((sp, blocks)) = sp.and_then(|sp| glm_sp_overlap::blocks(sp, ctx).map(|b| (sp, b)))
    else {
        return not_overlapped(pending);
    };
    let smallest = blocks.iter().map(|b| b.1).min().unwrap_or(0) as u32;
    let hc_mult = hc.hc_mult as u32;
    let pending_fits =
        |p: &glm_sp_moe_seam::Pending| !post && glm_sp_overlap::ahead_matches(p.out, &blocks);
    if !hc_site_blockable(post, sp.rows as u32, smallest, hc_mult, ctx)
        || pending.as_ref().is_some_and(|p| !pending_fits(p))
    {
        return not_overlapped(pending);
    }
    let h = ctx.config.hidden_size;
    let local = sp.local(normed, h);
    anyhow::ensure!(
        !(gather && extra.is_some()),
        "an overlapped seam gathers the normed rows or another tensor, not both"
    );
    let tensors: Vec<(DevicePtr, usize)> = match &extra {
        Some(e) => std::iter::once((e.ptr, e.width)).chain(e.also).collect(),
        None if gather => vec![(normed, h)],
        None => Vec::new(),
    };
    let group = extra.as_ref().map_or(1, |e| e.group);
    let gathered = (!tensors.is_empty()).then_some((&tensors[..], group));
    glm_sp_overlap::pipelined(
        sp,
        &blocks,
        pending
            .map(|p| p.out)
            .or((post && !reduced).then_some(normed)),
        gathered,
        h,
        ctx,
        stream,
        |(r0, rows)| {
            if let Some(p) = &pending {
                p.run_rows((r0, rows), ctx, stream)?;
            }
            let (out, block_hidden) = (local.offset(r0 * h * 2), hidden.offset(r0 * h * 2));
            let normed_too = hc_site_block(
                site,
                post.then_some(out),
                block_hidden,
                norm_for(out),
                (r0, rows),
                hc.sinkhorn_iters as u32,
                hc.hc_eps,
                ctx,
                stream,
            )?;
            if !normed_too {
                rms_norm(block_hidden, out, rows as u32)?;
            }
            if let Some(e) = &extra {
                (e.rows)((r0, rows))?;
            }
            Ok(())
        },
    )?;
    Ok(true)
}

/// A tensor an overlapped seam gathers per block instead of the normed rows
/// (`hc_seam_overlapped`'s `extra`): `[total, width]` BF16-width rows at
/// `ptr`, this rank's rows of each block written by `rows(block)` right
/// after the block is normed (the routed MoE's quantized input,
/// `MoeLayer::sp_route_rows`).
pub(crate) struct SeamGather<'a> {
    pub ptr: DevicePtr,
    pub width: usize,
    /// A second `[total, width]` tensor gathered with it (the f32 scales of
    /// the MLA input's E4M3 rows, `prefill/paged_glm_sp_gather.rs`).
    pub also: Option<(DevicePtr, usize)>,
    /// Blocks a gather covers.
    pub group: usize,
    pub rows: &'a dyn Fn((usize, usize)) -> Result<()>,
}
