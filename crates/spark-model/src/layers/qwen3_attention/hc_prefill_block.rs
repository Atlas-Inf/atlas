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
use crate::layers::{glm_sp_overlap, ops};

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
/// rows. `false` (nothing launched) when the seam does not overlap; the
/// caller then runs it as before.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_seam_overlapped(
    hc: &HcWeights,
    site: &HcSiteWeights,
    (post, gather): (bool, bool),
    hidden: DevicePtr,
    normed: DevicePtr,
    sp: Option<SpRows>,
    ctx: &ForwardContext,
    stream: u64,
    norm_for: impl Fn(DevicePtr) -> Option<HcNorm>,
    rms_norm: impl Fn(DevicePtr, DevicePtr, u32) -> Result<()>,
) -> Result<bool> {
    let Some((sp, blocks)) = sp.and_then(|sp| glm_sp_overlap::blocks(sp, ctx).map(|b| (sp, b)))
    else {
        return Ok(false);
    };
    let smallest = blocks.iter().map(|b| b.1).min().unwrap_or(0) as u32;
    let hc_mult = hc.hc_mult as u32;
    if !hc_site_blockable(post, sp.rows as u32, smallest, hc_mult, ctx) {
        return Ok(false);
    }
    let h = ctx.config.hidden_size;
    let local = sp.local(normed, h);
    glm_sp_overlap::pipelined(
        sp,
        &blocks,
        post.then_some(normed),
        gather.then_some(normed),
        h,
        ctx,
        stream,
        |(r0, rows)| {
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
            Ok(())
        },
    )?;
    Ok(true)
}
