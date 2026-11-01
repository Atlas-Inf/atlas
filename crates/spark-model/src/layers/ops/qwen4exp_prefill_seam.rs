// SPDX-License-Identifier: AGPL-3.0-only

//! The CROSS-LAYER mHC seam of a qwen4_exp prefill
//! (`ATLAS_QWEN4EXP_PREFILL_HC=1`).
//!
//! `qwen4exp_prefill_hc::hc_post_pre_seam` fuses a sublayer's `hc_post` into
//! the next site's collapse when both sit in one layer (attention/GDN ->
//! MoE). The other half of the seams cross a layer boundary: layer L's MoE
//! `hc_post` is followed by layer L+1's attention-site `hc_pre`, in another
//! layer object. So layer L leaves its post PENDING ([`defer_post`]) and
//! layer L+1 runs it inside its collapse ([`pre_with_pending`]): the same
//! bytes, one 40 KB/token highway read fewer, on 47 more seams.
//!
//! What keeps a deferred post from being lost or misapplied:
//! * the model's LAST layer never defers (its post feeds `hc_head` and, after
//!   the trunk, the MTP drafter's read of the streams);
//! * the one thing that touches the highway between two layers -- the PLE
//!   injection at the start of its layer -- flushes it first
//!   ([`flush_pending`]);
//! * a model's FIRST layer clears it ([`clear_pending`]): `hc_expand`
//!   rewrites the highway, so anything left over from a forward that failed
//!   part-way is void;
//! * highway taps (`ATLAS_QWEN4EXP_DUMP`) turn deferral off, so every tap
//!   still sees the post it always saw;
//! * only prefill defers; decode never sees a pending post;
//! * the layer before a sequence-parallel split never defers ([`NoDefer`]):
//!   its post covers every row, and the split then keeps only the local
//!   ones. Inside the split a pending post holds the local rows, which is
//!   what the next layer's collapse runs on.
//!
//! The pending post lives in a thread-local: a forward runs its layers in
//! order on one thread.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::cell::Cell;

use super::hyper_connection_dispatch::{HcVariant, hc_post_site};
use super::qwen4exp_prefill_hc::{hc_post_pre_seam, prefill_arm};
use crate::layers::qwen3_attention::{HcSiteWeights, HcWeights};

#[derive(Clone, Copy)]
struct PendingPost {
    block_out: DevicePtr,
    num_tokens: u32,
}

thread_local! {
    static PENDING: Cell<Option<PendingPost>> = const { Cell::new(None) };
    static BLOCKED: Cell<bool> = const { Cell::new(false) };
}

/// While alive, no layer defers its post: the layer before a sequence-
/// parallel split (`model::qwen4exp_prefill_sp`) posts every row itself,
/// before the split compacts the highway.
pub struct NoDefer(());

impl NoDefer {
    pub fn new() -> Self {
        BLOCKED.with(|b| b.set(true));
        Self(())
    }
}

impl Default for NoDefer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoDefer {
    fn drop(&mut self) {
        BLOCKED.with(|b| b.set(false));
    }
}

/// Drop any pending post: the first model layer is about to rewrite the
/// highway with `hc_expand`.
pub fn clear_pending() {
    PENDING.with(|p| p.set(None));
}

/// Leave this layer's final `hc_post(block_out)` to the next layer's
/// collapse. `false`: not deferred -- the caller posts now.
pub fn defer_post(
    gpu: &dyn GpuBackend,
    hc: &HcWeights,
    block_out: DevicePtr,
    num_tokens: u32,
    hidden: u32,
) -> bool {
    let Some(w) = &hc.ffn.lowrank else {
        return false;
    };
    if hc.is_last_model_layer
        || BLOCKED.with(Cell::get)
        || HcVariant::of(hc) != HcVariant::LowRank
        || crate::layers::ple::dump::tapping()
        || prefill_arm(gpu, w, num_tokens, hidden, hc.hc_mult as u32).is_none()
    {
        return false;
    }
    PENDING.with(|p| {
        p.set(Some(PendingPost {
            block_out,
            num_tokens,
        }))
    });
    true
}

/// Run a pending post now (before something else touches the highway).
#[allow(clippy::too_many_arguments)]
pub fn flush_pending(
    gpu: &dyn GpuBackend,
    hc_post_k: KernelHandle,
    hc: &HcWeights,
    streams: DevicePtr,
    post: DevicePtr,
    hidden: u32,
    stream: u64,
) -> Result<()> {
    if let Some(p) = PENDING.with(|c| c.take()) {
        hc_post_site(
            gpu,
            hc_post_k,
            hc,
            p.block_out,
            streams,
            post,
            DevicePtr::NULL,
            streams,
            p.num_tokens,
            hidden,
            stream,
        )?;
    }
    Ok(())
}

/// `site`'s collapse with the previous layer's pending post fused in.
/// `Ok(true)`: done (y, inj and the posted highway written). `Ok(false)`:
/// the caller runs its usual `hc_pre` -- either nothing was pending, or the
/// fused arm declined and the post has just run on its own.
#[allow(clippy::too_many_arguments)]
pub fn pre_with_pending(
    gpu: &dyn GpuBackend,
    hc_post_k: KernelHandle,
    hc: &HcWeights,
    site: &HcSiteWeights,
    streams: DevicePtr,
    y_out: DevicePtr,
    post: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden: u32,
    eps: f32,
    stream: u64,
) -> Result<bool> {
    let Some(p) = PENDING.with(|c| c.take()) else {
        return Ok(false);
    };
    if p.num_tokens == num_tokens
        && hc_post_pre_seam(
            gpu,
            hc,
            site,
            p.block_out,
            streams,
            y_out,
            post,
            scratch,
            num_tokens,
            hidden,
            eps,
            stream,
        )?
    {
        return Ok(true);
    }
    hc_post_site(
        gpu,
        hc_post_k,
        hc,
        p.block_out,
        streams,
        post,
        DevicePtr::NULL,
        streams,
        p.num_tokens,
        hidden,
        stream,
    )?;
    Ok(false)
}
