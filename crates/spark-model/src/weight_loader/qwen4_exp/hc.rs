// SPDX-License-Identifier: AGPL-3.0-only

//! The multi-hyperconnection weights: two sites per layer plus the
//! model-level mixer.
//!
//! Shapes, from the checkpoint (`hc_count = 4`, `hidden = 2560`, so
//! `hc_hidden = 10240`; `hc_lowrank = 320`):
//!
//! ```text
//! {lp}.attn_hyper_connection.hc_norm.weight                [10240]
//! {lp}.attn_hyper_connection.input_mix_weight_down.weight  [320, 10240]
//! {lp}.attn_hyper_connection.input_mix_weight_up.weight    [10240, 320]
//! {lp}.attn_hyper_connection.block_inject_weight.weight    [4, 10240]
//! {lp}.mlp_hyper_connection.*                              …same four
//! model.language_model.hyper_connection_mixer.*            …first THREE only
//! ```
//!
//! The model-level mixer has **no `block_inject_weight`** — it is built
//! `use_combine=False` in the reference and only collapses. It is also the
//! model's FINAL NORMALIZATION, which is why this checkpoint ships no
//! `model.norm.weight`.

use std::collections::HashSet;

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::WeightStore;

use crate::layers::qwen3_attention::{HcHeadWeights, HcLowRank, HcSiteWeights};
use crate::weight_map::dense;

/// Flips every `input_mix_weight_up` from the checkpoint's `[hc*H, rank]` to
/// the `[rank, hc*H]` the collapse kernels read.
///
/// WHY, in one line: the decode collapse gives one output dim to one thread
/// and contracts over `rank` sequentially, so in the checkpoint layout
/// adjacent threads walk rows 640 B apart — the kernel measured a flat
/// ~173 us regardless of T, ~38 GB/s of the part's ~273, and 23% of all decode
/// GPU time. Transposed, adjacent threads read adjacent bf16 and the
/// accumulation order is untouched, so it is coalesced AND bitwise identical.
/// The long form is in `hyper_connection.cu` under "WHY `up_w` IS STORED
/// TRANSPOSED", including the two kernel-side fixes that each got only one of
/// those two properties.
///
/// IN PLACE, through one shared staging buffer, because a second resident copy
/// is 6.55 MB x 97 sites = 635 MB on a box that loads at 113 of 119.6 GB.
///
/// IDEMPOTENT by device address. `load_head` is called once and its
/// `HcHeadWeights` cloned onto all 48 layers today, but a second transpose of
/// the same tensor would silently produce garbage rather than fail, so the
/// invariant is enforced here rather than assumed of every caller.
pub(super) struct UpTranspose<'a> {
    gpu: &'a dyn GpuBackend,
    kernel: KernelHandle,
    staging: DevicePtr,
    /// Rows of the CHECKPOINT layout, `hc_mult * hidden`.
    rows: u32,
    /// Columns of the checkpoint layout, `rank`.
    cols: u32,
    seen: HashSet<u64>,
}

impl<'a> UpTranspose<'a> {
    pub(super) fn new(gpu: &'a dyn GpuBackend, config: &ModelConfig) -> Result<Self> {
        let rows = (config.hc_mult * config.hidden_size) as u32;
        let cols = config.hc_lowrank as u32;
        let staging = gpu
            .alloc(rows as usize * cols as usize * 2)
            .context("qwen4_exp mHC: up_w transpose staging buffer")?;
        Ok(Self {
            gpu,
            kernel: gpu.kernel("hyper_connection", "hc_transpose_bf16")?,
            staging,
            rows,
            cols,
            seen: HashSet::new(),
        })
    }

    /// Transpose `up_w` in place. Copy out to staging, transpose back over the
    /// original — the two layouts are the same byte count, so the weight keeps
    /// its address and nothing downstream has to be re-pointed.
    fn apply(&mut self, up_w: DevicePtr) -> Result<()> {
        if up_w.is_null() || !self.seen.insert(up_w.0) {
            return Ok(());
        }
        let bytes = self.rows as usize * self.cols as usize * 2;
        self.gpu.copy_d2d(up_w, self.staging, bytes)?;
        let stream = self.gpu.default_stream();
        KernelLaunch::new(self.gpu, self.kernel)
            .grid([self.cols.div_ceil(32), self.rows.div_ceil(32), 1])
            .block([32, 32, 1])
            .arg_ptr(self.staging)
            .arg_ptr(up_w)
            .arg_u32(self.rows)
            .arg_u32(self.cols)
            .launch(stream)?;
        // Synchronous: the staging buffer is reused by the next site, so the
        // transpose has to have read it before that site's copy overwrites it.
        self.gpu.synchronize(stream)
    }

    /// Release the staging buffer. Not a `Drop` impl: freeing a device
    /// allocation can fail, and a loader that silently leaked 6.55 MB per
    /// model reload would be invisible until the box OOMed.
    pub(super) fn finish(self) -> Result<()> {
        self.gpu.free(self.staging)
    }
}

/// One hyper-connection site. `with_inject` is false only for the
/// model-level mixer.
fn load_site(
    store: &WeightStore,
    prefix: &str,
    rank: usize,
    with_inject: bool,
    tr: &mut UpTranspose<'_>,
) -> Result<HcLowRank> {
    let g = |name: &str| -> Result<DevicePtr> {
        dense(store, &format!("{prefix}.{name}.weight"))
            .map(|d| d.weight)
            .with_context(|| format!("qwen4_exp mHC: {prefix}.{name}.weight"))
    };
    let up_w = g("input_mix_weight_up")?;
    tr.apply(up_w)
        .with_context(|| format!("qwen4_exp mHC: transposing {prefix}.input_mix_weight_up"))?;
    Ok(HcLowRank {
        norm_w: g("hc_norm")?,
        down_w: g("input_mix_weight_down")?,
        up_w,
        inject_w: if with_inject {
            g("block_inject_weight")?
        } else {
            DevicePtr::NULL
        },
        rank,
    })
}

/// The Sinkhorn fields are NULL on this path: `lowrank.is_some()` is what
/// selects the kernel, and leaving these dangling would be a live footgun if
/// a future dispatch site forgot to branch.
fn site(lowrank: HcLowRank) -> HcSiteWeights {
    HcSiteWeights {
        hc_fn: DevicePtr::NULL,
        hc_base: DevicePtr::NULL,
        hc_scale: DevicePtr::NULL,
        lowrank: Some(lowrank),
    }
}

/// Both per-layer sites for one decoder layer.
pub(super) fn load_layer_sites(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    tr: &mut UpTranspose<'_>,
) -> Result<(HcSiteWeights, HcSiteWeights)> {
    let rank = config.hc_lowrank;
    let attn = load_site(
        store,
        &format!("{lp}.attn_hyper_connection"),
        rank,
        true,
        tr,
    )?;
    let ffn = load_site(store, &format!("{lp}.mlp_hyper_connection"), rank, true, tr)?;
    Ok((site(attn), site(ffn)))
}

/// The model-level mixer, replicated onto every layer but consumed only by
/// the last one.
pub(super) fn load_head(
    store: &WeightStore,
    config: &ModelConfig,
    tr: &mut UpTranspose<'_>,
) -> Result<HcHeadWeights> {
    let prefix = format!("{}.hyper_connection_mixer", super::embed_prefix(config));
    load_head_at(store, &prefix, config, tr)
}

/// The same mixer, read from an explicit prefix. The MTP block carries its own
/// (`mtp.hyper_connection_mixer`) which is structurally identical to the
/// model-level one — same three tensors, same `use_combine=false` collapse —
/// but sits outside `embed_prefix`, so it cannot go through [`load_head`].
pub(super) fn load_head_at(
    store: &WeightStore,
    prefix: &str,
    config: &ModelConfig,
    tr: &mut UpTranspose<'_>,
) -> Result<HcHeadWeights> {
    let lowrank = load_site(store, prefix, config.hc_lowrank, false, tr)?;
    Ok(HcHeadWeights {
        hc_fn: DevicePtr::NULL,
        hc_base: DevicePtr::NULL,
        hc_scale: DevicePtr::NULL,
        lowrank: Some(lowrank),
    })
}
