// SPDX-License-Identifier: AGPL-3.0-only

//! The SSM (Gated DeltaNet) layer of a score pass (`score.rs`).
//!
//! The projections before and after the recurrence run once over all rows,
//! exactly as `batched_layer.rs` does. The recurrence is the one ordered
//! step: the heads (each request's prefix[0..F)) run first, together, from
//! the zero state; then each head's conv window and recurrent state are
//! copied into its branches, and all branches run together from their fork
//! points. Each branch therefore sees exactly the state it would have had run
//! alone.
//!
//! A stream on a knowledge pack (`score_pack.rs`) enters from the pack's
//! state the same way a branch enters from its head's: the pointer tables name
//! the pack's slot as its entry state and conv window, read in place. Heads on
//! a pack then run their conv1d in one launch too, and their scan reads the
//! entry table instead of starting from zero.

#![allow(clippy::too_many_arguments)]

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use super::score_plan::ScorePlan;
use super::score_profile::{Phase, ScoreProfile};
use crate::layer::{
    BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers, SsmLayerState, TransformerLayer,
};
use crate::traits::SequenceState;

/// Pointer tables a layer gets from `score_stage_ssm_tables`, `n` entries
/// each: states, entry states, conv states, entry conv states.
pub(super) const TABLES: usize = 4;

fn ssm_ptrs(seq: &mut SequenceState, layer_idx: usize) -> Result<(DevicePtr, DevicePtr)> {
    let st = seq.layer_states[layer_idx]
        .as_any_mut()
        .downcast_mut::<SsmLayerState>()
        .ok_or_else(|| anyhow::anyhow!("score: layer {layer_idx} is not an SSM layer"))?;
    ensure!(
        st.h_prefill_stage.is_none(),
        "score: --ssm-h-dtype f16-pool is not supported"
    );
    Ok((st.h_state, st.conv_state))
}

/// The packed GDN buffers from row `off` for `len` rows.
fn gdn_slice(
    g: &GdnPrefillBuffers,
    off: usize,
    len: usize,
    ctx: &ForwardContext,
) -> GdnPrefillBuffers {
    let c = ctx.config;
    let key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
    let value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let (bf16, fp32) = (2usize, 4usize);
    GdnPrefillBuffers {
        qkv: g.qkv.offset(off * conv_dim * bf16),
        gate_beta: g
            .gate_beta
            .offset(off * c.linear_num_value_heads * 2 * fp32),
        output: g.output.offset(off * value_dim * bf16),
        z: g.z.offset(off * g.z_stride * bf16),
        z_stride: g.z_stride,
        total_len: len,
    }
}

impl TransformerModel {
    /// Upload the branches' `cu_seqlens`, rebased to the first branch row,
    /// once per pass (every SSM layer reuses it).
    pub(super) fn score_stage_branch_cu(
        &self,
        plan: &ScorePlan,
        meta: &BatchedAttnMetadata,
        off: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let cu = &meta.cu_seqlens_host;
        let base = cu[plan.heads];
        let rebased: Vec<i32> = cu[plan.heads..].iter().map(|c| c - base).collect();
        let dst = self.buffers.scratch().offset(off);
        // SAFETY: a live `&[i32]` reinterpreted as its own bytes.
        let bytes =
            unsafe { std::slice::from_raw_parts(rebased.as_ptr() as *const u8, rebased.len() * 4) };
        self.gpu.copy_h2d_async_retained(bytes, dst, stream)?;
        Ok(dst)
    }

    /// Once per pass: for every SSM layer, `[n]` state pointers then `[n]`
    /// entry-state pointers (a branch with a head reads the head's state, a
    /// stream on a pack without one the pack's), the tables `score_gdn_scan`
    /// takes; then the same two for the conv windows. `None` without SSM
    /// layers.
    pub(super) fn score_stage_ssm_tables(
        &self,
        plan: &ScorePlan,
        seqs: &mut [SequenceState],
        packs: &[&SequenceState],
        off: usize,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let n = seqs.len();
        let ssm: Vec<usize> = (0..self.layers.len())
            .filter(|&l| self.layers[l].is_ssm_layer())
            .collect();
        if ssm.is_empty() {
            return Ok(None);
        }
        ensure!(
            off + ssm.len() * TABLES * n * 8 <= self.buffers.scratch_bytes(),
            "score: SSM pointer tables exceed scratch ({n} streams)"
        );
        let mut tab: Vec<u64> = Vec::with_capacity(ssm.len() * TABLES * n);
        for &l in &ssm {
            let ptrs = seqs
                .iter_mut()
                .map(|s| ssm_ptrs(s, l))
                .collect::<Result<Vec<_>>>()?;
            let pack_ptrs = packs
                .iter()
                .map(|s| super::score_pack::ssm_ptrs_of(s, l))
                .collect::<Result<Vec<_>>>()?;
            let entry = |i: usize| match (plan.head_of[i], plan.reads_pack(i)) {
                (Some(h), _) => ptrs[h],
                (None, Some(p)) => pack_ptrs[p],
                (None, None) => ptrs[i],
            };
            let picks: [fn((DevicePtr, DevicePtr)) -> DevicePtr; 2] = [|p| p.0, |p| p.1];
            for pick in picks {
                tab.extend(ptrs.iter().map(|&p| pick(p).0));
                tab.extend((0..n).map(|i| pick(entry(i)).0));
            }
        }
        let bytes: Vec<u8> = tab.iter().flat_map(|p| p.to_le_bytes()).collect();
        let dst = self.buffers.scratch().offset(off);
        self.gpu.copy_h2d_async_retained(&bytes, dst, stream)?;
        Ok(Some(dst))
    }

    /// The recurrence as two single-launch scans (`score_gdn_scan`): the heads
    /// from zero, then every branch from its head's final state read in
    /// place (no copy). `Ok(false)`: the layer has no such scan and nothing
    /// ran.
    fn score_gdn_fused(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        plan: &ScorePlan,
        meta: &BatchedAttnMetadata,
        gdn: &GdnPrefillBuffers,
        branch_cu: DevicePtr,
        tabs: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let (n, heads, cu) = (plan.streams(), plan.heads, &meta.cu_seqlens_host);
        let max_len = |lo: usize, hi: usize| (lo..hi).map(|b| (cu[b + 1] - cu[b]) as u32).max();
        let g = gdn_slice(gdn, 0, cu[heads] as usize, ctx);
        // Heads start from zero, or from their pack's state (entry table).
        let heads_init = if (0..heads).any(|b| plan.reads_pack(b).is_some()) {
            tabs.offset(n * 8)
        } else {
            DevicePtr::NULL
        };
        if let Some(len) = max_len(0, heads)
            && !layer.score_gdn_scan(
                tabs,
                heads_init,
                &g,
                heads as u32,
                meta.cu_seqlens,
                len,
                ctx,
                stream,
            )?
        {
            return Ok(false);
        }
        let Some(len) = max_len(heads, n) else {
            return Ok(true);
        };
        let base = cu[heads] as usize;
        let g = gdn_slice(gdn, base, cu[n] as usize - base, ctx);
        let (h, init) = (tabs.offset(heads * 8), tabs.offset((n + heads) * 8));
        let nb = (n - heads) as u32;
        if layer.score_gdn_scan(h, init, &g, nb, branch_cu, len, ctx, stream)? {
            return Ok(true);
        }
        ensure!(
            heads == 0,
            "score: layer {layer_idx} scanned its heads but not its branches"
        );
        Ok(false)
    }

    /// The branches' conv1d in one launch (`score_conv1d`), each from its
    /// head's updated window read in place (no fork copy). `Ok(false)`: the
    /// layer has no such launch and nothing ran.
    fn score_conv1d_branches(
        &self,
        layer: &dyn TransformerLayer,
        plan: &ScorePlan,
        meta: &BatchedAttnMetadata,
        gdn: &GdnPrefillBuffers,
        branch_cu: DevicePtr,
        tabs: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let (n, heads, cu) = (plan.streams(), plan.heads, &meta.cu_seqlens_host);
        let at = |t: usize, s: usize| tabs.offset((t * n + s) * 8);
        let Some(len) = (heads..n).map(|b| (cu[b + 1] - cu[b]) as u32).max() else {
            return Ok(true);
        };
        let base = cu[heads] as usize;
        let g = gdn_slice(gdn, base, cu[n] as usize - base, ctx);
        let nb = (n - heads) as u32;
        layer.score_conv1d(
            [at(2, heads), at(3, heads)],
            base,
            &g,
            nb,
            branch_cu,
            len,
            ctx,
            stream,
        )
    }

    /// Heads on a pack: their conv1d in one launch (`score_conv1d`), each from
    /// its entry window (the pack's, or its own zeroed one) read in place.
    /// `Ok(false)`: no head reads a pack, or no such launch; nothing ran.
    fn score_conv1d_pack_heads(
        &self,
        layer: &dyn TransformerLayer,
        plan: &ScorePlan,
        meta: &BatchedAttnMetadata,
        gdn: &GdnPrefillBuffers,
        tabs: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let (n, heads, cu) = (plan.streams(), plan.heads, &meta.cu_seqlens_host);
        if !(0..heads).any(|b| plan.reads_pack(b).is_some()) {
            return Ok(false);
        }
        let at = |t: usize| tabs.offset(t * n * 8);
        let len = (0..heads)
            .map(|b| (cu[b + 1] - cu[b]) as u32)
            .max()
            .unwrap_or(0);
        let g = gdn_slice(gdn, 0, cu[heads] as usize, ctx);
        layer.score_conv1d(
            [at(2), at(3)],
            0,
            &g,
            heads as u32,
            meta.cu_seqlens,
            len,
            ctx,
            stream,
        )
    }

    /// Run the recurrence of the contiguous streams `lo..hi` together, from
    /// each stream's current state. `cu_dev` holds their `cu_seqlens` rebased
    /// to stream `lo`'s first row.
    fn score_gdn_group(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        seqs: &mut [&mut SequenceState],
        (lo, hi): (usize, usize),
        cu: &[i32],
        cu_dev: DevicePtr,
        gdn: &GdnPrefillBuffers,
        h_ptrs_off: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if lo == hi {
            return Ok(());
        }
        let base = cu[lo] as usize;
        let lens: Vec<usize> = (lo..hi).map(|b| (cu[b + 1] - cu[b]) as usize).collect();
        let group = &mut seqs[lo..hi];
        if group.len() == 1 {
            let g = gdn_slice(gdn, base, lens[0], ctx);
            return layer.prefill_gdn_full(
                group[0].layer_states[layer_idx].as_mut(),
                &g,
                ctx,
                stream,
            );
        }
        let gb = gdn_slice(gdn, base, lens.iter().sum(), ctx);
        let ptrs = self.stage_h_state_ptrs(layer_idx, group, h_ptrs_off, stream)?;
        let nb = group.len() as u32;
        let ran = if lens.iter().all(|&l| l == lens[0]) {
            layer.prefill_gdn_full_batched(ptrs, &gb, nb, lens[0] as u32, ctx, stream)?;
            true
        } else {
            let total_nt: usize = lens.iter().map(|l| l.div_ceil(64)).sum();
            let max_nc = lens.iter().map(|l| l.div_ceil(64)).max().unwrap_or(0) as u32;
            let max_sl = lens.iter().copied().max().unwrap_or(0) as u32;
            layer.prefill_gdn_full_batched_fla_varlen(
                ptrs, &gb, nb, cu_dev, max_nc, total_nt, max_sl, ctx, stream,
            )?
        };
        if !ran {
            // No varlen kernel for this layer: one stream at a time.
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!("score: no varlen GDN kernel; streams recur one at a time")
            });
            let mut off = base;
            for (seq, &len) in group.iter_mut().zip(&lens) {
                let g = gdn_slice(gdn, off, len, ctx);
                layer.prefill_gdn_full(seq.layer_states[layer_idx].as_mut(), &g, ctx, stream)?;
                off += len;
            }
        }
        Ok(())
    }

    /// Copy each head's conv window (`conv`) or recurrent state into its
    /// branches.
    fn score_fork(
        &self,
        seqs: &mut [&mut SequenceState],
        plan: &ScorePlan,
        layer_idx: usize,
        conv: bool,
        stream: u64,
    ) -> Result<()> {
        let bytes = if conv {
            self.ssm_pool.conv_bytes
        } else {
            self.ssm_pool.h_stored_bytes
        };
        for b in plan.heads..plan.streams() {
            let Some(h) = plan.head_of[b] else { continue };
            let pick = |p: (DevicePtr, DevicePtr)| if conv { p.1 } else { p.0 };
            let src = pick(ssm_ptrs(seqs[h], layer_idx)?);
            let dst = pick(ssm_ptrs(seqs[b], layer_idx)?);
            self.gpu.copy_d2d_async(src, dst, bytes, stream)?;
        }
        Ok(())
    }

    pub(super) fn score_ssm_layer(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        hidden: DevicePtr,
        residual: DevicePtr,
        seqs: &mut [&mut SequenceState],
        packs: &[&SequenceState],
        plan: &ScorePlan,
        meta: &BatchedAttnMetadata,
        gdn: &GdnPrefillBuffers,
        h_ptrs_off: usize,
        branch_cu: DevicePtr,
        tabs: Option<DevicePtr>,
        ctx: &ForwardContext,
        stream: u64,
        prof: &mut ScoreProfile,
    ) -> Result<()> {
        let n = seqs.len();
        let total = meta.total_tokens as usize;
        let cu = &meta.cu_seqlens_host;
        let heads = plan.heads;

        // Token-parallel projections once over every row.
        layer.prefill_phase1_proj_batched(hidden, residual, total, gdn, ctx, stream)?;
        prof.mark(self, stream, Phase::SsmProj)?;

        // conv1d: heads first, then fork their windows into the branches, or
        // run every branch in one launch from its head's window in place.
        // Heads on a pack run in one launch from their entry windows too; a
        // layer without these launches copies the pack's window in first.
        let heads_fused = match tabs {
            Some(t) => self.score_conv1d_pack_heads(layer, plan, meta, gdn, t, ctx, stream)?,
            None => false,
        };
        if !heads_fused && !packs.is_empty() {
            self.score_pack_seed(plan, seqs, packs, layer_idx, true, 0..heads, stream)?;
        }
        for b in (if heads_fused { heads } else { 0 })..n {
            if b == heads {
                let fused = match tabs {
                    Some(t) => self
                        .score_conv1d_branches(layer, plan, meta, gdn, branch_cu, t, ctx, stream)?,
                    None => false,
                };
                if fused {
                    break;
                }
                self.score_fork(seqs, plan, layer_idx, true, stream)?;
                if !packs.is_empty() {
                    self.score_pack_seed(plan, seqs, packs, layer_idx, true, heads..n, stream)?;
                }
            }
            let (off, len) = (cu[b] as usize, (cu[b + 1] - cu[b]) as usize);
            layer.prefill_phase1_conv1d_one(
                seqs[b].layer_states[layer_idx].as_mut(),
                off,
                len,
                gdn,
                ctx,
                stream,
            )?;
        }
        layer.prefill_phase1_l2_batched(total, gdn, ctx, stream)?;
        prof.mark(self, stream, Phase::SsmConv)?;

        // Recurrence: heads from zero (their cu_seqlens are the first heads+1
        // entries of the pass's, already based at 0), fork, then branches.
        let fused = match tabs {
            Some(t) => {
                self.score_gdn_fused(layer, layer_idx, plan, meta, gdn, branch_cu, t, ctx, stream)?
            }
            None => false,
        };
        if !fused {
            if !packs.is_empty() {
                self.score_pack_seed(plan, seqs, packs, layer_idx, false, 0..n, stream)?;
            }
            let g = (gdn, h_ptrs_off, ctx, stream);
            self.score_gdn_group(
                layer,
                layer_idx,
                seqs,
                (0, heads),
                cu,
                meta.cu_seqlens,
                g.0,
                g.1,
                g.2,
                g.3,
            )?;
            self.score_fork(seqs, plan, layer_idx, false, stream)?;
            self.score_gdn_group(
                layer,
                layer_idx,
                seqs,
                (heads, n),
                cu,
                branch_cu,
                g.0,
                g.1,
                g.2,
                g.3,
            )?;
        }
        prof.mark(self, stream, Phase::SsmGdn)?;

        // Gated norm, out-projection and MoE once over every row.
        layer.prefill_phase3(hidden, residual, total, gdn, 0, ctx, stream)?;
        prof.mark(self, stream, Phase::SsmOut)
    }
}
