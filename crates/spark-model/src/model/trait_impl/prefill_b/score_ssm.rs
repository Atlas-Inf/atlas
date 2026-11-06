// SPDX-License-Identifier: AGPL-3.0-only

//! The SSM (Gated DeltaNet) layer of a score pass (`score.rs`).
//!
//! The projections before and after the recurrence run once over all rows,
//! exactly as `batched_layer.rs` does. The recurrence is the one ordered
//! step: the head stream (prefix[0..F)) runs first from the zero state, then
//! its conv window and recurrent state are copied into every branch, and the
//! branches run together from that fork point. Each branch therefore sees
//! exactly the state it would have had run alone.

#![allow(clippy::too_many_arguments)]

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use super::score::{Phase, ScorePlan, ScoreProfile};
use crate::layer::{
    BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers, SsmLayerState, TransformerLayer,
};
use crate::traits::SequenceState;

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
        gate_beta: g.gate_beta.offset(off * c.linear_num_value_heads * 2 * fp32),
        output: g.output.offset(off * value_dim * bf16),
        z: g.z.offset(off * value_dim * bf16),
        total_len: len,
    }
}

impl TransformerModel {
    /// Upload the branches' `cu_seqlens`, rebased to the first branch row,
    /// once per pass (every SSM layer reuses it). NULL with one branch.
    pub(super) fn score_stage_branch_cu(
        &self,
        plan: &ScorePlan,
        meta: &BatchedAttnMetadata,
        off: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let first = usize::from(plan.has_head);
        let cu = &meta.cu_seqlens_host;
        let base = cu[first];
        let rebased: Vec<i32> = cu[first..].iter().map(|c| c - base).collect();
        let dst = self.buffers.scratch().offset(off);
        // SAFETY: a live `&[i32]` reinterpreted as its own bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(rebased.as_ptr() as *const u8, rebased.len() * 4)
        };
        self.gpu.copy_h2d_async_retained(bytes, dst, stream)?;
        Ok(dst)
    }

    pub(super) fn score_ssm_layer(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        hidden: DevicePtr,
        residual: DevicePtr,
        seqs: &mut [&mut SequenceState],
        has_head: bool,
        meta: &BatchedAttnMetadata,
        gdn: &GdnPrefillBuffers,
        h_ptrs_off: usize,
        branch_cu: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        prof: &mut ScoreProfile,
    ) -> Result<()> {
        let n = seqs.len();
        let total = meta.total_tokens as usize;
        let cu = &meta.cu_seqlens_host;
        let first = usize::from(has_head);
        let row = |b: usize| (cu[b] as usize, (cu[b + 1] - cu[b]) as usize);

        // Token-parallel projections once over every row.
        layer.prefill_phase1_proj_batched(hidden, residual, total, gdn, ctx, stream)?;
        prof.mark(self, stream, Phase::SsmProj)?;

        // conv1d: the head first, then fork its window into each branch.
        if has_head {
            let (off, len) = row(0);
            layer.prefill_phase1_conv1d_one(
                seqs[0].layer_states[layer_idx].as_mut(),
                off,
                len,
                gdn,
                ctx,
                stream,
            )?;
            let (_, src_conv) = ssm_ptrs(seqs[0], layer_idx)?;
            for seq in seqs[1..].iter_mut() {
                let (_, dst_conv) = ssm_ptrs(seq, layer_idx)?;
                self.gpu
                    .copy_d2d_async(src_conv, dst_conv, self.ssm_pool.conv_bytes, stream)?;
            }
        }
        for b in first..n {
            let (off, len) = row(b);
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

        // Recurrence: the head from zero, then fork h into each branch.
        if has_head {
            let (off, len) = row(0);
            let g0 = gdn_slice(gdn, off, len, ctx);
            layer.prefill_gdn_full(seqs[0].layer_states[layer_idx].as_mut(), &g0, ctx, stream)?;
            let (src_h, _) = ssm_ptrs(seqs[0], layer_idx)?;
            for seq in seqs[1..].iter_mut() {
                let (dst_h, _) = ssm_ptrs(seq, layer_idx)?;
                self.gpu
                    .copy_d2d_async(src_h, dst_h, self.ssm_pool.h_stored_bytes, stream)?;
            }
        }
        let branches = &mut seqs[first..];
        let nb = branches.len();
        let base = cu[first] as usize;
        let gb = gdn_slice(gdn, base, total - base, ctx);
        let lens: Vec<usize> = (first..n).map(|b| row(b).1).collect();
        let ptrs = self.stage_h_state_ptrs(layer_idx, branches, h_ptrs_off, stream)?;
        let ran = if lens.iter().all(|&l| l == lens[0]) {
            layer.prefill_gdn_full_batched(ptrs, &gb, nb as u32, lens[0] as u32, ctx, stream)?;
            true
        } else {
            let total_nt: usize = lens.iter().map(|l| l.div_ceil(64)).sum();
            let max_nc = lens.iter().map(|l| l.div_ceil(64)).max().unwrap_or(0) as u32;
            let max_sl = lens.iter().copied().max().unwrap_or(0) as u32;
            layer.prefill_gdn_full_batched_fla_varlen(
                ptrs, &gb, nb as u32, branch_cu, max_nc, total_nt, max_sl, ctx, stream,
            )?
        };
        if !ran {
            // No varlen kernel for this layer: one branch at a time.
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!("score: no varlen GDN kernel; branches recur one at a time")
            });
            let mut off = base;
            for (seq, &len) in branches.iter_mut().zip(&lens) {
                let g = gdn_slice(gdn, off, len, ctx);
                layer.prefill_gdn_full(seq.layer_states[layer_idx].as_mut(), &g, ctx, stream)?;
                off += len;
            }
        }

        prof.mark(self, stream, Phase::SsmGdn)?;
        // Gated norm, out-projection and MoE once over every row.
        layer.prefill_phase3(hidden, residual, total, gdn, 0, ctx, stream)?;
        prof.mark(self, stream, Phase::SsmOut)
    }
}
