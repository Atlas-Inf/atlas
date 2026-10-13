// SPDX-License-Identifier: AGPL-3.0-only

//! Owner-batched long-context verify for an MLA layer.
//!
//! Attention runs per owner, unchanged: each owner's `rows` rows are placed at
//! arena rows [0, rows) and go through the same validated eager multi-row
//! verify path (causal semantic index, sparse attention, cache writes) with
//! that owner's own metadata and state; `validate_glm_long_verify` admits the
//! width for the active lane (K3 on repaired MTP, 2..=8 on DFlash). The FFN
//! runs through `ffn_per_owner`; the mHC post is joint.

use super::*;
use crate::layer::AttnMetadataDev;
use crate::layer::glm_long_owner::{self as long_owner, GlmLongOwner, GlmLongStage};
use anyhow::{Context, ensure};

impl Qwen3AttentionLayer {
    pub(in crate::layers::qwen3_attention) fn decode_glm_long_owners_mla(
        &self,
        owners: &mut [GlmLongOwner<'_>],
        kv_cache: &mut PagedKvCache,
        stage: &GlmLongStage,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.mla.is_some() && self.hc.is_some() && !self.ffn.is_none(),
            "GLM long owner batch requires an mHC MLA layer with FFN"
        );
        ensure!(
            self.block_idx != 0 && self.block_idx + 1 != ctx.config.num_hidden_layers,
            "GLM long owner batch MLA cannot be the first or last layer"
        );
        if crate::speculative::glm_repair_policy::dflash_enabled() {
            return self.glm_long_owners_mla_chunk(owners, kv_cache, stage, ctx, stream);
        }
        let rows = long_owner::owner_rows(owners)?;
        let total = owners.len() * rows;
        let b = ctx.buffers;
        let r = stage.rows;
        let ffn_spans = [
            (b.hc_streams(), stage.highway, r.highway),
            (b.hc_post(), stage.post, r.post),
            (b.hc_comb(), stage.comb, r.comb),
            (b.norm_output(), stage.norm, r.hidden),
        ];
        // Every owner's incoming highway, before owner 0's attention reuses rows [0, rows).
        stage.copy(ctx.gpu, &ffn_spans[..1], 0, 0, total, true, stream)?;
        let row_owner = &[0usize; long_owner::MAX_OWNER_ROWS][..rows];
        let bs = kv_cache.block_size() as u32;
        for (owner, input) in owners.iter_mut().enumerate() {
            let first = owner * rows;
            stage.copy(ctx.gpu, &ffn_spans[..1], 0, first, rows, false, stream)?;
            let owner_ctx = ForwardContext {
                attn_metadata: Some(input.meta),
                midchunk_capture: None,
                ..*ctx
            };
            let mut c = ctx::MultiSeqCtx::new(
                self,
                &owner_ctx,
                b.hidden_states(),
                b.residual(),
                rows,
                &input.positions,
                bs,
                stream,
            );
            c.seq_slot = input.meta.seq_slot;
            let mut states: [&mut (dyn LayerState + 'static); 1] = [&mut *input.state];
            self.ms_hc_attention_norm_impl(
                &c,
                kv_cache,
                &owner_ctx,
                stream,
                Some(&mut states),
                Some(row_owner),
                // GLM layers carry no QSA indexer; the batched path is exact.
                false,
                &input.positions,
            )?
            .context("GLM long owner MLA produced no FFN phase")?;
            stage.copy(ctx.gpu, &ffn_spans, 0, first, rows, true, stream)?;
        }
        // Every owner's post-attention highway, mHC coefficients and FFN input.
        stage.copy(ctx.gpu, &ffn_spans, 0, 0, total, false, stream)?;
        let ffn_out = long_owner::ffn_per_owner(&self.ffn, owners.len(), rows, stage, ctx, stream)?;
        let positions: Vec<usize> = owners
            .iter()
            .flat_map(|o| o.positions.iter().copied())
            .collect();
        let joint = ctx::MultiSeqCtx::new(
            self,
            ctx,
            b.hidden_states(),
            b.residual(),
            total,
            &positions,
            bs,
            stream,
        );
        self.ms_hc_supplied_post(
            &joint,
            hc_ffn::HcFfnPhase {
                hc_streams: b.hc_streams(),
                post: b.hc_post(),
                comb: b.hc_comb(),
                diag_this: false,
            },
            ffn_out,
            ctx,
            stream,
        )
    }

    /// DFlash lane: the single-owner verify runs MLA layers as a causal
    /// prefill chunk, so here every row-wise step (mHC pre/post, norms, TP
    /// all-reduce, FFN) runs once over all owners' rows and only the chunk
    /// attention runs per owner, through the same prefill kernels.
    fn glm_long_owners_mla_chunk(
        &self,
        owners: &mut [GlmLongOwner<'_>],
        kv_cache: &mut PagedKvCache,
        stage: &GlmLongStage,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rows = long_owner::owner_rows(owners)?;
        let chunk_owners: Vec<_> = owners
            .iter()
            .enumerate()
            .map(|(owner, input)| owner_chunk(owner * rows, input))
            .collect();
        self.glm_mla_chunk_owners(
            &chunk_owners,
            Some(stage),
            kv_cache,
            ctx,
            stream,
            &mut || long_owner::ffn_per_owner(&self.ffn, owners.len(), rows, stage, ctx, stream),
        )
    }

    /// A prefill chunk of `num_tokens` rows (`ctx.attn_metadata` holds the
    /// chunk's paged metadata, with positions and slots covering every row)
    /// carrying verify owners at `num_tokens + owner * rows`: the chunk's
    /// pieces and one chunk owner per passenger share one row-wise pass.
    pub(in crate::layers::qwen3_attention) fn prefill_glm_passengers_mla(
        &self,
        num_tokens: usize,
        seq_len_start: usize,
        passengers: &[GlmLongOwner<'_>],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.mla.as_ref().is_some_and(|m| m.glm_indexer.is_some())
                && self.hc.is_some()
                && !self.ffn.is_none(),
            "GLM fused prefill + verify requires an mHC GLM MLA layer with FFN"
        );
        ensure!(
            self.block_idx != 0 && self.block_idx + 1 != ctx.config.num_hidden_layers,
            "GLM fused prefill + verify MLA cannot be the first or last layer"
        );
        let rows = long_owner::owner_rows(passengers)?;
        let meta = ctx
            .attn_metadata
            .context("GLM fused prefill + verify requires chunk metadata")?;
        let mut owners = crate::layers::qwen3_attention::prefill::glm_chunk_pieces(
            meta,
            seq_len_start,
            num_tokens,
            ctx.config.index_topk,
        );
        owners.extend(
            passengers
                .iter()
                .enumerate()
                .map(|(owner, input)| owner_chunk(num_tokens + owner * rows, input)),
        );
        let total = num_tokens + passengers.len() * rows;
        let b = ctx.buffers;
        self.glm_mla_chunk_owners(&owners, None, kv_cache, ctx, stream, &mut || {
            self.ffn
                .forward_prefill(b.norm_output(), total, ctx, stream)?;
            Ok(b.moe_output())
        })
    }

    /// GLM's MLA layer over chunk owners that tile the stacked rows: mHC
    /// pre/post, norms, TP all-reduce and `ffn` run once over every row and
    /// `glm_chunk_attention` per owner. `stash` keeps the mHC coefficients in
    /// the verify stage across the attention.
    fn glm_mla_chunk_owners(
        &self,
        owners: &[crate::layers::qwen3_attention::prefill::GlmChunkOwner],
        stash: Option<&GlmLongStage>,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
        ffn: &mut dyn FnMut() -> Result<DevicePtr>,
    ) -> Result<()> {
        let hc = self.hc.as_ref().context("GLM MLA chunk owners need mHC")?;
        ensure!(
            ops::HcVariant::of(hc).applies_block_input_norm() && self.post_attn_out_norm.is_none(),
            "GLM MLA chunk owners expect GLM's mHC norm layout"
        );
        let total: usize = owners.iter().map(|o| o.rows).sum();
        let n = total as u32;
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let b = ctx.buffers;
        let (hidden, normed) = (b.hidden_states(), b.norm_output());
        self.hc_pre_prefill(&hc.attn, hc, hidden, n, ctx, stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            hidden,
            &self.input_norm,
            normed,
            n,
            h as u32,
            eps,
            stream,
        )?;
        let coeffs = stash.map(|stage| {
            [
                (b.hc_post(), stage.post, stage.rows.post),
                (b.hc_comb(), stage.comb, stage.rows.comb),
            ]
        });
        if let (Some(stage), Some(coeffs)) = (stash, &coeffs) {
            stage.copy(ctx.gpu, coeffs, 0, 0, total, true, stream)?;
        }
        let attn_out =
            self.prefill_attention_glm_owners(owners, normed, total, kv_cache, ctx, stream)?;
        if let (Some(stage), Some(coeffs)) = (stash, &coeffs) {
            stage.copy(ctx.gpu, coeffs, 0, 0, total, false, stream)?;
        }
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(attn_out.0, total * h * 2, stream)?;
        }
        let streams = b.hc_streams();
        let (post, comb) = (b.hc_post(), b.hc_comb());
        let seam = crate::layers::qwen3_attention::hc_post_pre_prefill_fused(
            &hc.ffn,
            Some(attn_out),
            hidden,
            n,
            hc.hc_mult as u32,
            hc.sinkhorn_iters as u32,
            hc.hc_eps,
            ctx,
            stream,
        )?;
        if !seam {
            ops::hc_post_site(
                ctx.gpu,
                self.hc_post_k,
                hc,
                attn_out,
                streams,
                post,
                comb,
                streams,
                n,
                h as u32,
                stream,
            )?;
            self.hc_pre_prefill(&hc.ffn, hc, hidden, n, ctx, stream)?;
        }
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            hidden,
            &self.post_attn_norm,
            normed,
            n,
            h as u32,
            eps,
            stream,
        )?;
        let ffn_out = ffn()?;
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            ffn_out,
            streams,
            post,
            comb,
            streams,
            n,
            h as u32,
            stream,
        )
    }
}

/// One verify owner's `rows` rows at `row0` as a causal chunk owner.
fn owner_chunk(
    row0: usize,
    input: &GlmLongOwner<'_>,
) -> crate::layers::qwen3_attention::prefill::GlmChunkOwner {
    crate::layers::qwen3_attention::prefill::GlmChunkOwner {
        row0,
        rows: input.positions.len(),
        seq_len_start: input.positions[0],
        meta: AttnMetadataDev {
            num_seqs: 1,
            // Chunk-total length: the last row's causal extent.
            seq_len: input.meta.seq_len.offset((input.positions.len() - 1) * 4),
            ..input.meta
        },
    }
}
