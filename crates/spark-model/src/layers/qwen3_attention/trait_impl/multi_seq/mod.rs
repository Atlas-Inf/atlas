// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-sequence batched-decode body for [`super::super::Qwen3AttentionLayer`].
//! Split into phase modules under `_inner` delegation: `ctx`, `qkv`, `attn`, `ffn`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

mod attn;
mod c4;
mod ctx;
mod ffn;
pub(crate) use ffn::{
    grouped_routed_decode_enabled, grouped_routed_decode_min, pairwise_moe_decode_enabled,
};
mod glm_long_owner;
mod guard;
mod hc_ffn;
mod hc_generic;
mod hc_pre_site;
mod mla;
mod mla_gemv;
mod mla_glm;
mod mla_glm_sparse;
mod mla_independent;
mod nemotron_serial;
mod qkv;
mod qkv_dp4a;
mod qkv_fp8;
mod qsa_rows;
#[cfg(test)]
mod tests;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    /// `row_owner`: when `Some`, row `i` belongs to sequence `row_owner[i]`, and `states` is indexed
    /// by sequence. Used for per-sequence aux state (QSA indexer) to advance once per row in order.
    pub(in crate::layers::qwen3_attention) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        row_owner: Option<&[usize]>,
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.try_nemotron_attention_serial(
            hidden,
            residual,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            _block_tables,
            ctx,
            stream,
        )? {
            return Ok(());
        }
        // Pre-mutation QSA plan: all rows inert -> batched attention; an
        // ACTIVE row the per-row phase can serve -> `qsa_rows`; anything else
        // is refused HERE, before any layer state is touched (`guard.rs`).
        let qsa_rows = guard::plan_qsa_rows(self, seq_lens, num_seqs, row_owner, kv_cache, ctx)?;
        let bs = kv_cache.block_size() as u32;
        let mut c =
            ctx::MultiSeqCtx::new(self, ctx, hidden, residual, num_seqs, seq_lens, bs, stream);
        // Per-request LoRA routing slot buffer for this step (from metadata).
        if let Some(m) = ctx.attn_metadata.as_ref() {
            c.seq_slot = m.seq_slot;
        }

        // DeepSeek-V4 / Qwen4-exp: Manifold-Constrained Hyper-Connections.
        if self.hc.is_some() {
            return self.decode_multi_seq_inner_hc(
                c, states, row_owner, qsa_rows, seq_lens, kv_cache, ctx, stream,
            );
        }
        let _ = (states, row_owner); // Non-hc attention keeps no per-seq state.

        // ── Phase 1: RMS norm + residual for N tokens ──
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            c.hidden,
            &self.input_norm,
            c.normed,
            c.residual,
            c.n as u32,
            c.h as u32,
            c.eps,
            c.stream,
        )?;

        let meta = ctx
            .attn_metadata
            .expect("attention layer requires metadata");

        // ── Phases 2-6: attention ──
        // MLA models (Mistral-Small-4) take the dedicated absorbed-MLA
        // batched path (issue #84). The standard `ms_phase_qkv` reads
        // `attn.q_proj`, a NULL stub for MLA loaders — see `mla.rs`.
        let o_out = if let Some(ref _mla) = self.mla {
            self.ms_mla_decode(&c, kv_cache, meta)?
        } else {
            // ── Phase 2: QKV projections (batch3 / batch2 / sequential) ──
            self.ms_phase_qkv(&c)?;

            // ── Phase 3: RoPE per-sequence ──
            self.ms_phase_rope(&c, meta)?;

            // ── Phase 4: KV cache write ──
            self.ms_phase_cache_write(&c, kv_cache, meta)?;

            // ── Phase 5: paged decode attention (batched) ──
            let attn_out = self.ms_phase_paged_decode(&c, kv_cache, meta)?;

            // ── Phase 6: gate multiply + O projection ──
            self.ms_phase_o_proj(&c, attn_out)?
        };

        // TP all-reduce on o_out after o_proj (Megatron row-parallel
        // pattern). Mirrors decode_inner.rs and prefill_inner.rs. Without
        // this, multi-token decode (K=2 / K=3 / K=γ verify) under
        // tp_world_size>1 reads a partial attention output from each
        // rank, corrupting the FFN/MoE input and producing degenerate
        // logits — observed as `/`/`,` repetition spirals on
        // Qwen3.6 FP8 + TP=2 + MTP for HTML/code prompts.
        if c.fwd.config.tp_world_size > 1
            && let Some(comm) = c.fwd.comm
        {
            let bytes = c.n * c.h * c.bf16;
            comm.all_reduce_async(o_out.0, bytes, c.stream)?;
        }

        // ── Phase 7: residual + post-norm + MoE ──
        self.ms_phase_ffn(&c, o_out)?;

        Ok(())
    }

    /// Run serial phases while retaining upstream per-sequence auxiliary state.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_inner_hc<'a, 'b: 'a>(
        &self,
        c: ctx::MultiSeqCtx<'_>,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        row_owner: Option<&[usize]>,
        qsa_rows: bool,
        seq_lens: &[usize],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(phase) = self.ms_hc_attention_norm_impl(
            &c,
            kv_cache,
            ctx,
            stream,
            Some(states),
            row_owner,
            qsa_rows,
            seq_lens,
        )? {
            self.ms_hc_ffn_post(&c, phase, ctx, stream)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn ms_hc_attention_norm_impl<'a, 'b: 'a>(
        &self,
        c: &ctx::MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
        mut states: Option<&'a mut [&'b mut (dyn LayerState + 'static)]>,
        row_owner: Option<&[usize]>,
        qsa_rows: bool,
        seq_lens: &[usize],
    ) -> Result<Option<hc_ffn::HcFfnPhase>> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = c.n;
        let hc = self.hc.as_ref().unwrap();
        self.validate_glm_c4(c)?;
        let hc_mult = hc.hc_mult as u32;
        // GLM uses physical block_idx; mixed upstream models carry model indices.
        let (is_first_layer, is_last_layer) = if ctx.config.model_type == "glm5_next" {
            (
                self.block_idx == 0,
                self.block_idx + 1 == ctx.config.num_hidden_layers,
            )
        } else {
            (hc.is_first_model_layer, hc.is_last_model_layer)
        };
        let hc_streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let diag_this =
            std::env::var("ATLAS_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");

        if is_first_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                c.hidden,
                hc_streams,
                n as u32,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // ── Phase 1: collapse + norm for N tokens ──
        self.ms_hc_pre_site(
            &hc.attn, hc, hc_streams, c.hidden, post, comb, n, eps, ctx, stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                c.hidden,
                n * h,
                stream,
                &format!("V4-msdecode L{} hc_pre-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                n * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} post-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                n * (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} comb-attn", self.attn_layer_idx),
            );
        }
        if ops::HcVariant::of(hc).applies_block_input_norm() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                c.hidden,
                &self.input_norm,
                c.normed,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
        } else {
            // See `prefill_inner.rs`: `hc_norm` is the input norm on Qwen.
            ctx.gpu
                .copy_d2d_async(c.hidden, c.normed, n * h * 2, stream)?;
        }

        let meta = ctx
            .attn_metadata
            .expect("attention layer requires metadata");

        // ── Phases 2-6: attention ──
        let o_out = if let Some(ref _mla) = self.mla {
            self.ms_mla_decode(c, kv_cache, meta)?
        } else {
            self.ms_phase_qkv(c)?;
            self.ms_phase_rope(c, meta)?;
            self.ms_phase_cache_write(c, kv_cache, meta)?;
            // `qsa_rows` (decided pre-mutation) owns BOTH this choice and the
            // ingest loop below, so a row is never ingested twice.
            let attn_out = if qsa_rows {
                let states = states.as_deref_mut().ok_or_else(|| {
                    anyhow::anyhow!("QSA per-row phase requires per-sequence state")
                })?;
                self.ms_phase_attn_qsa_rows(c, states, row_owner, seq_lens, kv_cache, meta)?
            } else {
                self.ms_phase_paged_decode(c, kv_cache, meta)?
            };
            self.ms_phase_o_proj(c, attn_out)?
        };

        if c.fwd.config.tp_world_size > 1
            && let Some(comm) = c.fwd.comm
        {
            let bytes = c.n * c.h * c.bf16;
            comm.all_reduce_async(o_out.0, bytes, c.stream)?;
        }

        // ── QSA ingest continuity (all rows inert; see `qsa_rows.rs`) ──
        if !qsa_rows && self.qsa.is_some() {
            let states = states.ok_or_else(|| {
                anyhow::anyhow!("QSA batched HC requires actual per-sequence state")
            })?;
            self.ms_qsa_ingest_rows(c, states, row_owner, seq_lens, kv_cache, meta)?;
        }

        // Expand attention output back into multi-stream state.
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            o_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            n as u32,
            h as u32,
            stream,
        )?;
        if diag_this {
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-msdecode L{} hc_post-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                n * (hc_mult as usize) * h,
                stream,
                &format!(
                    "V4-msdecode L{} hc_post-attn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

        // Standalone attention (no FFN)
        if self.ffn.is_none() {
            if is_last_layer && let Some(ref head) = hc.head {
                ops::hc_head_site(
                    ctx.gpu,
                    self.hc_head_k,
                    hc_streams,
                    head,
                    hc,
                    c.hidden,
                    ctx.buffers.hc_lowrank_scratch(),
                    n as u32,
                    h as u32,
                    eps,
                    stream,
                )?;
                if diag_this {
                    super::diag_norm(
                        ctx.gpu,
                        c.hidden,
                        n * h,
                        stream,
                        &format!("V4-msdecode L{} hc_head", self.attn_layer_idx),
                    );
                }
            } else if is_last_layer {
                tracing::warn!(
                    "V4-msdecode L{}: hc_head SKIPPED (no head weights)",
                    self.attn_layer_idx
                );
            }
            return Ok(None);
        }

        // Phase 7: collapse/norm, with the variant-specific kernel ABI.
        self.ms_hc_pre_site(
            &hc.ffn, hc, hc_streams, c.hidden, post, comb, n, eps, ctx, stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                c.hidden,
                n * h,
                stream,
                &format!("V4-msdecode L{} hc_pre-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                n * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} post-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                n * (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} comb-ffn", self.attn_layer_idx),
            );
        }
        if ops::HcVariant::of(hc).applies_block_input_norm() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                c.hidden,
                &self.post_attn_norm,
                c.normed,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
        } else {
            ctx.gpu
                .copy_d2d_async(c.hidden, c.normed, n * h * 2, stream)?;
        }

        // GLM's caller consumes these shared-arena pointers immediately (or saves them).
        if ctx.config.model_type == "glm5_next" {
            return Ok(Some(hc_ffn::HcFfnPhase {
                hc_streams,
                post,
                comb,
                diag_this,
            }));
        }
        self.ms_hc_generic_finish(
            c,
            ctx,
            stream,
            hc_ffn::HcFfnPhase {
                hc_streams,
                post,
                comb,
                diag_this,
            },
            is_last_layer,
        )?;

        Ok(None)
    }
}
