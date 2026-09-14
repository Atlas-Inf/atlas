// SPDX-License-Identifier: AGPL-3.0-only

//! One draft step through the MTP module: the target's pre-mixer
//! residual in, a draft logit out. Split out of `qwen4exp_mtp.rs`
//! for the 500-LoC cap.

use super::*;

impl Qwen4ExpMtpHead {
    /// One draft step. Returns the drafted token id.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_one(
        &self,
        token: u32,
        position: usize,
        state: &mut Qwen4ExpMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size;
        let hc = ctx.config.hc_mult.max(1);
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;

        // The FP32 residual highway, `[1, hc_mult, hidden]`. On the first draft
        // this still holds the TARGET's final pre-mixer streams; on later ones
        // it holds the drafter's own from the previous position.
        let streams = ctx.buffers.hc_streams();

        // ── 1. Per-stream grouped norm of the incoming residual ──
        // Read out BEFORE step 3 overwrites `streams`.
        for s in 0..hc {
            ops::rms_norm_f32(
                ctx.gpu,
                self.rms_norm_f32_k,
                streams.offset(s * h * 4),
                self.module.pre_fc_norm_hidden.weight.offset(s * h * 2),
                self.normed_h.offset(s * h * 2),
                1,
                h as u32,
                eps,
                stream,
            )?;
        }

        // ── 2. Per-stream projection, shared weight ──
        for s in 0..hc {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                self.normed_h.offset(s * h * 2),
                &self.module.fc_hidden,
                self.h_streams.offset(s * h * 2),
                h as u32,
                h as u32,
                stream,
            )?;
        }

        // ── 3. Embedding branch, shared across streams ──
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu
            .copy_d2d_async(src, self.embed_buf, row_bytes, stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            self.embed_buf,
            &self.module.pre_fc_norm_embedding,
            self.normed_e,
            1,
            h as u32,
            eps,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            self.normed_e,
            &self.module.fc_embedding,
            self.e_branch,
            h as u32,
            h as u32,
            stream,
        )?;

        // Broadcast the shared embedding branch into every stream (this is what
        // overwrites the consumed input residual), then accumulate the
        // per-stream hidden branch onto it.
        ops::hc_expand(
            ctx.gpu,
            self.hc_expand_k,
            self.e_branch,
            streams,
            1,
            h as u32,
            hc as u32,
            stream,
        )?;
        for s in 0..hc {
            self.f32_add_bf16(
                ctx.gpu,
                streams.offset(s * h * 4),
                self.h_streams.offset(s * h * 2),
                h as u32,
                stream,
            )?;
        }

        // ── 4. Body: MIDDLE mHC + gated attention (QSA) + MoE ──
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }

        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let max_blocks = state.block_table.len() as u32;
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let meta_buf = pack_mtp_attn_meta(
            position as u32,
            global_slot,
            (state.seq_len + 1) as i32,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;

        let mtp_meta = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: max_blocks,
            num_seqs: 1,
            seq_slot: DevicePtr(0),
            moe_row_adapter: DevicePtr::NULL,
        };

        if let Some(tid_buf) = ctx.token_ids {
            ctx.gpu
                .copy_h2d_async(&token.to_le_bytes(), tid_buf, stream)?;
        }

        let mtp_ctx = ForwardContext {
            buffers: ctx.buffers,
            gpu: ctx.gpu,
            config: ctx.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: Some(mtp_meta),
            profile: ctx.profile,
            // Rank 0 only: the body holds ALL experts locally, so its MoE must
            // not issue an EP all-reduce no other rank will join.
            comm: None,
            // Host-built metadata + the H2D uploads above are illegal under
            // graph capture.
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: ctx.token_ids,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };

        // `decode_inner_hc` reads the persistent stream state from
        // `ctx.buffers.hc_streams()` and uses the `hidden` ARG as single-stream
        // scratch, so `hidden` must NOT alias the streams.
        let body_scratch = ctx.buffers.hidden_states();
        let residual = ctx.buffers.residual();
        let mut disk_block_ids: Vec<u32> = Vec::new();
        let mut disk_last_offloaded: Vec<u32> = vec![0u32; 1];
        self.module.body.decode(
            body_scratch,
            residual,
            state.body_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut disk_block_ids,
            &mut disk_last_offloaded,
            &mtp_ctx,
            stream,
        )?;
        drop(kv_cache);

        // ── 5. Head mixer: collapse the streams AND apply the final norm ──
        // qwen4_exp's mHC head is LOW-RANK. DeepSeek's proposer asserts
        // Sinkhorn here precisely because this path did not exist yet.
        let head = self
            .module
            .hc_head
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("qwen4exp_mtp: no hc_head (hc_mult == 0?)"))?;
        let lowrank = head.lowrank.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "qwen4exp_mtp: mHC head is not low-rank; this model has no Sinkhorn path"
            )
        })?;
        ops::hc_head_lowrank(
            ctx.gpu,
            self.hc_head_k,
            streams,
            lowrank,
            self.h_out,
            ctx.buffers.hc_lowrank_scratch(),
            1,
            h as u32,
            hc as u32,
            eps,
            stream,
        )?;

        // ── 6. Shared LM head. No separate final norm: the mixer's `hc_norm`
        //       IS it, which is why the checkpoint ships no `mtp.norm.weight`. ──
        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            self.h_out,
            &self.lm_head,
            logits,
            v,
            h as u32,
            stream,
        )?;

        let token_id = if let Some(bitmask) = grammar_bitmask {
            crate::layers::argmax_grammar_masked(ctx.gpu, logits, v as usize, bitmask, position)?
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, self.argmax_out, v, stream)?;
            let mut buf = [0u8; 4];
            ctx.gpu.copy_d2h(self.argmax_out, &mut buf)?;
            u32::from_le_bytes(buf)
        };

        state.seq_len += 1;
        Ok(token_id)
    }
}
