// SPDX-License-Identifier: AGPL-3.0-only

//! Native MTP proposal forward. Every proposal is subsequently target-verified.
use super::*;
impl Glm5MtpHead {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_inner(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            num_drafts <= 8,
            "GLM-5 native MTP admits at most eight drafts"
        );
        anyhow::ensure!(
            state.last_num_drafted == 0,
            "GLM-5 MTP proposal has an unverified tail"
        );
        anyhow::ensure!(
            state.seq_len.checked_add(1) == Some(position),
            "GLM-5 MTP requires caught-up context"
        );
        let capacity = self
            .max_seq_len
            .min(self.draft_config.max_position_embeddings);
        // Target verification consumes the pending root plus the drafts.
        // Reserve that root row before bounding the optional proposal tail.
        let num_drafts = bounded_draft_width(position, num_drafts, capacity)?;
        // Grammar requests keep full width; unmasked probabilities must not
        // decide an early stop for a different, grammar-masked distribution.
        state.confidence.begin(if grammar_bitmask.is_none() {
            self.confidence.stop_ln_threshold
        } else {
            None
        });
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut token = last_token;
        let mut hidden = target_hidden;
        for index in 0..num_drafts {
            let draft = self.forward_one(
                token,
                hidden,
                position + index,
                state,
                ctx,
                stream,
                grammar_bitmask,
            )?;
            drafts.push(draft);
            state.last_num_drafted = drafts.len();
            token = draft;
            if state.confidence.should_stop() {
                break;
            }
            hidden = self.arena.norm_output();
        }
        Ok(drafts)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        anyhow::ensure!(
            (token as usize) < self.draft_config.vocab_size,
            "GLM-5 MTP token exceeds vocabulary"
        );
        let h = self.draft_config.hidden_size;
        let h_u32 = u32::try_from(h).context("GLM-5 MTP: hidden size exceeds u32")?;
        let eps = self.draft_config.rms_norm_eps as f32;
        let row_bytes = h
            .checked_mul(2)
            .context("GLM-5 MTP: hidden row byte size overflows")?;

        // Upstream predictor input: [norm(embed) || norm(hidden)] -> hidden.
        let embed = self.arena.ssm_deinterleaved();
        let embed_src = self.embed_tokens.weight.offset(
            usize::try_from(token)
                .context("GLM-5 MTP: token id is negative")?
                .checked_mul(row_bytes)
                .context("GLM-5 MTP: embedding byte offset overflows")?,
        );
        ctx.gpu
            .copy_d2d_async(embed_src, embed, row_bytes, stream)?;
        if self.zero_first_embed && state.seq_len == 0 {
            ctx.gpu.memset_async(embed, 0, row_bytes, stream)?;
        }
        let eh_input = self.arena.ssm_qkvz();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed,
            &self.module.enorm,
            eh_input,
            1,
            h_u32,
            eps,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            target_hidden,
            &self.module.hnorm,
            eh_input.offset(row_bytes),
            1,
            h_u32,
            eps,
            stream,
        )?;
        let h_in = self.arena.hidden_states();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            eh_input,
            &self.module.eh_proj,
            h_in,
            h_u32,
            (2 * h)
                .try_into()
                .context("GLM-5 MTP: combined hidden size exceeds u32")?,
            stream,
        )?;
        // Append one predictor row to the private BF16 MLA cache.
        let mtp_meta = {
            let mut kv_cache = self.cache()?;
            let block_size = kv_cache.block_size();
            let needed = required_blocks(state.seq_len, block_size)?;
            while state.block_table.len() < needed {
                state.block_table.push(kv_cache.alloc_block()?);
            }
            let (_, slot) = append_cursor(
                &state.block_table,
                state.seq_len,
                block_size,
                self.max_blocks,
            )?;
            let meta_base = self.arena.scratch().offset(MTP_META_OFFSET);
            let meta_buf = pack_mtp_attn_meta(
                u32::try_from(position).context("GLM-5 MTP: position exceeds u32")?,
                slot,
                i32::try_from(
                    state
                        .seq_len
                        .checked_add(1)
                        .context("GLM-5 MTP: cache length overflows")?,
                )
                .context("GLM-5 MTP: cache length exceeds i32")?,
                &state.block_table,
                self.arena.scratch_bytes().saturating_sub(MTP_META_OFFSET),
            )?;
            ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;
            AttnMetadataDev {
                positions: meta_base,
                positions_h: meta_base,
                positions_w: meta_base,
                slot: meta_base.offset(8),
                seq_len: meta_base.offset(16),
                block_table: meta_base.offset(256),
                max_blocks_per_seq: state.block_table.len() as u32,
                num_seqs: 1,
                seq_slot: DevicePtr(0),
                moe_row_adapter: DevicePtr::NULL,
            }
        };

        // A private token-id row keeps even hash-MoE-style routing away from the
        // target's stable token-id buffer.
        let token_ids = ctx.token_ids.map(|_| self.arena.token_ids());
        if let Some(token_ids) = token_ids {
            ctx.gpu
                .copy_h2d_async(&token.to_le_bytes(), token_ids, stream)?;
        }
        // Clone the shared draft context, overriding only the fields this
        // single-row call needs; every other field is byte-identical to the
        // prefill/repair context.
        let mtp_ctx = ForwardContext {
            attn_metadata: Some(mtp_meta),
            profile: ctx.profile,
            token_ids,
            ..self.draft_context(ctx)
        };
        let vocab = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size
                .min(self.draft_config.vocab_size.try_into()?)
        } else {
            self.draft_config.vocab_size.try_into()?
        };
        let logits = self.arena.logits();
        let residual = self.arena.residual();
        ctx.gpu.memset_async(residual, 0, row_bytes, stream)?;
        let mut kv_cache = self.cache()?;
        let mut disk_ids = Vec::new();
        let mut disk_last = vec![0u32; 1];
        self.module.body.decode(
            h_in,
            residual,
            state.body_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut disk_ids,
            &mut disk_last,
            &mtp_ctx,
            stream,
        )?;
        drop(kv_cache);
        let draft_hidden = self.arena.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            h_in,
            &self.module.norm,
            draft_hidden,
            1,
            h_u32,
            eps,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            draft_hidden,
            &self.lm_head,
            logits,
            vocab,
            h_u32,
            stream,
        )?;
        let draft = if let Some(mask) = grammar_bitmask {
            crate::layers::argmax_grammar_masked(
                ctx.gpu,
                logits,
                usize::try_from(vocab)?,
                mask,
                position,
            )?
        } else if let Some(argmax_batch_lp) = self.confidence.argmax_batch_lp {
            let out = self.arena.scratch();
            ops::argmax_bf16_batch_lp(
                ctx.gpu,
                argmax_batch_lp,
                logits,
                out,
                out.offset(4),
                vocab,
                1,
                vocab,
                stream,
            )?;
            let mut bytes = [0u8; 8];
            ctx.gpu.copy_d2h_on_stream(out, &mut bytes, stream)?;
            let draft = u32::from_le_bytes(bytes[..4].try_into().expect("u32 width"));
            let logprob = f32::from_le_bytes(bytes[4..].try_into().expect("f32 width"));
            state.confidence.record(logprob)?;
            draft
        } else {
            let out = self.arena.scratch();
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out, vocab, stream)?;
            let mut bytes = [0u8; 4];
            ctx.gpu.copy_d2h_on_stream(out, &mut bytes, stream)?;
            u32::from_le_bytes(bytes)
        };
        state.seq_len = state
            .seq_len
            .checked_add(1)
            .context("GLM-5 MTP: cache length overflows")?;
        Ok(draft)
    }
}
