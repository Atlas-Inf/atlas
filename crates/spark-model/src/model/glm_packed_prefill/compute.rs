// SPDX-License-Identifier: AGPL-3.0-only

//! The traversal of a packed prefill (`model::glm_packed_prefill`), the same
//! on every rank.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use super::{meta_block_bytes, row_offsets, scratch_layout};
use crate::layer::glm_packed::GlmPackedOwner;
use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::traits::{Model, SequenceState};

impl TransformerModel {
    /// `PACKED_CHECK` (under `ATLAS_GLM_DET_TRACE`): the exact bytes at `ptr`, hashed
    /// (synchronizes `stream`).
    fn packed_check_hash(&self, ptr: DevicePtr, bytes: usize, stream: u64) -> Result<u64> {
        crate::det_trace::device_hash(self.gpu.as_ref(), stream, ptr, bytes, 32 << 20)
            .context("PACKED_CHECK: device read failed")
    }

    /// Both ranks: one traversal over every prompt's rows.
    pub(super) fn glm_packed_compute(
        &self,
        prompts: &[&[u32]],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let rows0 = row_offsets(&lens);
        let total: usize = lens.iter().sum();
        let hidden = self.buffers.hidden_states();

        // Every packed prompt runs the installed adapter (`-1`), as a chunk of
        // one does (`prefill_chunk_entry`).
        self.stamp_overlay_route(-1);
        // ── Arena and embeddings: a cold chunk's, at each prompt's rows ──
        if self.comm.is_some() {
            self.warm_zero_arena(stream)?;
        } else {
            self.buffers
                .zero_prefill_essentials(self.gpu.as_ref(), stream)?;
        }
        for ((prompt, seq), &row0) in prompts.iter().zip(seqs.iter()).zip(&rows0) {
            self.prefill_b_embed_chunk_at(
                seq,
                prompt,
                0,
                prompt.len(),
                hidden.offset(row0 * h * bf16),
                stream,
            )?;
        }
        let joint_tokens: Vec<u32> = prompts.iter().flat_map(|p| p.iter().copied()).collect();
        self.gpu.copy_h2d_async(
            // SAFETY: POD u32 vector viewed as its own bytes.
            unsafe { std::slice::from_raw_parts(joint_tokens.as_ptr() as *const u8, total * 4) },
            self.buffers.token_ids(),
            stream,
        )?;

        // ── Per prompt: a cold lookup's bookkeeping and its blocks ──
        let mut kv_cache = self.kv_cache.lock();
        for (prompt, seq) in prompts.iter().zip(seqs.iter_mut()) {
            seq.prompt_len = prompt.len();
            seq.prefix_ref_tokens.clear();
            seq.cached_prefix_tokens = 0;
            seq.cached_prefix_blocks = 0;
            seq.marconi_skip_to = 0;
            seq.prefix_lookup_skip = false;
            seq.prefix_lookup_applied = true;
            seq.marconi_exact_snap = None;
            let key = self.pc_key(seq, prompt).map(|k| k.into_owned());
            self.dflash_warm_begin(seq, key.as_deref(), 0, prompt.len(), stream)?;
            self.reserve_prefill_blocks(seq, prompt.len(), &mut kv_cache, stream)?;
            seq.kv_valid_tokens = prompt.len();
        }
        self.buffers.note_rows(total);

        // ── Metadata: each prompt's own, then the joint positions/slots ──
        let mrope = self.config.mrope_interleaved;
        let (joint_at, owners_at, _) =
            scratch_layout(&lens, self.config.num_experts_per_tok, mrope);
        let scratch = self.buffers.scratch();
        let mut metas: Vec<AttnMetadataDev> = Vec::with_capacity(seqs.len());
        for (k, (prompt, seq)) in prompts.iter().zip(seqs.iter_mut()).enumerate() {
            let rows = prompt.len();
            let base = scratch.offset(owners_at[k]);
            let layout = self.prefill_b_upload_meta_at(
                prompt,
                seq,
                0,
                rows,
                0,
                rows,
                0,
                &kv_cache,
                base,
                meta_block_bytes(rows, mrope),
                stream,
            )?;
            ensure!(
                layout.needs_paged,
                "GLM packed prefill expects paged metadata"
            );
            self.prefill_b_upload_paged(
                seq,
                rows,
                0,
                rows,
                base,
                layout.slot_offset,
                &kv_cache,
                stream,
            )?;
            // The metadata pack reuses one pinned staging buffer.
            self.gpu.synchronize(stream)?;
            let page = seq
                .chunked_prefill_meta
                .as_ref()
                .context("GLM packed prefill: paged metadata missing")?;
            let pos = |s: usize| {
                if layout.use_mrope {
                    base.offset(s * layout.pos_stream_bytes)
                } else {
                    base
                }
            };
            metas.push(AttnMetadataDev {
                positions: pos(0),
                positions_h: pos(1),
                positions_w: pos(2),
                slot: base.offset(layout.slot_offset),
                seq_len: page.seq_len,
                block_table: page.block_table,
                max_blocks_per_seq: seq.block_table.len() as u32,
                num_seqs: 1,
                seq_slot: DevicePtr::NULL,
                moe_row_adapter: DevicePtr::NULL,
            });
        }
        let joint = scratch.offset(joint_at);
        let stream_bytes = total * 4;
        let streams = if mrope { 3 } else { 1 };
        let joint_slots = joint.offset((stream_bytes * streams + 7) & !7);
        for (k, m) in metas.iter().enumerate() {
            let (row0, rows) = (rows0[k], lens[k]);
            let src = [m.positions, m.positions_h, m.positions_w];
            for (s, &src) in src.iter().enumerate().take(streams) {
                self.gpu.copy_d2d_async(
                    src,
                    joint.offset(s * stream_bytes + row0 * 4),
                    rows * 4,
                    stream,
                )?;
            }
            self.gpu
                .copy_d2d_async(m.slot, joint_slots.offset(row0 * 8), rows * 8, stream)?;
        }
        let joint_pos = |s: usize| {
            if mrope {
                joint.offset(s * stream_bytes)
            } else {
                joint
            }
        };
        // Force the H2D/D2D metadata before the layers (as a chunk does).
        self.gpu.synchronize(stream)?;

        let seq_slot =
            self.upload_seq_slot_uniform(-1, total, self.buffers.lora_seq_slot(), stream)?;
        let attn_metadata = AttnMetadataDev {
            positions: joint_pos(0),
            positions_h: joint_pos(1),
            positions_w: joint_pos(2),
            slot: joint_slots,
            seq_slot,
            ..metas[0]
        };
        let ctx = ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(attn_metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: Some(self.buffers.token_ids()),
            host_token_ids: None,
            routed_lora_layers: self.routed_slot_layers(-1),
            midchunk_capture: None,
            moe_lora_route: self.moe_lora_route(-1),
        };

        // ── Layers ──
        // The determinism tracer also proves the packing exact: one hash line
        // per prompt (`PACKED_CHECK`).
        let check = crate::det_trace::on();
        let mut layer_hash = vec![0u64; seqs.len()];
        let hc_row =
            self.config.hc_mult * h * crate::layers::ops::hc_elem_bytes(&self.config.model_type);
        {
            let _pin = crate::layers::glm_packed_scope::enter();
            for (i, layer) in self.layers.iter().enumerate() {
                let mut owners: Vec<GlmPackedOwner<'_>> = seqs
                    .iter_mut()
                    .zip(rows0.iter().zip(&lens))
                    .zip(&metas)
                    .map(|((seq, (&row0, &rows)), &meta)| GlmPackedOwner {
                        state: seq.layer_states[i].as_mut() as &mut (dyn LayerState + 'static),
                        row0,
                        rows,
                        meta,
                    })
                    .collect();
                layer
                    .prefill_glm_packed(hidden, total, &mut owners, &mut kv_cache, &ctx, stream)
                    .map_err(|e| anyhow::anyhow!("GLM packed prefill layer {i} failed: {e}"))?;
                drop(owners);
                for (k, seq) in seqs.iter_mut().enumerate() {
                    self.try_dflash_prefill_capture_layer_at(seq, i, 0, lens[k], rows0[k], stream)?;
                    if check {
                        let hash = self.packed_check_hash(
                            self.buffers.hc_streams().offset(rows0[k] * hc_row),
                            lens[k] * hc_row,
                            stream,
                        )?;
                        layer_hash[k] = atlas_tier::hash::mix64(layer_hash[k], hash);
                    }
                }
            }
        }

        // ── Per prompt: tokens, last-row norm + LM head, DFlash context ──
        let v = self.config.vocab_size;
        let normed = self.buffers.norm_output();
        // Each prompt's row through the same one-row `lm_head` a single-chunk
        // prefill runs, its logits then moved to their own row of the idle
        // MoE scratch (the logits buffer is the head's own working space).
        let park = self.buffers.expert_gate_out();
        let n = seqs.len();
        let mut logits = vec![DevicePtr::NULL; n];
        for k in 0..n {
            let last = hidden.offset((rows0[k] + lens[k] - 1) * h * bf16);
            let row_normed = normed.offset(k * h * bf16);
            self.final_norm_rows(last, row_normed, 1, stream)?;
            let out = self.lm_head(row_normed, stream)?;
            logits[k] = park.offset(k * v * 2);
            self.gpu.copy_d2d_async(out, logits[k], v * 2, stream)?;
        }
        let bs = kv_cache.block_size();
        drop(kv_cache);
        for (k, (prompt, seq)) in prompts.iter().zip(seqs.iter_mut()).enumerate() {
            seq.tokens.extend_from_slice(prompt);
            seq.seq_len = prompt.len();
            seq.last_decode_ckpt_block = seq.tokens.len() / bs;
            self.update_dflash_ctx_len_after_prefill(seq, 0, prompt.len())?;
            if check {
                let final_hash = self.packed_check_hash(
                    hidden.offset(rows0[k] * h * bf16),
                    lens[k] * h * bf16,
                    stream,
                )?;
                let logits_hash = self.packed_check_hash(logits[k], v * 2, stream)?;
                tracing::info!(
                    "PACKED_CHECK rank={} prompt={:016x} tokens={} packed={n}/{total} layers={:016x} final={final_hash:016x} logits={logits_hash:016x}",
                    self.config.ep_rank,
                    crate::det_trace::hash_bytes(
                        // SAFETY: POD u32 slice viewed as its own bytes.
                        unsafe {
                            std::slice::from_raw_parts(
                                prompt.as_ptr() as *const u8,
                                prompt.len() * 4,
                            )
                        }
                    ),
                    prompt.len(),
                    layer_hash[k],
                );
            }
        }
        Ok(logits)
    }
}
