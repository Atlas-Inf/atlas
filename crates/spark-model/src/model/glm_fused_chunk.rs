// SPDX-License-Identifier: AGPL-3.0-only

//! A GLM prefill chunk carrying DFlash verify owners (`layer::glm_long_owner`).
//!
//! While one request prefills, the other requests' verify rows ride its chunk
//! instead of taking their own target traversal between chunks: chunk rows at
//! arena rows `[0, C)`, owner `o`'s `R` rows at `[C + o*R, C + (o+1)*R)`.
//! Row-local work (projections, mHC, norms, FFN) reads each weight once for
//! both; each sequence advances only its own state
//! (`TransformerLayer::prefill_with_glm_passengers`). The chunk itself keeps
//! every ordinary prefill phase.
//!
//! Wire protocol, v2-addressed:
//!
//! ```text
//! EB  (seq_id prefill slot)  chunk_len, chunk_start, full_len, tokens[full_len],
//!                            width(N, R), slots[N], tokens[R*N]
//! ```
//!
//! Both ranks run the same traversal, which leaves every owner's final rows in
//! the verify stage like an owner-batched verify (E9); the owners' verdict
//! tails then follow as EA.

use super::TransformerModel;
use super::block_mgmt::ensure_blocks_through_decode;
use super::glm_long_verify::{decode_width, encode_width};
use crate::layer::AttnMetadataDev;
use crate::layer::glm_long_owner as owner;
use crate::layers::ops;
use crate::traits::{Model, SequenceState};
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

pub(super) const EP_CMD_GLM_FUSED_CHUNK: u32 = 0xFFFF_FFEB;

/// Verify owners riding one prefill chunk.
pub(super) struct Passengers<'a, 'b> {
    pub rows: usize,
    /// Owner-major verify tokens, `rows` per owner.
    pub tokens: &'a [u32],
    pub seqs: &'a mut [&'b mut SequenceState],
    /// Owner-major argmax IDs, set once the traversal finished.
    pub ids: Vec<u32>,
}

impl Passengers<'_, '_> {
    pub(super) fn total(&self) -> usize {
        self.seqs.len() * self.rows
    }
}

/// Device metadata of one fused traversal, staged after the chunk's.
pub(super) struct PassengerRun {
    /// Positions of every row, chunk rows first, one array per MRoPE stream.
    pub positions: [DevicePtr; 3],
    /// Cache slots of every row, chunk rows first.
    pub slots: DevicePtr,
    /// Each owner's own metadata for its rows.
    pub metas: Vec<AttnMetadataDev>,
    /// Each owner's DFlash hidden-save slot, when the drafter captures.
    pub save_slots: Option<Vec<usize>>,
}

/// Where the chunk's metadata landed (`prefill_b_upload_meta`).
pub(super) struct ChunkMeta {
    pub base: DevicePtr,
    pub pos_stream_bytes: usize,
    pub slot_offset: usize,
    pub use_mrope: bool,
}

fn align(bytes: usize) -> usize {
    bytes.div_ceil(256) * 256
}

/// Bytes of one owner's metadata block (`glm_long_upload_meta`).
fn owner_meta_bytes(rows: usize, max_blocks: usize) -> usize {
    align(super::glm_long_verify::META_BLOCK_TABLE + rows * max_blocks * 4)
}

impl TransformerModel {
    /// Whether a chunk of `chunk_len` rows of `prompt` may carry `owners`
    /// owners of `rows` rows each. Decided from the configuration and the
    /// request alone, so both ranks agree.
    pub(super) fn glm_fused_chunk_supported(
        &self,
        prompt: &[u32],
        seq: &SequenceState,
        chunk_len: usize,
        owners: usize,
        rows: usize,
    ) -> bool {
        crate::speculative::glm_repair_policy::dflash_enabled()
            && self.can_batch_glm_long_verify_impl(owners, rows)
            && chunk_len >= 2
            && chunk_len + owners * rows <= self.buffers.max_batch_tokens()
            && !self.prefix_cache.is_active()
            && seq.collect_prompt_logprobs.is_none()
            && !self.tokens_have_vision_pad(prompt)
    }

    /// After the chunk's metadata upload (`chunk` rows at `meta`): embed the
    /// owners' rows after the chunk, reserve their cache blocks, stage every
    /// owner's metadata and the joint positions/slots in scratch past the
    /// chunk's metadata, and prestage the owners' layer states.
    pub(super) fn glm_passengers_setup(
        &self,
        p: &mut Passengers<'_, '_>,
        chunk: usize,
        meta: &ChunkMeta,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<PassengerRun> {
        let h = self.config.hidden_size;
        let rows = p.rows;
        let total = chunk + p.total();
        let hidden = self.buffers.hidden_states();
        for (row, &token) in p.tokens.iter().enumerate() {
            self.embed(token, hidden.offset((chunk + row) * h * 2), stream)?;
        }
        let bs = kv_cache.block_size();
        for seq in p.seqs.iter_mut() {
            for t in 0..rows {
                ensure_blocks_through_decode(
                    seq,
                    (seq.seq_len + t) / bs,
                    kv_cache,
                    self.prefix_cache.as_ref(),
                    self.gpu.as_ref(),
                    stream,
                    self.levers.kv_poison,
                )?;
            }
        }

        // Scratch past the chunk's metadata: the joint arrays, then one
        // metadata block per owner.
        let scratch = self.buffers.scratch();
        let chunk_end = (meta.base.0 - scratch.0) as usize + meta.slot_offset + chunk * 8;
        let joint_at = align(chunk_end);
        let stream_bytes = align(total * 4);
        let slots_at = joint_at + 3 * stream_bytes;
        let owners_at = align(slots_at + total * 8);
        let mb = self.max_blocks_per_seq as usize;
        let owner_bytes = owner_meta_bytes(rows, mb);
        ensure!(
            owners_at + p.seqs.len() * owner_bytes <= self.buffers.scratch_bytes(),
            "GLM fused chunk metadata exceeds scratch"
        );
        let positions = scratch.offset(joint_at);
        let slots = scratch.offset(slots_at);
        for s in 0..3 {
            let src = if meta.use_mrope {
                meta.base.offset(s * meta.pos_stream_bytes)
            } else {
                meta.base
            };
            self.gpu
                .copy_d2d_async(src, positions.offset(s * stream_bytes), chunk * 4, stream)?;
        }
        self.gpu
            .copy_d2d_async(meta.base.offset(meta.slot_offset), slots, chunk * 8, stream)?;
        let mut owner_positions = Vec::with_capacity(p.total());
        let mut owner_slots = Vec::with_capacity(p.total());
        for seq in p.seqs.iter() {
            for pos in seq.seq_len..seq.seq_len + rows {
                let block = seq
                    .physical_block_for(pos / bs)
                    .context("GLM fused chunk owner block missing")?;
                owner_positions.push(pos as u32);
                owner_slots.push(block as i64 * bs as i64 + (pos % bs) as i64);
            }
        }
        fn bytes<T>(v: &[T]) -> &[u8] {
            // SAFETY: POD integer vectors; the byte view covers exactly `len`.
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
        }
        // Text rows share one position across the three MRoPE streams.
        for s in 0..3 {
            self.gpu.copy_h2d_async(
                bytes(&owner_positions),
                positions.offset(s * stream_bytes + chunk * 4),
                stream,
            )?;
        }
        self.gpu
            .copy_h2d_async(bytes(&owner_slots), slots.offset(chunk * 8), stream)?;

        let mut metas = Vec::with_capacity(p.seqs.len());
        for (o, seq) in p.seqs.iter().enumerate() {
            let own: Vec<(usize, &SequenceState)> =
                (0..rows).map(|t| (seq.seq_len + t, &**seq)).collect();
            let base = scratch.offset(owners_at + o * owner_bytes);
            metas.push(self.glm_long_upload_meta(base, &own, bs, stream)?);
        }
        for (seq, t) in p.seqs.iter_mut().zip(p.tokens.chunks_exact(rows)) {
            for (li, layer) in self.layers.iter().enumerate() {
                layer.verify_prestage(
                    t,
                    seq.layer_states[li].as_mut(),
                    self.gpu.as_ref(),
                    stream,
                )?;
            }
        }
        let save_slots = self
            .dflash_hidden_save
            .map(|_| {
                p.seqs
                    .iter()
                    .map(|s| s.dflash_hidden_save_slot())
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        Ok(PassengerRun {
            positions: [
                positions,
                positions.offset(stream_bytes),
                positions.offset(2 * stream_bytes),
            ],
            slots,
            metas,
            save_slots,
        })
    }

    /// One layer's owners over their own rows and states.
    pub(super) fn glm_passenger_owners<'s>(
        p: &'s mut Passengers<'_, '_>,
        run: &PassengerRun,
        li: usize,
    ) -> Vec<owner::GlmLongOwner<'s>> {
        let rows = p.rows;
        p.seqs
            .iter_mut()
            .zip(&run.metas)
            .map(|(seq, &meta)| owner::GlmLongOwner {
                positions: (seq.seq_len..seq.seq_len + rows).collect(),
                state: seq.layer_states[li].as_mut(),
                meta,
            })
            .collect()
    }

    /// After the layers and before the chunk's finalize (which reuses the
    /// logits rows): the owners' final norm and argmax (with each owner's
    /// min_tokens ban), their final rows into the verify stage, and each
    /// owner advanced by its rows.
    pub(super) fn glm_passengers_finish(
        &self,
        p: &mut Passengers<'_, '_>,
        chunk: usize,
        stream: u64,
    ) -> Result<()> {
        let stage = self
            .glm_long_stage
            .context("GLM fused chunk needs the long owner stage")?;
        let h = self.config.hidden_size;
        let rows = p.rows;
        let total = p.total();
        let b = &self.buffers;
        let normed = b.norm_output().offset(chunk * h * 2);
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            b.hidden_states().offset(chunk * h * 2),
            &self.final_norm,
            normed,
            total as u32,
            h as u32,
            self.config.rms_norm_eps as f32,
            stream,
        )?;
        let ban = p.seqs.first().map(|s| s.eos_ban).unwrap_or_default();
        let ban_rows = p.seqs.iter().enumerate().fold(0u64, |mask, (o, seq)| {
            mask | seq
                .eos_ban
                .row_mask(seq.seq_len, rows)
                .checked_shl((o * rows) as u32)
                .unwrap_or(0)
        });
        let argmax = b.scratch();
        if !self.glm_split_head_argmax(normed, total, argmax, (ban_rows, &ban), None, stream)? {
            self.lm_head_batched(normed, total as u32, b.logits(), stream)?;
            let vocab = self.config.vocab_size;
            for r in 0..total {
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    b.logits().offset(r * vocab * 2),
                    argmax.offset(r * 4),
                    vocab as u32,
                    stream,
                )?;
            }
        }
        // The stage spans of an owner-batched verify: the owners' rows start
        // at arena row `chunk`, their logits at row 0.
        let r = stage.rows;
        for (arena, dst, row_bytes, row0) in [
            (b.hidden_states(), stage.hidden, r.hidden, chunk),
            (b.norm_output(), stage.norm, r.hidden, chunk),
            (b.hc_streams(), stage.highway, r.highway, chunk),
            (b.logits(), stage.logits, r.logits, 0),
        ] {
            self.gpu.copy_d2d_async(
                arena.offset(row0 * row_bytes),
                dst,
                total * row_bytes,
                stream,
            )?;
        }
        let mut buf = vec![0u8; total * 4];
        self.gpu.copy_d2h_on_stream(argmax, &mut buf, stream)?;
        p.ids = buf
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        for (seq, t) in p.seqs.iter_mut().zip(p.tokens.chunks_exact(rows)) {
            seq.tokens.extend_from_slice(t);
            seq.seq_len += rows;
        }
        Ok(())
    }

    /// The traversal both ranks run.
    #[allow(clippy::too_many_arguments)]
    fn glm_fused_chunk_compute(
        &self,
        prompt: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        rows: usize,
        tokens: &[u32],
        owners: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<(DevicePtr, Vec<u32>)> {
        ensure!(
            self.glm_fused_chunk_supported(prompt, seq, chunk_len, owners.len(), rows)
                && tokens.len() == owners.len() * rows
                && chunk_start + chunk_len <= prompt.len(),
            "GLM fused chunk refused: {chunk_len} rows + {} owners x {rows} rows",
            owners.len()
        );
        let is_last = chunk_start + chunk_len >= prompt.len();
        let mut p = Passengers {
            rows,
            tokens,
            seqs: owners,
            ids: Vec::new(),
        };
        let logits = self.prefill_chunk_dispatch_with(
            prompt,
            seq,
            chunk_start,
            chunk_len,
            is_last,
            stream,
            Some(&mut p),
        )?;
        ensure!(
            p.ids.len() == p.total(),
            "GLM fused chunk returned {} of {} verify IDs",
            p.ids.len(),
            p.total()
        );
        Ok((logits, p.ids))
    }

    /// Head: announce and run a chunk of `prompt` carrying `owners` owners of
    /// `rows` rows each (`tokens` owner-major). Returns the chunk's logits
    /// (`DevicePtr::NULL` before the last chunk) and the owners' argmax IDs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_chunk_with_glm_owners_impl(
        &self,
        prompt: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        rows: usize,
        tokens: &[u32],
        owners: &mut [&mut SequenceState],
    ) -> Result<(DevicePtr, Vec<u32>)> {
        ensure!(
            self.glm_fused_chunk_supported(prompt, seq, chunk_len, owners.len(), rows),
            "GLM fused chunk is not available"
        );
        let slots: Vec<u32> = owners.iter().map(|s| s.slot_idx as u32).collect();
        self.ep_broadcast_seq_and_cmd(seq.slot_idx as u32, EP_CMD_GLM_FUSED_CHUNK, true)?;
        for word in [chunk_len, chunk_start, prompt.len()] {
            self.ep_broadcast_u32(word as u32)?;
        }
        self.ep_broadcast_tokens(prompt)?;
        self.ep_broadcast_u32(encode_width(owners.len(), rows))?;
        self.ep_broadcast_tokens(&slots)?;
        self.ep_broadcast_tokens(tokens)?;
        let stream = self.gpu.default_stream();
        self.glm_fused_chunk_compute(
            prompt,
            seq,
            chunk_start,
            chunk_len,
            rows,
            tokens,
            owners,
            stream,
        )
    }

    /// Worker side of EB: the same traversal, then the chunk's SSM
    /// normalization (mirroring the head, as after an ordinary chunk).
    pub(super) fn glm_fused_receive(
        &self,
        seq_id: u32,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        let chunk_len = self.ep_broadcast_u32(0)? as usize;
        let chunk_start = self.ep_broadcast_u32(0)? as usize;
        let full_len = self.ep_broadcast_u32(0)? as usize;
        let prompt = self.ep_broadcast_tokens(&vec![0u32; full_len])?;
        let (n, rows) = decode_width(self.ep_broadcast_u32(0)?);
        ensure!(
            owner::width_supported(n, rows) && n < slots.len(),
            "GLM fused chunk width {n} x {rows} rows"
        );
        let ids = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let tokens = self.ep_broadcast_tokens(&vec![0u32; n * rows])?;
        let mut refs: Vec<(usize, &mut SequenceState)> = slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, s)| s.as_mut().map(|s| (i, s)))
            .collect();
        let mut take = |id: usize| {
            refs.iter()
                .position(|(i, _)| *i == id)
                .map(|at| refs.swap_remove(at).1)
                .with_context(|| format!("GLM fused chunk slot {id} unallocated or repeated"))
        };
        let seq = take(seq_id as usize)?;
        let mut owners = Vec::with_capacity(n);
        for &id in &ids {
            owners.push(take(id as usize)?);
        }
        self.sync_secondary()?;
        let stream = self.gpu.default_stream();
        self.glm_fused_chunk_compute(
            &prompt,
            seq,
            chunk_start,
            chunk_len,
            rows,
            &tokens,
            &mut owners,
            stream,
        )?;
        if let Err(e) = self.normalize_ssm_states(seq, stream) {
            tracing::warn!("Worker SSM state normalization failed: {e:#}");
        }
        Ok(true)
    }
}
