// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 3: stage and upload positions (T,H,W for MRoPE) + slot table
//! into the per-chunk metadata buffer carved out of `scratch`. Returns
//! the layout descriptor used by phase 3b/4.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::request_local::request_local;
use crate::traits::SequenceState;

pub(in crate::model) struct MetaLayout {
    pub meta_base: DevicePtr,
    pub slot_offset: usize,
    pub pos_stream_bytes: usize,
    pub use_mrope: bool,
    pub needs_paged: bool,
}

impl TransformerModel {
    pub(super) fn prefill_b_upload_meta(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        proc_start: usize,
        proc_count: usize,
        // Rows after the chunk's that share its MoE routing (fused verify).
        extra_rows: usize,
        effective_seq_len_start: usize,
        kv_cache: &PagedKvCache,
        stream: u64,
    ) -> Result<MetaLayout> {
        // Single-stream entry point: lay metadata at the default offset
        // after the MoE topk staging area.
        let moe_scratch_bytes = (proc_count + extra_rows) * self.config.num_experts_per_tok * 4 * 2;
        let meta_offset = (moe_scratch_bytes + 7) & !7;
        let meta_base = self.buffers.scratch().offset(meta_offset);
        self.prefill_b_upload_meta_at(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            proc_start,
            proc_count,
            effective_seq_len_start,
            kv_cache,
            meta_base,
            self.buffers.scratch_bytes().saturating_sub(meta_offset),
            stream,
        )
    }

    /// Build positions + slots metadata for `proc_count` tokens and upload
    /// to the caller-provided `meta_base` device pointer.
    ///
    /// `meta_region_bytes` is how much room this metadata block owns AT
    /// `meta_base` — the tail of the scratch arena for the single-stream entry
    /// point, one per-stream slice for the Q12 batched one. It is a required
    /// parameter because `meta_base` is a bare `DevicePtr` that carries no size,
    /// so without it the pack can only be bounded against the HOST staging
    /// buffer and would happily run off the end of the DEVICE allocation. Used by both the
    /// single-stream entry point above and Q12 batched prefill (multiple
    /// per-stream metadata blocks concatenated in one big scratch region).
    pub(in crate::model) fn prefill_b_upload_meta_at(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        proc_start: usize,
        proc_count: usize,
        effective_seq_len_start: usize,
        kv_cache: &PagedKvCache,
        meta_base: DevicePtr,
        meta_region_bytes: usize,
        stream: u64,
    ) -> Result<MetaLayout> {
        // The processed rows are this chunk's uncached tail.
        debug_assert!(
            proc_start >= chunk_start && proc_start + proc_count <= chunk_start + chunk_len,
            "processed range {proc_start}+{proc_count} outside chunk {chunk_start}+{chunk_len}"
        );
        // MRoPE-interleaved packs three u32 position streams (T, H, W).
        let use_mrope = self.config.mrope_interleaved;
        let pos_stream_bytes = proc_count * 4;
        let slot_offset = if use_mrope {
            (pos_stream_bytes * 3 + 7) & !7
        } else {
            (pos_stream_bytes + 7) & !7
        };
        // GLM-5 MLA attends through the paged cache from the first chunk on.
        // ATLAS_QWEN4EXP_PREFILL_ROWINV attends paged from the first chunk on
        // too (`qwen3_attention::prefill_inner_hc`).
        let needs_paged = effective_seq_len_start > 0
            || self.config.model_type == "glm5_next"
            || (self.config.model_type == "qwen4_exp" && crate::layers::ops::qwen4exp_rowinv::on());

        // Lock staging, build positions plus non-paged slots, and upload.
        {
            // SAFETY: Single-threaded scheduler access (see TransformerModel Send/Sync docs).
            let stg = unsafe { &mut *self.pinned_staging.get() };
            stg.positions.clear();
            stg.positions
                .extend(proc_start as u32..(proc_start + proc_count) as u32);

            // Build (T, H, W) streams matching HF Qwen3-VL's
            // `get_rope_index`/`get_vision_position_ids`:
            //   - text token: T = H = W = current_pos, increment current_pos by 1.
            //   - image pad run: T is constant, H/W are the post-merge
            //     row/column streams, and current_pos advances by the long
            //     spatial side.
            //   - Qwen3.8 video pad planes are one timestamp-separated
            //     `grid_t=1` range each, so the walker resets T/H/W at every
            //     plane while timestamps and marker boundaries move current_pos
            //     as ordinary text.
            // This matters because Qwen3-VL/3.6/3.8 was trained with
            // structured 3-D positions, not a linear position over the entire
            // multimodal prompt.
            if use_mrope {
                stg.positions_h.clear();
                stg.positions_w.clear();
                // Per-sequence grids: this request's items only (the model
                // holds no vision state between requests).
                let grids = seq.vision.grids.clone();
                // Both pad tokens: consumed identically here — the item's
                // t_len already says which it is — but the scan has to
                // recognize both, or a video run would be walked one text
                // token at a time.
                let (pad_id, video_pad_id) = self.vision_pad_ids();
                let is_pad = |tok: u32| tok == pad_id || tok == video_pad_id;
                let glm_vision = self.config.vision.as_ref().filter(|v| v.is_glm5_next);
                let qwen3vl = self
                    .config
                    .vision
                    .as_ref()
                    .is_some_and(|v| v.is_qwen3vl_video);
                // Both walkers build against the prompt from position 0 and
                // slice the processed range: text after an image keeps the
                // image's compressed position base in every later chunk, an
                // item straddling a chunk boundary keeps its plane, and a
                // later chunk's item takes its own grid. (A chunk-local walk
                // restarted each chunk at the linear position and at item 0.)
                // GLM's expanded video and Qwen3.8's timestamped video both
                // have structural text tokens between frame planes, which
                // only the complete prompt shows.
                let end = proc_start
                    .checked_add(proc_count)
                    .filter(|&end| end <= tokens.len())
                    .ok_or_else(|| request_local("MRoPE position range exceeds the prompt"))?;
                let have_vision = !grids.is_empty() && tokens[..end].iter().copied().any(is_pad);

                if have_vision {
                    stg.positions.clear();
                    // Co-dispatch: this request owns grids[grid_base .. grid_base+owned]
                    // of the shared packed vision_grids (0/all for legacy).
                    let grid_base = seq.vision.grid_base;
                    let owned = seq.vision.owned_images;
                    let grid_hi = if owned > 0 {
                        (grid_base + owned).min(grids.len())
                    } else {
                        grids.len()
                    };
                    let (mut full_t, mut full_h, mut full_w) = (Vec::new(), Vec::new(), Vec::new());
                    if let Some(v) = glm_vision {
                        // GLM-5 expands a video into per-frame image markers
                        // and timestamp text; its structural markers require
                        // the GLM-specific walker.
                        mrope_pos::build_glm5(
                            tokens,
                            &grids,
                            grid_base,
                            grid_hi,
                            0,
                            v.image_start_token_id,
                            v.image_end_token_id,
                            v.video_start_token_id,
                            v.video_end_token_id,
                            pad_id,
                            video_pad_id,
                            &mut full_t,
                            &mut full_h,
                            &mut full_w,
                        );
                    } else if qwen3vl {
                        // Qwen3-VL's nested per-group form has timestamp and
                        // boundary text between the t=1 group planes.
                        mrope_pos::build_qwen3vl(
                            tokens,
                            &grids,
                            grid_base,
                            grid_hi,
                            0,
                            pad_id,
                            video_pad_id,
                            &mut full_t,
                            &mut full_h,
                            &mut full_w,
                        );
                    } else {
                        mrope_pos::build(
                            &tokens[..end],
                            &grids,
                            grid_base,
                            grid_hi,
                            0,
                            pad_id,
                            video_pad_id,
                            &mut full_t,
                            &mut full_h,
                            &mut full_w,
                        );
                    }
                    // A run the walker pushed past `end` (an item straddling
                    // it) is sliced off; a walk shorter than `end` means the
                    // placeholders and grids disagree. Every rank decides that
                    // from the same tokens and grids (`request_local`).
                    if end > full_t.len() || end > full_h.len() || end > full_w.len() {
                        return Err(request_local(format!(
                            "MRoPE position range {proc_start}..{end} exceeds the {} walked positions",
                            full_t.len()
                        )));
                    }
                    stg.positions.extend_from_slice(&full_t[proc_start..end]);
                    stg.positions_h.extend_from_slice(&full_h[proc_start..end]);
                    stg.positions_w.extend_from_slice(&full_w[proc_start..end]);
                } else {
                    stg.positions_h.extend_from_slice(&stg.positions);
                    stg.positions_w.extend_from_slice(&stg.positions);
                }
            }

            // Build the slot table before packing: the packer borrows the
            // staging struct, so every reusable `Vec` has to be final first.
            if !needs_paged {
                let bs = kv_cache.block_size();
                stg.slots.clear();
                stg.slots
                    .extend((proc_start..proc_start + proc_count).map(|i| {
                        let block_idx = seq
                            .physical_block_for(i / bs)
                            .unwrap_or(self.dummy_kv_block);
                        (block_idx as i64) * (bs as i64) + ((i % bs) as i64)
                    }));
            }

            // The MRoPE+vision arm above REBUILDS `positions` from the
            // prompt's pad-token runs (one entry per text token, `gh*gw` per
            // image run) instead of from `proc_count`, then slices
            // `proc_start..proc_start + proc_count`. `put_prefix_at` still
            // carries the check: each stream contributes exactly `proc_count`
            // elements or the pack is refused.
            //
            // The DESTINATION bound used to be missing here entirely — the only
            // check was an `assert!(cursor <= stg.bytes)` AFTER the writes had
            // already landed, which is too late to keep them inside the
            // allocation. The packer checks each field before writing it.
            //
            // Rounding `slot_offset` up to 8 leaves up to 4 pad bytes after the
            // position streams that no copy writes; they are still initialised
            // (see the `pinned_pack` module docs).
            let mut pack = stg.packer_for(meta_region_bytes);
            pack.put_prefix_at("positions", 0, &stg.positions, proc_count)?;
            if use_mrope {
                let h_at = pos_stream_bytes;
                let w_at = h_at + pos_stream_bytes;
                pack.put_prefix_at("positions_h", h_at, &stg.positions_h, proc_count)?;
                pack.put_prefix_at("positions_w", w_at, &stg.positions_w, proc_count)?;
            }
            if !needs_paged {
                pack.put_prefix_at("slots", slot_offset, &stg.slots, proc_count)?;
            }
            self.gpu
                .copy_h2d_async_retained(pack.packed(), meta_base, stream)?;
        }

        Ok(MetaLayout {
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
        })
    }
}

#[path = "mrope_pos.rs"]
pub(crate) mod mrope_pos;
