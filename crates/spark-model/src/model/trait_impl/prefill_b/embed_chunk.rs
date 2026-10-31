// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 1+1b: embed chunk tokens to hidden buffer + overlay vision-pad
//! positions with pre-computed vision encoder embeddings.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;

use super::super::super::types::TransformerModel;
use crate::layers::ops;
use crate::request_local::request_local;
use crate::traits::SequenceState;

/// `ATLAS_QWEN4EXP_PREFILL_HOST_IDS=1`: the prefill forward hands its layers
/// the host copy of the token ids it uploaded (`ForwardContext::
/// host_token_ids`). PLE otherwise reads them back from the device with a
/// blocking copy in the middle of the pass, which drains the stream and
/// leaves the GPU idle while the host then hashes the n-grams (5.5 ms at 16K
/// on the pair, nsys `sqpf-p3s7-r0`). The ids are the same bytes: the copy
/// mirrors the last upload into the stable `token_ids` buffer.
pub(in crate::model) fn host_ids_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_HOST_IDS").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

thread_local! {
    /// The ids last uploaded into the stable `token_ids` buffer.
    static STAGED_IDS: std::cell::RefCell<Option<Vec<u32>>> =
        const { std::cell::RefCell::new(None) };
}

/// The host copy of the ids in `token_ids`, taken by the forward that reads
/// them (`None`: switch off, or no prefill upload since the last take).
pub(super) fn take_staged_ids() -> Option<Vec<u32>> {
    STAGED_IDS.with(|s| s.borrow_mut().take())
}

/// Where each placeholder of `tokens[start..start + count]` that the prefill
/// COMPUTES reads its encoder row: `(offset in the range, physical buf_out
/// row)`.
///
/// The encoder output follows the complete prompt's placeholder order, less
/// the leading `skip_rows` of items the prefix cache covered wholly (their
/// rows are never produced). `row_idx` is a placeholder's index in that
/// packed order (`row_base` offsets a co-dispatched slice); `buf_out` holds
/// the last encode's window, re-based to physical 0 at `rows_base`, of
/// `pending` rows.
///
/// A placeholder before `computed_from` needs no row: its KV is restored from
/// the cache and no layer reads its embedding. That includes the cached head
/// of an item the cache covers only in part, whose rows the window holds
/// only from the chunk that first computes one of its placeholders: such a
/// head still advances `row_idx`, so the computed tail maps to its own rows.
/// Requiring rows for it failed every chunk lying wholly in the cache with a
/// straddling item in it (the issue-180 history on a warm cache).
///
/// A computed placeholder outside the window, or one belonging to a wholly
/// cached item, is request-local (every rank decides it from the same tokens
/// and window): a prompt with more placeholders than its items have rows
/// fails alone.
#[allow(clippy::too_many_arguments)]
pub(in crate::model) fn vision_splice_plan(
    tokens: &[u32],
    start: usize,
    count: usize,
    is_pad: impl Fn(u32) -> bool,
    row_base: usize,
    skip_rows: usize,
    computed_from: usize,
    rows_base: usize,
    pending: usize,
) -> Result<Vec<(usize, usize)>> {
    let prior = tokens[..start].iter().filter(|&&t| is_pad(t)).count();
    let mut pad_idx = prior;
    let mut row_idx = row_base
        .checked_add(prior.saturating_sub(skip_rows))
        .ok_or_else(|| request_local("vision pad overlay row offset overflow"))?;
    let mut plan = Vec::new();
    for (i, &tok) in tokens[start..start + count].iter().enumerate() {
        if !is_pad(tok) {
            continue;
        }
        let pos = start + i;
        if pad_idx < skip_rows {
            if pos >= computed_from {
                return Err(request_local(format!(
                    "cached vision placeholder at {pos} is computed (from {computed_from})"
                )));
            }
        } else {
            if pos >= computed_from {
                let physical = row_idx.checked_sub(rows_base).ok_or_else(|| {
                    request_local(format!(
                        "vision row {row_idx} already consumed (base {rows_base})"
                    ))
                })?;
                if physical >= pending {
                    return Err(request_local(format!(
                        "vision pad overlay exceeds the resident window: row {row_idx} at {pos}, \
                         window {rows_base}..{}, skipped={skip_rows}",
                        rows_base + pending
                    )));
                }
                plan.push((i, physical));
            }
            row_idx += 1;
        }
        pad_idx += 1;
    }
    Ok(plan)
}

impl TransformerModel {
    pub(super) fn prefill_b_embed_chunk(
        &self,
        seq: &SequenceState,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<()> {
        // Single-stream entry point: write to the arena's hidden buffer at offset 0.
        let hidden = self.buffers.hidden_states();
        self.prefill_b_embed_chunk_at(seq, tokens, chunk_start, chunk_len, hidden, stream)
    }

    /// Embed `chunk_len` tokens into `hidden_dst` starting at position 0
    /// of the destination, then apply embedding scale + vision-pad overlay.
    /// Used by both the single-stream entry point above (writing into the
    /// arena's `hidden_states()`) and by Q12 batched prefill (writing into
    /// per-stream offsets of a shared stacked-streams buffer).
    pub(in crate::model) fn prefill_b_embed_chunk_at(
        &self,
        seq: &SequenceState,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        hidden_dst: spark_runtime::gpu::DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let _tp = std::time::Instant::now();

        // ── 1. Embed chunk tokens → [chunk_len, H] contiguous at hidden_dst ──
        // Upload token IDs to device and do a single batched embed kernel launch
        // instead of chunk_len individual D2D copies.
        {
            let chunk_tokens = &tokens[chunk_start..chunk_start + chunk_len];
            // SAFETY: `chunk_tokens` is sliced on the line above with an END
            // bound of `chunk_start + chunk_len`, so its length IS `chunk_len`
            // (an out-of-range chunk panics in that slice index first) and the
            // byte length is `chunk_tokens.len() * size_of::<u32>()` over a live
            // `&[u32]`.
            let token_ids_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(chunk_tokens.as_ptr() as *const u8, chunk_len * 4)
            };
            let token_ids_dev = self.buffers.scratch(); // temporary, overwritten by MoE later
            self.gpu
                .copy_h2d_async(token_ids_bytes, token_ids_dev, stream)?;
            // Also stage this chunk's token IDs into the STABLE token_ids buffer
            // (scratch is reused by MoE routing). DeepSeek-V4 hash-MoE reads
            // `tid2eid[token_id]` per token in this same chunk order.
            self.gpu
                .copy_h2d_async(token_ids_bytes, self.buffers.token_ids(), stream)?;
            STAGED_IDS
                .with(|s| *s.borrow_mut() = host_ids_requested().then(|| chunk_tokens.to_vec()));
            let t_h2d = _tp.elapsed();
            if self.has_ngram_embedding() {
                // THE chunked-prefill embed. n-gram hashes read behind the
                // chunk, so hand it the earlier tokens of the prompt as well.
                let cs = chunk_start.saturating_sub(self.ngram_lookbehind());
                self.embed_tokens_fused(
                    &tokens[cs..chunk_start + chunk_len],
                    chunk_len,
                    hidden_dst,
                    stream,
                )?;
            } else {
                ops::batched_embed(
                    self.gpu.as_ref(),
                    self.batched_embed_kernel,
                    token_ids_dev,
                    self.embed_tokens.weight,
                    hidden_dst,
                    chunk_len as u32,
                    h as u32,
                    stream,
                )?;
            }
            let t_gather = _tp.elapsed();
            if std::env::var("ATLAS_DUMP_EMBED").ok().as_deref() == Some("1") {
                self.gpu.synchronize(stream)?;
                let offset = (chunk_len - 1) * h * 2;
                let mut buf = vec![0u8; h * 2];
                let _ = self.gpu.copy_d2h(hidden_dst.offset(offset), &mut buf);
                let v: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                tracing::info!(
                    "ATLAS_EMBED post-batched_embed (chunk_start={}, last_tok_id={}): |x|={:.4} first5={:?}",
                    chunk_start,
                    tokens[chunk_start + chunk_len - 1],
                    n,
                    &v[..5]
                );
            }
            // Feature-2: overlay overridden vocab rows AFTER the gather, BEFORE
            // the embed scale (the override row is a raw embed row that must
            // also be scaled). `token_ids()` holds this chunk's ids (staged
            // above); uniform-active route (seq_slot NULL). No-op when no
            // overlay is installed.
            self.apply_embed_overlay(
                self.buffers.token_ids(),
                spark_runtime::gpu::DevicePtr(0),
                hidden_dst,
                chunk_len as u32,
                stream,
            )?;
            let t_overlay = _tp.elapsed();
            self.scale_embeddings(hidden_dst, chunk_len, stream)?;
            let t_scale = _tp.elapsed();
            if std::env::var_os("ATLAS_PROFILE_PREFILL").is_some() {
                tracing::info!(
                    "embed profile (len={}): h2d={:?} gather={:?} overlay={:?} scale={:?}",
                    chunk_len,
                    t_h2d,
                    t_gather.saturating_sub(t_h2d),
                    t_overlay.saturating_sub(t_gather),
                    t_scale.saturating_sub(t_overlay),
                );
            }
            if std::env::var("ATLAS_DUMP_EMBED").ok().as_deref() == Some("1") {
                self.gpu.synchronize(stream)?;
                let offset = (chunk_len - 1) * h * 2;
                let mut buf = vec![0u8; h * 2];
                let _ = self.gpu.copy_d2h(hidden_dst.offset(offset), &mut buf);
                let v: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                tracing::info!(
                    "ATLAS_EMBED post-scale_embeddings: |x|={:.4} first5={:?}",
                    n,
                    &v[..5]
                );
            }
        }

        // ── 1b. Overwrite image_pad token positions with vision encoder embeddings ──
        self.splice_vision_rows(seq, tokens, chunk_start, chunk_len, hidden_dst, stream)?;

        Ok(())
    }

    /// Overwrite the vision placeholders of `tokens[start..start + count]`,
    /// embedded at `hidden_dst` row 0 onward, with their encoder rows.
    ///
    /// Vision embeddings are pre-computed by prepare_vision_embed() (or the
    /// per-chunk encode, `vision_cache`) and stored in the VisionEncoder's
    /// buf_out buffer ([rows, out_hidden_size] BF16) in prompt order, less the
    /// leading `skip_rows` of items the prefix cache covered: their
    /// placeholders all sit at positions this prefill does not compute, so
    /// their rows are left as the raw token embedding no layer reads.
    ///
    /// `buf_out` is the single shared scratch row buffer: each per-chunk
    /// encode writes the items this chunk splices at row 0 (a straddling item
    /// is re-encoded at every chunk holding one of its placeholders), so the
    /// resident window is exactly this chunk's splice range — `rows_base` is
    /// its first virtual row and `rows_encoded` its row count. A pad mapping
    /// below the base belongs to an earlier chunk and is refused.
    pub(in crate::model) fn splice_vision_rows(
        &self,
        seq: &SequenceState,
        tokens: &[u32],
        start: usize,
        count: usize,
        hidden_dst: spark_runtime::gpu::DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        // BF16 residual is the shipping config (2 bytes/element).
        let elem_bytes = 2usize;
        let pending = seq.vision.rows_encoded;
        let skip_rows = seq.vision.skip_rows;
        let computed_from = seq.vision.computed_from;
        let Some(ve) = &self.vision_encoder else {
            return Ok(());
        };
        // A text request. A vision one is always planned: a computed
        // placeholder with no resident row is an error, never a silent raw
        // token embedding.
        if pending == 0 && skip_rows == 0 && seq.vision.grids.is_empty() {
            return Ok(());
        }
        let chunk_tokens = &tokens[start..start + count];
        // EITHER pad token. Matching only the image one meant a video's
        // positions were skipped entirely — no encoder row was copied over
        // them, the hidden state kept the raw token embedding, and the model
        // described a featureless gray field while every token count looked
        // correct.
        let (image_pad, video_pad) = self.vision_pad_ids();
        let chunk_pad_rows = chunk_tokens
            .iter()
            .filter(|&&tok| tok == image_pad || tok == video_pad)
            .count();
        if chunk_pad_rows == 0 {
            return Ok(());
        }
        let prior_pad_rows = crate::model::vision_transport::pad_rows_before_chunk(
            tokens, start, image_pad, video_pad,
        );
        // Co-dispatch: this request's slice starts at vision_row_base in the
        // shared packed buf_out (0 for the legacy single encode).
        let row_base = seq.vision.row_base;
        let owned_images = seq.vision.owned_images;
        let slice_rows = seq.vision.slice_rows;
        if owned_images > 0
            && !(slice_rows > 0
                && prior_pad_rows
                    .checked_add(chunk_pad_rows)
                    .is_some_and(|end| end <= slice_rows))
        {
            return Err(request_local(format!(
                "vision pad rows exceed co-dispatched slice: prior={prior_pad_rows}, chunk={chunk_pad_rows}, slice={slice_rows}"
            )));
        }
        // Physical rows from the packed prompt order (`vision_splice_plan`).
        // Co-dispatch reads a slice of the shared packed output with the base
        // at 0 (the batched encode wrote every request's rows; nothing has
        // been re-encoded since).
        let plan = vision_splice_plan(
            tokens,
            start,
            count,
            |t| t == image_pad || t == video_pad,
            row_base,
            skip_rows,
            computed_from,
            seq.vision.rows_base,
            pending,
        )?;
        for (i, physical) in plan {
            let src = ve.out_row().offset(physical * ve.out_hidden_size() * 2);
            let dst = hidden_dst.offset(i * h * elem_bytes);
            self.gpu
                .copy_d2d_async(src, dst, ve.out_hidden_size() * 2, stream)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::vision_splice_plan;

    const IMAGE: u32 = 10;
    const VIDEO: u32 = 11;
    const TEXT: u32 = 1;
    fn pad(t: u32) -> bool {
        t == IMAGE || t == VIDEO
    }

    /// Physical rows the computed placeholders of `[start, start + count)` read.
    fn rows(
        tokens: &[u32],
        start: usize,
        count: usize,
        args: (usize, usize, usize, usize, usize),
    ) -> Vec<usize> {
        let (row_base, skip, computed_from, rows_base, pending) = args;
        vision_splice_plan(
            tokens,
            start,
            count,
            pad,
            row_base,
            skip,
            computed_from,
            rows_base,
            pending,
        )
        .unwrap()
        .into_iter()
        .map(|(_, p)| p)
        .collect()
    }

    #[test]
    fn vision_rows_follow_the_full_prompt_across_chunks() {
        let tokens = [TEXT, IMAGE, IMAGE, TEXT, IMAGE, VIDEO, VIDEO, TEXT];
        // A co-dispatched slice at row 4 of a window based at 0.
        assert_eq!(rows(&tokens, 0, 3, (4, 0, 0, 0, 9)), [4, 5]);
        assert_eq!(rows(&tokens, 3, 3, (4, 0, 0, 0, 9)), [6, 7]);
        assert_eq!(rows(&tokens, 6, 2, (4, 0, 0, 0, 9)), [8]);
    }

    #[test]
    fn vision_rows_fail_closed_when_the_packed_slice_is_short() {
        let err = vision_splice_plan(&[1, 10, 10], 1, 2, pad, 3, 0, 0, 0, 4)
            .expect_err("two rows cannot fit in one packed row");
        assert!(err.to_string().contains("exceeds the resident window"));
    }

    #[test]
    fn a_placeholder_past_the_encoded_rows_fails_only_its_request() {
        // One 2-row image, then a stray placeholder (a pad id the request's
        // items do not explain) in a later chunk: every rank reaches this
        // check alike, so the error must be request-local, not rank-fatal.
        let tokens = [TEXT, IMAGE, IMAGE, TEXT, IMAGE, TEXT];
        let err = vision_splice_plan(&tokens, 3, 3, pad, 0, 0, 0, 0, 2).expect_err("stray pad");
        assert!(crate::request_local::is_request_local(&err));
    }

    #[test]
    fn cached_items_shift_rows_down_by_the_skipped_rows() {
        // Item 0 (2 rows, cached) at 1..3, item 1 (2 rows, encoded) at 4..6.
        let tokens = [TEXT, IMAGE, IMAGE, TEXT, IMAGE, IMAGE, TEXT];
        assert_eq!(rows(&tokens, 3, 4, (0, 2, 3, 0, 2)), [0, 1]);
        // A chunk over the cached item computes nothing.
        assert!(rows(&tokens, 0, 3, (0, 2, 3, 0, 2)).is_empty());
        assert!(vision_splice_plan(&tokens, 3, 4, pad, 0, 2, 3, 0, 1).is_err());
    }

    #[test]
    fn a_cached_head_of_a_straddling_item_needs_no_rows() {
        // Item 0 (3 rows) at 1..4 is cached up to position 3: its head (1, 2)
        // has no row and the window is empty; the chunk over it must pass
        // (it failed with "exceeds the resident window", issue 180), and the
        // chunk that computes its tail reads its own rows 2.. of the item.
        let tokens = [TEXT, IMAGE, IMAGE, IMAGE, TEXT];
        assert!(rows(&tokens, 0, 3, (0, 0, 3, 0, 0)).is_empty());
        // The later chunk re-encodes item 0 whole: window [0, 3).
        assert_eq!(rows(&tokens, 3, 2, (0, 0, 3, 0, 3)), [2]);
    }

    #[test]
    fn a_per_chunk_window_covers_exactly_its_own_splice_range() {
        // Item 0 (2 rows) pads at 1..3, item 1 (3 rows) pads at 4..7. Chunk 1
        // = [3, 8) holds item 1's pads only: its window is virtual [2, 5) —
        // 3 rows at physical 0, base 2.
        let tokens = [TEXT, IMAGE, IMAGE, TEXT, IMAGE, IMAGE, IMAGE, TEXT];
        assert_eq!(rows(&tokens, 3, 5, (0, 0, 0, 2, 3)), [0, 1, 2]);
        assert!(vision_splice_plan(&tokens, 3, 5, pad, 0, 0, 0, 2, 2).is_err());
    }
}
