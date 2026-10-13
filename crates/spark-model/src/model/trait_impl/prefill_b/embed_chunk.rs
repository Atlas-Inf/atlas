// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 1+1b: embed chunk tokens to hidden buffer + overlay vision-pad
//! positions with pre-computed vision encoder embeddings.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::{Result, bail, ensure};

use super::super::super::types::TransformerModel;
use crate::layers::ops;

/// Return the packed vision-row range covered by one prompt chunk. The
/// encoder output follows the complete prompt's pad-token order, so a chunk
/// starting in the middle of a media run must begin after every earlier image
/// or video pad token. `row_base` is the request slice in co-dispatch output;
/// `pending` is the complete packed output row count.
fn vision_pad_row_range(
    tokens: &[u32],
    chunk_start: usize,
    chunk_len: usize,
    image_pad: u32,
    video_pad: u32,
    row_base: usize,
    pending: usize,
) -> Result<(usize, usize)> {
    let chunk_tokens = &tokens[chunk_start..chunk_start + chunk_len];
    let is_pad = |tok: &&u32| **tok == image_pad || **tok == video_pad;
    let prior_pad_count = tokens[..chunk_start].iter().filter(is_pad).count();
    let chunk_pad_count = chunk_tokens.iter().filter(is_pad).count();
    let start = row_base
        .checked_add(prior_pad_count)
        .ok_or_else(|| anyhow::anyhow!("vision pad overlay row offset overflow"))?;
    let end = start
        .checked_add(chunk_pad_count)
        .ok_or_else(|| anyhow::anyhow!("vision pad overlay row count overflow"))?;
    if end > pending {
        bail!(
            "vision pad overlay exceeds packed encoder output: base={}, prior={}, chunk={}, pending={}",
            row_base,
            prior_pad_count,
            chunk_pad_count,
            pending
        );
    }
    Ok((start, end))
}

impl TransformerModel {
    pub(super) fn prefill_b_embed_chunk(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<()> {
        // Single-stream entry point: write to the arena's hidden buffer at offset 0.
        let hidden = self.buffers.hidden_states();
        self.prefill_b_embed_chunk_at(tokens, chunk_start, chunk_len, hidden, stream)
    }

    /// Embed `chunk_len` tokens into `hidden_dst` starting at position 0
    /// of the destination, then apply embedding scale + vision-pad overlay.
    /// Used by both the single-stream entry point above (writing into the
    /// arena's `hidden_states()`) and by Q12 batched prefill (writing into
    /// per-stream offsets of a shared stacked-streams buffer).
    pub(in crate::model) fn prefill_b_embed_chunk_at(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        hidden_dst: spark_runtime::gpu::DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        // BF16 residual is the shipping config (2 bytes/element).
        let elem_bytes = 2usize;
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
        // Vision embeddings are pre-computed by prepare_vision_embed() and stored in
        // the VisionEncoder's buf_out buffer ([total_patches, out_hidden_size] BF16).
        {
            let pending = *self.vision_embed_patches.lock();
            if pending > 0
                && let Some(ve) = &self.vision_encoder
            {
                let chunk_tokens = &tokens[chunk_start..chunk_start + chunk_len];
                // EITHER pad token. Matching only the image one meant a
                // video's positions were skipped entirely — no encoder row was
                // copied over them, the hidden state kept the raw token
                // embedding, and the model described a featureless gray field
                // while every token count looked correct.
                let (image_pad, video_pad) = self.vision_pad_ids();
                let prior_pad_rows = crate::model::vision_transport::pad_rows_before_chunk(
                    tokens,
                    chunk_start,
                    image_pad,
                    video_pad,
                );
                // Co-dispatch: this request's slice starts at vision_row_base
                // in the shared packed buf_out (0 for the legacy single encode).
                let row_base = *self.vision_row_base.lock();
                let owned_images = *self.vision_owned_images.lock();
                let slice_rows = *self.vision_slice_rows.lock();
                let chunk_pad_rows = chunk_tokens
                    .iter()
                    .filter(|&&tok| tok == image_pad || tok == video_pad)
                    .count();
                ensure!(
                    row_base
                        .checked_add(prior_pad_rows)
                        .and_then(|v| v.checked_add(chunk_pad_rows))
                        .is_some_and(|end| end <= pending),
                    "vision pad rows exceed encoded rows: base={row_base}, prior={prior_pad_rows}, chunk={chunk_pad_rows}, encoded={pending}"
                );
                if owned_images > 0 {
                    ensure!(
                        slice_rows > 0
                            && prior_pad_rows
                                .checked_add(chunk_pad_rows)
                                .is_some_and(|end| end <= slice_rows),
                        "vision pad rows exceed co-dispatched slice: prior={prior_pad_rows}, chunk={chunk_pad_rows}, slice={slice_rows}"
                    );
                }
                // The encoder output is packed in prompt order, while this
                // function may be called for several chunks. Starting at
                // zero for every chunk would repeat earlier vision rows.
                let (mut row_idx, _) = vision_pad_row_range(
                    tokens,
                    chunk_start,
                    chunk_len,
                    image_pad,
                    video_pad,
                    row_base,
                    pending,
                )?;

                for (i, &tok) in chunk_tokens.iter().enumerate() {
                    if tok == image_pad || tok == video_pad {
                        let src = ve.buf_out.offset(row_idx * ve.out_hidden_size * 2);
                        let dst = hidden_dst.offset(i * h * elem_bytes);
                        self.gpu
                            .copy_d2d_async(src, dst, ve.out_hidden_size * 2, stream)?;
                        row_idx += 1;
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::vision_pad_row_range;

    #[test]
    fn vision_rows_follow_the_full_prompt_across_chunks() {
        let image = 10;
        let video = 11;
        let text = 1;
        let tokens = [text, image, image, text, image, video, video, text];

        assert_eq!(
            vision_pad_row_range(&tokens, 0, 3, image, video, 4, 9).unwrap(),
            // `row_base` is already included in this absolute range; the
            // copy site must not add it a second time.
            (4, 6)
        );
        assert_eq!(
            vision_pad_row_range(&tokens, 3, 3, image, video, 4, 9).unwrap(),
            (6, 8)
        );
        assert_eq!(
            vision_pad_row_range(&tokens, 6, 2, image, video, 4, 9).unwrap(),
            (8, 9)
        );
    }

    #[test]
    fn vision_rows_fail_closed_when_the_packed_slice_is_short() {
        let err = vision_pad_row_range(&[1, 10, 10], 1, 2, 10, 11, 3, 4)
            .expect_err("two rows cannot fit in one packed row");
        assert!(err.to_string().contains("exceeds packed encoder output"));
    }
}
