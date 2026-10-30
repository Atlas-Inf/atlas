// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence vision state and the per-chunk vision encode.
//!
//! # Encoding only what the prefill computes
//!
//! On the chunked path the head stages the items into the SEQUENCE instead of
//! encoding them ([`TransformerModel::prepare_vision_embed_deferred_dispatch`])
//! and the command carries only the grids and hashes. Once the chunk-0 lookup
//! has the match every rank agreed on, [`TransformerModel::vision_resolve_deferred`]
//! encodes the items with a computed placeholder inside chunk 0, and on a
//! multi-rank world broadcasts those rows. Every later chunk encodes its own
//! items the same way, right before its embed; an item straddling a chunk
//! boundary is re-encoded at each chunk it reaches. Leading items wholly
//! inside a cached prefix skip the ViT; their grids stay for MRoPE. The
//! splice offsets its rows by the rows it skipped (`skip_rows`) and by the
//! last encode's window (`rows_base`/`rows_encoded`). `buf_out` is shared
//! scratch: an encode first drains the prefill stream, whose queued splice
//! may still be reading the previous chunk's rows.

use anyhow::Result;

use super::types::TransformerModel;
use crate::VisionItem;
use crate::request_local::request_local;
use crate::traits::SequenceState;

#[path = "vision_cache_items.rs"]
mod items;
pub(crate) use items::{cached_items, first_pad_positions, items_in_range, last_pad_positions};

/// The encode of `items` (prompt-order indices into `counts`, each item's
/// rows): its place in the prompt-order packed output of the items the cache
/// did not cover wholly (the virtual row its first item starts at: every row
/// before it, less the `skip_rows` of wholly cached leading items), and its
/// row count. An empty selection keeps the previous window's base.
pub(crate) fn encode_window(
    counts: &[usize],
    items: &[usize],
    rows_base: usize,
    skip_rows: usize,
) -> (usize, usize) {
    let a = items
        .first()
        .map_or(rows_base.saturating_sub(skip_rows), |&first| {
            let before: usize = counts[..first].iter().sum();
            // An encoded item is never one the cache covers wholly.
            debug_assert!(
                before >= skip_rows,
                "item {first} starts inside the skipped rows"
            );
            before - skip_rows.min(before)
        });
    (a, items.iter().map(|&i| counts[i]).sum())
}

/// Each staged item's placeholder (= encoder row) count, in prompt order.
fn item_rows(seq: &SequenceState) -> Vec<usize> {
    seq.vision
        .grids
        .iter()
        .map(|&(t, h, w)| t * h * w)
        .collect()
}

impl TransformerModel {
    /// Before the embed of chunk `span` (`(start, len)`): a later chunk of a
    /// deferred prefill encodes its own items here, on every rank alike. True
    /// when the embed must follow the prefix lookup instead, where a deferred
    /// prefill's chunk 0 encodes (once the cached prefix is agreed).
    pub(in crate::model) fn vision_embed_after(
        &self,
        seq: &mut SequenceState,
        tokens: &[u32],
        span: (usize, usize),
        stream: u64,
    ) -> Result<bool> {
        self.vision_encode_chunk_items(seq, tokens, span.0, span.0 + span.1, stream)?;
        Ok(seq.vision.deferred)
    }

    /// Derive the per-item post-merge grids and hashes of `images` into `seq`.
    /// Shared by every staging path: MRoPE and the worker's command need them
    /// before any encode runs.
    fn stage_vision_grids(seq: &mut SequenceState, images: &[VisionItem], sms: usize) {
        seq.vision.grids = images
            .iter()
            .map(|it| (it.t_len(), it.grid_h / sms, it.grid_w / sms))
            .collect();
        seq.vision.hashes = images.iter().map(|it| it.content_hash).collect();
    }

    /// The vision config's spatial merge (1 without one).
    fn vision_merge(&self) -> usize {
        self.config
            .vision
            .as_ref()
            .map_or(1, |v| v.spatial_merge_size.max(1))
    }

    /// Trait entry for [`Self::stage_vision_grids`] (the co-dispatch path's
    /// rows are already encoded; only the geometry is staged).
    pub(in crate::model) fn stage_vision_grids_dispatch(
        &self,
        seq: &mut SequenceState,
        images: &[VisionItem],
    ) {
        Self::stage_vision_grids(seq, images, self.vision_merge());
    }

    /// Head, chunked path: stage `images` into `seq` for the encode the
    /// chunk-0 lookup resolves. The grids and hashes are set now (MRoPE and
    /// the worker's command need them); `buf_out` holds nothing yet. Every
    /// tower encodes per chunk: that is what keeps interleaved image prompts
    /// from reading each other's rows. An encode at admission left a long
    /// prompt's rows in the shared `buf_out` until its last chunk, and an
    /// image prompt admitted in between overwrote them (the long prompt then
    /// described the other prompt's image).
    pub(in crate::model) fn prepare_vision_embed_deferred_dispatch(
        &self,
        seq: &mut SequenceState,
        images: Vec<VisionItem>,
    ) -> Result<()> {
        if self.vision_encoder.is_none() {
            return self.prepare_vision_embed_dispatch(seq, images);
        }
        seq.vision.rows_encoded = 0;
        seq.vision.rows_base = 0;
        seq.vision.skip_rows = 0;
        seq.vision.computed_from = 0;
        Self::stage_vision_grids(seq, &images, self.vision_merge());
        // Moved, not copied: the request's only copy of the pixels.
        seq.vision.items = images;
        seq.vision.deferred = true;
        Ok(())
    }

    /// Run a deferred encode once the chunk-0 lookup has fixed the first
    /// computed position (module docs). Every rank calls this at the same
    /// point. The ranks min-reduce `computed_from` first, so they size the
    /// payload broadcast alike even if their restore depths ever differed;
    /// encoding from an earlier position is always correct. No-op otherwise.
    ///
    /// Encodes only the items whose first computed placeholder falls inside
    /// `[computed_from, chunk_end)`; a later chunk encodes its own items the
    /// same way (`vision_encode_chunk_items`).
    pub(in crate::model) fn vision_resolve_deferred(
        &self,
        seq: &mut SequenceState,
        tokens: &[u32],
        computed_from: usize,
        chunk_end: usize,
        stream: u64,
    ) -> Result<()> {
        if !seq.vision.deferred {
            return Ok(());
        }
        // `deferred` stays set: it now means "this prefill encodes per chunk"
        // (later chunks encode their own items right before their embed), not
        // "the chunk-0 lookup is still pending".
        let computed_from = if self.multi_rank_protocol_active() {
            let local = u32::try_from(computed_from).unwrap_or(u32::MAX);
            self.ep_min_u32(local)? as usize
        } else {
            computed_from
        };
        let counts = item_rows(seq);
        let is_pad = |t| self.is_vision_pad(t);
        let (skip, skip_rows) = cached_items(tokens, is_pad, &counts, computed_from);
        // Before the encode: `vision_encode_items` subtracts `skip_rows` when
        // it places the run's window (`rows_base`), so the cached leading rows
        // must already be recorded here.
        seq.vision.skip_rows = skip_rows;
        seq.vision.computed_from = computed_from;
        let items: Vec<usize> = items_in_range(
            &first_pad_positions(tokens, is_pad, &counts),
            &last_pad_positions(tokens, is_pad, &counts),
            computed_from,
            computed_from,
            chunk_end,
        );
        let rows = self.vision_encode_items(seq, &items, stream)?;
        tracing::info!(
            "Vision encode: {skip} of {} items cached ({skip_rows} rows), encoded {rows} rows \
             for positions from {computed_from}",
            counts.len()
        );
        Ok(())
    }

    /// Encode the items of `seq` whose placeholder range intersects this
    /// chunk's range `[chunk_start, chunk_end)` (module docs). Called for
    /// every chunk after the first, right before its embed; chunk 0's encode
    /// ran inside the lookup (`vision_resolve_deferred`). Only a DEFERRED
    /// prefill reaches here: an immediate or co-dispatched encode already
    /// wrote every row this prefill splices, and re-encoding would clobber
    /// the shared `buf_out` window.
    pub(in crate::model) fn vision_encode_chunk_items(
        &self,
        seq: &mut SequenceState,
        tokens: &[u32],
        chunk_start: usize,
        chunk_end: usize,
        stream: u64,
    ) -> Result<()> {
        // Grids, not items: a worker holds no pixels but must reach the
        // encode's broadcast (its rows arrive by receive, `vision_encode_items`).
        // Chunk 0's items belong to its lookup, which alone knows the cached
        // prefix: a path that embeds before its lookup must not encode early.
        if !seq.vision.deferred || seq.vision.grids.is_empty() || !seq.prefix_lookup_applied {
            return Ok(());
        }
        let counts = item_rows(seq);
        let is_pad = |t| self.is_vision_pad(t);
        let items: Vec<usize> = items_in_range(
            &first_pad_positions(tokens, is_pad, &counts),
            &last_pad_positions(tokens, is_pad, &counts),
            seq.vision.computed_from,
            chunk_start,
            chunk_end,
        );
        self.vision_encode_items(seq, &items, stream)?;
        Ok(())
    }

    /// Encode the named items of `seq` into `buf_out` at row 0 and broadcast
    /// them on a multi-rank world. Returns the row count written. The rows
    /// are complete before the splice reads them on the prefill stream and
    /// before the communicator's stream sends them.
    ///
    /// A worker (receive-only for vision) holds no pixels: it skips the
    /// encode and receives the head's broadcast into its own `buf_out`
    /// instead, so both ranks end with the same resident rows.
    fn vision_encode_items(
        &self,
        seq: &mut SequenceState,
        items: &[usize],
        stream: u64,
    ) -> Result<usize> {
        let ve = self
            .vision_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("vision encode without an encoder"))?;
        let counts = item_rows(seq);
        // Computed on BOTH ranks (the worker has no pixels but must slide
        // its window identically), so it cannot depend on the items alone.
        let (a, rows) = encode_window(&counts, items, seq.vision.rows_base, seq.vision.skip_rows);
        // Every rank derives `rows` from the same grids, so every rank fails
        // here alike, before the encode and any collective. The API admits
        // only requests whose every chunk fits (`vision_limits`).
        if rows > ve.output_rows() {
            return Err(request_local(format!(
                "vision chunk needs {rows} rows, the encoder output window holds {}",
                ve.output_rows()
            )));
        }
        // `buf_out` is shared: the previous chunk's splice (this or another
        // sequence's) copies out of it on the prefill `stream`, while the
        // encoder and the broadcast write it on the default stream.
        let default = self.gpu.default_stream();
        if rows > 0 && stream != default {
            self.gpu.synchronize(stream)?;
        }
        // The head (rank 0, or the only rank) alone holds pixels: a worker
        // (receive-only for vision) skips the encode and takes the rows from
        // the broadcast below. A head without pixels for rows it must encode
        // fails through the agreement below rather than leave the splice
        // reading whatever `buf_out` held.
        let head = self.comm.as_ref().is_none_or(|c| c.rank() == 0);
        let encoded = if rows == 0 || !head {
            Ok(())
        } else if seq.vision.items.is_empty() {
            Err(anyhow::anyhow!(
                "the head holds no pixels for {rows} vision rows"
            ))
        } else {
            self.vision_encode_on_head(seq, items, rows)
        };
        // A head-only failure is agreed before the payload broadcast, which
        // the worker is about to enter: both ranks then fail the request at
        // the same point (`request_local`) instead of leaving the worker in a
        // collective the head never issues.
        if rows > 0 && self.multi_rank_protocol_active() {
            let agreed = self.ep_min_u32_head_late(u32::from(encoded.is_ok()))?;
            if agreed == 0 {
                return Err(request_local(match encoded {
                    Err(e) => format!("vision encode failed: {e:#}"),
                    Ok(()) => "vision encode failed on another rank".to_string(),
                }));
            }
        } else if let Err(e) = encoded {
            return Err(request_local(format!("vision encode failed: {e:#}")));
        }
        // The window is exactly this chunk's splice range: every item it
        // holds has a placeholder in this chunk, and every item of earlier
        // chunks was either fully spliced there or is re-encoded here. So
        // `rows_base` is the run's virtual start (in the prompt-order packed
        // non-cached output) and `rows_encoded` the run's rows — the previous
        // window is obsolete. An empty selection encodes nothing and leaves
        // the window alone (the chunk splices no pads).
        if rows > 0 {
            seq.vision.rows_base = a;
            seq.vision.rows_encoded = rows;
        }
        self.gpu.synchronize(default)?;
        if rows > 0 && self.multi_rank_protocol_active() {
            let comm = self
                .comm
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("deferred vision payload without communicator"))?;
            comm.broadcast(ve.out_row().0, rows * ve.out_hidden_size() * 2, 0)?;
        }
        Ok(rows)
    }
}

#[path = "vision_cache_head.rs"]
mod head;

#[cfg(test)]
#[path = "vision_cache_stream_tests.rs"]
mod stream_tests;
#[cfg(test)]
#[path = "vision_cache_tests.rs"]
mod tests;
