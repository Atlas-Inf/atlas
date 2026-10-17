// SPDX-License-Identifier: AGPL-3.0-only

//! Prefix caching for vision prompts (issue #58).
//!
//! # Keys
//!
//! Every image of a prompt is the same run of placeholder tokens, so the radix
//! tree cannot be keyed on the prompt as the model sees it. At chunk 0 each
//! rank stamps `SequenceState::vision_cache_key`: the prompt with every
//! placeholder replaced by an id derived from its item's content hash and its
//! index in the item ([`vision_cache_key`]). Every prefix-cache call takes its
//! tokens through [`TransformerModel::pc_key`]; embedding, the splice, MRoPE
//! and the drafters keep reading the real tokens. The head computes the hashes
//! (`VisionItem::content_hash`) and sends them with the grids
//! (`EP_CMD_VISION_STATE`), so a worker rebuilds the same key and its radix
//! walk agrees with the head's before the EP min-sync.
//!
//! Only GLM-5 prompts are keyed. Its MRoPE walker rebuilds positions from the
//! complete prompt, so a prefill that starts after a cached image positions
//! the rest exactly as a full one would; the generic walker is chunk-local.
//!
//! The generated tokens of a vision sequence are not cached: decode writes
//! them at linear positions while a prefill of the same text writes the
//! compressed MRoPE positions, so their KV would not equal a recompute.
//! A next turn reuses the previous prompt and re-prefills the reply.
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
//! inside the cached prefix skip the ViT; their grids stay for MRoPE. The
//! splice offsets its rows by the rows it skipped (`skip_rows`) and by the
//! last encode's window (`rows_base`/`rows_encoded`). `buf_out` is shared
//! scratch: an encode first drains the prefill stream, whose queued splice
//! may still be reading the previous chunk's rows.

use std::borrow::Cow;
use std::sync::OnceLock;

use anyhow::{Result, ensure};
use spark_runtime::prefix_cache::vision_key::vision_cache_key;

use super::types::TransformerModel;
use crate::VisionItem;
use crate::request_local::request_local;
use crate::traits::SequenceState;

/// `ATLAS_VISION_PREFIX_CACHE=0` turns vision prefix caching off (vision
/// prompts then neither look up nor insert, as before). Read once.
fn switch_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_VISION_PREFIX_CACHE").as_deref() != Ok("0"))
}

#[path = "vision_cache_items.rs"]
mod items;
pub(crate) use items::{cached_items, first_pad_positions, items_in_range, last_pad_positions};

impl TransformerModel {
    /// Whether vision prompts are keyed for the prefix cache (module docs).
    pub(super) fn vision_prefix_cache_ok(&self) -> bool {
        self.prefix_cache.is_active()
            && self.vision_encoder.is_some()
            && self.config.vision.as_ref().is_some_and(|v| v.is_glm5_next)
            && switch_on()
    }

    /// Stamp `seq`'s cache key from the sequence's own vision state. Chunk 0
    /// only; a prompt without placeholders, or whose items cannot be keyed,
    /// keeps an empty key (and therefore stays uncached if it has
    /// placeholders).
    pub(in crate::model) fn stamp_vision_cache_key(&self, tokens: &[u32], seq: &mut SequenceState) {
        seq.vision_cache_key.clear();
        if !self.vision_prefix_cache_ok() || !self.tokens_have_vision_pad(tokens) {
            return;
        }
        let grids = &seq.vision.grids;
        let hashes = &seq.vision.hashes;
        if grids.len() != hashes.len() {
            return;
        }
        let items: Vec<(u128, usize)> = grids
            .iter()
            .zip(hashes)
            .map(|(&(t, h, w), &hash)| (hash, t * h * w))
            .collect();
        if let Some(key) = vision_cache_key(tokens, |t| self.is_vision_pad(t), &items) {
            seq.vision_cache_key = key;
        }
    }

    /// `tokens` (a prefix of `seq`'s stream from position 0) with the
    /// stamped key over the positions it covers.
    pub(in crate::model) fn pc_key_tokens<'a>(
        &self,
        seq: &SequenceState,
        tokens: &'a [u32],
    ) -> Cow<'a, [u32]> {
        let key = &seq.vision_cache_key;
        if key.is_empty() {
            return Cow::Borrowed(tokens);
        }
        let n = key.len().min(tokens.len());
        let mut out = Vec::with_capacity(tokens.len());
        out.extend_from_slice(&key[..n]);
        out.extend_from_slice(&tokens[n..]);
        Cow::Owned(out)
    }

    /// The prefix-cache key for `tokens` (a prefix of `seq`'s stream from
    /// position 0), or `None` when it holds a placeholder the key does not
    /// cover: such a prefix must not be looked up or cached.
    pub(in crate::model) fn pc_key<'a>(
        &self,
        seq: &SequenceState,
        tokens: &'a [u32],
    ) -> Option<Cow<'a, [u32]>> {
        let covered = seq.vision_cache_key.len().min(tokens.len());
        (!self.tokens_have_vision_pad(&tokens[covered..])).then(|| self.pc_key_tokens(seq, tokens))
    }

    /// Whether the current request's encode waits for its chunk-0 lookup.
    pub(in crate::model) fn vision_encode_deferred(seq: &SequenceState) -> bool {
        seq.vision.deferred
    }

    /// Clear a sequence's per-request vision state (the sequence is leaving
    /// the prefill path).
    pub(in crate::model) fn clear_seq_vision_state(seq: &mut SequenceState) {
        seq.vision = Default::default();
    }

    /// Derive the per-item post-merge grids and hashes of `images` into `seq`.
    /// Shared by every staging path: the key, MRoPE and the worker's command
    /// need them before any encode runs.
    fn stage_vision_grids(seq: &mut SequenceState, images: &[VisionItem], sms: usize) {
        seq.vision.grids = images
            .iter()
            .map(|it| (it.t_len(), it.grid_h / sms, it.grid_w / sms))
            .collect();
        seq.vision.hashes = images.iter().map(|it| it.content_hash).collect();
    }

    /// Trait entry for [`Self::stage_vision_grids`] (the co-dispatch path's
    /// rows are already encoded; only the geometry is staged).
    pub(in crate::model) fn stage_vision_grids_dispatch(
        &self,
        seq: &mut SequenceState,
        images: &[VisionItem],
    ) {
        let sms = self
            .config
            .vision
            .as_ref()
            .map_or(1, |v| v.spatial_merge_size.max(1));
        Self::stage_vision_grids(seq, images, sms);
    }

    /// Head, chunked path: stage `images` into `seq` for the encode the
    /// chunk-0 lookup resolves. The grids and hashes are set now (the key,
    /// MRoPE and the worker's command need them); `buf_out` holds nothing
    /// yet. GLM-5 encodes per chunk whether or not the prompt is cached:
    /// that, not the cache, is what keeps interleaved image prompts from
    /// reading each other's rows. Other towers encode at once.
    pub(in crate::model) fn prepare_vision_embed_deferred_dispatch(
        &self,
        seq: &mut SequenceState,
        images: &[VisionItem],
    ) -> Result<()> {
        if self.vision_encoder.is_none() {
            return self.prepare_vision_embed_dispatch(seq, images);
        }
        let sms = self
            .config
            .vision
            .as_ref()
            .map_or(1, |v| v.spatial_merge_size.max(1));
        seq.vision.rows_encoded = 0;
        seq.vision.rows_base = 0;
        seq.vision.skip_rows = 0;
        seq.vision.computed_from = 0;
        Self::stage_vision_grids(seq, images, sms);
        seq.vision.items = images.to_vec();
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
        // `vision_deferred` stays set: it now means "this prefill encodes
        // per chunk" (later chunks encode their own items right before their
        // embed), not "the chunk-0 lookup is still pending".
        let computed_from = if self.multi_rank_protocol_active() {
            let local = u32::try_from(computed_from).unwrap_or(u32::MAX);
            self.ep_min_u32(local)? as usize
        } else {
            computed_from
        };
        let counts: Vec<usize> = seq
            .vision
            .grids
            .iter()
            .map(|&(t, h, w)| t * h * w)
            .collect();
        let (skip, skip_rows) =
            cached_items(tokens, |t| self.is_vision_pad(t), &counts, computed_from);
        // Before the encode: `vision_encode_items` subtracts `skip_rows` when
        // it places the run's window (`rows_base`), so the cached leading rows
        // must already be recorded here.
        seq.vision.skip_rows = skip_rows;
        seq.vision.computed_from = computed_from;
        let items: Vec<usize> = items_in_range(
            &first_pad_positions(tokens, |t| self.is_vision_pad(t), &counts),
            &last_pad_positions(tokens, |t| self.is_vision_pad(t), &counts),
            computed_from,
            computed_from,
            chunk_end,
        );
        let rows = self.vision_encode_items(seq, &items, stream)?;
        tracing::info!(
            "Vision prefix cache: {skip} of {} items cached ({skip_rows} rows), encoded {rows} rows \
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
        let counts: Vec<usize> = seq
            .vision
            .grids
            .iter()
            .map(|&(t, h, w)| t * h * w)
            .collect();
        let items: Vec<usize> = items_in_range(
            &first_pad_positions(tokens, |t| self.is_vision_pad(t), &counts),
            &last_pad_positions(tokens, |t| self.is_vision_pad(t), &counts),
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
        let grids = &seq.vision.grids;
        // The encoded run's place in the prompt-order packed non-cached
        // output: every item before the first encoded one contributes its
        // rows (cached items' rows are skipped by the splice, not packed).
        // Computed on BOTH ranks (the worker has no pixels but must slide
        // its window identically), so it cannot depend on the items alone.
        let a: usize = items
            .first()
            .map_or(seq.vision.rows_base, |&first| {
                (0..first)
                    .map(|i| {
                        let (t, h, w) = grids[i];
                        t * h * w
                    })
                    .sum()
            })
            .saturating_sub(seq.vision.skip_rows);
        let rows: usize = items
            .iter()
            .map(|&i| {
                let (t, h, w) = grids[i];
                t * h * w
            })
            .sum();
        // `buf_out` is shared: the previous chunk's splice (this or another
        // sequence's) copies out of it on the prefill `stream`, while the
        // encoder and the broadcast write it on the default stream.
        let default = self.gpu.default_stream();
        if rows > 0 && stream != default {
            self.gpu.synchronize(stream)?;
        }
        // The head alone holds pixels: a worker (receive-only for vision)
        // skips the encode and takes the rows from the broadcast below.
        let encoded = if !items.is_empty() && rows > 0 && !seq.vision.items.is_empty() {
            self.vision_encode_on_head(seq, items, rows)
        } else {
            Ok(())
        };
        // A head-only failure is agreed before the payload broadcast, which
        // the worker is about to enter: both ranks then fail the request at
        // the same point (`request_local`) instead of leaving the worker in a
        // collective the head never issues.
        if rows > 0 && self.multi_rank_protocol_active() {
            let agreed = self.ep_min_u32(u32::from(encoded.is_ok()))?;
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
        self.gpu.synchronize(self.gpu.default_stream())?;
        if rows > 0 && self.multi_rank_protocol_active() {
            if rows > ve.output_rows() {
                return Err(request_local(
                    "deferred vision rows exceed encoder capacity",
                ));
            }
            let comm = self
                .comm
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("deferred vision payload without communicator"))?;
            comm.broadcast(ve.out_row().0, rows * ve.out_hidden_size() * 2, 0)?;
        }
        Ok(rows)
    }
}

impl TransformerModel {
    /// The head's encode of `items` into `buf_out` row 0, checked against
    /// the grids staged at admission (the key and the worker used those).
    fn vision_encode_on_head(
        &self,
        seq: &SequenceState,
        items: &[usize],
        rows: usize,
    ) -> Result<()> {
        let ve = self
            .vision_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("vision encode without an encoder"))?;
        let grids = &seq.vision.grids;
        ensure!(
            grids.len() == seq.vision.items.len(),
            "vision items and grids disagree"
        );
        let refs: Vec<&VisionItem> = items.iter().map(|&i| &seq.vision.items[i]).collect();
        let per_group = ve.forward_items(&refs, self.gpu.as_ref(), self.gpu.default_stream())?;
        // The grids were derived before the encode (the key and the worker
        // needed them); the encoder's own geometry must agree.
        let mut group = 0usize;
        for &i in items {
            let item = &seq.vision.items[i];
            let &(eh, ew, _) = per_group
                .get(group)
                .ok_or_else(|| anyhow::anyhow!("vision encoder returned {group} groups"))?;
            ensure!(
                (item.t_len(), eh, ew) == grids[i],
                "deferred vision grid {:?} disagrees with the encoder's {:?}",
                grids[i],
                (item.t_len(), eh, ew)
            );
            group += item.t_len();
        }
        let encoded: usize = per_group.iter().map(|&(_, _, m)| m).sum();
        ensure!(
            encoded == rows,
            "deferred vision encode produced {encoded} rows, the grids expect {rows}"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "vision_cache_tests.rs"]
mod tests;
