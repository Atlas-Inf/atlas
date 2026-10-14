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
//! On the chunked path the head stages the items instead of encoding them
//! ([`TransformerModel::prepare_vision_embed_deferred_dispatch`]) and the
//! command carries only the grids and hashes. Once the chunk-0 lookup has the
//! match every rank agreed on, [`TransformerModel::vision_resolve_deferred`]
//! encodes the items from the first one that reaches a computed position,
//! and on a multi-rank world broadcasts those rows. Leading items wholly
//! inside the cached prefix skip the ViT; their grids stay for MRoPE. The
//! splice offsets its rows by the rows it skipped (`skip_rows`).

use std::borrow::Cow;
use std::sync::OnceLock;

use anyhow::{Result, ensure};
use spark_runtime::prefix_cache::vision_key::vision_cache_key;

use super::types::TransformerModel;
use crate::VisionItem;
use crate::traits::SequenceState;

/// The current vision request's content identity and deferred-encode state,
/// beside `vision_image_grids` / `vision_embed_patches`.
#[derive(Default)]
pub(crate) struct VisionCacheState {
    /// Content hash per item, parallel to `vision_image_grids`.
    pub hashes: Vec<u128>,
    /// An encode the chunk-0 lookup still has to run: the items on the head,
    /// empty on a worker (its rows arrive by broadcast).
    pub deferred: Option<Vec<VisionItem>>,
    /// Leading encoder rows `buf_out` does not hold: the cached items'.
    pub skip_rows: usize,
    /// First prompt position the prefill computes. Every skipped row's
    /// placeholder sits before it.
    pub computed_from: usize,
}

/// `ATLAS_VISION_PREFIX_CACHE=0` turns vision prefix caching off (vision
/// prompts then neither look up nor insert, as before). Read once.
fn switch_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_VISION_PREFIX_CACHE").as_deref() != Ok("0"))
}

/// Leading items whose every placeholder sits before `computed_from`, and
/// their placeholder (= encoder row) count. `counts` holds each item's
/// placeholders in prompt order.
pub(crate) fn cached_items(
    tokens: &[u32],
    is_pad: impl Fn(u32) -> bool,
    counts: &[usize],
    computed_from: usize,
) -> (usize, usize) {
    let (mut item, mut rows, mut seen) = (0usize, 0usize, 0usize);
    for (pos, &t) in tokens.iter().enumerate() {
        if item == counts.len() || pos >= computed_from {
            break;
        }
        if is_pad(t) {
            seen += 1;
            if seen == counts[item] {
                rows += seen;
                item += 1;
                seen = 0;
            }
        }
    }
    (item, rows)
}

impl TransformerModel {
    /// Whether vision prompts are keyed for the prefix cache (module docs).
    pub(super) fn vision_prefix_cache_ok(&self) -> bool {
        self.prefix_cache.is_active()
            && self.vision_encoder.is_some()
            && self.config.vision.as_ref().is_some_and(|v| v.is_glm5_next)
            && switch_on()
    }

    /// Stamp `seq`'s cache key from the current vision state. Chunk 0 only;
    /// a prompt without placeholders, or whose items cannot be keyed, keeps
    /// an empty key (and therefore stays uncached if it has placeholders).
    pub(in crate::model) fn stamp_vision_cache_key(&self, tokens: &[u32], seq: &mut SequenceState) {
        seq.vision_cache_key.clear();
        if !self.vision_prefix_cache_ok() || !self.tokens_have_vision_pad(tokens) {
            return;
        }
        let grids = self.vision_image_grids.lock();
        let hashes = &self.vision_cache.lock().hashes;
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
    pub(in crate::model) fn vision_encode_deferred(&self) -> bool {
        self.vision_cache.lock().deferred.is_some()
    }

    /// Clear every piece of per-request vision state.
    pub(in crate::model) fn clear_vision_state(&self) {
        *self.vision_embed_patches.lock() = 0;
        self.vision_image_grids.lock().clear();
        *self.vision_cache.lock() = VisionCacheState::default();
        *self.vision_row_base.lock() = 0;
        *self.vision_grid_base.lock() = 0;
        *self.vision_owned_images.lock() = 0;
        *self.vision_slice_rows.lock() = 0;
    }

    /// Head, chunked path: stage `images` for the encode the chunk-0 lookup
    /// resolves. The grids and hashes are set now (the key, MRoPE and the
    /// worker's command need them); `buf_out` holds nothing yet. Encodes at
    /// once when vision prompts are not cached.
    pub(in crate::model) fn prepare_vision_embed_deferred_dispatch(
        &self,
        images: &[VisionItem],
    ) -> Result<()> {
        if !self.vision_prefix_cache_ok() {
            return self.prepare_vision_embed_dispatch(images);
        }
        let sms = self
            .config
            .vision
            .as_ref()
            .map_or(1, |v| v.spatial_merge_size.max(1));
        *self.vision_embed_patches.lock() = 0;
        *self.vision_image_grids.lock() = images
            .iter()
            .map(|it| (it.t_len(), it.grid_h / sms, it.grid_w / sms))
            .collect();
        *self.vision_cache.lock() = VisionCacheState {
            hashes: images.iter().map(|it| it.content_hash).collect(),
            deferred: Some(images.to_vec()),
            ..Default::default()
        };
        Ok(())
    }

    /// Run a deferred encode once the chunk-0 lookup has fixed the first
    /// computed position (module docs). Every rank calls this at the same
    /// point. The ranks min-reduce `computed_from` first, so they size the
    /// payload broadcast alike even if their restore depths ever differed;
    /// encoding from an earlier position is always correct. No-op otherwise.
    pub(in crate::model) fn vision_resolve_deferred(
        &self,
        tokens: &[u32],
        computed_from: usize,
    ) -> Result<()> {
        let Some(items) = self.vision_cache.lock().deferred.take() else {
            return Ok(());
        };
        let computed_from = if self.multi_rank_protocol_active() {
            let local = u32::try_from(computed_from).unwrap_or(u32::MAX);
            self.ep_min_u32(local)? as usize
        } else {
            computed_from
        };
        let ve = self
            .vision_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("deferred vision encode without an encoder"))?;
        let grids = self.vision_image_grids.lock().clone();
        let counts: Vec<usize> = grids.iter().map(|&(t, h, w)| t * h * w).collect();
        let (skip, skip_rows) =
            cached_items(tokens, |t| self.is_vision_pad(t), &counts, computed_from);
        let rows: usize = counts[skip..].iter().sum();
        let stream = self.gpu.default_stream();
        if !items.is_empty() && rows > 0 {
            ensure!(
                items.len() == grids.len(),
                "deferred vision items and grids disagree"
            );
            let refs: Vec<&VisionItem> = items[skip..].iter().collect();
            let per_group = ve.forward_items(&refs, self.gpu.as_ref(), stream)?;
            // The grids were derived before the encode (the key and the
            // worker needed them); the encoder's own geometry must agree.
            let mut group = 0usize;
            for (item, &(t_len, h, w)) in items[skip..].iter().zip(&grids[skip..]) {
                let (eh, ew, _) = per_group[group];
                ensure!(
                    (item.t_len(), eh, ew) == (t_len, h, w),
                    "deferred vision grid {:?} disagrees with the encoder's {:?}",
                    (t_len, h, w),
                    (item.t_len(), eh, ew)
                );
                group += item.t_len();
            }
            let encoded: usize = per_group.iter().map(|&(_, _, m)| m).sum();
            ensure!(
                encoded == rows,
                "deferred vision encode produced {encoded} rows, the grids expect {rows}"
            );
        }
        *self.vision_embed_patches.lock() = rows;
        *self.vision_slice_rows.lock() = rows;
        {
            let mut st = self.vision_cache.lock();
            st.skip_rows = skip_rows;
            st.computed_from = computed_from;
        }
        // The rows must be complete before the splice reads them on the
        // prefill stream and before the communicator's stream sends them.
        self.gpu.synchronize(stream)?;
        if rows > 0 && self.multi_rank_protocol_active() {
            ensure!(
                rows <= ve.output_rows(),
                "deferred vision rows exceed encoder capacity"
            );
            let comm = self
                .comm
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("deferred vision payload without communicator"))?;
            comm.broadcast(ve.buf_out.0, rows * ve.out_hidden_size * 2, 0)?;
        }
        tracing::info!(
            "Vision prefix cache: {skip} of {} items cached ({skip_rows} rows), encoded {rows} rows \
             for positions from {computed_from}",
            grids.len()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::cached_items;

    const PAD: u32 = 9;

    #[test]
    fn only_items_wholly_before_the_computed_position_are_cached() {
        // text, item 0 (2 rows) at 1..3, text, item 1 (3 rows) at 4..7, text.
        let t = [1, PAD, PAD, 2, PAD, PAD, PAD, 3];
        let pad = |x| x == PAD;
        assert_eq!(cached_items(&t, pad, &[2, 3], 0), (0, 0));
        assert_eq!(
            cached_items(&t, pad, &[2, 3], 2),
            (0, 0),
            "item 0 is half computed"
        );
        assert_eq!(cached_items(&t, pad, &[2, 3], 3), (1, 2));
        assert_eq!(
            cached_items(&t, pad, &[2, 3], 6),
            (1, 2),
            "item 1 is partly computed"
        );
        assert_eq!(cached_items(&t, pad, &[2, 3], 7), (2, 5));
        assert_eq!(cached_items(&t, pad, &[2, 3], 99), (2, 5));
    }

    #[test]
    fn split_video_runs_count_by_item() {
        // One item of 3 placeholders split by timestamp text.
        let t = [PAD, 5, PAD, PAD, 6];
        assert_eq!(cached_items(&t, |x| x == PAD, &[3], 3), (0, 0));
        assert_eq!(cached_items(&t, |x| x == PAD, &[3], 4), (1, 3));
    }
}
