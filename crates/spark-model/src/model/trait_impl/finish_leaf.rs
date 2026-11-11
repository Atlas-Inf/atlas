// SPDX-License-Identifier: AGPL-3.0-only

//! End-of-turn SSM snapshot on a block boundary
//! (`ATLAS_GLM_PC_FINISH_LEAF=1`, default off).
//!
//! # What it is for
//!
//! A prefill saves one SSM checkpoint, a block or two below the prompt end.
//! The next turn of the conversation restores it and re-prefills everything
//! after it: the gap to the prompt end, all of the previous turn's output and
//! the new message. With this flag the sequence also keeps the state it had
//! at the last save boundary its decode crossed (its leaf), and the next turn
//! restores there instead.
//!
//! # Why the older leaves were wrong
//!
//! The finish leaf (`finish_leaf_snapshot`) copies the LIVE state at finish
//! and labels it `seq.tokens.len()`. Production turns it off, along with the
//! prompt-end and decode-cadence snapshots (`ATLAS_MARCONI_PREFILL_ONLY`),
//! because restores gave wrong answers. What the code shows:
//!
//! 1. *Position.* A verify step that finished the sequence returned before
//!    its commit (fixed separately), so the head's state was not the state
//!    at `seq.tokens.len()`: with KDA records the h state had none of the
//!    step's rows, the conv state had all `k` rows including the rejected
//!    drafts, and the aux rewind had not run.
//! 2. *Rank symmetry.* `cache_sequence` runs on the head only, so only the
//!    head held the leaf. It came into reach two turns later (its blocks
//!    are then prompt blocks), and when the message in between was shorter
//!    than the 17-32 token tail gap it was the head's deepest snapshot while
//!    the worker's was the prefill checkpoint: two restore depths, so every
//!    TP collective paired one rank's row `i` with the other rank's row
//!    `i + d`. The extra head-only snapshots also made the two 16-slot pools
//!    evict different victims, which desynchronizes restores of any kind.
//! 3. *Alignment.* `seq.tokens.len()` is rarely a block boundary and a lookup
//!    only returns snapshots at or below its block-aligned match, so the
//!    leaf could not serve the turn it was saved for. With sub-block matching
//!    it would pair with a partial block whose tail holds rejected rows.
//! 4. *Reachability.* The radix insert at finish left the block under the
//!    prompt end without a sequence reference (see [`owned_from`]), so no
//!    block of the generated output was ever matched by the next turn.
//!
//! The prompt-end leaf is exact as saved (both ranks, end of prefill), but it
//! sits mid-block 17-32 tokens above the tail checkpoint and buys at most
//! that, so it stays off.
//!
//! # What this does instead
//!
//! The state is saved at the moment it is exactly on a boundary, by model
//! code that both ranks run from rank-identical inputs:
//!
//! * A records commit folds the accepted rows of a verify step into the h
//!   state one row at a time. When the step crosses a boundary the fold is
//!   issued in two launches, to the boundary and from it, with the save
//!   between. `kda_commit_records` applies the same per-row fold from and to
//!   FP32 memory, so the split is bit-identical to the single launch. The
//!   conv state at the boundary is the verify's own snapshot after that row.
//! * A plain decode step advances the state in place by one token, so the
//!   live state is the boundary state whenever the new length is a boundary.
//!
//! The save reuses one Marconi snapshot slot per sequence (a rolling slot),
//! so a turn costs one slot however long it is. At finish the head tells the
//! worker to cache its mirror of the sequence too ([`EP_CMD_CACHE_SEQUENCE`]),
//! which is what brings the leaf into reach: a lookup only returns a snapshot
//! at or below its block match, and the blocks of the output are cached at
//! finish. A restore goes through the usual gate plus the restore-depth rank
//! agreement (`pc_policy`), so a rank that lost its copy makes both ranks
//! fall back together.
//!
//! # The snapshot pool
//!
//! A leaf serves one turn of one conversation, and only if that turn's prompt
//! reproduces the output up to the boundary (not a reasoning turn whose
//! history drops the thinking, nor a client that re-renders the output). So
//! it is a second-class entry (`spark_runtime::radix_tree`, `snapshot_leaf`)
//! and it never costs a conversation a checkpoint:
//!
//! * The rolling slot is registered as a leaf from its first save and moved
//!   boundary by boundary. No slot is held outside the index: a prefill
//!   checkpoint that needs a slot takes any leaf, a decoding sequence's
//!   included, before it takes a frontier. The sequence asks for its slot
//!   back before each save and takes another when it is gone.
//! * A rolling slot comes from the free list, from superseded history or
//!   from another leaf. It never evicts a conversation's frontier checkpoint
//!   or a branch point; when none of the three is available the turn takes
//!   no leaf.
//! * A leaf does not supersede the prefill checkpoint under it, which stays
//!   the conversation's protected restore point.
//! * The turn that restores a leaf settles it
//!   ([`TransformerModel::finish_leaf_restored`]). Normally it saves a tail
//!   checkpoint of its own and the leaf becomes superseded history, which
//!   that save and the turn's own rolling slot then reuse. A turn that
//!   restores at or above its tail cut saves no checkpoint (its new message
//!   is shorter than the tail gap), and the leaf is promoted to one.
//!
//! # Stream order
//!
//! A leaf's slot can change hands between two writes (evicted, then reused
//! by a tail checkpoint or by another sequence's leaf) and the writes come
//! from different streams: the secondary stream for a verify commit, the
//! default stream for a plain decode, the prefill stream for a checkpoint
//! and for a restore's read. With the flag on every one of them waits on the
//! snapshot event before it touches a slot and records it after, so they
//! form one order whatever the streams.
//!
//! # Preconditions
//!
//! The flag is ignored with a warning unless all of these hold:
//!
//! * `ATLAS_PREFIX_SUBBLOCK=0`: the leaf only pairs with a whole-block match.
//! * `ATLAS_MARCONI_PREFILL_ONLY=1`: the legacy leaves above stay off.
//! * `ATLAS_GLM_PC_EVICT=1`: the rules above are classes of the chain-aware
//!   victim, which also makes both ranks choose victims the same way (it
//!   does not use `session_hash`, which only the head knows).
//!
//! # Numerics
//!
//! Exact, but not the bits a cold prefill of the same prompt produces. The
//! restored state and the KV rows of the previous turn's output come from the
//! decode kernels that generated that output (the state the model really
//! had), where the flag-off path recomputes both with the prefill kernels.
//! The next prefill also starts at a different row, so its pass shapes differ
//! (see the accumulation-order note in `pc_policy`).
//!
//! # Cost
//!
//! One D2D copy of the sequence's SSM state per rank (74 MiB; about 3 ms in
//! `scripts/finish-leaf-bench`, a preliminary standalone reading on a shared
//! GB10) each time decode crosses a save boundary
//! (`ATLAS_GLM_PC_FINISH_LEAF_BLOCKS` blocks apart, default 4), on the stream
//! that advanced the state. For a verify commit that is the secondary
//! stream, where the copy overlaps the head's next propose. A plain decode
//! step saves on the default stream ahead of its logits read, so there the
//! copy is not hidden. Every request pays it, including the ones whose leaf
//! is never restored. The DFlash drafter's context on the next turn shrinks
//! with the re-prefill it replaces (it is filled from the rows a prefill
//! processes); a wider span keeps more of it.
//!
//! # Rank env parity
//!
//! Both ranks must run with the same `ATLAS_GLM_PC_FINISH_LEAF`,
//! `ATLAS_GLM_PC_FINISH_LEAF_BLOCKS` and the three variables above. The flag
//! itself adds no collective (a mismatch in it or in the span loses restores
//! and logs an error on the worker, it does not hang), but
//! `ATLAS_GLM_PC_EVICT` does (see "Rank env parity" in `pc_policy`). The
//! ranks compare all of them at startup (`model::startup_parity`).
//!
//! # Known limits
//!
//! * A preempted sequence is cached through the same command and resumes
//!   through `Model::prefill`, which takes no rank agreement (as in base,
//!   where the worker resumes through the chunk path) and does not take part
//!   in the stream order above. Keep the flag off where preemption can run
//!   (`--swap-space-gb` above 0) until resume takes the chunk path.
//! * The spill tier (`ATLAS_SSM_TIER`) reads slots outside that order too,
//!   and would spill leaves it should drop. Keep it off with the flag.
//!
//! # qwen4_exp (`ATLAS_QWEN4EXP_FINISH_LEAF=1`)
//!
//! The same leaf for a model with per-sequence aux state, saved at the end
//! of the step that crossed a save boundary rather than on it, with its aux
//! (see `finish_leaf/qwen4exp.rs`). It shares everything above:
//! preconditions, pool classes, stream order, mirrored caching and the rank
//! agreement.

use anyhow::Result;

use super::super::ssm_batched_copy::run_ssm_state_copies;
use super::super::types::TransformerModel;
use crate::traits::SequenceState;

mod cache;
mod flag;
mod qwen4exp;
mod rolling;
pub(in crate::model) use flag::{enabled, qwen4exp_enabled, qwen4exp_requested, span_blocks};
pub(crate) use rolling::LeafCell;
use rolling::{FinishLeaf, leaf_copies, leaf_save, owned_from};

/// Head to worker: cache your mirror of this slot's sequence (then the usual
/// free follows). Sent only with the flag on.
pub(in crate::model) const EP_CMD_CACHE_SEQUENCE: u32 = 0xFFFF_FFF8;

impl TransformerModel {
    /// The save span in tokens when `seq` can carry a rolling leaf now.
    fn finish_leaf_span(&self, seq: &SequenceState) -> Option<usize> {
        let ok = self.leaf_on()
            && self.ssm_snapshots.is_enabled()
            && self.prefix_cache.is_active()
            && self.config.num_ssm_layers() > 0
            && self.ssm_pool.h_stored_bytes == self.ssm_snapshots.h_bytes()
            // A leaf of a model with aux state carries it (`qwen4exp`).
            && (!self.requires_aux_state() || self.qwen4exp_leaf_on())
            && seq.slot_idx < self.ssm_pool.max_slots
            && seq.hss_window_start() == 0
            && seq.seq_len == seq.tokens.len()
            && !self.seq_ssm_h_is_f16(seq);
        ok.then(|| {
            self.kv_cache
                .lock()
                .block_size()
                .saturating_mul(span_blocks())
        })
    }

    /// A snapshot slot for a rolling leaf: free, or taken from superseded
    /// history or another leaf. Never from a frontier (see "The snapshot
    /// pool" above).
    fn finish_leaf_reserve(&self, session_hash: u64) -> Option<usize> {
        let snaps = &self.ssm_snapshots;
        snaps.reserve_tail_slot(session_hash).or_else(|| {
            snaps.free(self.prefix_cache.evict_snapshot_for_leaf()?);
            snaps.reserve_tail_slot(session_hash)
        })
    }

    /// Take `seq`'s rolling leaf out of the index, if it is still there:
    /// the slot is then the caller's to write or to free.
    fn finish_leaf_take(&self, seq: &SequenceState) -> Option<usize> {
        self.finish_leaf_take_leaf(seq, seq.finish_leaf.take()?)
    }

    /// [`Self::finish_leaf_take`] of `leaf`, already taken out of its cell.
    fn finish_leaf_take_leaf(&self, seq: &SequenceState, leaf: FinishLeaf) -> Option<usize> {
        let prefix = seq.tokens.get(..leaf.tokens)?;
        self.prefix_cache
            .take_leaf_snapshot(prefix, leaf.snap, seq.adapter_id)
            .then_some(leaf.snap)
    }

    /// Order `stream` after every snapshot copy recorded so far (see "Stream
    /// order" above). With the flag off this is nothing: the prefill
    /// checkpoint calls it too.
    pub(super) fn finish_leaf_wait_copies(&self, stream: u64) -> Result<()> {
        if self.leaf_on() {
            self.wait_snapshot_saves_dispatch(stream)?;
        }
        Ok(())
    }

    /// Record the snapshot copy just issued on `stream`, for the next one to
    /// wait on. Nothing with the flag off.
    pub(super) fn finish_leaf_record_copy(&self, stream: u64) -> Result<()> {
        if self.leaf_on() {
            self.record_snapshot_save_dispatch(stream)?;
        }
        Ok(())
    }

    /// Save the state `seq` has on `stream` as its rolling leaf at `at`
    /// tokens (see [`leaf_copies`] for `conv_row`) and register it there. A
    /// failed save, or a checkpoint already at that prefix, leaves the turn
    /// without a leaf until the next boundary.
    fn finish_leaf_save(
        &self,
        seq: &SequenceState,
        at: usize,
        conv_row: Option<usize>,
        stream: u64,
    ) {
        // Below the restore floor a leaf could never be restored.
        if at < crate::model::mtp_carry::marconi_min_tokens() {
            return;
        }
        let Some(snap) = self
            .finish_leaf_take(seq)
            .or_else(|| self.finish_leaf_reserve(seq.session_hash))
        else {
            return;
        };
        if self.finish_leaf_write(seq, snap, at, conv_row, stream) {
            seq.finish_leaf.set(FinishLeaf { snap, tokens: at });
        }
    }

    /// Write the state `seq` has on `stream` into snapshot slot `snap` (h,
    /// conv per [`leaf_copies`], and the aux state of a model that has any)
    /// and register it as a leaf at `at` tokens. `false`: the slot is freed
    /// (a failed save, or a checkpoint already at that prefix).
    fn finish_leaf_write(
        &self,
        seq: &SequenceState,
        snap: usize,
        at: usize,
        conv_row: Option<usize>,
        stream: u64,
    ) -> bool {
        let (h, conv) = leaf_copies(
            &self.ssm_pool,
            &self.ssm_snapshots,
            seq.slot_idx,
            snap,
            conv_row,
        );
        let saved = self
            .finish_leaf_wait_copies(stream)
            .and_then(|()| run_ssm_state_copies(self.gpu.as_ref(), &h, &conv, stream))
            .and_then(|()| self.finish_leaf_record_copy(stream))
            .and_then(|()| self.finish_leaf_aux(seq, snap, stream));
        if let Err(e) = saved {
            tracing::warn!("finish-leaf save at token {at}: {e:#}");
            self.ssm_snapshots.free(snap);
            return false;
        }
        let displaced = self.prefix_cache.insert_leaf_snapshot(
            &seq.tokens[..at],
            snap,
            seq.session_hash,
            seq.adapter_id,
        );
        if let Some(old) = displaced {
            self.ssm_snapshots.free(old);
        }
        tracing::debug!(
            "finish-leaf save: slot {} token {at} snapshot {snap}",
            seq.slot_idx
        );
        displaced != Some(snap)
    }

    /// Attach the aux state `seq` has now (PLE carry, QSA keys) to `snap`.
    /// Nothing for a model without aux state.
    fn finish_leaf_aux(&self, seq: &SequenceState, snap: usize, stream: u64) -> Result<()> {
        if !self.requires_aux_state() {
            return Ok(());
        }
        // The slot's previous blobs, refilled in place.
        let mut aux = self.ssm_snapshots.take_aux(snap);
        self.collect_aux_states_into(seq, stream, &mut aux)?;
        anyhow::ensure!(!aux.is_empty(), "no aux state collected");
        self.ssm_snapshots.set_aux(snap, aux);
        Ok(())
    }

    /// KDA records commit of the first `rows` rows of a `k`-row verify
    /// (`seq.tokens` already rolled back to the accepted prefix, see
    /// [`leaf_save`]), saving the rolling leaf where the step crosses a save
    /// boundary: fold to the boundary, save, fold on.
    pub(super) fn commit_kda_records_with_leaf(
        &self,
        seq: &mut SequenceState,
        rows: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        let rewind = rows < k;
        let save = self
            .finish_leaf_span(seq)
            .and_then(|span| leaf_save(seq.tokens.len(), rows, k, span));
        let Some(save) = save else {
            return self.commit_kda_records(seq, 0..rows, rewind, stream);
        };
        let r = save.rows;
        self.commit_kda_records(seq, 0..r, rewind && r == rows, stream)?;
        self.finish_leaf_save(seq, save.at, save.conv_row, stream);
        if r < rows {
            self.commit_kda_records(seq, r..rows, rewind, stream)?;
        }
        Ok(())
    }

    /// A plain decode step advanced `seq` in place: save the rolling leaf
    /// when it now sits on a save boundary.
    pub(in crate::model) fn finish_leaf_after_decode(&self, seq: &SequenceState) {
        let end = seq.tokens.len();
        if self.qwen4exp_leaf_on() {
            return self.qwen4exp_leaf_after_step(seq, 1, self.gpu.default_stream());
        }
        if !enabled() || seq.finish_leaf.get().is_some_and(|l| l.tokens == end) {
            return;
        }
        if let Some(span) = self.finish_leaf_span(seq)
            && let Some(save) = leaf_save(end, 1, 1, span)
        {
            self.finish_leaf_save(seq, save.at, save.conv_row, self.gpu.default_stream());
        }
    }

    /// A prefill of `tokens` restored at `restored` tokens on `stream`:
    /// settle the leaf it restored from, if it is one. When the prompt runs
    /// at most two blocks past the restore, that is at or above the tail cut
    /// and the prefill saves no checkpoint of its own, so the leaf becomes
    /// the conversation's checkpoint (qwen4_exp's in-pass checkpoint also
    /// needs a 64-token chunk boundary under the cut:
    /// `tail_checkpoint_follows`). Otherwise the leaf is dead history once
    /// the new checkpoint is saved, and says so now so that save can take its
    /// slot. Both ranks run this from the agreed restore depth.
    pub(super) fn finish_leaf_restored(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        restored: usize,
        bs: usize,
        stream: u64,
    ) -> Result<()> {
        if !self.leaf_on() || restored == 0 {
            return Ok(());
        }
        let keep = !self.tail_checkpoint_follows(tokens.len(), restored, bs);
        self.prefix_cache
            .settle_leaf_snapshot(&tokens[..restored], seq.adapter_id, keep);
        // The restore's read is a snapshot copy too ("Stream order").
        self.finish_leaf_record_copy(stream)
    }

    /// Head: have the worker cache its mirror of `seq` (it holds the same
    /// blocks and rolling leaf), so both ranks match and restore alike.
    pub(super) fn finish_leaf_mirror_cache(&self, seq: &SequenceState) {
        let head = self.comm.as_ref().is_some_and(|c| c.rank() == 0);
        if self.leaf_on() && head && seq.slot_idx < self.ssm_pool.max_slots {
            let sent = self.ep_broadcast_seq_and_cmd(
                seq.slot_idx as u32,
                EP_CMD_CACHE_SEQUENCE,
                self.ep_protocol_v2,
            );
            if let Err(e) = sent {
                tracing::error!("finish-leaf: EP broadcast cache-sequence: {e:#}");
            }
        }
    }

    /// Worker: the head's cache command. The head sends it only with the
    /// flag on, so a worker without it runs a different environment: nothing
    /// hangs (the flag adds no collective), but no leaf can be restored.
    pub(in crate::model) fn finish_leaf_cache_command(&self, seq: &SequenceState) {
        if !self.leaf_on() {
            tracing::error!(
                "finish-leaf: cache command from the head, but ATLAS_GLM_PC_FINISH_LEAF / \
                 ATLAS_QWEN4EXP_FINISH_LEAF is off on this rank: the ranks' environments differ"
            );
        }
        self.cache_sequence_dispatch(seq);
    }

    /// `cache_sequence` with the flag on, on either rank: cache the whole
    /// blocks of `seq.tokens`, which brings the rolling leaf into reach of
    /// the next turn, and leave the leaf to the index.
    fn finish_leaf_cache(&self, seq: &SequenceState, bs: usize) {
        let acquired = self.prefix_cache.insert(
            &seq.tokens,
            &seq.block_table,
            &seq.disk_block_ids,
            bs,
            owned_from(seq.prompt_len, bs),
            seq.adapter_id,
        );
        super::super::block_mgmt::cache_acquires_refs(&acquired, &mut self.kv_cache.lock());
        let Some(leaf) = self.finish_leaf_settle_at_finish(seq, bs) else {
            return;
        };
        // Still this sequence's, and a prefix of what it ended as?
        let kept = leaf.tokens <= seq.tokens.len()
            && self
                .prefix_cache
                .snapshot_at(&seq.tokens, leaf.tokens, seq.adapter_id)
                == Some(leaf.snap);
        if kept {
            tracing::info!(
                "finish-leaf: snapshot {} registered at token {} of {}",
                leaf.snap,
                leaf.tokens,
                seq.tokens.len()
            );
        } else {
            tracing::info!("finish-leaf: leaf at token {} gone", leaf.tokens);
        }
    }

    /// A sequence freed without being cached (abort, vision, slid window):
    /// its leaf is out of reach for good, so take it back and free the slot.
    pub(super) fn finish_leaf_release(&self, seq: &SequenceState) {
        let prev = seq.finish_leaf.take_prev();
        let prev = prev.and_then(|l| self.finish_leaf_take_leaf(seq, l));
        for snap in self.finish_leaf_take(seq).into_iter().chain(prev) {
            self.ssm_snapshots.free(snap);
        }
    }
}

#[cfg(test)]
#[path = "finish_leaf_tests.rs"]
mod tests;
