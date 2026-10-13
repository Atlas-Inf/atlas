// SPDX-License-Identifier: AGPL-3.0-only

//! KV admission for a prefill chunk or a decode/verify step: every rank
//! reserves the step's blocks (and the per-sequence state that grows with
//! them, `TransformerLayer::reserve_aux`) and all ranks agree on the outcome
//! BEFORE its forward pass issues a collective.
//!
//! The head cannot decide alone: each rank evicts from its own prefix cache
//! (rank-local, and it diverges — see the F83 note in `prefix_lookup`), so
//! free-plus-evictable capacity differs per rank. Each rank therefore tries
//! its own reservation and one min-vote over `Admission` decides for all.
//! On refusal every rank undoes its reservation and returns
//! [`KvAdmissionRefused`] from the same point, having mutated no sequence or
//! GPU state for the chunk, so the worker simply waits for its next command
//! while the head either frees a victim and re-sends the chunk (exhaustion)
//! or fails just this request (any other reservation error), releasing the
//! slot (`spark-server` `prefill_preempt`).
//!
//! Decode and verify steps vote only when the step needs a NEW block. Whether
//! it does is a function of the sequence length and the block-table length,
//! which every rank shares, so all ranks vote at the same points: one
//! collective per new block, not per step. An exhausted pool then preempts or
//! fails a request on the head (`spark-server` `preempt`) instead of killing
//! the worker, which used to exit on its own allocation failure and take the
//! pair down with it.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::PrefixCache;

use super::block_mgmt::{ensure_blocks_through_decode, ensure_blocks_through_prefill};
use super::types::TransformerModel;
use crate::traits::SequenceState;

#[cfg(test)]
#[path = "kv_admission_tests.rs"]
mod tests;

/// Every rank refused a prefill chunk's or a decode step's KV reservation
/// before its forward pass and rolled it back; `by_peer` when a peer's
/// outcome, worse than this rank's, decided it.
#[derive(Debug)]
pub struct KvAdmissionRefused {
    pub by_peer: bool,
    /// Out of KV blocks, so freeing a victim can make a retry fit (any
    /// other reservation failure would only repeat).
    pub retryable: bool,
    /// A decode/verify step's blocks rather than a prefill chunk's.
    pub decode: bool,
}

/// Shared by the message and [`is_retryable_refusal_text`].
const REFUSED_BY: &str = "refused by";

/// Shared by [`PartialBatchStep`]'s message and [`is_retryable_refusal_text`].
const ALREADY_RAN: &str = "already ran this step";

impl std::fmt::Display for KvAdmissionRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let by = if self.by_peer {
            "a peer rank"
        } else {
            "this rank"
        };
        let what = if self.decode {
            "decode step"
        } else {
            "prefill chunk"
        };
        if self.retryable {
            write!(
                f,
                "KV cache exhausted: {what} {REFUSED_BY} {by} before its forward pass"
            )
        } else {
            write!(
                f,
                "{what} KV reservation failed on {by} before its forward pass \
                 (rolled back on every rank, not retried)"
            )
        }
    }
}

impl std::error::Error for KvAdmissionRefused {}

/// Context on an error from a per-sequence decode loop (`decode_batch`'s
/// highway and MLA fallbacks) raised after `advanced` of its `n` sequences
/// ran their step: a retry of the batch would feed those sequences the same
/// token again, so nothing retries it, whatever it wraps. An agreed refusal
/// underneath still stopped every rank at the same sequence, so the pair
/// keeps serving and only these requests fail.
#[derive(Debug)]
pub struct PartialBatchStep {
    pub advanced: usize,
    pub n: usize,
}

impl std::fmt::Display for PartialBatchStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} of {} sequences {ALREADY_RAN}; the batch is not retried",
            self.advanced, self.n
        )
    }
}

/// `e` from sequence `advanced` of an `n`-sequence per-sequence loop: marked
/// [`PartialBatchStep`] once an earlier sequence has run, unchanged before.
pub fn after_partial_step(e: anyhow::Error, advanced: usize, n: usize) -> anyhow::Error {
    if advanced == 0 {
        e
    } else {
        e.context(PartialBatchStep { advanced, n })
    }
}

/// Whether `e` is marked [`PartialBatchStep`].
pub fn is_partial_batch_step(e: &anyhow::Error) -> bool {
    e.downcast_ref::<PartialBatchStep>().is_some()
}

/// The agreed, rolled-back refusal `e` carries, if any.
pub fn kv_admission_refusal(e: &anyhow::Error) -> Option<&KvAdmissionRefused> {
    e.downcast_ref()
}

/// Whether an error already rendered to text (`format!("{e:#}")`, which is
/// how the scheduler records an engine error) was an agreed, RETRYABLE
/// refusal: every rank rolled the step back, so the sequence is intact and
/// freeing a victim can make a retry fit. Not after a [`PartialBatchStep`].
pub fn is_retryable_refusal_text(text: &str) -> bool {
    text.contains("KV cache exhausted: ")
        && text.contains(REFUSED_BY)
        && !text.contains(ALREADY_RAN)
}

/// One rank's reservation outcome, ordered so that the all-rank minimum is
/// the verdict: the worst outcome on any rank decides for every rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Admission {
    Failed = 0,
    Exhausted = 1,
    Admitted = 2,
}

impl Admission {
    fn of(local: &Result<()>) -> Self {
        match local {
            Ok(()) => Self::Admitted,
            Err(e) if format!("{e:#}").contains("KV cache exhausted") => Self::Exhausted,
            Err(_) => Self::Failed,
        }
    }

    fn from_word(word: u32) -> Self {
        match word {
            0 => Self::Failed,
            1 => Self::Exhausted,
            _ => Self::Admitted,
        }
    }
}

/// EP worker: an agreed refusal is not fatal. The head re-sends the chunk
/// after preempting, or releases the slot (`0xFFFFFFF1`); keep serving.
pub(super) fn worker_step_outcome(step: Result<bool>) -> Result<bool> {
    match step {
        Err(e) if kv_admission_refusal(&e).is_some() => {
            tracing::warn!("EP worker: {e:#}; waiting for the head's retry or release");
            Ok(true)
        }
        step => step,
    }
}

/// EP worker: the head sends a chunk from its own recorded progress (a retry
/// resumes past a completed tail-split half), so this rank must stand there
/// too; anything else would desync the chunk's collectives.
pub(super) fn check_worker_chunk_start(seq: &SequenceState, chunk_start: usize) -> Result<()> {
    anyhow::ensure!(
        seq.seq_len == chunk_start,
        "EP worker: prefill chunk starts at {chunk_start} but slot {} is at {} \
         (head/worker prefill progress diverged)",
        seq.slot_idx,
        seq.seq_len
    );
    Ok(())
}

/// Reserve `seq`'s prefill blocks through `abs_block_idx`; `vote` turns this
/// rank's outcome into the all-rank verdict. Transactional: unless every
/// rank admits, this call's blocks (and HSS disk ids) are released again. A
/// local error other than exhaustion stays in the refusal's chain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn admit(
    seq: &mut SequenceState,
    abs_block_idx: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
    kv_poison: bool,
    vote: impl FnOnce(Admission) -> Result<Admission>,
) -> Result<()> {
    admit_with(seq, kv_cache, false, vote, |seq, kv| {
        ensure_blocks_through_prefill(seq, abs_block_idx, kv, prefix_cache, gpu, stream, kv_poison)
    })
}

/// [`admit`] with the reservation supplied: `reserve` grows `seq` (blocks
/// first, then whatever rides with them) and may leave blocks pushed when it
/// fails; they are released unless every rank admits.
pub(crate) fn admit_with(
    seq: &mut SequenceState,
    kv_cache: &mut PagedKvCache,
    decode: bool,
    vote: impl FnOnce(Admission) -> Result<Admission>,
    reserve: impl FnOnce(&mut SequenceState, &mut PagedKvCache) -> Result<()>,
) -> Result<()> {
    let (blocks, disk_ids) = (seq.block_table.len(), seq.disk_block_ids.len());
    let local = reserve(seq, kv_cache);
    let mine = Admission::of(&local);
    let agreed = vote(mine)?.min(mine);
    if agreed == Admission::Admitted {
        return Ok(());
    }
    kv_cache.free_blocks(&seq.block_table.split_off(blocks));
    let ids = seq
        .disk_block_ids
        .split_off(disk_ids.min(seq.disk_block_ids.len()));
    if !ids.is_empty() {
        let _ = spark_storage::with_local(|hss| {
            for &id in &ids {
                hss.dec_disk_ref(id);
            }
            Ok(())
        });
    }
    let refused = KvAdmissionRefused {
        by_peer: agreed < mine,
        retryable: agreed == Admission::Exhausted,
        decode,
    };
    Err(match local {
        Err(e) if mine == Admission::Failed => e.context(refused),
        _ => refused.into(),
    })
}

/// Whether reserving through `abs_block_idx` allocates: the block lies past
/// the sequence's window. A function of state every rank shares.
pub(crate) fn needs_new_block(seq: &SequenceState, abs_block_idx: usize) -> bool {
    let bt_len = seq.block_table.len();
    !(bt_len > 0 && abs_block_idx < seq.hss_window_start() + bt_len)
}

impl TransformerModel {
    /// [`admit`] the blocks for positions `..end_pos` of this chunk. A
    /// multi-rank world always votes, so every rank issues the same
    /// collectives whether or not it had to allocate.
    pub(in crate::model) fn reserve_prefill_blocks(
        &self,
        seq: &mut SequenceState,
        end_pos: usize,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<()> {
        let bs = kv_cache.block_size();
        let last_block = (end_pos - 1) / bs;
        admit_with(
            seq,
            kv_cache,
            false,
            |mine| self.admission_vote(mine),
            |seq, kv| {
                ensure_blocks_through_prefill(
                    seq,
                    last_block,
                    kv,
                    self.prefix_cache.as_ref(),
                    self.gpu.as_ref(),
                    stream,
                    self.levers.kv_poison,
                )?;
                self.reserve_aux_through(seq, (last_block + 1) * bs, stream)
            },
        )
    }

    /// Make `seq` hold logical block `abs_block_idx` for a decode or verify
    /// step, before its forward pass; every decode and verify path allocates
    /// through here. A step that needs no new block only checks the write
    /// window, exactly as before. One that does reserves the block and the
    /// state riding with it and, on a multi-rank model, votes (module doc).
    pub(in crate::model) fn reserve_decode_blocks(
        &self,
        seq: &mut SequenceState,
        abs_block_idx: usize,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<()> {
        let ensure = |seq: &mut SequenceState, kv: &mut PagedKvCache| {
            ensure_blocks_through_decode(
                seq,
                abs_block_idx,
                kv,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
                self.levers.kv_poison,
            )
        };
        // The HSS sliding window frees blocks as it grows, which a refusal
        // could not undo; it keeps the unvoted path.
        if !needs_new_block(seq, abs_block_idx) || kv_cache.config().cache_blocks_per_seq.is_some()
        {
            return ensure(seq, kv_cache);
        }
        let bs = kv_cache.block_size();
        admit_with(
            seq,
            kv_cache,
            true,
            |mine| self.admission_vote(mine),
            |seq, kv| {
                ensure(seq, kv)?;
                self.reserve_aux_through(seq, (abs_block_idx + 1) * bs, stream)
            },
        )
    }

    /// The all-rank minimum of `mine` (this rank's own off a pair).
    fn admission_vote(&self, mine: Admission) -> Result<Admission> {
        if self.multi_rank_protocol_active() {
            Ok(Admission::from_word(self.ep_min_u32(mine as u32)?))
        } else {
            Ok(mine)
        }
    }

    /// Grow every layer's per-sequence aux state to `tokens` positions.
    /// Failing to is running out of memory for those tokens, so it reports
    /// as exhaustion (a retryable refusal).
    fn reserve_aux_through(
        &self,
        seq: &mut SequenceState,
        tokens: usize,
        stream: u64,
    ) -> Result<()> {
        for (layer, st) in self.layers.iter().zip(seq.layer_states.iter_mut()) {
            layer
                .reserve_aux(st.as_mut(), tokens, self.gpu.as_ref(), stream)
                .map_err(|e| {
                    e.context(format!(
                        "KV cache exhausted: per-sequence indexer state could not grow \
                         to {tokens} tokens"
                    ))
                })?;
        }
        Ok(())
    }
}
