// SPDX-License-Identifier: AGPL-3.0-only

//! KV admission for a prefill chunk: every rank reserves the chunk's blocks
//! and all ranks agree on the outcome BEFORE its forward pass issues a
//! collective.
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

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::PrefixCache;

use super::block_mgmt::ensure_blocks_through_prefill;
use super::types::TransformerModel;
use crate::traits::SequenceState;

#[cfg(test)]
#[path = "kv_admission_tests.rs"]
mod tests;

/// Every rank refused a prefill chunk's KV reservation before its forward
/// pass and rolled it back; `by_peer` when a peer's outcome, worse than this
/// rank's, decided it.
#[derive(Debug)]
pub struct KvAdmissionRefused {
    pub by_peer: bool,
    /// Out of KV blocks, so freeing a victim can make a retry fit (any
    /// other reservation failure would only repeat).
    pub retryable: bool,
}

impl std::fmt::Display for KvAdmissionRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let by = if self.by_peer {
            "a peer rank"
        } else {
            "this rank"
        };
        if self.retryable {
            write!(
                f,
                "KV cache exhausted: prefill chunk refused by {by} before its forward pass"
            )
        } else {
            write!(
                f,
                "prefill chunk KV reservation failed on {by} before its forward pass \
                 (rolled back on every rank, not retried)"
            )
        }
    }
}

impl std::error::Error for KvAdmissionRefused {}

/// The agreed, rolled-back refusal `e` carries, if any.
pub fn kv_admission_refusal(e: &anyhow::Error) -> Option<&KvAdmissionRefused> {
    e.downcast_ref()
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

/// Reserve `seq`'s blocks through `abs_block_idx`; `vote` turns this rank's
/// outcome into the all-rank verdict. Transactional: unless every rank
/// admits, this call's blocks (and HSS disk ids) are released again. A
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
    let (blocks, disk_ids) = (seq.block_table.len(), seq.disk_block_ids.len());
    let local = ensure_blocks_through_prefill(
        seq,
        abs_block_idx,
        kv_cache,
        prefix_cache,
        gpu,
        stream,
        kv_poison,
    );
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
    };
    Err(match local {
        Err(e) if mine == Admission::Failed => e.context(refused),
        _ => refused.into(),
    })
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
        let last_block = (end_pos - 1) / kv_cache.block_size();
        admit(
            seq,
            last_block,
            kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
            self.levers.kv_poison,
            |mine| {
                if self.multi_rank_protocol_active() {
                    Ok(Admission::from_word(self.ep_min_u32(mine as u32)?))
                } else {
                    Ok(mine)
                }
            },
        )
    }
}
