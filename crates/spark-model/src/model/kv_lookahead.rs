// SPDX-License-Identifier: AGPL-3.0-only

//! KV lookahead for the batched verify (`ATLAS_QWEN4EXP_KV_LOOKAHEAD=<tokens>`,
//! default 0 = off; both ranks must agree, `startup_parity`).
//!
//! A decode or verify step votes on its KV admission whenever a sequence
//! crosses into a new block (`kv_admission.rs`): two rooted 4-byte NCCL
//! broadcasts with a host sync between, and the head waits there for the
//! worker to reach the same vote. At C8 x ~3.4 tokens a step that is ~1.7
//! votes a step, and in a real code wave (nsys, rank 0, 2026-10-07) the
//! head spent ~1 ms a step in those broadcasts: each pair is the 24 KV-poison
//! memsets of the new block, then `H2D:4, bcast, bcast(~0.5 ms), D2H:4`.
//!
//! With the switch, a batched verify that has to vote at all reserves every
//! sequence's blocks through `last position + lookahead` and votes ONCE for
//! the whole batch; the next ~lookahead / tokens-per-step steps need no new
//! block and no vote. Every input of the decision (sequence lengths, block
//! table lengths, row counts, the lookahead) is shared by the ranks, so they
//! vote at the same points. A refused lookahead vote (the pool cannot take
//! the extra blocks on some rank) falls back to the per-sequence votes of
//! exactly the step's needs, so the switch never refuses a step the default
//! admits. Block ids only place KV rows; every row is still written before it
//! is read, so tokens are unchanged.

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::kv_cache::PagedKvCache;

use super::kv_admission::{Admission, needs_new_block};
use super::types::TransformerModel;
use crate::traits::SequenceState;

/// `ATLAS_QWEN4EXP_KV_LOOKAHEAD`: tokens past a verify's last row to reserve
/// when it votes (0, unset or unparsable: off).
pub(crate) fn lookahead_tokens() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_QWEN4EXP_KV_LOOKAHEAD")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// Each sequence's block target for a step whose rows end at `last[i]`:
/// `None` when no sequence needs a new block (no vote), else every
/// sequence's last block through `last[i] + lookahead`, the lookahead held
/// to block `cap` (past the context it would only ever be refused).
fn plan(
    last: &[usize],
    needs: &[bool],
    lookahead: usize,
    bs: usize,
    cap: usize,
) -> Option<Vec<usize>> {
    needs.iter().any(|&n| n).then(|| {
        last.iter()
            .map(|&p| ((p + lookahead) / bs).min(cap).max(p / bs))
            .collect()
    })
}

impl TransformerModel {
    /// Reserve the blocks of a batched verify whose sequence `i` writes rows
    /// through position `last[i]` (module docs).
    pub(in crate::model) fn reserve_verify_blocks(
        &self,
        seqs: &mut [&mut SequenceState],
        last: &[usize],
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<()> {
        let bs = kv_cache.block_size();
        let la = lookahead_tokens();
        let needs: Vec<bool> = seqs
            .iter()
            .zip(last)
            .map(|(s, &p)| needs_new_block(s, p / bs))
            .collect();
        let hss = kv_cache.config().cache_blocks_per_seq.is_some();
        // The context's last whole block: its aux reach stays in the indexer.
        let cap = (self.max_blocks_per_seq as usize).saturating_sub(2);
        // qwen4_exp only: GLM's tail slots would release the block being
        // written for a lookahead a refusal then undoes.
        if la > 0
            && !hss
            && self.config.model_type == "qwen4_exp"
            && let Some(targets) = plan(last, &needs, la, bs, cap)
            && !self.reserve_blocks_one_vote(seqs, &targets, kv_cache, stream)?
        {
            tracing::debug!("KV lookahead refused; per-sequence votes");
        }
        // Covered blocks take the no-vote path; the rest vote as before.
        for (seq, &p) in seqs.iter_mut().zip(last) {
            self.reserve_decode_blocks(seq, p / bs, kv_cache, stream)?;
        }
        Ok(())
    }

    /// Reserve every `seqs[i]` through block `targets[i]`, then ONE vote.
    /// Transactional like `admit_with`: unless every rank admits (`false`),
    /// every block this call pushed is released. `Err` only from the vote's
    /// collective itself.
    fn reserve_blocks_one_vote(
        &self,
        seqs: &mut [&mut SequenceState],
        targets: &[usize],
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<bool> {
        let bs = kv_cache.block_size();
        let before: Vec<usize> = seqs.iter().map(|s| s.block_table.len()).collect();
        let mut mine = Admission::Admitted;
        for (seq, &t) in seqs.iter_mut().zip(targets) {
            if !needs_new_block(seq, t) {
                continue;
            }
            let local = super::block_mgmt::ensure_blocks_through_decode(
                seq,
                t,
                kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
                self.levers.kv_poison,
            )
            .and_then(|()| self.reserve_aux_through(seq, (t + 1) * bs, stream));
            mine = mine.min(Admission::of(&local));
            if mine != Admission::Admitted {
                break;
            }
        }
        let agreed = self.admission_vote(mine)?.min(mine);
        if agreed == Admission::Admitted {
            return Ok(true);
        }
        for (seq, &n) in seqs.iter_mut().zip(&before) {
            if seq.block_table.len() > n {
                kv_cache.free_blocks(&seq.block_table.split_off(n));
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::plan;

    #[test]
    fn a_vote_tops_every_sequence_up_by_the_lookahead() {
        // No new block anywhere: no vote.
        assert_eq!(plan(&[10, 40], &[false, false], 64, 16, 99), None);
        // One sequence crosses: both reserve through last + 64.
        assert_eq!(
            plan(&[16, 40], &[true, false], 64, 16, 99),
            Some(vec![5, 6])
        );
        assert_eq!(plan(&[15], &[true], 0, 16, 99), Some(vec![0]));
        // Near the context's end the lookahead stops at the cap, never
        // below the step's own block.
        assert_eq!(plan(&[16, 40], &[true, false], 64, 16, 4), Some(vec![4, 4]));
        assert_eq!(plan(&[90], &[true], 64, 16, 4), Some(vec![5]));
    }
}
