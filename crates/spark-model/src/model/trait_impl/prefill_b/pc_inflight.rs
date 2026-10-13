// SPDX-License-Identifier: AGPL-3.0-only

//! In-flight shared-prefix checkpoints (`ATLAS_GLM_PC_INFLIGHT=1`, default
//! off): the model half. The scheduler half is
//! `spark-server/src/scheduler/shared_prefix.rs`.
//!
//! Agentic clients open several NEW conversations at once with the same long
//! system prompt and tool list. Branch-point checkpoints (`pc_policy`,
//! `ATLAS_GLM_PC_BRANCH`) plant a snapshot where a CACHED path forks, so they
//! see nothing while the sharing requests are all still queued or
//! prefilling: each one recomputes the shared prefix, or replays it from the
//! nearest chunk-boundary checkpoint. With this switch the head's scheduler
//! finds the requests that share a prefix with a prefill in flight, asks that
//! prefill (the leader) to plant a checkpoint at the shared boundary, and
//! holds the others until the leader has computed past it; they then restore
//! there and replay only their own suffix.
//!
//! # The plant and rank lockstep
//!
//! Only the head has the queue, so only the head decides. It sends the
//! leader's sequence [`EP_CMD_PC_PLANT`](crate::traits::EP_CMD_PC_PLANT) and
//! the position through the command stream, before the chunk command it must
//! precede, and the worker records the same `pc_plant_at` when it reads it
//! (`pc_plant_receive`). At every chunk, after the prefix lookup, each rank
//! turns the request into `pc_branch_at` (the branch-checkpoint split and
//! save, unchanged) if [`plant_target`] says it is still ahead. Its inputs
//! are the request, the chunk start, the rank-agreed restore depth
//! (`pc_agree_restore`), the prompt and the model config, identical on both
//! ranks, so both split the same chunk at the same row and save the same
//! checkpoint, with no extra collective. A request that arrives after its
//! position was computed, or at or under the restore depth, or past the tail
//! cut, plants nothing on either rank. A planted request replaces a branch
//! point the chunk-0 fork test chose: it has requests waiting on it.
//!
//! The restore at the planted checkpoint is an ordinary Marconi restore
//! (rank-agreed depth), so a rank whose pool lost the snapshot makes both
//! recompute, as for any other restore.
//!
//! # Exactness
//!
//! The same caveat as branch checkpoints (`pc_policy`, "Accumulation
//! order"): the leader runs `[start, at)` and `[at, end)` where an unplanted
//! prefill runs one pass, and its followers restore at `at` and run `[at,
//! total)` where a cold run of their prompt would run other chunk shapes.
//! Same kernels and math; the accumulation order differs, so outputs can
//! differ bitwise from a cold run and a near-tied greedy argmax can flip, as
//! after any Marconi intermediate restore. The switch is off by default for
//! that reason. The scheduler only plants for requests it sees sharing a
//! prefix, so traffic without one keeps its chunk shapes.
//!
//! # Rank env parity
//!
//! `ATLAS_GLM_PC_INFLIGHT` is read by the head's scheduler only: the worker
//! follows the plant commands whatever its own value, so the switch is not in
//! `startup_parity` (its threshold is `ATLAS_GLM_PC_BRANCH_MIN`, read on the
//! head for the same purpose).

use anyhow::Result;

use super::super::super::types::TransformerModel;
use super::pc_policy::{glm_pc_branch_min_tokens, tail_cut};
use crate::traits::SequenceState;

/// `ATLAS_GLM_PC_INFLIGHT=1`: in-flight shared-prefix checkpoints. Read once.
pub(in crate::model) fn glm_pc_inflight_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PC_INFLIGHT").as_deref() == Ok("1"))
}

/// The checkpoint a chunk starting at `chunk_start` plants for the request
/// `plant`: the position when it is block-aligned, still ahead of the chunk
/// (a position the chunk starts at was already passed: the previous chunk's
/// end saved it or did not), above the restore depth `skip_to`, and below
/// the tail cut of a `total`-token prompt (which already gets a checkpoint).
pub(super) fn plant_target(
    plant: Option<usize>,
    chunk_start: usize,
    skip_to: usize,
    total: usize,
    bs: usize,
) -> Option<usize> {
    plant.filter(|&at| {
        at > 0
            && at.is_multiple_of(bs)
            && at > chunk_start
            && at > skip_to
            && at < tail_cut(total, bs)
    })
}

impl TransformerModel {
    /// Whether prefix checkpoints can be planted for this model at all.
    fn pc_plants_possible(&self) -> bool {
        self.config.num_ssm_layers() > 0
            && self.ssm_snapshots.is_enabled()
            && self.prefix_cache.is_active()
    }

    /// [`crate::traits::Model::pc_inflight_min_tokens`].
    pub(in crate::model) fn pc_inflight_min_dispatch(&self) -> Option<usize> {
        (glm_pc_inflight_enabled() && self.pc_plants_possible()).then(glm_pc_branch_min_tokens)
    }

    /// Worker side of `EP_CMD_PC_PLANT`: record the head's request.
    pub(in crate::model) fn pc_plant_receive(&self, seq: &mut SequenceState) -> Result<()> {
        let at = self.ep_broadcast_u32(0)? as usize;
        seq.pc_plant_at = Some(at);
        Ok(())
    }

    /// Turn a pending plant into this chunk's branch checkpoint (after the
    /// prefix lookup, so `marconi_skip_to` is the agreed restore depth).
    /// Every input is the same on every rank.
    pub(super) fn pc_apply_plant(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        bs: usize,
    ) {
        let target = plant_target(
            seq.pc_plant_at,
            chunk_start,
            seq.marconi_skip_to,
            tokens.len(),
            bs,
        );
        let Some(at) = target else {
            return;
        };
        if seq.pc_branch_at == Some(at)
            || !self.pc_plants_possible()
            || self.tokens_have_vision_pad(tokens)
        {
            return;
        }
        seq.pc_branch_at = Some(at);
        tracing::info!(
            "pc in-flight checkpoint planted at token {at} (chunk at {chunk_start}, \
             restore depth {}, {}-token prompt)",
            seq.marconi_skip_to,
            tokens.len(),
        );
    }
}

#[cfg(test)]
#[path = "pc_inflight_tests.rs"]
mod tests;
