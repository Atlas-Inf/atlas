// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence fallback for the batched DFlash proposer: once any drafter
//! state has been advanced, an error must degrade THAT sequence to empty
//! drafts rather than propagate — the scheduler-level fallback would
//! re-prepare every sequence and double-advance the lifecycle. Split out of
//! `batch_propose.rs` for the file-size cap.

use anyhow::Result;

use spark_runtime::gpu::DevicePtr;

use super::batch_plan::{BatchSeqPlan, plan_prepared_fallback};
use super::{BlockDiffusionDraftHead, DflashProposerState, DflashScratch, SequenceGeneration};

impl BlockDiffusionDraftHead {
    /// Per-sequence fallback once `prepared` of `n` sequences have had their
    /// drafter state advanced: prepared sequences run `forward_prepared`
    /// (no re-prepare — that would double-advance the lifecycle), the
    /// `failed_at` index gets empty drafts (its prepare may be half-advanced),
    /// and the rest run the full serial `propose_drafts`.
    ///
    /// TOTAL: a per-sequence failure (owner validation, `forward_prepared`,
    /// or the serial `propose_drafts`) degrades THAT sequence to empty
    /// drafts with one WARN — propagating Err here would bounce the whole
    /// batch back to the scheduler's fallback, which re-prepares every
    /// sequence and double-advances the lifecycle this plan exists to
    /// prevent. Only the state downcast stays a hard Err: a mismatch there
    /// is a programmer error, not a runtime condition.
    ///
    /// NOTE: `Skip` / empty drafts do NOT roll back a half-advanced
    /// prepare — a sequence that died after `lifecycle.advance` + ctx-slot
    /// append keeps that state. The serial path shares the same
    /// advance → append → alloc ordering, so this hazard is pre-existing,
    /// not introduced by the batched seam.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepared_fallback_batch(
        &self,
        n: usize,
        prepared: usize,
        failed_at: Option<usize>,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn crate::speculative::ProposerState],
        expected_owners: &[SequenceGeneration],
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<Vec<Vec<u32>>> {
        let plans = plan_prepared_fallback(n, prepared, failed_at);
        let (_, scratch, markov_embed, markov_bias) = self.lane(0, ctx.gpu.default_stream());
        let mut out = Vec::with_capacity(n);
        for (i, plan) in plans.iter().enumerate() {
            match plan {
                BatchSeqPlan::Skip => out.push(Vec::new()),
                BatchSeqPlan::ForwardPrepared => {
                    // A downcast failure means the batch was built from
                    // non-DFlash proposer state — a programmer error, so it
                    // stays a hard Err.
                    let dstate = states[i]
                        .as_any_mut()
                        .downcast_mut::<DflashProposerState>()
                        .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
                    match self.fallback_forward_prepared(
                        i,
                        dstate,
                        scratch,
                        markov_embed,
                        markov_bias,
                        last_tokens,
                        positions,
                        num_drafts,
                        expected_owners,
                        ctx,
                        stream,
                    ) {
                        Ok(drafts) => out.push(drafts),
                        Err(e) => {
                            tracing::warn!(
                                "DFlash per-sequence fallback: sequence {i}/{n} prepared-forward \
                                 failed; empty drafts this step: {e:#}"
                            );
                            out.push(Vec::new());
                        }
                    }
                }
                BatchSeqPlan::SerialPropose => {
                    match self.propose_drafts(
                        last_tokens[i],
                        target_hiddens[i],
                        positions[i],
                        num_drafts,
                        states[i],
                        Some(expected_owners[i]),
                        ctx,
                        stream,
                        None,
                        None,
                        Some(target_hiddens[i]),
                    ) {
                        Ok(drafts) => out.push(drafts),
                        Err(e) => {
                            tracing::warn!(
                                "DFlash per-sequence fallback: sequence {i}/{n} serial propose \
                                 failed; empty drafts this step: {e:#}"
                            );
                            out.push(Vec::new());
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// The `ForwardPrepared` arm of [`Self::prepared_fallback_batch`],
    /// factored so its `?`s collapse into the caller's per-sequence
    /// warn-and-empty mapping. Validates the owner and rebuilds the same
    /// `option_b_arg` the serial setup computes (paged block table +
    /// effective ctx count, zeroed by the ablation).
    #[allow(clippy::too_many_arguments)]
    fn fallback_forward_prepared(
        &self,
        i: usize,
        dstate: &mut DflashProposerState,
        scratch: &DflashScratch,
        markov_embed: DevicePtr,
        markov_bias: DevicePtr,
        last_tokens: &[u32],
        positions: &[usize],
        num_drafts: usize,
        expected_owners: &[SequenceGeneration],
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let owner = self.validate_dflash_owner(dstate, Some(expected_owners[i]))?;
        // ctx_count_drafter was set by the prepare pass.
        let option_b_arg = if self.startup.option_b_enabled {
            let effective_ctx = if self.startup.option_b_no_ctx {
                0
            } else {
                dstate.ctx_count_drafter as u32
            };
            Some((
                dstate.block_table_dev.ok_or_else(|| {
                    anyhow::anyhow!("DFlash prepared fallback: state has no block table")
                })?,
                effective_ctx,
            ))
        } else {
            None
        };
        self.forward_prepared(
            scratch,
            markov_embed,
            markov_bias,
            0,
            last_tokens[i],
            positions[i],
            num_drafts,
            dstate,
            owner,
            option_b_arg,
            false,
            ctx,
            stream,
        )
    }
}
