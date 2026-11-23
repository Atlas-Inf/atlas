// SPDX-License-Identifier: AGPL-3.0-only

//! One head pass for a step's Inkling 2 proposes.
//!
//! With two or more live sequences each propose runs alone, and each reads
//! the target's vocabulary head once for its block (926 MB, about 3.8 ms on
//! one GB10): the memory stream, not the rows, sets that cost. A step that
//! verifies few drafts per sequence needs few head rows, so the scheduler
//! brackets its per-sequence proposes as a group:
//!
//! 1. [`BlockDiffusionDraftHead::inkling_group_begin`] plans it: `members`
//!    sequences of `1 + steps` rows each, if they fit [`GROUP_ROWS`].
//! 2. Each propose runs its block as before; its tail, instead of the head
//!    pass, copies its normed rows, its recurrent context and its anchor
//!    token here ([`BlockDiffusionDraftHead::inkling_group_hold`]) and
//!    returns no drafts.
//! 3. [`BlockDiffusionDraftHead::inkling_group_finish`] runs the head once
//!    over every held row, then each member's draft steps, and reads all the
//!    drafts back in one copy.
//!
//! A row's logits come from the same batched GEMV, each row computed as the
//! one-row kernel computes it, so a member's drafts are the ones its own
//! tail would have produced. A propose that captures its graphs runs its
//! tail itself and stays out (`InklingScratch::set_inline_tail`).

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::tail::TailRows;
use super::{FOLD_K, InklingDspark, RANK};
use crate::layers::dflash_head::{BlockDiffusionDraftHead, DflashScratch};

/// Head rows one group pass holds: the batched head GEMV's width.
pub(super) const GROUP_ROWS: usize = 8;
/// Members a group holds (each has an anchor row and at least one draft).
const GROUP_MEMBERS: usize = GROUP_ROWS / 2;

#[derive(Clone, Copy, Default)]
struct Plan {
    /// Draft steps per member.
    steps: usize,
    /// Members the group may hold; 0 is no group.
    room: usize,
    held: usize,
}

/// The group's device buffers and the current step's plan.
pub(super) struct TailGroup {
    /// `[GROUP_ROWS, hidden]` BF16: the members' normed rows, packed.
    head_in: DevicePtr,
    /// `[GROUP_ROWS, vocab]` BF16 base logits.
    logits: DevicePtr,
    /// Per member, the recurrent head's context (`gamma * 3 * RANK` f32).
    ctx: DevicePtr,
    ctx_bytes: usize,
    /// Per member, its anchor token.
    anchors: DevicePtr,
    /// Per member, `gamma` tokens: row 0's argmax, then the drafts.
    tokens: DevicePtr,
    gamma: usize,
    raw: DevicePtr,
    state: DevicePtr,
    latent: DevicePtr,
    bias: DevicePtr,
    plan: parking_lot::Mutex<Plan>,
}

impl TailGroup {
    pub(super) fn alloc(
        gpu: &dyn GpuBackend,
        hidden: usize,
        gamma: usize,
        vocab: usize,
    ) -> Result<Self> {
        let bf16 = 2;
        let a = |bytes: usize| -> Result<DevicePtr> {
            let p = gpu.alloc(bytes).context("Inkling 2 DSpark group scratch")?;
            gpu.copy_h2d(&vec![0u8; bytes], p)?;
            Ok(p)
        };
        let ctx_bytes = gamma * 3 * RANK * 4;
        Ok(Self {
            head_in: a(GROUP_ROWS * hidden * bf16)?,
            logits: a(GROUP_ROWS * vocab * bf16)?,
            ctx: a(GROUP_MEMBERS * ctx_bytes)?,
            ctx_bytes,
            anchors: a(GROUP_MEMBERS * 4)?,
            tokens: a(GROUP_MEMBERS * gamma * 4)?,
            gamma,
            raw: a(3 * RANK * 4)?,
            state: a(RANK * bf16)?,
            latent: a(FOLD_K * bf16)?,
            bias: a(vocab * bf16)?,
            plan: Default::default(),
        })
    }
}

/// The plan for `members` proposes of `steps` drafts each, if one head pass
/// holds them.
fn plan_for(members: usize, steps: usize) -> Option<Plan> {
    (members >= 2 && steps >= 1 && members * (steps + 1) <= GROUP_ROWS).then_some(Plan {
        steps,
        room: members,
        held: 0,
    })
}

impl BlockDiffusionDraftHead {
    /// Plan one head pass for the next `members` proposes of `drafts` drafts
    /// each. `false`: no group, every propose runs its own tail.
    pub(in crate::layers::dflash_head) fn inkling_group_begin(
        &self,
        members: usize,
        drafts: usize,
    ) -> bool {
        let Some(ink) = self.inkling.as_ref() else {
            return false;
        };
        let steps = self.draft_cap(drafts).min(self.gamma.saturating_sub(1));
        let plan = plan_for(members, steps);
        *ink.group.plan.lock() = plan.unwrap_or_default();
        plan.is_some()
    }

    /// Whether a group is waiting for more tails.
    pub(in crate::layers::dflash_head) fn inkling_group_open(&self) -> bool {
        self.inkling.as_ref().is_some_and(|ink| {
            let plan = ink.group.plan.lock();
            plan.held < plan.room
        })
    }

    /// In a tail, after the final norm: hold this sequence's rows for the
    /// group's head pass. `false`: no group has room for it, or the tail must
    /// run here; the caller goes on with the head and the draft steps.
    pub(in crate::layers::dflash_head) fn inkling_group_hold(
        &self,
        ink: &InklingDspark,
        gpu: &dyn GpuBackend,
        normed: DevicePtr,
        scratch: &DflashScratch,
        stream: u64,
    ) -> Result<bool> {
        let sc = scratch
            .inkling
            .as_ref()
            .context("Inkling 2 DSpark lane scratch")?;
        let g = &ink.group;
        let mut plan = g.plan.lock();
        if plan.held >= plan.room || sc.inline_tail.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(false);
        }
        let (m, rows, row_bytes) = (plan.held, plan.steps + 1, self.hidden_size * 2);
        gpu.copy_d2d_async(
            normed,
            g.head_in.offset(m * rows * row_bytes),
            rows * row_bytes,
            stream,
        )?;
        self.inkling_rnn_ctx(
            ink,
            gpu,
            normed,
            sc.f_rnn,
            g.ctx.offset(m * g.ctx_bytes),
            plan.steps,
            stream,
        )?;
        gpu.copy_d2d_async(scratch.markov_prev_dev, g.anchors.offset(m * 4), 4, stream)?;
        plan.held += 1;
        sc.held.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(true)
    }

    /// Close the group: one head pass over the held rows, each member's
    /// draft steps, one readback. The drafts of the held proposes, in the
    /// order they were held (empty when none was).
    pub(in crate::layers::dflash_head) fn inkling_group_finish(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<Vec<u32>>> {
        let Some(ink) = self.inkling.as_ref() else {
            return Ok(Vec::new());
        };
        let g = &ink.group;
        let plan = std::mem::take(&mut *g.plan.lock());
        if plan.held == 0 {
            return Ok(Vec::new());
        }
        let rows = plan.steps + 1;
        ensure!(
            plan.held * rows <= GROUP_ROWS && plan.steps < g.gamma,
            "Inkling 2 DSpark group of {} x {rows} rows exceeds {GROUP_ROWS}",
            plan.held
        );
        self.inkling_head_rows(gpu, g.head_in, g.logits, (plan.held * rows) as u32, stream)?;
        for m in 0..plan.held {
            self.inkling_steps(
                ink,
                gpu,
                &TailRows {
                    logits: g.logits.offset(m * rows * self.vocab_size * 2),
                    ctx: g.ctx.offset(m * g.ctx_bytes),
                    anchor: g.anchors.offset(m * 4),
                    tokens: g.tokens.offset(m * g.gamma * 4),
                    raw: g.raw,
                    state: g.state,
                    latent: g.latent,
                    bias: g.bias,
                },
                plan.steps,
                stream,
            )?;
        }
        let mut host = vec![0u8; plan.held * g.gamma * 4];
        gpu.copy_d2h_on_stream(g.tokens, &mut host, stream)?;
        Ok(host
            .chunks_exact(g.gamma * 4)
            .map(|member| {
                // tokens 1..=steps are the drafts (token 0 is the anchor row's)
                member[4..(plan.steps + 1) * 4]
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .collect())
    }

    /// The head over `rows` normed rows: the NVFP4 head when the drafter has
    /// one, else the BF16 projection, as a propose's own tail picks.
    fn inkling_head_rows(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        logits: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        if self.kernels.w4a16_gemm.0 != 0
            && let Some(nvfp4) = self.lm_head_nvfp4.as_ref()
        {
            let kernel = match rows {
                1..=4 => self.kernels.w4a16_gemv_batch4,
                _ => self.kernels.w4a16_gemv_batch8,
            };
            ensure!(
                kernel.0 != 0,
                "Inkling 2 DSpark group: no batched NVFP4 head"
            );
            return crate::layers::ops::w4a16_gemv_batchm(
                gpu,
                kernel,
                input,
                nvfp4,
                logits,
                rows,
                self.vocab_size as u32,
                self.hidden_size as u32,
                stream,
            );
        }
        self.project_head(gpu, input, logits, rows, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::plan_for;

    #[test]
    fn a_group_is_two_or_more_proposes_that_fit_one_head_pass() {
        // two sequences of up to three drafts, four of one
        for (members, steps) in [(2, 1), (2, 2), (2, 3), (3, 1), (4, 1)] {
            let plan = plan_for(members, steps).expect("fits");
            assert_eq!((plan.steps, plan.room, plan.held), (steps, members, 0));
        }
        // one propose is not a group; nine rows and more do not fit
        for (members, steps) in [(1, 3), (2, 4), (2, 7), (3, 2), (4, 2), (5, 1), (2, 0)] {
            assert!(plan_for(members, steps).is_none(), "{members} x {steps}");
        }
    }
}
