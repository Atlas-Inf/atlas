// SPDX-License-Identifier: AGPL-3.0-only

//! The Inkling 2 DSpark tail: the recurrent Markov head over a block's
//! base logits (see the parent module).

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::{FOLD_K, InklingDspark, RANK};
use crate::layers::dflash_head::{BlockDiffusionDraftHead, DflashScratch};
use crate::layers::ops;

impl BlockDiffusionDraftHead {
    /// The draft tokens of rows 1..γ from the block's logits (rows 1..γ of
    /// `logits` hold the base logits of the drafter's normed rows `normed`):
    /// per step, the recurrence from the previous token, its fold over the
    /// head added to that row's logits, then the argmax. Row 0 (the anchor)
    /// is argmaxed as is; it is not a draft.
    pub(in crate::layers::dflash_head) fn inkling_tail(
        &self,
        ink: &InklingDspark,
        gpu: &dyn GpuBackend,
        normed: DevicePtr,
        scratch: &DflashScratch,
        stream: u64,
    ) -> Result<()> {
        let sc = scratch
            .inkling
            .as_ref()
            .context("Inkling 2 DSpark lane scratch")?;
        let steps = sc
            .short_tail()
            .unwrap_or(self.gamma - 1)
            .min(self.gamma - 1);
        self.inkling_rnn_ctx(ink, gpu, normed, sc.f_rnn, sc.ctx, steps, stream)?;
        self.inkling_steps(
            ink,
            gpu,
            &TailRows {
                logits: scratch.logits,
                ctx: sc.ctx,
                anchor: scratch.markov_prev_dev,
                tokens: scratch.draft_tokens_dev,
                // after the block's `gamma` tokens (`draft_conf::record_bytes`)
                conf: ink
                    .conf
                    .then(|| scratch.draft_tokens_dev.offset(self.gamma * 4)),
                raw: sc.raw,
                state: sc.state,
                latent: sc.latent,
                bias: sc.bias,
            },
            steps,
            stream,
        )
    }

    /// The recurrent head's context for `steps` draft steps of `normed`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn inkling_rnn_ctx(
        &self,
        ink: &InklingDspark,
        gpu: &dyn GpuBackend,
        normed: DevicePtr,
        f_rnn: DevicePtr,
        ctx: DevicePtr,
        steps: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, ink.k.rnn_ctx)
            .grid([(3 * RANK / 8) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ink.w.rnn_proj.weight)
            .arg_ptr(ink.w.rnn_bias)
            .arg_ptr(normed)
            .arg_ptr(f_rnn)
            .arg_ptr(ctx)
            .arg_u32(steps as u32)
            .arg_u32(self.hidden_size as u32)
            .launch(stream)
    }

    /// Row 0's argmax, then `steps` draft steps over `rows` (see
    /// [`Self::inkling_tail`]).
    pub(super) fn inkling_steps(
        &self,
        ink: &InklingDspark,
        gpu: &dyn GpuBackend,
        rows: &TailRows,
        steps: usize,
        stream: u64,
    ) -> Result<()> {
        let (h, v) = (self.hidden_size as u32, self.vocab_size);
        let bf16 = 2usize;
        let row = |i: usize| rows.logits.offset(i * v * bf16);
        ops::argmax_bf16(
            gpu,
            self.kernels.argmax,
            row(0),
            rows.tokens,
            v as u32,
            stream,
        )?;
        gpu.memset_async(rows.state, 0, RANK * bf16, stream)?;
        for i in 0..steps {
            // step i drafts row i + 1 from the token before it (the anchor first)
            let prev = if i == 0 {
                rows.anchor
            } else {
                rows.tokens.offset(i * 4)
            };
            KernelLaunch::new(gpu, ink.k.rnn_raw)
                .grid([(3 * RANK / 8) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(ink.w.rnn_proj.weight)
                .arg_ptr(rows.ctx.offset(i * 3 * RANK * 4))
                .arg_ptr(rows.state)
                .arg_ptr(ink.w.markov_w1.weight)
                .arg_ptr(prev)
                .arg_ptr(rows.raw)
                .arg_u32(h)
                .launch(stream)?;
            KernelLaunch::new(gpu, ink.k.rnn_gate)
                .grid([1, 1, 1])
                .block([RANK as u32, 1, 1])
                .arg_ptr(rows.raw)
                .arg_ptr(rows.state)
                .arg_ptr(rows.latent)
                .arg_u32(FOLD_K as u32)
                .launch(stream)?;
            ops::dense_gemv(
                gpu,
                self.kernels.dense_gemv,
                rows.latent,
                &ink.fold,
                rows.bias,
                v as u32,
                FOLD_K as u32,
                stream,
            )?;
            ops::residual_add(
                gpu,
                self.kernels.residual_add,
                row(i + 1),
                rows.bias,
                v as u32,
                stream,
            )?;
            let token = rows.tokens.offset((i + 1) * 4);
            match rows.conf {
                // the pick and the drafter's confidence in it, word `i + 1`
                Some(conf) => KernelLaunch::new(gpu, ink.k.pick)
                    .grid([1, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(row(i + 1))
                    .arg_ptr(token)
                    .arg_ptr(conf.offset((i + 1) * 4))
                    .arg_u32(v as u32)
                    .arg_f32(super::LOGIT_SOFTCAP)
                    .launch(stream)?,
                None => ops::argmax_bf16(
                    gpu,
                    self.kernels.argmax,
                    row(i + 1),
                    token,
                    v as u32,
                    stream,
                )?,
            }
        }
        Ok(())
    }
}

/// Where one sequence's tail reads and writes.
pub(super) struct TailRows {
    /// `[1 + steps, vocab]` BF16 base logits, row 0 the anchor's.
    pub(super) logits: DevicePtr,
    /// The recurrent head's context ([`BlockDiffusionDraftHead::inkling_rnn_ctx`]).
    pub(super) ctx: DevicePtr,
    /// The anchor token (one `u32`).
    pub(super) anchor: DevicePtr,
    /// `[1 + steps]` `u32` out: row 0's argmax, then the drafts.
    pub(super) tokens: DevicePtr,
    /// `[1 + steps]` `f32` out, when confidences are reported: word `r` is
    /// row `r`'s (word 0 is not written; the anchor row is not a draft).
    pub(super) conf: Option<DevicePtr>,
    pub(super) raw: DevicePtr,
    pub(super) state: DevicePtr,
    pub(super) latent: DevicePtr,
    pub(super) bias: DevicePtr,
}
