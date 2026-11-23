// SPDX-License-Identifier: AGPL-3.0-only

//! Inkling 2's DSpark over a staged batch of sequences (`batch_propose.rs`).
//!
//! The batched propose runs the drafter's layers and the vocabulary head over
//! every sequence's block at once (`batch_query_embed` -> `batch_norm` ->
//! `batch_logits`). Inkling adds a step before them and replaces the tail:
//!
//! - before the layers, each sequence's block input (its anchor's embedding,
//!   then `mask_embed`) and its recurrent-head context row `f_rnn`, as
//!   [`BlockDiffusionDraftHead::inkling_prepare`] does for one sequence;
//! - after the head, the recurrent Markov tail for every sequence at once:
//!   one launch a kernel a draft step, and the fold over the vocabulary head
//!   (105 MB) read once a step for up to eight sequences instead of once a
//!   sequence. Each sequence's tokens and confidences land in its record of
//!   `batch_tokens`, in the layout the serial tail writes.
//!
//! The tail computes every row as the serial tail does (the fold's batched
//! GEMV is row-exact with the one-row GEMV, the pick adds the bias as
//! `bf16_residual_add` does), so given the same base logits the drafts are
//! the serial drafts. The layers and head over more rows can round
//! differently; the target verifies every draft either way.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::{FOLD_K, Pick, RANK, pick};
use crate::layers::dflash_head::BlockDiffusionDraftHead;
use crate::layers::ops;
use crate::weight_loader::dflash_loader::INKLING2_DSPARK_TAPS;

/// Rows one `dense_gemv_bf16_batchm` launch takes (its MAX_M).
const FOLD_ROWS: usize = 8;

/// The batch tail's per-sequence buffers, for up to `capacity` sequences.
pub(super) struct BatchScratch {
    capacity: usize,
    /// `[capacity, H]` BF16: each sequence's recurrent-head context row.
    f_rnn: DevicePtr,
    /// `[capacity, gamma, 3R]` f32: each sequence's per-step gate context.
    ctx: DevicePtr,
    /// `[capacity, 3R]` f32
    raw: DevicePtr,
    /// `[capacity, R]` BF16
    state: DevicePtr,
    /// `[capacity, FOLD_K]` BF16
    latent: DevicePtr,
    /// `[capacity, vocab]` BF16: each sequence's Markov bias this step.
    bias: DevicePtr,
}

impl BatchScratch {
    pub(super) fn alloc(
        gpu: &dyn GpuBackend,
        capacity: usize,
        hidden: usize,
        gamma: usize,
        vocab: usize,
    ) -> Result<Self> {
        let a = |bytes: usize| -> Result<DevicePtr> {
            let p = gpu.alloc(bytes)?;
            gpu.copy_h2d(&vec![0u8; bytes], p)?;
            Ok(p)
        };
        let (bf16, f32b) = (2, 4);
        Ok(Self {
            capacity,
            f_rnn: a(capacity * hidden * bf16)?,
            ctx: a(capacity * gamma * 3 * RANK * f32b)?,
            raw: a(capacity * 3 * RANK * f32b)?,
            state: a(capacity * RANK * bf16)?,
            latent: a(capacity * FOLD_K * bf16)?,
            bias: a(capacity * vocab * bf16)?,
        })
    }
}

/// The fold launches for `n` latent rows: `(first row, rows)` of at most
/// [`FOLD_ROWS`] each.
fn fold_waves(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n)
        .step_by(FOLD_ROWS)
        .map(move |first| (first, (n - first).min(FOLD_ROWS)))
}

impl BlockDiffusionDraftHead {
    /// Before the staged layers: each sequence's block input over its gamma
    /// rows of `batch_query_embed` (row 0 the anchor's embedding, the rest
    /// `mask_embed`) and its `f_rnn` from the newest of its `accs` rows (zero
    /// without context).
    pub(in crate::layers::dflash_head) fn inkling_batch_prepare(
        &self,
        gpu: &dyn GpuBackend,
        accs: &[Option<(DevicePtr, usize)>],
        stream: u64,
    ) -> Result<()> {
        let Some(ink) = self.inkling.as_ref() else {
            return Ok(());
        };
        let b = &ink.batch;
        ensure!(
            !accs.is_empty() && accs.len() <= b.capacity,
            "Inkling 2 DSpark batch of {} sequences, capacity {}",
            accs.len(),
            b.capacity
        );
        let h = self.hidden_size;
        KernelLaunch::new(gpu, ink.k.block_input)
            .grid([self.gamma as u32, accs.len() as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(self.batch_query_embed)
            .arg_ptr(ink.w.target_embed_norm)
            .arg_ptr(ink.w.mask_embed)
            .arg_f32(ink.w.target_eps)
            .arg_u32(h as u32)
            .launch(stream)?;
        let row_bytes = INKLING2_DSPARK_TAPS * h * 2;
        for (s, acc) in accs.iter().enumerate() {
            let out = b.f_rnn.offset(s * h * 2);
            match *acc {
                Some((base, rows)) if rows > 0 => ink.mix(
                    gpu,
                    base.offset((rows - 1) * row_bytes),
                    1,
                    h as u32,
                    self.num_layers,
                    ink.w.markov_norm.weight,
                    self.rms_norm_eps,
                    out,
                    stream,
                )?,
                _ => gpu.memset_async(out, 0, h * 2, stream)?,
            }
        }
        Ok(())
    }

    /// After the staged head: the Markov tail of `n` sequences, `steps` drafts
    /// each, over `batch_norm` and `batch_logits`. Sequence `s`'s record in
    /// `batch_tokens` gets row 0's pick in word 0 and its drafts in words
    /// 1..=steps, and with confidences reported those in words gamma + 1..
    pub(in crate::layers::dflash_head) fn inkling_batch_tail(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        steps: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(ink) = self.inkling.as_ref() else {
            return Ok(());
        };
        let b = &ink.batch;
        ensure!(
            n >= 1 && n <= b.capacity && steps < self.gamma,
            "Inkling 2 DSpark batch tail of {n} x {steps} (capacity {}, gamma {})",
            b.capacity,
            self.gamma
        );
        let (h, v, g) = (self.hidden_size, self.vocab_size, self.gamma);
        let outs = 3 * RANK;
        let record = (self.draft_record_bytes() / 4) as u32;
        let tokens = self.batch_tokens;
        KernelLaunch::new(gpu, ink.k.rnn_ctx)
            .grid([(outs / 8) as u32, n as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(ink.w.rnn_proj.weight)
            .arg_ptr(ink.w.rnn_bias)
            .arg_ptr(self.batch_norm)
            .arg_ptr(b.f_rnn)
            .arg_ptr(b.ctx)
            .arg_u32(steps as u32)
            .arg_u32(h as u32)
            .arg_u32((g * h) as u32)
            .arg_u32(h as u32)
            .arg_u32((g * outs) as u32)
            .launch(stream)?;
        // row 0, the anchor's own pick: not a draft
        pick(
            gpu,
            ink,
            &Pick {
                logits: self.batch_logits,
                logits_stride: (g * v) as u64,
                bias: DevicePtr::NULL,
                bias_stride: 0,
                token: tokens,
                token_stride: record,
                conf: DevicePtr::NULL,
                conf_stride: 0,
                vocab: v,
            },
            n,
            stream,
        )?;
        gpu.memset_async(b.state, 0, n * RANK * 2, stream)?;
        for i in 0..steps {
            // step i drafts row i + 1 from the token before it (the anchor first)
            let (prev, prev_stride) = if i == 0 {
                (self.batch_markov_prev, 1)
            } else {
                (tokens.offset(i * 4), record)
            };
            KernelLaunch::new(gpu, ink.k.rnn_raw)
                .grid([(outs / 8) as u32, n as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(ink.w.rnn_proj.weight)
                .arg_ptr(b.ctx.offset(i * outs * 4))
                .arg_ptr(b.state)
                .arg_ptr(ink.w.markov_w1.weight)
                .arg_ptr(prev)
                .arg_ptr(b.raw)
                .arg_u32(h as u32)
                .arg_u32((g * outs) as u32)
                .arg_u32(RANK as u32)
                .arg_u32(prev_stride)
                .arg_u32(outs as u32)
                .launch(stream)?;
            KernelLaunch::new(gpu, ink.k.rnn_gate)
                .grid([n as u32, 1, 1])
                .block([RANK as u32, 1, 1])
                .arg_ptr(b.raw)
                .arg_ptr(b.state)
                .arg_ptr(b.latent)
                .arg_u32(FOLD_K as u32)
                .arg_u32(outs as u32)
                .arg_u32(RANK as u32)
                .arg_u32(FOLD_K as u32)
                .launch(stream)?;
            for (first, rows) in fold_waves(n) {
                ops::dense_gemv_batchm(
                    gpu,
                    self.kernels.dense_gemv_batchm,
                    b.latent.offset(first * FOLD_K * 2),
                    &ink.fold,
                    b.bias.offset(first * v * 2),
                    rows as u32,
                    v as u32,
                    FOLD_K as u32,
                    v as u32,
                    stream,
                )?;
            }
            pick(
                gpu,
                ink,
                &Pick {
                    logits: self.batch_logits.offset((i + 1) * v * 2),
                    logits_stride: (g * v) as u64,
                    bias: b.bias,
                    bias_stride: v as u32,
                    token: tokens.offset((i + 1) * 4),
                    token_stride: record,
                    conf: if ink.conf {
                        tokens.offset((g + i + 1) * 4)
                    } else {
                        DevicePtr::NULL
                    },
                    conf_stride: record,
                    vocab: v,
                },
                n,
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::fold_waves;

    #[test]
    fn the_fold_runs_in_waves_of_eight_rows() {
        assert_eq!(fold_waves(1).collect::<Vec<_>>(), [(0, 1)]);
        assert_eq!(fold_waves(8).collect::<Vec<_>>(), [(0, 8)]);
        assert_eq!(fold_waves(13).collect::<Vec<_>>(), [(0, 8), (8, 5)]);
        assert_eq!(fold_waves(32).count(), 4);
    }
}
