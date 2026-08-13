// SPDX-License-Identifier: AGPL-3.0-only

//! DSpark sequential Markov fixup on DFlash block logits.
//!
//! After the parallel backbone writes `[γ, vocab]` logits, sample left to
//! right: `logits[i] += markov_w2 @ markov_w1[prev]`, then argmax. Official
//! Lightning DSpark is Markov-only (no confidence head).

use anyhow::Result;

use super::BlockDiffusionDraftHead;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

impl BlockDiffusionDraftHead {
    /// Argmax each γ row. When Markov weights are bound, add the sequential
    /// transition bias first using `last_token` as the position-0 previous.
    pub(super) fn argmax_block_logits(
        &self,
        last_token: u32,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let bf16 = 2usize;
        if let (Some(w1), Some(w2)) = (self.markov_w1.as_ref(), self.markov_w2.as_ref()) {
            if self.markov_rank > 0 && !self.markov_embed.is_null() {
                return self.argmax_with_markov(last_token, w1, w2, gpu, stream);
            }
        }
        for i in 0..self.gamma {
            let logits_row = self.scratch.logits.offset(i * self.vocab_size * bf16);
            let token_slot = self.scratch.draft_tokens_dev.offset(i * 4);
            ops::argmax_bf16(
                gpu,
                self.kernels.argmax,
                logits_row,
                token_slot,
                self.vocab_size as u32,
                stream,
            )?;
        }
        Ok(())
    }

    fn argmax_with_markov(
        &self,
        last_token: u32,
        w1: &DenseWeight,
        w2: &DenseWeight,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let bf16 = 2usize;
        let rank = self.markov_rank;
        // Seed prev = last_token on device. No per-step DtoH — the chain
        // stays on-device so Option B graph capture does not hit CUDA 900.
        let last_bytes = last_token.to_le_bytes();
        gpu.copy_h2d(&last_bytes, self.scratch.slot_mapping_dev)?;
        // Row 0 = anchor bonus: plain argmax, never Markov-biased.
        ops::argmax_bf16(
            gpu,
            self.kernels.argmax,
            self.scratch.logits,
            self.scratch.draft_tokens_dev,
            self.vocab_size as u32,
            stream,
        )?;
        // Rows 1..γ−1 = mask rows. prev is last_token for row 1, then the
        // just-written draft_tokens_dev[i-1].
        for i in 1..self.gamma {
            let prev_dev = if i == 1 {
                self.scratch.slot_mapping_dev
            } else {
                self.scratch.draft_tokens_dev.offset((i - 1) * 4)
            };
            ops::batched_embed(
                gpu,
                self.kernels.batched_embed,
                prev_dev,
                w1.weight,
                self.markov_embed,
                1,
                rank as u32,
                stream,
            )?;
            ops::dense_gemv(
                gpu,
                self.kernels.dense_gemv,
                self.markov_embed,
                w2,
                self.markov_bias,
                self.vocab_size as u32,
                rank as u32,
                stream,
            )?;
            let logits_row = self.scratch.logits.offset(i * self.vocab_size * bf16);
            ops::residual_add(
                gpu,
                self.kernels.residual_add,
                logits_row,
                self.markov_bias,
                self.vocab_size as u32,
                stream,
            )?;
            let token_slot = self.scratch.draft_tokens_dev.offset(i * 4);
            ops::argmax_bf16(
                gpu,
                self.kernels.argmax,
                logits_row,
                token_slot,
                self.vocab_size as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
