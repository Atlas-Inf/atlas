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
        let mut prev = last_token as usize;
        let mut tok_bytes = [0u8; 4];
        for i in 0..self.gamma {
            let src = w1.weight.offset(prev * rank * bf16);
            gpu.copy_d2d_async(src, self.markov_embed, rank * bf16, stream)?;
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
            gpu.copy_d2h_async(token_slot, &mut tok_bytes, stream)?;
            gpu.synchronize(stream)?;
            prev = u32::from_le_bytes(tok_bytes) as usize;
        }
        Ok(())
    }
}
