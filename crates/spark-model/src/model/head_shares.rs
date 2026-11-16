// SPDX-License-Identifier: AGPL-3.0-only
//! The single-row LM head with the vocabulary in one contiguous share a
//! rank, for a model whose ranks all run every layer (DeepSeek-V4.1: its
//! ranks each hold parts of every wide projection and end a token with the
//! same hidden row).
//!
//! The BF16 head is `[vocab, hidden]`, 1.3 GB for DeepSeek-V4.1, and one row
//! through all of it is about as long as ten of the model's layers. Here a
//! rank projects its share of the vocabulary, through a pointer offset into
//! the resident head, and the ranks gather the shares into the logits row.
//! Every rank then holds the whole row, so sampling reads what it always
//! did.
//!
//! Exactness: `dense_gemv_bf16` computes output column `n` from weight row
//! `n` and the activation row alone (`qwen4exp_lmhead_split`, which is the
//! same idea over a pair), so a share is the bytes of its columns of the
//! full projection.
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::qwen4exp_lmhead_split::HeadArith;
use super::types::TransformerModel;

const BF16: usize = 2;

impl TransformerModel {
    /// The head of one row in shares, into `self.buffers.logits()`.
    /// `Ok(false)`, nothing launched: the caller runs the full head. Every
    /// reason to decline is configuration the ranks share.
    pub(in crate::model) fn head_in_shares(&self, normed: DevicePtr, stream: u64) -> Result<bool> {
        let (world, rank) = (self.config.ep_world_size, self.config.ep_rank);
        let vocab = self.config.vocab_size;
        let Some(comm) = self.comm.as_ref() else {
            return Ok(false);
        };
        if self.config.model_type != "deepseek_v41"
            || world < 2
            || !vocab.is_multiple_of(world)
            || !self.plain_bf16_head()
        {
            return Ok(false);
        }
        let share = vocab / world;
        if !comm.supports_gather_async(share * BF16) {
            return Ok(false);
        }
        let logits = self.buffers.logits();
        self.head_project(
            HeadArith::Gemv,
            normed,
            rank * share,
            share,
            1,
            logits.offset(rank * share * BF16),
            share,
            stream,
        )?;
        if !comm.gather_async(logits.0, (share * BF16) as i64, share * BF16, stream)? {
            // The transport would not take it after all: the row whole, here.
            self.lm_head(normed, stream)?;
        }
        Ok(true)
    }
}
