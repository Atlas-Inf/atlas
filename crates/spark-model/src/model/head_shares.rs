// SPDX-License-Identifier: AGPL-3.0-only
//! The LM head with the vocabulary in one contiguous share a rank, for a
//! model whose ranks all run every layer (DeepSeek-V4.1: its ranks each hold
//! parts of every wide projection and end a token with the same hidden row):
//! the row of a decode step, and the rows of a verify.
//!
//! The BF16 head is `[vocab, hidden]`, 1.3 GB for DeepSeek-V4.1, and one row
//! through all of it is about as long as ten of the model's layers. Here a
//! rank projects its share of the vocabulary, through a pointer offset into
//! the resident head, and the ranks gather the shares into the logits rows.
//! Every rank then holds the whole rows, so sampling reads what it always
//! did.
//!
//! Exactness: `dense_gemv_bf16` computes output column `n` from weight row
//! `n` and the activation row alone (`qwen4exp_lmhead_split`, which is the
//! same idea over a pair), so a share is the bytes of its columns of the
//! full projection; and a verify's rows go through `dense_gemv_bf16_batchm`,
//! whose rows are that kernel's bytes, where its full head was a GEMM that
//! adds in another order.
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::qwen4exp_lmhead_split::HeadArith;
use super::types::TransformerModel;

const BF16: usize = 2;

impl TransformerModel {
    /// The vocabulary a rank projects, when the head goes in shares.
    /// `None`: the caller runs the full head. Every reason to decline is
    /// configuration the ranks share.
    fn head_share(&self) -> Option<usize> {
        let (world, vocab) = (self.config.ep_world_size, self.config.vocab_size);
        let comm = self.comm.as_ref()?;
        let shares = self.config.model_type == "deepseek_v41"
            && world >= 2
            && vocab.is_multiple_of(world)
            && self.plain_bf16_head()
            && comm.supports_gather_async(vocab / world * BF16);
        shares.then_some(vocab / world)
    }

    /// The head of the first `rows` rows of `normed` in shares, into the
    /// rows of `self.buffers.logits()`: a rank's share of every row in one
    /// pass over its share of the head, then a gather a row. `Ok(false)`:
    /// the caller runs the full head, as every rank's does.
    pub(in crate::model) fn head_rows_in_shares(
        &self,
        normed: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<bool> {
        let (Some(share), Some(comm)) = (self.head_share(), self.comm.as_ref()) else {
            return Ok(false);
        };
        let (rank, vocab) = (self.config.ep_rank, self.config.vocab_size);
        let logits = self.buffers.logits();
        self.head_project(
            HeadArith::Gemv,
            normed,
            rank * share,
            share,
            rows,
            logits.offset(rank * share * BF16),
            vocab,
            stream,
        )?;
        for row in 0..rows {
            let row = logits.offset(row * vocab * BF16);
            if !comm.gather_async(row.0, (share * BF16) as i64, share * BF16, stream)? {
                // The transport would not take it after all.
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The head of one row in shares, into `self.buffers.logits()`.
    /// `Ok(false)`, nothing launched: the caller runs the full head.
    pub(in crate::model) fn head_in_shares(&self, normed: DevicePtr, stream: u64) -> Result<bool> {
        if self.head_share().is_none() {
            return Ok(false);
        }
        if !self.head_rows_in_shares(normed, 1, stream)? {
            // The transport would not take it after all: the row whole, here.
            self.lm_head(normed, stream)?;
        }
        Ok(true)
    }
}
