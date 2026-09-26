// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash2 bilinear candidate selector over the staged `[B, gamma]` rows.
//!
//! Mirrors the serial `argmax_block_logits` selector arm: ONE projected-hidden
//! GEMM `batch_norm [B*gamma, H] @ hidden_projection^T -> [B*gamma, rank]`
//! (weights read once for the whole batch), then one
//! `dflash2_candidate_selector` launch per sequence — the chain is sequential
//! within a sequence, so it never merges across boundaries. Only the device
//! kernel is supported here; the host fallback stays serial-only.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::BlockDiffusionDraftHead;
use crate::layers::ops;

impl BlockDiffusionDraftHead {
    pub(super) fn run_batched_dflash2_tail(
        &self,
        batch_size: u32,
        last_tokens: &[u32],
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let selector = self
            .candidate_selector
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DFlash batched tail without a candidate selector"))?;
        let kernel = self
            .kernels
            .dflash2_candidate_selector
            .filter(|k| k.0 != 0)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "DFlash batched tail: dflash2_candidate_selector device kernel is \
                     unresolved; the host fallback stays serial-only"
                )
            })?;
        anyhow::ensure!(
            self.batch_dflash2_projected != DevicePtr::NULL,
            "DFlash batched selector scratch is null"
        );
        anyhow::ensure!(
            last_tokens.len() == batch_size as usize,
            "DFlash batched selector last_tokens {} != batch {}",
            last_tokens.len(),
            batch_size
        );
        let gamma =
            u32::try_from(self.gamma).map_err(|_| anyhow::anyhow!("DFlash gamma exceeds u32"))?;
        let hidden = u32::try_from(self.hidden_size)
            .map_err(|_| anyhow::anyhow!("DFlash hidden width exceeds u32"))?;
        let vocab = u32::try_from(self.vocab_size)
            .map_err(|_| anyhow::anyhow!("DFlash vocab exceeds u32"))?;
        let rank = u32::try_from(selector.rank)
            .map_err(|_| anyhow::anyhow!("DFlash selector rank exceeds u32"))?;
        let total_rows = batch_size
            .checked_mul(gamma)
            .ok_or_else(|| anyhow::anyhow!("DFlash batched selector rows overflow"))?;

        // (a) One projected-hidden GEMM over all B*gamma rows.
        self.drafter_dense_gemm(
            ctx.gpu,
            self.batch_norm,
            &selector.hidden_projection,
            self.batch_dflash2_projected,
            total_rows,
            rank,
            hidden,
            stream,
        )?;

        // (b) One selector launch per sequence on that sequence's slices.
        let seq_logits_bytes = (gamma as usize)
            .checked_mul(vocab as usize)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("DFlash selector logits stride overflow"))?;
        let seq_projected_bytes = (gamma as usize)
            .checked_mul(selector.rank)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("DFlash selector projected stride overflow"))?;
        let seq_token_bytes = (gamma as usize)
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("DFlash selector token stride overflow"))?;
        for sequence in 0..batch_size as usize {
            ops::dflash2_candidate_selector(
                ctx.gpu,
                kernel,
                self.batch_logits.offset(sequence * seq_logits_bytes),
                self.batch_dflash2_projected
                    .offset(sequence * seq_projected_bytes),
                selector.predecessor_codebook.weight,
                selector.successor_codebook.weight,
                self.batch_tokens.offset(sequence * seq_token_bytes),
                last_tokens[sequence],
                gamma,
                vocab,
                rank,
                selector.top_k as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
