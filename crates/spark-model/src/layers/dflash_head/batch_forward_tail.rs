// SPDX-License-Identifier: AGPL-3.0-only

//! Staged tail for the native `[B, gamma]` path: final norm + shared LM
//! head + argmax (`run_batched_tail_base`), then the batch-wide Markov
//! rescore (`run_batched_markov`). Split from `batch_forward.rs` at the
//! layer-body / tail seam; both stay `pub(super)` methods on
//! `BlockDiffusionDraftHead`.

use anyhow::Result;

use super::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    /// Stage final norm, shared LM head, and unbiased per-row argmax. The token
    /// buffer is consumed only by the forthcoming batch-wide Markov stage.
    pub(super) fn run_batched_tail_base(
        &self,
        batch_rows: u32,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hidden = u32::try_from(self.hidden_size)
            .map_err(|_| anyhow::anyhow!("DFlash hidden width exceeds u32"))?;
        let vocab = u32::try_from(self.vocab_size)
            .map_err(|_| anyhow::anyhow!("DFlash vocab exceeds u32"))?;
        crate::layers::ops::rms_norm(
            ctx.gpu,
            self.kernels.rms_norm,
            self.batch_query_embed,
            &self.norm,
            self.batch_norm,
            batch_rows,
            hidden,
            self.rms_norm_eps,
            stream,
        )?;
        if let Some(fp8) = self
            .lm_head_shared_fp8
            .as_ref()
            .filter(|_| matches!(self.quant, super::DflashQuantization::Fp8Weights))
        {
            crate::layers::ops::fp8_gemm_n128_row_scaled(
                ctx.gpu,
                self.kernels.fp8_gemm_n128_row_scaled,
                self.batch_norm,
                fp8,
                self.batch_logits,
                batch_rows,
                vocab,
                hidden,
                stream,
            )?;
        } else if let Some(nvfp4) = self.lm_head_nvfp4.as_ref() {
            let kernel = match batch_rows {
                1..=4 => self.kernels.w4a16_gemv_batch4,
                5..=8 => self.kernels.w4a16_gemv_batch8,
                9..=32 => self.kernels.w4a16_gemv_batch16,
                _ => spark_runtime::gpu::KernelHandle(0),
            };
            if kernel.0 != 0 {
                let mut row = 0u32;
                while row < batch_rows {
                    let rows = (batch_rows - row).min(16);
                    crate::layers::ops::w4a16_gemv_batchm(
                        ctx.gpu,
                        kernel,
                        self.batch_norm.offset(row as usize * hidden as usize * 2),
                        nvfp4,
                        self.batch_logits.offset(row as usize * vocab as usize * 2),
                        rows,
                        vocab,
                        hidden,
                        stream,
                    )?;
                    row += rows;
                }
            } else if self.startup.native_batch_authoritative {
                anyhow::bail!(
                    "Lightning DSpark exact NVFP4 LM-head batch kernel is unresolved for rows={batch_rows}; batch4/8/16 with <=16-row waves is mandatory"
                );
            } else {
                anyhow::ensure!(
                    self.kernels.w4a16_gemm.0 != 0,
                    "DFlash batched NVFP4 LM head kernel is unresolved"
                );
                crate::layers::ops::w4a16_gemm(
                    ctx.gpu,
                    self.kernels.w4a16_gemm,
                    self.batch_norm,
                    nvfp4,
                    self.batch_logits,
                    batch_rows,
                    vocab,
                    hidden,
                    stream,
                )?;
            }
        } else {
            crate::layers::ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.kernels.dense_gemm_pipelined,
                self.batch_norm,
                &crate::weight_map::DenseWeight {
                    weight: self.lm_head_shared,
                },
                self.batch_logits,
                batch_rows,
                vocab,
                hidden,
                stream,
            )?;
        }
        crate::layers::ops::argmax_bf16_batch(
            ctx.gpu,
            self.kernels.argmax_batch,
            self.batch_logits,
            self.batch_tokens,
            vocab,
            batch_rows,
            vocab,
            stream,
        )
    }

    /// Apply DSpark Markov bias depth-serial and batch-wide. Row 0 remains the
    /// unbiased anchor; rows 1..gamma are overwritten in `batch_tokens`.
    pub(super) fn run_batched_markov(
        &self,
        batch_size: u32,
        gamma: u32,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.markov_rank == 0 {
            return Ok(());
        }
        let (w1, w2) = self
            .markov_w1
            .as_ref()
            .zip(self.markov_w2.as_ref())
            .ok_or_else(|| anyhow::anyhow!("DFlash batched Markov weights are missing"))?;
        anyhow::ensure!(
            !self.batch_markov_embed.is_null() && !self.batch_markov_bias.is_null(),
            "DFlash batched Markov scratch is null"
        );
        let rank = u32::try_from(self.markov_rank)
            .map_err(|_| anyhow::anyhow!("DFlash Markov rank exceeds u32"))?;
        let vocab = u32::try_from(self.vocab_size)
            .map_err(|_| anyhow::anyhow!("DFlash vocab exceeds u32"))?;
        let row_stride = gamma
            .checked_mul(vocab)
            .ok_or_else(|| anyhow::anyhow!("DFlash Markov row stride overflow"))?;
        for depth in 1..gamma as usize {
            crate::layers::ops::batched_embed(
                ctx.gpu,
                self.kernels.batched_embed,
                self.batch_markov_prev,
                w1.weight,
                self.batch_markov_embed,
                batch_size,
                rank,
                stream,
            )?;
            crate::layers::ops::dense_gemv_batchm(
                ctx.gpu,
                self.kernels.dense_gemv_batchm,
                self.batch_markov_embed,
                w2,
                self.batch_markov_bias,
                batch_size,
                vocab,
                rank,
                vocab,
                stream,
            )?;
            crate::layers::ops::dflash_batch_add_depth_bias(
                ctx.gpu,
                self.kernels.batch_markov_add_bias,
                self.batch_logits,
                self.batch_markov_bias,
                batch_size,
                gamma,
                vocab,
                depth as u32,
                stream,
            )?;
            let logits_offset = depth
                .checked_mul(self.vocab_size)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| anyhow::anyhow!("DFlash Markov logits offset overflow"))?;
            crate::layers::ops::argmax_bf16_batch(
                ctx.gpu,
                self.kernels.argmax_batch,
                self.batch_logits.offset(logits_offset),
                self.batch_markov_prev,
                vocab,
                batch_size,
                row_stride,
                stream,
            )?;
            crate::layers::ops::dflash_batch_store_depth_tokens(
                ctx.gpu,
                self.kernels.batch_markov_store_tokens,
                self.batch_tokens,
                self.batch_markov_prev,
                batch_size,
                gamma,
                depth as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
