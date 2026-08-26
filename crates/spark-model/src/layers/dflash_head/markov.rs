// SPDX-License-Identifier: AGPL-3.0-only

//! DSpark sequential Markov fixup on DFlash block logits.
//!
//! After the parallel backbone writes `[γ, vocab]` logits, sample left to
//! right: `logits[i] += markov_w2 @ markov_w1[prev]`, then argmax. Official
//! Lightning DSpark is Markov-only (no confidence head).

use anyhow::Result;

use super::{BlockDiffusionDraftHead, DflashScratch};
use crate::layers::ops;
use crate::weight_map::DenseWeight;
use spark_runtime::gpu::DevicePtr;

impl BlockDiffusionDraftHead {
    /// Argmax each γ row. When Markov weights are bound, add the sequential
    /// transition bias first using `last_token` as the position-0 previous.
    pub(super) fn argmax_block_logits(
        &self,
        last_token: u32,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        stream: u64,
        scratch: &DflashScratch,
        markov_embed: DevicePtr,
        markov_bias: DevicePtr,
    ) -> Result<()> {
        let bf16 = 2usize;
        if let (Some(w1), Some(w2)) = (self.markov_w1.as_ref(), self.markov_w2.as_ref())
            && self.markov_rank > 0
            && !markov_embed.is_null()
        {
            return self.argmax_with_markov(
                last_token,
                w1,
                w2,
                gpu,
                stream,
                scratch,
                markov_embed,
                markov_bias,
            );
        }
        for i in 0..self.gamma {
            let logits_row = scratch.logits.offset(i * self.vocab_size * bf16);
            let token_slot = scratch.draft_tokens_dev.offset(i * 4);
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
        scratch: &DflashScratch,
        markov_embed: DevicePtr,
        markov_bias: DevicePtr,
    ) -> Result<()> {
        let bf16 = 2usize;
        let rank = self.markov_rank;
        // Seed prev = last_token on device. No per-step DtoH — the chain
        // stays on-device so Option B graph capture does not hit CUDA 900.
        // Pageable 4-byte last_token: copy_h2d_async stages it (no stream
        // sync). copy_h2d would cuStreamSynchronize and CUDA-900 under
        // Option B tail capture.
        // last_token must already live in markov_prev_dev (seeded outside
        // the captured tail). Do not H2D a stack [u8;4] here — that
        // pointer dies after capture and every replay uses garbage prev.
        let _ = last_token;
        // Row 0 = anchor bonus: plain argmax, never Markov-biased.
        ops::argmax_bf16(
            gpu,
            self.kernels.argmax,
            scratch.logits,
            scratch.draft_tokens_dev,
            self.vocab_size as u32,
            stream,
        )?;
        // Rows 1..γ−1 = mask rows. prev is last_token for row 1, then the
        // just-written draft_tokens_dev[i-1].
        for i in 1..self.gamma {
            let prev_dev = if i == 1 {
                scratch.markov_prev_dev
            } else {
                scratch.draft_tokens_dev.offset((i - 1) * 4)
            };
            ops::batched_embed(
                gpu,
                self.kernels.batched_embed,
                prev_dev,
                w1.weight,
                markov_embed,
                1,
                rank as u32,
                stream,
            )?;
            ops::dense_gemv(
                gpu,
                self.kernels.dense_gemv,
                markov_embed,
                w2,
                markov_bias,
                self.vocab_size as u32,
                rank as u32,
                stream,
            )?;
            let logits_row = scratch.logits.offset(i * self.vocab_size * bf16);
            ops::residual_add(
                gpu,
                self.kernels.residual_add,
                logits_row,
                markov_bias,
                self.vocab_size as u32,
                stream,
            )?;
            let token_slot = scratch.draft_tokens_dev.offset(i * 4);
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

    /// Write `last_token` into the stable device slot the tail graph reads.
    /// Must run on `stream` BEFORE begin_capture(tail) / launch_graph(tail).
    pub(super) fn seed_markov_prev(
        &self,
        last_token: u32,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        stream: u64,
        scratch: &DflashScratch,
    ) -> Result<()> {
        let ptr = scratch
            .markov_prev_host_pinned
            .load(std::sync::atomic::Ordering::Relaxed);
        anyhow::ensure!(!ptr.is_null(), "markov_prev_host_pinned is null");
        unsafe {
            std::ptr::write(ptr as *mut u32, last_token);
        }
        let host = unsafe { std::slice::from_raw_parts(ptr, 4) };
        gpu.copy_h2d_async(host, scratch.markov_prev_dev, stream)
    }
}
