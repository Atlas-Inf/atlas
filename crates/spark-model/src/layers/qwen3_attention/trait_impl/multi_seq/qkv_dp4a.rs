// SPDX-License-Identifier: AGPL-3.0-only

//! W4A8 DP4A M=4 arm for the attention NVFP4 verify projections (q/k/v/o).
//! Mirrors `model/trait_impl/lm_head_dp4a.rs`: the NVFP4 weight layout is
//! already what the DP4A kernel consumes, so the only addition is an int8
//! activation quant per call — the q/k/v calls re-quantize `normed` (three
//! extra ~6us launches/layer), accepted so the arm stays at the single
//! `wide_verify_gemm` dispatch site that o_proj also routes through.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::weight_map::QuantizedWeight;

impl Qwen3AttentionLayer {
    /// `true` when the guard-free M=4 DP4A arm applies: `ATLAS_W4A16_DP4A=1`,
    /// m == 4 exactly (the `_d4` kernel writes all four rows), both kernel
    /// handles resolved, and the shared int8 scratch non-null.
    pub(super) fn dp4a_batch4_ready(&self, c: &MultiSeqCtx<'_>, m: u32) -> bool {
        ops::dp4a_batch4_eligible(
            m,
            ops::dp4a_enabled(),
            self.dp4a_quant_batch4_k,
            self.dp4a_gemv_batch4_k,
            c.fwd.buffers.ffn_act_a(),
            c.fwd.buffers.ffn_act_scale(),
        )
    }

    /// Quantize `input` ([m, k] BF16) into the shared int8 scratch.
    /// Caller must have checked `dp4a_batch4_ready`.
    ///
    /// Reuses the dense-FFN int8 scratch: the attention block and the FFN run
    /// sequentially on one stream and the FFN re-quantizes its own input, so
    /// the lifetimes do not overlap.
    pub(super) fn dp4a_quant_input(
        &self,
        c: &MultiSeqCtx<'_>,
        input: DevicePtr,
        m: u32,
        k: u32,
    ) -> Result<()> {
        ops::quantize_act_int8_batch4(
            c.fwd.gpu,
            self.dp4a_quant_batch4_k,
            input,
            c.fwd.buffers.ffn_act_a(),
            c.fwd.buffers.ffn_act_scale(),
            m,
            k,
            c.stream,
        )
    }

    /// DP4A batch4 GEMV from the just-quantized scratch into `output`
    /// ([m, n] BF16 — the same row-major layout `w4a16_gemv_batchm` writes).
    pub(super) fn dp4a_gemv_prequant(
        &self,
        c: &MultiSeqCtx<'_>,
        w_base: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        ops::w4a16_gemv_dp4a_batch4(
            c.fwd.gpu,
            self.dp4a_gemv_batch4_k,
            c.fwd.buffers.ffn_act_a(),
            c.fwd.buffers.ffn_act_scale(),
            w_base,
            output,
            m,
            n,
            k,
            c.stream,
        )?;
        ops::dp4a_arm_active_once();
        Ok(())
    }

    /// Full M=4 arm for `wide_verify_gemm`: quantize `input`, run the DP4A
    /// GEMV. Returns `false` when the arm does not apply so the caller's
    /// ladder is left untouched.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dp4a_verify_gemv(
        &self,
        c: &MultiSeqCtx<'_>,
        input: DevicePtr,
        w_base: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<bool> {
        if !self.dp4a_batch4_ready(c, m) {
            return Ok(false);
        }
        self.dp4a_quant_input(c, input, m, k)?;
        self.dp4a_gemv_prequant(c, w_base, output, m, n, k)?;
        Ok(true)
    }
}
