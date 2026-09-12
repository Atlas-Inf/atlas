// SPDX-License-Identifier: AGPL-3.0-only

//! W4A8 DP4A arm for the K=4 verify LM head.
//!
//! Split out of `impl_a3.rs` to keep that file inside the 500-LoC cap.
//!
//! The NVFP4 LM head is already the layout the DP4A path consumes
//! (U8 `[V, K/2]` + F8 `[V, K/16]`), so the only addition is a hoisted int8
//! activation quant. Measured at `[248320, 5120]` M=4: the float
//! `w4a16_gemv_batch4` tier is 5979.7 us isolated — 6367.8 us in-situ in the
//! K=4 profile, i.e. ~120 GB/s — against 3772.6 us for
//! `w4a16_gemv_dp4a_batch4_d4`, a 1.58x win. Opt-in behind
//! `ATLAS_W4A16_DP4A`, exactly like the dense FFN.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::layers::ops;

impl TransformerModel {
    /// Project the K=4 verify rows through the LM head with W4A8 DP4A.
    ///
    /// Returns `false` when the arm does not apply — flag off, kernels absent,
    /// scratch NULL, or the head is not NVFP4 — so the caller's ladder is left
    /// untouched and targets without the kernels are unaffected.
    ///
    /// `num_tokens` must be 4 exactly: the guard-free M=4 DP4A kernel writes
    /// all four rows unconditionally, so dispatching it at M=3 would write a
    /// row the caller never asked for. The guarded `_dyn` sibling is not wired
    /// here.
    pub(crate) fn lm_head_dp4a_batch4(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        logits: DevicePtr,
        h: u32,
        v: u32,
        stream: u64,
    ) -> Result<bool> {
        if num_tokens != 4
            || !ops::dp4a_enabled()
            || self.lm_head_dp4a_gemv_kernel.0 == 0
            || self.lm_head_dp4a_quant_kernel.0 == 0
        {
            return Ok(false);
        }
        let Some(ref nvfp4) = self.lm_head_nvfp4 else {
            return Ok(false);
        };
        // The dense-FFN int8 scratch is sized for a wider K and the LM head
        // runs after every FFN layer in a step, so the lifetimes do not
        // overlap and no second scratch is needed.
        let quantized = self.buffers.ffn_act_a();
        let scales = self.buffers.ffn_act_scale();
        if quantized.is_null() || scales.is_null() {
            return Ok(false);
        }
        ops::quantize_act_int8_batch4(
            self.gpu.as_ref(),
            self.lm_head_dp4a_quant_kernel,
            hidden,
            quantized,
            scales,
            num_tokens,
            h,
            stream,
        )?;
        ops::w4a16_gemv_dp4a_batch4(
            self.gpu.as_ref(),
            self.lm_head_dp4a_gemv_kernel,
            quantized,
            scales,
            nvfp4,
            logits,
            num_tokens,
            v,
            h,
            stream,
        )?;
        Ok(true)
    }
}
