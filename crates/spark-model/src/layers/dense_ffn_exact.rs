// SPDX-License-Identifier: AGPL-3.0-only

//! The dense FFN's verify rows under the dense exact verify
//! (`ATLAS_EXACT_VERIFY=1`, `model/exact_verify.rs`).
//!
//! Serial decode (`DenseFfnLayer::forward`, NVFP4 weights) runs gate and up
//! through `w4a16_gemv_dual[_sw]`, SiLU-mul, then down through
//! `w4a16_gemv[_sw]`: every projection row is `w4a16_gemv`'s bytes. The
//! verify's 2/3-row arms read gate+up with `w4a16_gemv_dual_batch2/3`, a
//! different accumulation (~1 output in 1000 a BF16 ULP off), and the 4..8
//! row arm may take a tensor-core or DP4A tier, and 9..=16 rows (a DFlash
//! K=γ verify) the `forward_prefill` tile GEMMs. Here every projection runs
//! the scalar `w4a16_gemv_batch2/3` / `w4a16_gemv_batchN` template (batch16
//! past 8 rows, `W4a16BatchmTiers::exact_kernel`), whose row `r` is
//! `w4a16_gemv` on row `r` with the weight read once for all rows
//! (`examples/w4a16_batch_bitparity_microtest.rs`).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{DenseFfnLayer, FfnActivation};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

impl DenseFfnLayer {
    /// Serial decode takes the plain NVFP4 split-SiLU arm: no native, Q2,
    /// FP8 or BF16 copy, no DP4A, no decode-via-GEMM (the exact lever
    /// refuses that and the fused-SiLU switch at build).
    fn serial_is_plain_nvfp4(&self, ctx: &ForwardContext) -> bool {
        self.native.is_none()
            && self.q2_weights.is_none()
            && self.fp8_weights.is_none()
            && self.bf16_weights.is_none()
            && self.activation == FfnActivation::SiLU
            && self.act_mul.0 != 0
            && self.w4a16_gemv.0 != 0
            && !self.weights.gate_proj.weight.is_null()
            && !(ops::dp4a_enabled() && self.dp4a_gemv_k.0 != 0)
            && !ctx.levers.decode_ffn_via_gemm
    }

    /// `m` verify rows of `input` through the FFN into `moe_output`, each row
    /// serial decode's bytes: 2..=8 rows, or up to the lever's widest verify
    /// (`ModelLevers::dense_exact_rows`, 16 beside a DFlash drafter).
    /// `Ok(false)`, nothing launched, when the dense exact verify is off or
    /// serial decode takes another arm.
    pub(super) fn forward_rows_exact(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !ctx.levers.dense_exact_verify()
            || !((2..=8).contains(&m) || ctx.levers.dense_exact_rows(m as usize))
            || !self.serial_is_plain_nvfp4(ctx)
        {
            return Ok(false);
        }
        let tier = self.w4a16_batchm.exact_kernel(m, self.w4a16_gemv_batch16);
        if m > 3 && tier.0 == 0 {
            return Ok(false);
        }
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let (gate, up, out) = (
            ctx.buffers.expert_gate_out(),
            ctx.buffers.expert_up_out(),
            ctx.buffers.moe_output(),
        );
        let rows = |x: DevicePtr, w: &QuantizedWeight, y: DevicePtr, n: u32, k: u32| match m {
            2 => ops::w4a16_gemv_batch2(ctx.gpu, self.w4a16_gemv_batch2, x, w, y, n, k, stream),
            3 => ops::w4a16_gemv_batch3(ctx.gpu, self.w4a16_gemv_batch3, x, w, y, n, k, stream),
            _ => ops::w4a16_gemv_batchm(ctx.gpu, tier, x, w, y, m, n, k, stream),
        };
        rows(input, &self.weights.gate_proj, gate, inter, h)?;
        rows(input, &self.weights.up_proj, up, inter, h)?;
        ops::silu_mul(ctx.gpu, self.act_mul, gate, up, gate, m * inter, stream)?;
        rows(gate, &self.weights.down_proj, out, h, inter)?;
        Ok(true)
    }
}
