// SPDX-License-Identifier: AGPL-3.0-only
//! Optional target-only BF16 prefill cache; packed verification weights remain.
use super::*;
use anyhow::ensure;
use atlas_core::config::ModelConfig;
impl DenseFfnLayer {
    /// Called only by the deferred target factory pass, before arena/KV sizing.
    pub(crate) fn cache_glm_prefill_bf16(
        &mut self,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        crate::factory::glm_dense_cache::validate(config)?;
        ensure!(
            self.prefill_bf16_weights.is_none()
                && self.bf16_weights.is_none()
                && self.fp8_weights.is_none()
                && self.q2_weights.is_none()
                && self.lora.is_none(),
            "dense BF16 prefill cache conflicts with existing weights/adapters"
        );
        ensure!(
            self.dequant_nvfp4_bf16_k.0 != 0,
            "dense BF16 dequant kernel missing"
        );
        let sources = [
            &self.weights.gate_proj,
            &self.weights.up_proj,
            &self.weights.down_proj,
        ];
        for w in sources {
            ensure!(
                !w.weight.is_null()
                    && !w.weight_scale.is_null()
                    && !w.has_per_row_scale2()
                    && w.weight_scale_2.is_finite()
                    && w.weight_scale_2 > 0.,
                "invalid dense BF16 source weight/scales"
            );
        }
        let mut ptrs = Vec::with_capacity(3);
        let stream = gpu.default_stream();
        let result = (|| -> Result<()> {
            for (w, (n, k)) in
                sources
                    .into_iter()
                    .zip([(12288, 4096), (12288, 4096), (4096, 12288)])
            {
                let ptr = gpu.alloc(n as usize * k as usize * 2)?;
                ptrs.push(ptr);
                ops::dequant_nvfp4_to_bf16(
                    gpu,
                    self.dequant_nvfp4_bf16_k,
                    w.weight,
                    w.weight_scale,
                    ptr,
                    w.weight_scale_2,
                    n,
                    k,
                    stream,
                )?;
            }
            gpu.synchronize(stream)?;
            Ok(())
        })();
        if let Err(error) = result {
            // Synchronize before releasing buffers referenced by queued work.
            let _ = gpu.synchronize(stream);
            for ptr in ptrs {
                let _ = gpu.free(ptr);
            }
            return Err(error);
        }
        self.prefill_bf16_weights = Some(DenseFfnWeightsBf16 {
            gate_proj: DenseWeight { weight: ptrs[0] },
            up_proj: DenseWeight { weight: ptrs[1] },
            down_proj: DenseWeight { weight: ptrs[2] },
        });
        Ok(())
    }

    /// SiLU FFN over `rows` verify rows on the W4A16 tensor-core tier for that
    /// width; false when the tier or a scalar-scale NVFP4 weight is missing.
    /// `ATLAS_GLM_CANONICAL_VERIFY`: a GLM dense FFN of 1..=32 verify rows on
    /// the canonical tensor-core rows below, whichever entry the width takes
    /// (one row's prefill GEMM, the scalar K2 / K3 GEMVs, the batch-M tiers,
    /// the owner-batched rows), so a row's bits do not follow the width.
    pub(super) fn try_glm_canonical_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if ctx.config.model_type != "glm5_next"
            || !crate::layers::canonical_verify::enabled()
            || !(1..=crate::layers::canonical_verify::MAX_ROWS as usize).contains(&rows)
            || self.q2_weights.is_some()
        {
            return Ok(false);
        }
        self.glm_verify_rows_tc(input, rows as u32, ctx, stream)
    }

    fn glm_verify_rows_tc(
        &self,
        input: DevicePtr,
        rows: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let tc = crate::layers::w4a16_gemv_tiers::tc_kernel(rows);
        let w = &self.weights;
        let usable = |q: &crate::weight_map::QuantizedWeight| {
            !q.weight.is_null() && !q.weight_scale.is_null() && !q.has_per_row_scale2()
        };
        if tc.0 == 0
            || self.activation != FfnActivation::SiLU
            || self.lora.is_some()
            || ![&w.gate_proj, &w.up_proj, &w.down_proj]
                .into_iter()
                .all(usable)
        {
            return Ok(false);
        }
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let (gate, up) = (ctx.buffers.expert_gate_out(), ctx.buffers.expert_up_out());
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            tc,
            input,
            &w.gate_proj,
            gate,
            rows,
            inter,
            h,
            stream,
        )?;
        ops::w4a16_gemv_batchm(ctx.gpu, tc, input, &w.up_proj, up, rows, inter, h, stream)?;
        ops::silu_mul(ctx.gpu, self.act_mul, gate, up, gate, rows * inter, stream)?;
        let out = ctx.buffers.moe_output();
        ops::w4a16_gemv_batchm(ctx.gpu, tc, gate, &w.down_proj, out, rows, h, inter, stream)?;
        Ok(true)
    }

    pub(super) fn try_glm_prefill_bf16(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let transient = self.prefill_bf16_weights.is_none()
            && crate::factory::glm_dense_cache::transient()
            && self.dequant_nvfp4_bf16_k.0 != 0;
        if self.prefill_bf16_weights.is_none() && !transient {
            return Ok(false);
        }
        // All existing native speculative verification widths retain W4A16,
        // including fallbacks that happen to enter forward_prefill_inner.
        if rows <= 8 {
            return Ok(false);
        }
        // Owner-batched verify blocks (9..=32 rows) read the NVFP4 weights once
        // on the tensor-core tiers instead of dequantizing them to BF16 per step.
        if rows <= 32 && self.glm_verify_rows_tc(input, rows as u32, ctx, stream)? {
            return Ok(true);
        }
        crate::factory::glm_dense_cache::validate(ctx.config)?;
        ensure!(
            ctx.dispatch.cublas_gemm,
            "dense BF16 prefill cache requires cuBLAS GEMM"
        );
        ensure!(
            self.activation == FfnActivation::SiLU && self.lora.is_none(),
            "dense BF16 prefill supports unadapted SiLU only"
        );
        let m = u32::try_from(rows)?;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let gate = ctx.buffers.expert_gate_out();
        let up = ctx.buffers.expert_up_out();
        // Cached BF16 weight `i` (gate, up, down), or the NVFP4 weight
        // dequantized into `expert_down_out`, idle for a dense layer; each
        // GEMM reads it before the next dequant on this stream overwrites it.
        let weight = |i: usize, n: u32, k: u32| -> Result<DevicePtr> {
            if let Some(w) = &self.prefill_bf16_weights {
                return Ok([w.gate_proj.weight, w.up_proj.weight, w.down_proj.weight][i]);
            }
            let src = [
                &self.weights.gate_proj,
                &self.weights.up_proj,
                &self.weights.down_proj,
            ][i];
            let scratch = ctx.buffers.expert_down_out();
            ensure!(
                ctx.buffers.sizes().expert_down_out >= n as usize * k as usize * 2,
                "dense BF16 transient weight exceeds expert_down_out"
            );
            ops::dequant_nvfp4_to_bf16(
                ctx.gpu,
                self.dequant_nvfp4_bf16_k,
                src.weight,
                src.weight_scale,
                scratch,
                src.weight_scale_2,
                n,
                k,
                stream,
            )?;
            Ok(scratch)
        };
        ops::cublas_bf16_proj_dense(input, weight(0, inter, h)?, gate, m, inter, h, stream)?;
        ops::cublas_bf16_proj_dense(input, weight(1, inter, h)?, up, m, inter, h, stream)?;
        ops::silu_mul(ctx.gpu, self.act_mul, gate, up, gate, m * inter, stream)?;
        ops::cublas_bf16_proj_dense(
            gate,
            weight(2, h, inter)?,
            ctx.buffers.moe_output(),
            m,
            h,
            inter,
            stream,
        )?;
        Ok(true)
    }
}
#[cfg(test)]
#[path = "dense_ffn_prefill_bf16_tests.rs"]
mod tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "dense_ffn_prefill_bf16_native.rs"]
mod native;
