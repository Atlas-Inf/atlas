// SPDX-License-Identifier: AGPL-3.0-only

//! Shared-expert phase of `MoeLayer::forward_prefill`.
//!
//! Hoisted from `forward_prefill.rs` to keep that file under the 500 LoC
//! cap. The single entry point [`MoeLayer::run_shared_expert_prefill`]
//! mirrors the original block 1:1 — same control flow, same kernel
//! launches, same buffer wiring.

use super::*;

impl MoeLayer {
    /// Shared-expert path of the prefill pipeline (gate + up GEMM → SiLU →
    /// down GEMM). Runs sequentially on the supplied `aux` stream when
    /// `use_overlap == false`; otherwise issues an event so the routed
    /// path can wait on completion.
    ///
    /// Skips entirely when `shared_inter == 0` (e.g. Qwen3-VL-30B has no
    /// shared expert). Launching kernels with N=0 returns
    /// CUDA_ERROR_INVALID_VALUE (grid.x=0).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_shared_expert_prefill(
        &self,
        input: DevicePtr,
        n: u32,
        h: u32,
        shared_inter: u32,
        aux: u64,
        stream: u64,
        use_overlap: bool,
        ctx: &ForwardContext,
    ) -> Result<()> {
        if shared_inter == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            !self.shared_fp8_cache.verify
                || (!use_overlap && !ctx.graph_capture && !ctx.gpu.stream_is_capturing(aux)),
            "shared FP8 VERIFY requires eager, non-overlapped execution"
        );
        if use_overlap {
            // Ensure secondary stream sees `input` (produced by prior default-stream work)
            ctx.gpu.record_event(self.event_a, stream)?;
            ctx.gpu.stream_wait_event(aux, self.event_a)?;
        }

        let shared_gate_out = ctx.buffers.ssm_deinterleaved();
        let shared_up_out = ctx.buffers.ssm_qkvz();
        let shared_down_out = ctx.buffers.attn_output();
        if self.independent_grouped(ctx, n) {
            anyhow::ensure!(
                !use_overlap,
                "independent small-row shared work is sequential"
            );
            return self.independent_shared_expert(input, n, ctx, aux);
        }
        if self.glm_c2_grouped(ctx, n) {
            anyhow::ensure!(
                !use_overlap,
                "C2 compact shared work must remain sequential"
            );
            return self.c2_shared_expert(input, ctx, aux);
        }
        if self.glm_c4_grouped(ctx, n) {
            self.c4_shared_expert(
                input,
                shared_gate_out,
                shared_up_out,
                shared_down_out,
                h,
                shared_inter,
                ctx,
                aux,
            )?;
            if use_overlap {
                ctx.gpu.record_event(self.event_b, aux)?;
            }
            return Ok(());
        }
        if self.glm_c3_grouped(ctx, n) {
            self.c3_shared_expert(
                input,
                shared_gate_out,
                shared_up_out,
                shared_down_out,
                h,
                shared_inter,
                n,
                ctx,
                aux,
            )?;
            if use_overlap {
                ctx.gpu.record_event(self.event_b, aux)?;
            }
            return Ok(());
        }
        if self.run_bf16_shared_expert(
            input,
            n,
            h,
            shared_inter,
            shared_gate_out,
            shared_up_out,
            shared_down_out,
            ctx,
            aux,
        )? {
            if use_overlap {
                ctx.gpu.record_event(self.event_b, aux)?;
            }
            return Ok(());
        }

        // GLM's K=5 verifier enters the grouped routed-expert pipeline to
        // amortize its expert weights, but the generic shared-expert prefill
        // kernels pad this five-row problem to a much wider GEMM tile. Reuse
        // the exact-M decode kernels already proven by forward_k5: they read
        // each native NVFP4 projection once while computing only five rows.
        let batch5 = self.w4a16_batchm.kernel(5);
        let exact_k5 = n == 5
            && std::env::var("ATLAS_GLM_K5_BATCHED_SHARED").as_deref() == Ok("1")
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && batch5.0 != 0
            && !self.weights.shared_expert.gate_proj.is_null()
            && !self.weights.shared_expert.up_proj.is_null()
            && !self.weights.shared_expert.down_proj.is_null();
        let tc = crate::layers::w4a16_gemv_tiers::tc_kernel(n);
        if n > 8
            && tc.0 != 0
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && !self.weights.shared_expert.gate_proj.is_null()
            && !self.weights.shared_expert.up_proj.is_null()
            && !self.weights.shared_expert.down_proj.is_null()
        {
            self.run_shared_tc(
                tc,
                input,
                shared_gate_out,
                shared_up_out,
                shared_down_out,
                n,
                h,
                shared_inter,
                ctx,
                aux,
            )?;
            if use_overlap {
                ctx.gpu.record_event(self.event_b, aux)?;
            }
            return Ok(());
        }
        if exact_k5 {
            self.run_exact_k5_shared(
                input,
                shared_gate_out,
                shared_up_out,
                shared_down_out,
                h,
                shared_inter,
                ctx,
                aux,
            )?;
            if use_overlap {
                ctx.gpu.record_event(self.event_b, aux)?;
            }
            return Ok(());
        }

        // Shared gate + up GEMM on aux stream
        if let (Some(sg_fp8), Some(su_fp8)) = (self.shared_gate_fp8, self.shared_up_fp8) {
            self.run_shared_fp8_cache(
                0,
                input,
                sg_fp8,
                shared_gate_out,
                n,
                shared_inter,
                h,
                ctx,
                aux,
            )?;
            self.run_shared_fp8_cache(
                1,
                input,
                su_fp8,
                shared_up_out,
                n,
                shared_inter,
                h,
                ctx,
                aux,
            )?;
        } else if let (Some(sg), Some(su), Some(_sd)) =
            (&self.shared_gate_t, &self.shared_up_t, &self.shared_down_t)
        {
            self.run_shared_m16(
                shared_m16::SharedProjection::Gate,
                input,
                sg,
                shared_gate_out,
                n,
                shared_inter,
                h,
                ctx,
                aux,
                use_overlap,
            )?;
            self.run_shared_m16(
                shared_m16::SharedProjection::Up,
                input,
                su,
                shared_up_out,
                n,
                shared_inter,
                h,
                ctx,
                aux,
                use_overlap,
            )?;
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                input,
                &self.weights.shared_expert.gate_proj,
                shared_gate_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                input,
                &self.weights.shared_expert.up_proj,
                shared_up_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
        }

        // Shared activation (SiLU or GeGLU) + down GEMM on aux stream
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            shared_gate_out,
            shared_up_out,
            shared_gate_out,
            n * shared_inter,
            aux,
        )?;
        if let Some(sd_fp8) = self.shared_down_fp8 {
            self.run_shared_fp8_cache(
                2,
                shared_gate_out,
                sd_fp8,
                shared_down_out,
                n,
                h,
                shared_inter,
                ctx,
                aux,
            )?;
        } else if let Some(sd) = &self.shared_down_t {
            self.run_shared_m16(
                shared_m16::SharedProjection::Down,
                shared_gate_out,
                sd,
                shared_down_out,
                n,
                h,
                shared_inter,
                ctx,
                aux,
                use_overlap,
            )?;
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                shared_gate_out,
                &self.weights.shared_expert.down_proj,
                shared_down_out,
                n,
                h,
                shared_inter,
                aux,
            )?;
        }

        if use_overlap {
            ctx.gpu.record_event(self.event_b, aux)?;
        }
        Ok(())
    }
}
