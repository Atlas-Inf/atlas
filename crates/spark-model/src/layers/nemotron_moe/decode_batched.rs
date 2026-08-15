// SPDX-License-Identifier: AGPL-3.0-only

//! K-token MoE verify: batched gate + one expert GEMV + one relu²-down
//! (blockIdx.z = token). Same math as `decode_direct_moe`. Latent / native-FP8
//! shared falls back to the per-token path.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::NemotronMoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl NemotronMoeLayer {
    pub(super) fn decode_batched_direct(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if num_tokens <= 1 || self.moe_latent_size > 0 {
            for t in 0..num_tokens {
                let off = t * ctx.config.hidden_size * 2;
                self.decode_inner(hidden.offset(off), residual.offset(off), ctx, stream)?;
            }
            return Ok(());
        }
        let native_fp8 = self
            .weights
            .shared_down_fp8
            .as_ref()
            .filter(|_| self.w8a16_gemv_k.0 != 0 && self.moe_relu2_elementwise_k.0 != 0)
            .is_some();
        if native_fp8 {
            for t in 0..num_tokens {
                let off = t * ctx.config.hidden_size * 2;
                self.decode_inner(hidden.offset(off), residual.offset(off), ctx, stream)?;
            }
            return Ok(());
        }

        let h = ctx.config.hidden_size;
        let n = num_tokens as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = self.top_k as u32;
        let inter = self.moe_inter as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let scale = ctx.config.routed_scaling_factor as f32;
        let bf16 = 2usize;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;
        let gate_logits = ctx.buffers.gate_logits();
        self.dense_gemm_prefill(
            ctx.gpu,
            normed,
            &self.weights.gate,
            gate_logits,
            n,
            num_experts,
            h as u32,
            stream,
        )?;

        let scratch = ctx.buffers.scratch();
        let indices = scratch;
        let weights = scratch.offset(num_tokens * top_k as usize * 4);
        for t in 0..num_tokens {
            ops::moe_topk_sigmoid(
                ctx.gpu,
                self.topk_sigmoid_k,
                gate_logits.offset(t * num_experts as usize * bf16),
                self.weights.e_score_correction_bias.weight,
                indices.offset(t * top_k as usize * 4),
                weights.offset(t * top_k as usize * 4),
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                scale,
                stream,
            )?;
        }

        let expert_up_out = ctx.buffers.expert_up_out();
        // Marlin n<=4 slots decode: NOT bit-exact vs serial GEMV on the
        // fixed engine (G3 A/B: prime prompt diverges). Opt-in until the
        // slots path re-gates lossless. GEMV batched IS bit-exact.
        if self.marlin.is_some()
            && num_tokens <= 4
            && std::env::var("ATLAS_MOE_MARLIN_DECODE").is_ok()
        {
            return self.decode_batched_marlin(hidden, residual, num_tokens, ctx, stream);
        }
        let grouped = n >= 2
            && self.moe_expert_gemv_wide_grouped_k.0 != 0
            && std::env::var("ATLAS_MOE_EXPERT_GROUPED").is_ok();
        if grouped {
            ops::moe_expert_gemv_wide_grouped(
                ctx.gpu,
                self.moe_expert_gemv_wide_grouped_k,
                normed,
                self.up_ptrs.packed_ptrs,
                self.up_ptrs.scale_ptrs,
                self.up_ptrs.scale2_vals,
                expert_up_out,
                indices,
                inter,
                h as u32,
                top_k,
                0,
                n,
                num_experts,
                stream,
            )?;
        } else if self.moe_expert_gemv_wide_k.0 != 0
            && std::env::var("ATLAS_NO_MOE_EXPERT_WIDE").is_err()
        {
            ops::moe_expert_gemv_wide(
                ctx.gpu,
                self.moe_expert_gemv_wide_k,
                normed,
                self.up_ptrs.packed_ptrs,
                self.up_ptrs.scale_ptrs,
                self.up_ptrs.scale2_vals,
                expert_up_out,
                indices,
                inter,
                h as u32,
                top_k,
                0,
                n,
                stream,
            )?;
        } else {
        ops::moe_expert_gemv(
            ctx.gpu,
            self.moe_expert_gemv_k,
            normed,
            self.up_ptrs.packed_ptrs,
            self.up_ptrs.scale_ptrs,
            self.up_ptrs.scale2_vals,
            expert_up_out,
            indices,
            inter,
            h as u32,
            top_k,
            0,
            n,
            stream,
        )?;
        }

        let shared_up = ctx.buffers.ssm_qkvz();
        if n <= 4 && self.w4a16_gemv_batch4_k.0 != 0 {
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                self.w4a16_gemv_batch4_k,
                normed,
                &self.weights.shared_up,
                shared_up,
                n,
                shared_inter,
                h as u32,
                stream,
            )?;
        } else if n <= 16 && self.w4a16_gemv_batch16_k.0 != 0 {
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                self.w4a16_gemv_batch16_k,
                normed,
                &self.weights.shared_up,
                shared_up,
                n,
                shared_inter,
                h as u32,
                stream,
            )?;
        } else {
            self.prefill_shared_up(normed, shared_up, n, h, shared_inter, ctx, stream)?;
        }

        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_down = ctx.buffers.ssm_deinterleaved();
        let smem = (shared_inter.max(inter) as usize) * 4;
        let down_wide = self.relu2_down_wide_k.0 != 0
            && std::env::var("ATLAS_NO_MOE_DOWN_WIDE").is_err();
        if down_wide {
            KernelLaunch::new(ctx.gpu, self.relu2_down_wide_k)
                .grid([div_ceil(h as u32, 64), top_k + 1, n])
                .block([256, 1, 1])
                .shared_mem(smem as u32)
                .arg_ptr(expert_up_out)
                .arg_ptr(self.down_ptrs.packed_ptrs)
                .arg_ptr(self.down_ptrs.scale_ptrs)
                .arg_ptr(self.down_ptrs.scale2_vals)
                .arg_ptr(expert_down_out)
                .arg_ptr(indices)
                .arg_ptr(shared_up)
                .arg_ptr(self.weights.shared_down.weight)
                .arg_ptr(self.weights.shared_down.weight_scale)
                .arg_f32(self.weights.shared_down.weight_scale_2)
                .arg_ptr(shared_down)
                .arg_u32(h as u32)
                .arg_u32(inter)
                .arg_u32(shared_inter)
                .arg_u32(h as u32)
                .arg_u32(top_k)
                .launch(stream)?;
        } else {
            KernelLaunch::new(ctx.gpu, self.relu2_down_shared_k)
                .grid([div_ceil(h as u32, 8), top_k + 1, n])
                .block([128, 1, 1])
                .shared_mem(smem as u32)
                .arg_ptr(expert_up_out)
                .arg_ptr(self.down_ptrs.packed_ptrs)
                .arg_ptr(self.down_ptrs.scale_ptrs)
                .arg_ptr(self.down_ptrs.scale2_vals)
                .arg_ptr(expert_down_out)
                .arg_ptr(indices)
                .arg_ptr(shared_up)
                .arg_ptr(self.weights.shared_down.weight)
                .arg_ptr(self.weights.shared_down.weight_scale)
                .arg_f32(self.weights.shared_down.weight_scale_2)
                .arg_ptr(shared_down)
                .arg_u32(h as u32)
                .arg_u32(inter)
                .arg_u32(shared_inter)
                .arg_u32(h as u32)
                .arg_u32(top_k)
                .launch(stream)?;
        }

        let output = ctx.buffers.moe_output();
        for t in 0..num_tokens {
            let off_h = t * h * bf16;
            KernelLaunch::new(ctx.gpu, self.weighted_sum_scale_k)
                .grid([div_ceil(h as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(output.offset(off_h))
                .arg_ptr(expert_down_out.offset(t * top_k as usize * h * bf16))
                .arg_ptr(weights.offset(t * top_k as usize * 4))
                .arg_ptr(shared_down.offset(off_h))
                .arg_u32(h as u32)
                .arg_u32(top_k)
                .arg_f32(1.0f32)
                .launch(stream)?;
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden.offset(off_h),
                output.offset(off_h),
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
