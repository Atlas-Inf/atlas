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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateProjection {
    /// Same per-row K iteration/reduction as serial `dense_gemv` while sharing
    /// the router weight read across K verify rows.
    Batchm,
    /// Existing tiled prefill path for widths outside the exact batchm bound.
    Tiled,
}

fn gate_projection(num_tokens: usize, batchm_ready: bool) -> GateProjection {
    if batchm_ready && (2..=ops::DENSE_GEMV_BATCHM_MAX_M as usize).contains(&num_tokens) {
        GateProjection::Batchm
    } else {
        GateProjection::Tiled
    }
}

impl NemotronMoeLayer {
    pub(super) fn decode_batched_direct(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_batched_direct_with_gate_chunks(hidden, residual, num_tokens, None, ctx, stream)
    }

    pub(super) fn decode_verify_multi_shared(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        ks: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let num_tokens = ks.iter().sum();
        self.decode_batched_direct_with_gate_chunks(
            hidden,
            residual,
            num_tokens,
            Some(ks),
            ctx,
            stream,
        )
    }

    fn decode_batched_direct_with_gate_chunks(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gate_chunks: Option<&[usize]>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let marlin_all_widths =
            std::env::var("ATLAS_LIGHTNING_MOE_MARLIN_ALL_WIDTHS").as_deref() == Ok("1");
        if (num_tokens <= 1 || self.moe_latent_size > 0) && !marlin_all_widths {
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
        if native_fp8 && !marlin_all_widths {
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
        if let Some(chunks) = gate_chunks {
            anyhow::ensure!(
                chunks.iter().sum::<usize>() == num_tokens,
                "MoE verify gate chunks do not cover R={num_tokens}"
            );
            let mut row = 0usize;
            for &k in chunks {
                match gate_projection(k, self.dense_gemv_batchm_k.0 != 0) {
                    GateProjection::Batchm => ops::dense_gemv_batchm(
                        ctx.gpu,
                        self.dense_gemv_batchm_k,
                        normed.offset(row * h * bf16),
                        &self.weights.gate,
                        gate_logits.offset(row * num_experts as usize * bf16),
                        k as u32,
                        num_experts,
                        h as u32,
                        num_experts,
                        stream,
                    )?,
                    GateProjection::Tiled => self.dense_gemm_prefill(
                        ctx.gpu,
                        normed.offset(row * h * bf16),
                        &self.weights.gate,
                        gate_logits.offset(row * num_experts as usize * bf16),
                        k as u32,
                        num_experts,
                        h as u32,
                        stream,
                    )?,
                }
                row += k;
            }
        } else {
            match gate_projection(num_tokens, self.dense_gemv_batchm_k.0 != 0) {
                GateProjection::Batchm => ops::dense_gemv_batchm(
                    ctx.gpu,
                    self.dense_gemv_batchm_k,
                    normed,
                    &self.weights.gate,
                    gate_logits,
                    n,
                    num_experts,
                    h as u32,
                    num_experts,
                    stream,
                )?,
                GateProjection::Tiled => self.dense_gemm_prefill(
                    ctx.gpu,
                    normed,
                    &self.weights.gate,
                    gate_logits,
                    n,
                    num_experts,
                    h as u32,
                    stream,
                )?,
            }
        }

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
            && (num_tokens <= 4 || marlin_all_widths)
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
                false,
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
        } else if n <= 32 && self.w4a16_gemv_batch32_k.0 != 0 {
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                self.w4a16_gemv_batch32_k,
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
        let down_wide =
            self.relu2_down_wide_k.0 != 0 && std::env::var("ATLAS_NO_MOE_DOWN_WIDE").is_err();
        if grouped {
            // Routed DOWN: consume route-major UP rows, apply relu² in
            // registers, and share each expert's DOWN weight stream across all
            // matching routes without an intermediate BF16 activation store.
            ops::moe_expert_gemv_wide_grouped(
                ctx.gpu,
                self.moe_expert_gemv_wide_grouped_k,
                expert_up_out,
                self.down_ptrs.packed_ptrs,
                self.down_ptrs.scale_ptrs,
                self.down_ptrs.scale2_vals,
                expert_down_out,
                indices,
                h as u32,
                inter,
                top_k,
                inter,
                n,
                num_experts,
                true,
                stream,
            )?;

            if self.shared_relu2_down_grouped_k.0 != 0 {
                KernelLaunch::new(ctx.gpu, self.shared_relu2_down_grouped_k)
                    .grid([div_ceil(h as u32, 32), 1, 1])
                    .block([256, 1, 1])
                    .shared_mem((64 + 8 * shared_inter as usize * bf16) as u32)
                    .arg_ptr(shared_up)
                    .arg_ptr(self.weights.shared_down.weight)
                    .arg_ptr(self.weights.shared_down.weight_scale)
                    .arg_f32(self.weights.shared_down.weight_scale_2)
                    .arg_ptr(shared_down)
                    .arg_u32(n)
                    .arg_u32(h as u32)
                    .arg_u32(shared_inter)
                    .launch(stream)?;
            } else {
                // Fallback retains the prior exact shared-only launch.
                let (shared_kernel, shared_tile, shared_block) = if down_wide {
                    (self.relu2_down_wide_k, 64u32, 256u32)
                } else {
                    (self.relu2_down_shared_k, 8u32, 128u32)
                };
                KernelLaunch::new(ctx.gpu, shared_kernel)
                    .grid([div_ceil(h as u32, shared_tile), 1, n])
                    .block([shared_block, 1, 1])
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
                    .arg_u32(0)
                    .launch(stream)?;
            }
        } else if down_wide {
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

#[cfg(test)]
mod gate_projection_tests {
    use super::{GateProjection, gate_projection};

    #[test]
    fn kverify_uses_bit_exact_batchm_only_inside_its_proven_width() {
        for n in 2..=8 {
            assert_eq!(gate_projection(n, true), GateProjection::Batchm);
        }
        for n in [0, 1, 9, 16, 32] {
            assert_eq!(gate_projection(n, true), GateProjection::Tiled);
        }
        assert_eq!(gate_projection(4, false), GateProjection::Tiled);
    }
}
