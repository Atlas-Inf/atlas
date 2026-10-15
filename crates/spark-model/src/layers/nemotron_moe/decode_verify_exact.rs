// SPDX-License-Identifier: AGPL-3.0-only

//! Exact batched K-row verify for LatentMoE (`moe_verify_exact` lever).
//!
//! `decode_batched_direct_with_gate_chunks` bails to the per-token
//! `decode_inner` loop whenever `moe_latent_size > 0`, so a K=8 verify pays
//! 8 full passes over the dense projections and re-reads each matched
//! expert's weights per row.
//!
//! Batched exactly:
//!   - dense projections (gate/fc1/fc2) via `dense_gemv_bf16_batchm`
//!   - routed expert up + relu^2-down via `moe_expert_gemv_wide_grouped`
//!     (union grouping — each distinct expert's weights load once across
//!     the K rows; per-row K-order identical, proven byte-identical in
//!     `union_verify_gpu_tests`)
//!   - shared expert **per-row-decode-identical**: `w4a16_decode_gemv` UP
//!     (matching decode's sw/plain selection) plus ONE launch of the fused
//!     `moe_expert_relu2_down_shared` with `slot_base=top_k` and
//!     grid.z=m — the kernel's shared-slot math is unchanged, it just runs
//!     on m rows at once. NOT the w4a16 batch twin: the fused slot's
//!     pair-accumulation differs from w4a16's reduction tree.
//!
//! FP8-native shared checkpoints (Puzzle) keep the w8a16 batch twins
//! instead, matching their own decode path.
//!
//! Readiness (`verify_exact_moe_ready`) requires the batchm dense twin +
//! the grouped kernel + a resolvable shared path (fused relu2-down kernel
//! for NVFP4, FP8 twins for native-FP8 checkpoints).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::NemotronMoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl NemotronMoeLayer {
    /// All kernels the exact batched LatentMoE verify needs are resolved.
    /// The shared expert runs via the NVFP4 batch twin (`w4a16_gemv_batch16`,
    /// proven byte-identical to `w4a16_gemv` at omni dims); the native-FP8
    /// twin is used when the checkpoint actually ships FP8 shared experts.
    /// The routed up/down stage goes through `moe_expert_gemv_wide_grouped`
    /// (union grouping — each distinct expert's weights load once, per-row
    /// accumulation order identical, proven byte-identical to the serial
    /// kernels in `union_verify_gpu_tests`).
    pub(super) fn verify_exact_moe_ready(&self) -> bool {
        // NVFP4 shared: decode-identical — per-row `w4a16_decode_gemv` UP +
        // the shared-slot twin `moe_relu2_down_shared_slot` (grid.z=m).
        let shared_nvfp4 = self.w4a16_gemv_k.0 != 0 && self.relu2_down_shared_slot_k.0 != 0;
        let shared_fp8 = self.w8a16_gemv_batch16_k.0 != 0
            && self.moe_relu2_elementwise_k.0 != 0
            && self.weights.shared_up_fp8.is_some()
            && self.weights.shared_down_fp8.is_some();
        self.dense_gemv_batchm_k.0 != 0
            && self.moe_expert_gemv_wide_grouped_k.0 != 0
            && (shared_nvfp4 || shared_fp8)
    }

    /// K-row LatentMoE verify, byte-identical to `decode_inner` per row.
    /// `hidden`/`residual` are K contiguous `[h]` rows; scratch rows are
    /// carved out of the (prefill-sized) arena buffers, all sized `[m,*]`
    /// with m >> K.
    pub(super) fn decode_verify_exact_latent(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let n = num_tokens as u32;
        let m = num_tokens;
        let inter = self.moe_inter as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = self.top_k as u32;
        let latent = self.moe_latent_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let scale = ctx.config.routed_scaling_factor as f32;
        let bf16 = 2usize;

        let fc1 = self.weights.fc1_latent_proj.as_ref().unwrap();
        let fc2 = self.weights.fc2_latent_proj.as_ref().unwrap();

        // 1. RMS norm over all rows — same elementwise kernel decode uses.
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h,
            eps,
            stream,
        )?;

        // 2. Router gate for all rows in one weight pass (bit-identical twin).
        //    gate_logits rows are `num_experts` BF16 apart.
        let gate_logits = ctx.buffers.gate_logits();
        ops::dense_gemv_batchm(
            ctx.gpu,
            self.dense_gemv_batchm_k,
            normed,
            &self.weights.gate,
            gate_logits,
            n,
            num_experts,
            h,
            num_experts,
            stream,
        )?;

        // 3. Top-k per row (inputs differ; kernel is cheap). The grouped
        //    kernels read expert_indices[tok*top_k + slot] — contiguous — so
        //    indices live in the first m*topk u32s of scratch and weights in
        //    the second m*topk f32s (same buffers decode uses, laid flat).
        let scratch = ctx.buffers.scratch();
        let idx_base = scratch;
        let wgt_base = scratch.offset(m * top_k as usize * 4);
        for t in 0..m {
            ops::moe_topk_sigmoid(
                ctx.gpu,
                self.topk_sigmoid_k,
                gate_logits.offset(t * num_experts as usize * bf16),
                self.weights.e_score_correction_bias.weight,
                idx_base.offset(t * top_k as usize * 4),
                wgt_base.offset(t * top_k as usize * 4),
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                scale,
                stream,
            )?;
        }

        // 4. fc1_latent for all rows: [m, H] -> [m, latent] (bit-identical).
        let latent_buf = ctx.buffers.ssm_ba();
        ops::dense_gemv_batchm(
            ctx.gpu,
            self.dense_gemv_batchm_k,
            normed,
            fc1,
            latent_buf,
            n,
            latent,
            h,
            latent,
            stream,
        )?;

        // 5. Routed experts, union-grouped: each distinct expert across the
        //    K rows streams its weights once; each matched row accumulates in
        //    exactly the serial kernel's K-order (byte-identical — proven in
        //    union_verify_gpu_tests). UP and the relu2-fused DOWN both go
        //    through moe_expert_gemv_wide_grouped.
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_up_out = ctx.buffers.ssm_qkvz();
        let shared_down_out = ctx.buffers.ssm_deinterleaved();
        crate::layers::moe::union_stats::maybe_sample_expert_union(
            ctx,
            idx_base,
            n as usize,
            top_k as usize,
            stream,
        );
        let union = match (
            self.moe_union_match_k.0,
            self.moe_expert_gemv_wide_grouped_u_k.0,
            self.moe_union_scratch,
        ) {
            (0, _, _) | (_, 0, _) | (_, _, None) => None,
            (_, _, Some(u_buf)) => Some(u_buf),
        };
        if let Some(u_buf) = union {
            ops::moe_union_match(
                ctx.gpu,
                self.moe_union_match_k,
                idx_base,
                u_buf,
                n,
                top_k,
                stream,
            )?;
            ops::moe_expert_gemv_wide_grouped_u(
                ctx.gpu,
                self.moe_expert_gemv_wide_grouped_u_k,
                latent_buf,
                self.up_ptrs.packed_ptrs,
                self.up_ptrs.scale_ptrs,
                self.up_ptrs.scale2_vals,
                expert_up_out,
                u_buf,
                inter,
                latent,
                top_k,
                0,
                n,
                false,
                stream,
            )?;
            ops::moe_expert_gemv_wide_grouped_u(
                ctx.gpu,
                self.moe_expert_gemv_wide_grouped_u_k,
                expert_up_out,
                self.down_ptrs.packed_ptrs,
                self.down_ptrs.scale_ptrs,
                self.down_ptrs.scale2_vals,
                expert_down_out,
                u_buf,
                latent,
                inter,
                top_k,
                inter,
                n,
                true,
                stream,
            )?;
        } else {
            ops::moe_expert_gemv_wide_grouped(
                ctx.gpu,
                self.moe_expert_gemv_wide_grouped_k,
                latent_buf,
                self.up_ptrs.packed_ptrs,
                self.up_ptrs.scale_ptrs,
                self.up_ptrs.scale2_vals,
                expert_up_out,
                idx_base,
                inter,
                latent,
                top_k,
                0,
                n,
                num_experts,
                false,
                stream,
            )?;
            ops::moe_expert_gemv_wide_grouped(
                ctx.gpu,
                self.moe_expert_gemv_wide_grouped_k,
                expert_up_out,
                self.down_ptrs.packed_ptrs,
                self.down_ptrs.scale_ptrs,
                self.down_ptrs.scale2_vals,
                expert_down_out,
                idx_base,
                latent,
                inter,
                top_k,
                inter,
                n,
                num_experts,
                true,
                stream,
            )?;
        }

        // 6. Shared expert — per-row exactness: decode computes the shared
        //    DOWN inside `moe_expert_relu2_down_shared`'s extra slot (relu²
        //    in fp32 smem, pair-multiply accumulation, warp reduce), which is
        //    NOT the w4a16 GEMV's math. Reproduce decode's sequence per row:
        //    w4a16_decode_gemv (the same sw/plain selector decode uses) for
        //    UP, then ONE shared-slot launch with grid.z=m —
        //    each z computes the shared slot for that row with identical
        //    per-element math. The FP8-native branch (other checkpoints)
        //        keeps its batched twins, proven byte-identical.
        if let Some(ref fp8_up) = self.weights.shared_up_fp8 {
            let fp8_down = self.weights.shared_down_fp8.as_ref().unwrap();
            ops::w8a16_gemv_batch4_div(
                ctx.gpu,
                self.w8a16_gemv_batch16_k,
                normed,
                fp8_up.weight,
                fp8_up.row_scale,
                shared_up_out,
                n,
                shared_inter,
                h,
                4,
                stream,
            )?;
            KernelLaunch::new(ctx.gpu, self.moe_relu2_elementwise_k)
                .grid([div_ceil(n * shared_inter, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(shared_up_out)
                .arg_u32(n * shared_inter)
                .launch(stream)?;
            ops::w8a16_gemv_batch4_div(
                ctx.gpu,
                self.w8a16_gemv_batch16_k,
                shared_up_out,
                fp8_down.weight,
                fp8_down.row_scale,
                shared_down_out,
                n,
                h,
                shared_inter,
                4,
                stream,
            )?;
        } else {
            // NVFP4 shared: per-row decode-identical sequence.
            for t in 0..m {
                ops::w4a16_decode_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    self.w4a16_gemv_sw_k,
                    ctx.levers.gemv_sw,
                    normed.offset(t * h as usize * 2),
                    &self.weights.shared_up,
                    shared_up_out.offset(t * shared_inter as usize * 2),
                    shared_inter,
                    h,
                    stream,
                )?;
            }
            // One fused launch, shared slot only, one row per grid.z — the
            // kernel's relu² smem stage + pair-accumulation is byte-identical
            // to decode's fused slot.
            KernelLaunch::new(ctx.gpu, self.relu2_down_shared_slot_k)
                .grid([div_ceil(h, 8), 1, m as u32])
                .block([128, 1, 1])
                .shared_mem(shared_inter * 4)
                .arg_ptr(shared_up_out) // sh_up_in [m, shared_inter]
                .arg_ptr(self.weights.shared_down.weight)
                .arg_ptr(self.weights.shared_down.weight_scale)
                .arg_f32(self.weights.shared_down.weight_scale_2)
                .arg_ptr(shared_down_out) // sh_down_out [m, h]
                .arg_u32(h) // N_shared
                .arg_u32(shared_inter) // K_shared
                .launch(stream)?;
        }

        // 7. Weighted sum in latent space, per row: [top_k, L] -> combined[L].
        //    combined rows reuse latent_buf (step-2 inputs already consumed);
        //    dummy shared input zeroed once for all rows.
        let dummy_shared = ctx.buffers.expert_gate_out();
        ctx.gpu
            .memset_async(dummy_shared, 0, m * latent as usize * bf16, stream)?;
        for t in 0..m {
            KernelLaunch::new(ctx.gpu, self.weighted_sum_scale_k)
                .grid([div_ceil(latent, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(latent_buf.offset(t * latent as usize * bf16))
                .arg_ptr(expert_down_out.offset(t * top_k as usize * latent as usize * bf16))
                .arg_ptr(wgt_base.offset(t * top_k as usize * 4))
                .arg_ptr(dummy_shared.offset(t * latent as usize * bf16))
                .arg_u32(latent)
                .arg_u32(top_k)
                .arg_f32(1.0f32)
                .launch(stream)?;
        }

        // 8. fc2_latent for all rows: [m, latent] -> routed_out [m, H].
        let routed_out = ctx.buffers.moe_output();
        ops::dense_gemv_batchm(
            ctx.gpu,
            self.dense_gemv_batchm_k,
            latent_buf,
            fc2,
            routed_out,
            n,
            h,
            latent,
            h,
            stream,
        )?;

        // 9. routed + shared, then residual — batched elementwise adds over
        //    the packed [m, H] block, identical per element to decode's.
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            routed_out,
            shared_down_out,
            n * h,
            stream,
        )?;
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            routed_out,
            n * h,
            stream,
        )
    }
}
