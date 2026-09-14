// SPDX-License-Identifier: AGPL-3.0-only
//! Load-time Marlin sidecar + graph-safe batched decode.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::NemotronMoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

const GROUP: usize = 16;
const SMS: u32 = 48;
const SMEM: u32 = 96 * 1024;
const SORTED_CAP: usize = 2048;

pub(super) struct MarlinSidecar {
    pub up_w: DevicePtr,
    pub up_s: DevicePtr,
    pub up_gs: DevicePtr,
    pub down_w: DevicePtr,
    pub down_s: DevicePtr,
    pub down_gs: DevicePtr,
    pub locks: DevicePtr,
    pub c_tmp: DevicePtr,
    pub sorted_ids: DevicePtr,
    pub expert_ids: DevicePtr,
    pub n_post: DevicePtr,
    pub a_exp: DevicePtr,
    pub moe_up_k: KernelHandle,
    pub moe_down_k: KernelHandle,
    pub lin_up_k: KernelHandle,
    pub lin_down_k: KernelHandle,
    pub cfg4_up_k: KernelHandle,
    pub cfg4_down_k: KernelHandle,
    pub pack_rows_k: KernelHandle,
    pub lin_up_out: DevicePtr,
    pub lin_dn_out: DevicePtr,
    pub pack_k: KernelHandle,
    pub scatter_k: KernelHandle,
    pub slot_up_k: KernelHandle,
    pub slot_dn_k: KernelHandle,
    pub slot_eids: DevicePtr,
    pub slot_map: DevicePtr,
    pub slot_a: DevicePtr,
    pub slot_up: DevicePtr,
    pub slot_dn: DevicePtr,
    pub slot_bars: DevicePtr,
    pub align_k: KernelHandle,
    pub repeat_k: KernelHandle,
    pub up_n: i32,
    pub up_k: i32,
    pub down_n: i32,
    pub down_k: i32,
    pub e: i32,
}

mod build;

impl NemotronMoeLayer {
    pub(super) fn decode_batched_marlin(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if std::env::var_os("ATLAS_MOE_MARLIN_LINEAR").is_some() {
            return self.decode_batched_marlin_linear(hidden, residual, num_tokens, ctx, stream);
        }
        if std::env::var_os("ATLAS_MOE_MARLIN_GROUPED").is_none() {
            return self.decode_batched_marlin_slots(hidden, residual, num_tokens, ctx, stream);
        }
        let Some(m) = self.marlin.as_ref() else {
            bail!("marlin sidecar missing");
        };
        let h = ctx.config.hidden_size;
        let n = num_tokens as u32;
        let top_k = self.top_k as u32;
        let inter = self.moe_inter as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let scale = ctx.config.routed_scaling_factor as f32;
        let bf16 = 2usize;
        let num_experts = ctx.config.num_experts as u32;

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

        ops::marlin_align_block8(
            ctx.gpu,
            m.align_k,
            indices,
            m.sorted_ids,
            m.expert_ids,
            m.n_post,
            n as i32,
            top_k as i32,
            m.e,
            SORTED_CAP as i32,
            stream,
        )?;
        KernelLaunch::new(ctx.gpu, m.repeat_k)
            .grid([n, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(normed)
            .arg_ptr(m.a_exp)
            .arg_i32(n as i32)
            .arg_i32(h as i32)
            .arg_i32(top_k as i32)
            .launch(stream)?;

        let expert_up_out = ctx.buffers.expert_up_out();
        let ng_up = m.up_k / GROUP as i32;
        ctx.gpu.memset_async(m.locks, 0, 48 * 16, stream)?;
        ops::marlin_moe_nvfp4(
            ctx.gpu,
            m.moe_up_k,
            m.a_exp,
            m.up_w,
            expert_up_out,
            m.c_tmp,
            m.up_s,
            m.up_gs,
            m.sorted_ids,
            m.expert_ids,
            m.n_post,
            top_k as i32,
            ng_up,
            n as i32,
            m.up_n,
            m.up_k,
            m.locks,
            stream,
        )?;
        ops::relu_squared_inplace(
            ctx.gpu,
            self.moe_relu2_elementwise_k,
            expert_up_out,
            n * top_k * inter,
            stream,
        )?;

        let expert_down_out = ctx.buffers.expert_down_out();
        let ng_dn = m.down_k / GROUP as i32;
        ctx.gpu.memset_async(m.locks, 0, 48 * 16, stream)?;
        ops::marlin_moe_nvfp4(
            ctx.gpu,
            m.moe_down_k,
            expert_up_out,
            m.down_w,
            expert_down_out,
            m.c_tmp,
            m.down_s,
            m.down_gs,
            m.sorted_ids,
            m.expert_ids,
            m.n_post,
            top_k as i32,
            ng_dn,
            n as i32,
            m.down_n,
            m.down_k,
            m.locks,
            stream,
        )?;

        // shared down via existing wide kernel on the shared slot only is hard;
        // run the live shared-down half by calling relu2_down on shared_up only
        // through the existing fused kernel with top_k=0? Use w4a16 on shared.
        let shared_down = ctx.buffers.ssm_deinterleaved();
        if n <= 4 && self.w4a16_gemv_batch4_k.0 != 0 {
            // relu2 shared_up then down
            ops::relu_squared_inplace(
                ctx.gpu,
                self.moe_relu2_elementwise_k,
                shared_up,
                n * shared_inter,
                stream,
            )?;
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                self.w4a16_gemv_batch4_k,
                shared_up,
                &self.weights.shared_down,
                shared_down,
                n,
                h as u32,
                shared_inter,
                stream,
            )?;
        } else {
            ops::relu_squared_inplace(
                ctx.gpu,
                self.moe_relu2_elementwise_k,
                shared_up,
                n * shared_inter,
                stream,
            )?;
            self.prefill_shared_up(
                shared_up,
                shared_down,
                n,
                shared_inter as usize,
                h as u32,
                ctx,
                stream,
            )?;
        }

        let output = ctx.buffers.moe_output();
        for t in 0..num_tokens {
            let off_h = t * h * bf16;
            KernelLaunch::new(ctx.gpu, self.weighted_sum_scale_k)
                .grid([spark_runtime::kernel_args::div_ceil(h as u32, 256), 1, 1])
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
