// SPDX-License-Identifier: AGPL-3.0-only
//! Option B: linear Marlin per unique expert. Eager only (D2H topk).

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

use super::NemotronMoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

const GROUP: i32 = 16;
const SMS: u32 = 48;
const SMEM: u32 = 96 * 1024;
const M_MAX: usize = 8;

impl NemotronMoeLayer {
    pub(super) fn decode_batched_marlin_linear(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(m) = self.marlin.as_ref() else {
            bail!("marlin sidecar missing");
        };
        if m.lin_up_k.0 == 0 || m.lin_down_k.0 == 0 {
            bail!("linear Marlin kernels missing");
        }
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

        ctx.gpu.synchronize(stream)?;
        let mut ids = vec![0i32; num_tokens * top_k as usize];
        let mut raw = vec![0u8; ids.len() * 4];
        ctx.gpu.copy_d2h(indices, &mut raw)?;
        for (i, chunk) in raw.chunks_exact(4).enumerate() {
            ids[i] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        let e = m.e as usize;
        let mut slots: Vec<Vec<(usize, usize)>> = vec![Vec::new(); e];
        for t in 0..num_tokens {
            for s in 0..top_k as usize {
                let eid = ids[t * top_k as usize + s];
                if eid >= 0 && (eid as usize) < e {
                    slots[eid as usize].push((t, s));
                }
            }
        }

        let expert_down_out = ctx.buffers.expert_down_out();
        let up_w_b = (m.up_k as usize / 16) * (m.up_n as usize * 16 / 8) * 4;
        let down_w_b = (m.down_k as usize / 16) * (m.down_n as usize * 16 / 8) * 4;
        let up_s_b = (m.up_k as usize / GROUP as usize) * m.up_n as usize;
        let down_s_b = (m.down_k as usize / GROUP as usize) * m.down_n as usize;
        let ng_up = m.up_k / GROUP;
        let ng_dn = m.down_k / GROUP;

        for (eid, hits) in slots.iter().enumerate() {
            if hits.is_empty() {
                continue;
            }
            for chunk in hits.chunks(M_MAX) {
                let mm = chunk.len();
                for (row, &(t, _)) in chunk.iter().enumerate() {
                    ctx.gpu.copy_d2d_async(
                        normed.offset(t * h * bf16),
                        m.a_exp.offset(row * h * bf16),
                        h * bf16,
                        stream,
                    )?;
                }
                ctx.gpu
                    .memset_async(m.locks, 0, SMS as usize * 16, stream)?;
                ops::marlin_nvfp4_m8(
                    ctx.gpu,
                    m.lin_up_k,
                    m.a_exp,
                    DevicePtr(m.up_w.0 + (eid * up_w_b) as u64),
                    m.lin_up_out,
                    m.c_tmp,
                    DevicePtr(m.up_s.0 + (eid * up_s_b) as u64),
                    DevicePtr(m.up_gs.0 + (eid * 4) as u64),
                    m.locks,
                    mm as i32,
                    m.up_n,
                    m.up_k,
                    m.up_k,
                    ng_up,
                    SMS,
                    SMEM,
                    stream,
                )?;
                ops::relu_squared_inplace(
                    ctx.gpu,
                    self.moe_relu2_elementwise_k,
                    m.lin_up_out,
                    (mm as u32) * inter,
                    stream,
                )?;
                ctx.gpu
                    .memset_async(m.locks, 0, SMS as usize * 16, stream)?;
                ops::marlin_nvfp4_m8(
                    ctx.gpu,
                    m.lin_down_k,
                    m.lin_up_out,
                    DevicePtr(m.down_w.0 + (eid * down_w_b) as u64),
                    m.lin_dn_out,
                    m.c_tmp,
                    DevicePtr(m.down_s.0 + (eid * down_s_b) as u64),
                    DevicePtr(m.down_gs.0 + (eid * 4) as u64),
                    m.locks,
                    mm as i32,
                    m.down_n,
                    m.down_k,
                    m.down_k,
                    ng_dn,
                    SMS,
                    SMEM,
                    stream,
                )?;
                for (row, &(t, s)) in chunk.iter().enumerate() {
                    ctx.gpu.copy_d2d_async(
                        m.lin_dn_out.offset(row * h * bf16),
                        expert_down_out.offset((t * top_k as usize + s) * h * bf16),
                        h * bf16,
                        stream,
                    )?;
                }
            }
        }

        let shared_down = ctx.buffers.ssm_deinterleaved();
        ops::relu_squared_inplace(
            ctx.gpu,
            self.moe_relu2_elementwise_k,
            shared_up,
            n * shared_inter,
            stream,
        )?;
        if n <= 4 && self.w4a16_gemv_batch4_k.0 != 0 {
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
