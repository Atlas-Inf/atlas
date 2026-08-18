// SPDX-License-Identifier: AGPL-3.0-only
//! Graph-safe linear Marlin: device pack unique experts into MARLIN_SLOTS.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

use super::NemotronMoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

const GROUP: i32 = 16;
const SMS: u32 = 48;
const SMEM: u32 = 96 * 1024;
const SLOTS: i32 = ops::MARLIN_SLOTS;
const M_TILE: i32 = ops::MARLIN_M_TILE;

impl NemotronMoeLayer {
    pub(super) fn decode_batched_marlin_slots(
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
        if m.pack_k.0 == 0 || m.slot_up_k.0 == 0 || m.slot_dn_k.0 == 0 || m.scatter_k.0 == 0 {
            bail!("slot Marlin kernels missing");
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
        let up_only = std::env::var("ATLAS_MOE_MARLIN_UP_ONLY").as_deref() == Ok("1");
        let down_only = std::env::var("ATLAS_MOE_MARLIN_DOWN_ONLY").as_deref() == Ok("1");
        anyhow::ensure!(
            !(up_only && down_only),
            "Marlin UP-only and DOWN-only are mutually exclusive"
        );
        if down_only {
            anyhow::ensure!(m.gather_k.0 != 0, "Marlin DOWN-only gather kernel missing");
        }
        // KEEP shape is n<=4 (C=4 AR, C=1 DSpark verify). One launch over
        // n=8 (~41 unique) still garbles with 64 slots + zeroed scatter.
        // Wave the proven tile; shared expert stays one n-wide GEMM.
        const WAVE: usize = 4;

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

        let expert_down_out = ctx.buffers.expert_down_out();
        ctx.gpu.memset_async(
            expert_down_out,
            0,
            num_tokens * top_k as usize * h * bf16,
            stream,
        )?;
        let ng_up = m.up_k / GROUP;
        let ng_dn = m.down_k / GROUP;
        let mut wave_off = 0usize;
        while wave_off < num_tokens {
            let cn = (num_tokens - wave_off).min(WAVE);
            ctx.gpu
                .memset_async(m.slot_a, 0, (SLOTS * M_TILE) as usize * h * bf16, stream)?;
            ops::marlin_pack_slots(
                ctx.gpu,
                m.pack_k,
                indices.offset(wave_off * top_k as usize * 4),
                normed.offset(wave_off * h * bf16),
                m.slot_eids,
                m.slot_map,
                m.slot_a,
                m.n_post,
                cn as i32,
                top_k as i32,
                m.e,
                h as i32,
                stream,
            )?;
            if down_only {
                let expert_up_out = ctx.buffers.expert_up_out();
                let wave_up =
                    expert_up_out.offset(wave_off * top_k as usize * inter as usize * bf16);
                ops::moe_expert_gemv_wide(
                    ctx.gpu,
                    self.moe_expert_gemv_wide_k,
                    normed.offset(wave_off * h * bf16),
                    self.up_ptrs.packed_ptrs,
                    self.up_ptrs.scale_ptrs,
                    self.up_ptrs.scale2_vals,
                    wave_up,
                    indices.offset(wave_off * top_k as usize * 4),
                    inter,
                    h as u32,
                    top_k,
                    0,
                    cn as u32,
                    stream,
                )?;
                ops::relu_squared_inplace(
                    ctx.gpu,
                    self.moe_relu2_elementwise_k,
                    wave_up,
                    (cn as u32) * top_k * inter,
                    stream,
                )?;
                ops::marlin_gather_slots(
                    ctx.gpu,
                    m.gather_k,
                    wave_up,
                    m.slot_map,
                    m.slot_up,
                    inter as i32,
                    stream,
                )?;
            } else {
                ctx.gpu
                    .memset_async(m.locks, 0, (SLOTS as usize) * 256 * 4, stream)?;
                ctx.gpu
                    .memset_async(m.slot_bars, 0, (SLOTS as usize) * 4, stream)?;
                ops::marlin_nvfp4_m8_allslots(
                    ctx.gpu,
                    m.slot_up_k,
                    m.slot_a,
                    m.up_w,
                    m.slot_up,
                    m.c_tmp,
                    m.up_s,
                    m.up_gs,
                    m.slot_eids,
                    m.n_post,
                    m.slot_bars,
                    ng_up,
                    M_TILE,
                    m.up_n,
                    m.up_k,
                    m.up_k,
                    m.locks,
                    SMS,
                    SMEM,
                    stream,
                )?;
                ops::relu_squared_inplace(
                    ctx.gpu,
                    self.moe_relu2_elementwise_k,
                    m.slot_up,
                    (SLOTS * M_TILE) as u32 * inter,
                    stream,
                )?;
            }
            if up_only {
                let expert_up_out = ctx.buffers.expert_up_out();
                let wave_up =
                    expert_up_out.offset(wave_off * top_k as usize * inter as usize * bf16);
                ops::marlin_scatter_slots(
                    ctx.gpu,
                    m.scatter_k,
                    m.slot_up,
                    m.slot_map,
                    wave_up,
                    inter as i32,
                    stream,
                )?;
                ops::moe_expert_gemv_wide(
                    ctx.gpu,
                    self.moe_expert_gemv_wide_k,
                    wave_up,
                    self.down_ptrs.packed_ptrs,
                    self.down_ptrs.scale_ptrs,
                    self.down_ptrs.scale2_vals,
                    expert_down_out.offset(wave_off * top_k as usize * h * bf16),
                    indices.offset(wave_off * top_k as usize * 4),
                    h as u32,
                    inter,
                    top_k,
                    inter,
                    cn as u32,
                    stream,
                )?;
            } else {
                ctx.gpu
                    .memset_async(m.locks, 0, (SLOTS as usize) * 256 * 4, stream)?;
                ctx.gpu
                    .memset_async(m.slot_bars, 0, (SLOTS as usize) * 4, stream)?;
                ops::marlin_nvfp4_m8_allslots(
                    ctx.gpu,
                    m.slot_dn_k,
                    m.slot_up,
                    m.down_w,
                    m.slot_dn,
                    m.c_tmp,
                    m.down_s,
                    m.down_gs,
                    m.slot_eids,
                    m.n_post,
                    m.slot_bars,
                    ng_dn,
                    M_TILE,
                    m.down_n,
                    m.down_k,
                    m.down_k,
                    m.locks,
                    SMS,
                    SMEM,
                    stream,
                )?;
                ops::marlin_scatter_slots(
                    ctx.gpu,
                    m.scatter_k,
                    m.slot_dn,
                    m.slot_map,
                    expert_down_out.offset(wave_off * top_k as usize * h * bf16),
                    h as i32,
                    stream,
                )?;
            }
            wave_off += cn;
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
