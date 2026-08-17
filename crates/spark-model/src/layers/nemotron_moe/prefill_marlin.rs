// SPDX-License-Identifier: AGPL-3.0-only

//! Marlin MoE prefill (NVIDIA fused_marlin_moe family).
//!
//! Sorted layout from `moe_sort_by_expert` (proven): tokens sorted by expert,
//! contiguous segments. Per expert ONE launch of the plain m8 Marlin kernels
//! (the template's persistent grid tiles prob_m/8 internally, m8 handles the
//! remainder). The activation pack gathers `normed[sorted_token_ids]` into the
//! sorted layout — staged in `expert_down_out` (dead until the DOWN phase).
//!
//! UP:   atlas_marlin_nvfp4_m8        K=2688 %128==0
//! DOWN: atlas_marlin_nvfp4_m8_k64n128 K=1856 %64==0
//!
//! Opt-in: ATLAS_MOE_MARLIN=1 (sidecar + decode path) AND
//! ATLAS_MOE_MARLIN_PREFILL=1. Falls back to the sorted grouped GEMM.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

use super::NemotronMoeLayer;
use super::prefill_sorted::SortedPrefillCtx;
use crate::layer::ForwardContext;
use crate::layers::ops;

const GROUP: i32 = 16;
const SMS: u32 = 48;
const SMEM: u32 = 96 * 1024;

impl NemotronMoeLayer {
    pub(super) fn prefill_marlin_path(
        &self,
        p: &SortedPrefillCtx,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let m = self.marlin.as_ref().expect("marlin sidecar present");
        let te = (p.n * p.top_k) as usize;
        let h = p.h;
        let inter = p.inter as usize;
        let bf16 = 2usize;
        let ne = p.num_experts as usize;

        // Sort arrays live in the gate_logits region (same layout as
        // prefill_sorted): sorted_token_ids @+0 [te], expert_offsets @+te*8.
        let sorted_token_ids = p.gate_logits;
        let sorted_expert_ids = p.gate_logits.offset(te * 4);
        let expert_offsets = p.gate_logits.offset(te * 4 * 2);
        let token_to_perm = p
            .gate_logits
            .offset(te * 4 * 2 + (p.num_experts as usize + 1) * 4);

        // ── 0. Batched routing + sort (the sorted path runs these inside) ──
        KernelLaunch::new(ctx.gpu, self.topk_sigmoid_batched_k)
            .grid([1, p.n, 1])
            .block([256, 1, 1])
            .arg_ptr(p.gate_logits)
            .arg_ptr(self.weights.e_score_correction_bias.weight)
            .arg_ptr(p.indices_dev)
            .arg_ptr(p.weights_dev)
            .arg_u32(p.num_experts)
            .arg_u32(p.top_k)
            .arg_u32(if ctx.config.norm_topk_prob { 1 } else { 0 })
            .arg_f32(p.scale)
            .arg_u32(p.n)
            .launch(stream)?;
        ops::moe_sort_by_expert(
            ctx.gpu,
            self.moe_sort_k,
            p.indices_dev,
            sorted_token_ids,
            sorted_expert_ids,
            expert_offsets,
            token_to_perm,
            te as u32,
            p.num_experts,
            p.top_k,
            stream,
        )?;

        // ── 1. Gather sorted rows into expert_down_out (free until the DOWN) ──
        let packed_a = ctx.buffers.expert_down_out();
        ops::marlin_pack_rows(
            ctx.gpu,
            m.pack_rows_k,
            sorted_token_ids,
            p.normed,
            packed_a,
            te as i32,
            h as i32,
            stream,
        )?;

        // ── 2. D2H expert offsets (eager prefill) ──
        ctx.gpu.synchronize(stream)?;
        let mut off_bytes = vec![0u8; (ne + 1) * 4];
        ctx.gpu.copy_d2h(expert_offsets, &mut off_bytes)?;
        let off: Vec<usize> = off_bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();

        // ── 3. UP per expert ──
        // cfg3-style (128 threads, tm=2 → 32-row tiles) is KEEP: harness A/B
        // exact vs m8 at M=32/64 + offsets. The 256-thread cfg4 in this
        // vendored template is BROKEN (cols 0-15/32-47 of every tile).
        // ATLAS_NO_MOE_MARLIN_PREFILL_CFG4=1 reverts to m8 chunks of 8.
        let use_cfg4 = std::env::var("ATLAS_NO_MOE_MARLIN_PREFILL_CFG4").is_err()
            && m.cfg4_up_k.0 != 0
            && m.cfg4_down_k.0 != 0;
        let up_w_b = (m.up_k as usize / 16) * (m.up_n as usize * 16 / 8) * 4;
        let up_s_b = (m.up_k as usize / GROUP as usize) * m.up_n as usize;
        let ng_up = m.up_k / GROUP;
        let up_out = ctx.buffers.expert_up_out();
        for e in 0..ne {
            let me_ = off[e + 1] - off[e];
            let mut row = 0;
            while row < me_ {
                let (mm, base) = if use_cfg4 {
                    // one launch per expert, padded to a 32-multiple
                    (((me_ + 31) & !31), off[e])
                } else {
                    ((me_ - row).min(8), off[e] + row)
                };
                ctx.gpu
                    .memset_async(m.locks, 0, SMS as usize * 16, stream)?;
                ops::marlin_nvfp4_m8(
                    ctx.gpu,
                    if use_cfg4 { m.cfg4_up_k } else { m.lin_up_k },
                    packed_a.offset(base * h * bf16),
                    DevicePtr(m.up_w.0 + (e * up_w_b) as u64),
                    up_out.offset(base * inter * bf16),
                    m.c_tmp,
                    DevicePtr(m.up_s.0 + (e * up_s_b) as u64),
                    DevicePtr(m.up_gs.0 + (e * 4) as u64),
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
                row += mm;
            }
        }

        // ── 4. relu² over the whole sorted UP output ──
        ops::relu_squared_inplace(
            ctx.gpu,
            self.moe_relu2_elementwise_k,
            up_out,
            (te * inter) as u32,
            stream,
        )?;

        // ── 5. DOWN per expert (k64n128: K=1856 % 64 == 0) ──
        let down_w_b = (m.down_k as usize / 16) * (m.down_n as usize * 16 / 8) * 4;
        let down_s_b = (m.down_k as usize / GROUP as usize) * m.down_n as usize;
        let ng_dn = m.down_k / GROUP;
        let dn_out = ctx.buffers.expert_down_out();
        for e in 0..ne {
            let me_ = off[e + 1] - off[e];
            let mut row = 0;
            while row < me_ {
                let (mm, base) = if use_cfg4 {
                    (((me_ + 31) & !31), off[e])
                } else {
                    ((me_ - row).min(8), off[e] + row)
                };
                ctx.gpu
                    .memset_async(m.locks, 0, SMS as usize * 16, stream)?;
                ops::marlin_nvfp4_m8(
                    ctx.gpu,
                    if use_cfg4 {
                        m.cfg4_down_k
                    } else {
                        m.lin_down_k
                    },
                    up_out.offset(base * inter * bf16),
                    DevicePtr(m.down_w.0 + (e * down_w_b) as u64),
                    dn_out.offset(base * h * bf16),
                    m.c_tmp,
                    DevicePtr(m.down_s.0 + (e * down_s_b) as u64),
                    DevicePtr(m.down_gs.0 + (e * 4) as u64),
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
                row += mm;
            }
        }

        // ── 6. Unpermute + weighted reduce (sorted layout — proven kernel) ──
        let routed_out = ctx.buffers.moe_output();
        let token_to_perm = p.gate_logits.offset(te * 4 * 2 + (ne + 1) * 4);
        ops::moe_unpermute_reduce_indexed(
            ctx.gpu,
            self.moe_unpermute_reduce_k,
            dn_out,
            routed_out,
            token_to_perm,
            p.weights_dev,
            h as u32,
            p.n,
            p.top_k,
            stream,
        )?;

        // ── 7. Shared expert down + blend + residual (mirrors sorted path 5g) ──
        let shared_up_out_base = p.shared_up_out_base;
        let shared_down_out = ctx.buffers.ssm_deinterleaved();
        let shared_relu2_n = p.n * p.shared_inter;
        KernelLaunch::new(ctx.gpu, self.moe_relu2_elementwise_k)
            .grid([
                spark_runtime::kernel_args::div_ceil(shared_relu2_n, 256),
                1,
                1,
            ])
            .block([256, 1, 1])
            .arg_ptr(shared_up_out_base)
            .arg_u32(shared_relu2_n)
            .launch(stream)?;
        let native_down = self
            .weights
            .shared_down_fp8
            .as_ref()
            .filter(|_| self.w8a16_gemm_pipelined_k.0 != 0 || self.w8a16_gemm_k.0 != 0);
        if let Some(fp8w) = native_down {
            let (kern, pipelined) = if self.w8a16_gemm_pipelined_k.0 != 0 {
                (self.w8a16_gemm_pipelined_k, true)
            } else {
                (self.w8a16_gemm_k, false)
            };
            if pipelined {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    kern,
                    shared_up_out_base,
                    fp8w.weight,
                    fp8w.row_scale,
                    shared_down_out,
                    p.n,
                    h as u32,
                    p.shared_inter,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    kern,
                    shared_up_out_base,
                    fp8w.weight,
                    fp8w.row_scale,
                    shared_down_out,
                    p.n,
                    h as u32,
                    p.shared_inter,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                shared_up_out_base,
                &self.weights.shared_down,
                shared_down_out,
                p.n,
                h as u32,
                p.shared_inter,
                stream,
            )?;
        }
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            routed_out,
            shared_down_out,
            (p.num_tokens * h) as u32,
            stream,
        )?;
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            p.hidden,
            routed_out,
            (p.num_tokens * h) as u32,
            stream,
        )?;
        Ok(())
    }
}
