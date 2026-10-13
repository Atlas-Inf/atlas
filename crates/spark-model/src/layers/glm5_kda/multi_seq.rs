// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 KDA multi-sequence decode.
//!
//! The large stateless projections are evaluated once for the whole decode
//! batch. Convolution and recurrence use validated device-indexed pools when
//! available, retaining the per-sequence fallback. At width three, the FFN can also use Atlas's existing
//! grouped path; width two remains sequential because measurements on GB10
//! show no benefit there.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::Glm5KdaLayer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;

/// `ATLAS_GLM_KDA_MULTI_SEQ=1`: a 2- or 3-sequence decode step runs this
/// layer once over every row (one TP reduce of all rows) instead of once a
/// sequence. Both ranks must run the same value (`model::startup_parity`).
pub(crate) fn enabled() -> bool {
    std::env::var("ATLAS_GLM_KDA_MULTI_SEQ").ok().as_deref() == Some("1")
}

impl Glm5KdaLayer {
    pub(super) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let independent = crate::model::glm_independent::selected(ctx, num_seqs)?;
        ensure!(
            independent
                || crate::model::glm_c4::batched_kda_rows(
                    num_seqs,
                    enabled(),
                    crate::model::glm_c4::enabled(&ctx.config.model_type),
                )?,
            "GLM KDA batched decode supports C2/C3 and opted-in C4"
        );
        ensure!(states.len() >= num_seqs, "GLM KDA state batch is truncated");
        if independent {
            crate::model::glm_independent::validate_runtime(
                ctx.config,
                ctx.comm.map_or(0, |c| c.world_size()),
                crate::model::ep_protocol_v2_requested(),
                true,
            )?;
            crate::model::glm_independent::validate_scratch(
                ctx.buffers.sizes(),
                self.hc.hc_mult,
                num_seqs,
            )?;
        } else if num_seqs == 4 {
            crate::model::glm_c4::validate_runtime(
                ctx.config,
                ctx.comm.map_or(0, |comm| comm.world_size()),
                crate::model::ep_protocol_v2_requested(),
                true,
            )?;
            ensure!(
                self.hidden_size == 4096 && self.heads == 32 && self.dim == 128,
                "GLM C4 requires local KDA P4096/dim128"
            );
            crate::model::glm_c4::validate_projection_handles(
                self.w4a16_gemv_batchm.kernel(4).0,
                self.dense_gemv_batchm_k.0,
            )?;
            crate::model::glm_c4::validate_scratch(ctx.buffers.sizes(), self.hc.hc_mult)?;
        }
        for state in states.iter().take(num_seqs) {
            let state = state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("GLM KDA requires independent SSM states"))?;
            ensure!(!state.h_is_f16, "GLM KDA requires FP32 recurrent state");
        }
        let indexed_core = self.prepare_indexed_core(num_seqs, ctx)?;
        ensure!(
            !independent || indexed_core.is_some(),
            "independent KDA must use actual indexed state pair"
        );

        let n = num_seqs;
        let m = n as u32;
        let h = self.hidden_size;
        let h_u32 = h as u32;
        let p = self.heads * self.dim;
        let p_u32 = p as u32;
        let bf16 = 2usize;
        let profile = std::env::var("ATLAS_GLM_KDA_MS_PROFILE").ok().as_deref() == Some("1")
            && !ctx.graph_capture;
        let mixer_t0 = if profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // mHC is batch-aware. Keeping the highway in [N, hc, H] layout also
        // fixes the generic fallback's scratch reuse between distinct rows.
        if self.layer_idx == 0 {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                ctx.buffers.hc_streams(),
                m,
                h_u32,
                self.hc.hc_mult as u32,
                stream,
            )?;
        }
        self.hc_pre(&self.hc.attn, hidden, m, ctx, stream)?;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.input_norm,
            normed,
            m,
            h_u32,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;

        // Q/K/V/O are the dominant KDA weight streams. The batch2/3 GEMV
        // kernels preserve each row's scalar accumulation order while reading
        // the NVFP4 weights once.
        let projected = ctx.buffers.qkv_output();
        let plane_bytes = n * p * bf16;
        self.project_hot_multi_decode(
            normed,
            &self.weights.q_proj,
            projected,
            m,
            p_u32,
            h_u32,
            ctx,
            stream,
        )?;
        self.project_hot_multi_decode(
            normed,
            &self.weights.k_proj,
            projected.offset(plane_bytes),
            m,
            p_u32,
            h_u32,
            ctx,
            stream,
        )?;
        self.project_hot_multi_decode(
            normed,
            &self.weights.v_proj,
            projected.offset(2 * plane_bytes),
            m,
            p_u32,
            h_u32,
            ctx,
            stream,
        )?;

        let beta = projected.offset(3 * plane_bytes);
        self.project_dense_multi_decode(
            normed,
            &self.weights.b_proj,
            beta,
            m,
            self.heads as u32,
            h_u32,
            ctx,
            stream,
        )?;
        let fa = beta.offset(n * self.heads * bf16);
        self.project_dense_multi_decode(
            normed,
            &self.weights.f_a_proj,
            fa,
            m,
            self.dim as u32,
            h_u32,
            ctx,
            stream,
        )?;
        let ga = fa.offset(n * self.dim * bf16);
        self.project_dense_multi_decode(
            normed,
            &self.weights.g_a_proj,
            ga,
            m,
            self.dim as u32,
            h_u32,
            ctx,
            stream,
        )?;

        let g1 = ctx.buffers.ssm_deinterleaved();
        self.project_dense_multi_decode(
            fa,
            &self.weights.f_b_proj,
            g1,
            m,
            p_u32,
            self.dim as u32,
            ctx,
            stream,
        )?;
        let g2 = g1.offset(plane_bytes);
        self.project_dense_multi_decode(
            ga,
            &self.weights.g_b_proj,
            g2,
            m,
            p_u32,
            self.dim as u32,
            ctx,
            stream,
        )?;

        let packed = ctx.buffers.ssm_qkvz();
        ops::kda_pack_qkv(ctx.gpu, self.pack_k, projected, packed, m, p_u32, stream)?;

        // Recurrent state is sequence-private. Only submission of this core
        // changes; all projections and subsequent normalization stay identical.
        let convolved = ctx.buffers.ssm_conv_out_f32();
        let core_out = ctx.buffers.attn_output();
        let packed_row_bytes = 3 * p * bf16;
        let gate_row_bytes = p * bf16;
        let beta_row_bytes = self.heads * bf16;
        if let Some(core) = indexed_core {
            self.run_indexed_core(core, ctx, stream)?;
        } else {
            for i in 0..n {
                let state = states[i]
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| {
                        anyhow::anyhow!("GLM-5 KDA expected SsmLayerState for row {i}")
                    })?;
                ensure!(!state.h_is_f16, "GLM-5 KDA requires FP32 recurrent state");
                ops::conv1d_update_prefill(
                    ctx.gpu,
                    self.conv_prefill_k,
                    self.conv_prefill_tp_k,
                    state.conv_state,
                    packed.offset(i * packed_row_bytes),
                    &self.weights.conv,
                    DevicePtr::NULL,
                    convolved.offset(i * packed_row_bytes),
                    (3 * p) as u32,
                    self.conv_width as u32,
                    1,
                    (3 * p) as u32,
                    (3 * p) as u32,
                    stream,
                )?;
                self.run_recurrent(
                    convolved.offset(i * packed_row_bytes),
                    g1.offset(i * gate_row_bytes),
                    beta.offset(i * beta_row_bytes),
                    state.h_state,
                    core_out.offset(i * gate_row_bytes),
                    1,
                    true,
                    ctx,
                    stream,
                )?;
            }
        }

        let gated = projected;
        ops::kda_sigmoid_gated_norm(
            ctx.gpu,
            self.gated_norm_k,
            core_out,
            g2,
            self.weights.o_norm.weight,
            gated,
            m,
            self.heads as u32,
            self.dim as u32,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        self.project_hot_multi_decode(
            gated,
            &self.weights.o_proj,
            normed,
            m,
            h_u32,
            p_u32,
            ctx,
            stream,
        )?;
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(normed.0, n * h * bf16, stream)?;
        }
        self.hc_post(normed, m, ctx, stream)?;

        let mixer_us = if let Some(t0) = mixer_t0 {
            ctx.gpu.synchronize(stream)?;
            t0.elapsed().as_micros()
        } else {
            0
        };
        let ffn_t0 = if profile {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Keep mixer and FFN batching independently switchable for diagnosis.
        // Atlas's K=3 FFN path emits a contiguous [N,H] result, so mHC can
        // consume the entire batch once.
        self.hc_pre(&self.hc.ffn, hidden, m, ctx, stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.post_attn_norm,
            normed,
            m,
            h_u32,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        let compact_c2 = if independent || crate::model::glm_independent::ffn_rows_selected(ctx, n)?
        {
            Some(self.ffn.forward_independent(normed, n, ctx, stream)?)
        } else if n == 2 {
            self.ffn.try_forward_c2_compact(normed, ctx, stream)?
        } else {
            None
        };
        if let Some(output) = compact_c2 {
            self.hc_post(output, m, ctx, stream)?;
        } else if n == 4 {
            let output = self.ffn.forward_c4(normed, ctx, stream)?;
            self.hc_post(output, m, ctx, stream)?;
        } else if super::kda_batched_ffn_enabled() && n == 3 {
            self.ffn.forward_k3(normed, ctx, stream)?;
            self.hc_post(ctx.buffers.moe_output(), m, ctx, stream)?;
        } else {
            // Conservative fallback: consume shared MoE output before the
            // next row overwrites it.
            for i in 0..n {
                let ffn_out = self.ffn.forward(normed.offset(i * h * bf16), ctx, stream)?;
                let hc_streams_i = ctx.buffers.hc_streams().offset(
                    i * self.hc.hc_mult * h * super::super::ops::hc_elem_bytes("glm5_next"),
                );
                let post_i = ctx
                    .buffers
                    .hc_post()
                    .offset(i * self.hc.hc_mult * size_of::<f32>());
                let comb_i = ctx
                    .buffers
                    .hc_comb()
                    .offset(i * self.hc.hc_mult * self.hc.hc_mult * size_of::<f32>());
                ops::hc_post(
                    ctx.gpu,
                    self.hc_post_k,
                    ffn_out,
                    hc_streams_i,
                    post_i,
                    comb_i,
                    hc_streams_i,
                    1,
                    h_u32,
                    self.hc.hc_mult as u32,
                    stream,
                )?;
            }
        }

        if self.layer_idx + 1 == ctx.config.num_hidden_layers {
            ops::hc_contract(
                ctx.gpu,
                self.hc_contract_k,
                ctx.buffers.hc_streams(),
                hidden,
                m,
                h_u32,
                self.hc.hc_mult as u32,
                stream,
            )?;
        }

        if let Some(t0) = ffn_t0 {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "ATLAS_GLM_KDA_MS_PROFILE n={n} layer={} mixer={}us ffn={}us",
                self.layer_idx,
                mixer_us,
                t0.elapsed().as_micros(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::enabled;

    #[test]
    fn glm_kda_multiseq_is_experimentally_gated() {
        // The environment is intentionally not mutated in parallel tests;
        // absence is the conservative default.
        if std::env::var_os("ATLAS_GLM_KDA_MULTI_SEQ").is_none() {
            assert!(!enabled());
        }
    }
}
