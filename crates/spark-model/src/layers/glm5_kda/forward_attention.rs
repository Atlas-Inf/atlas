// SPDX-License-Identifier: AGPL-3.0-only

//! Attention through the FFN input normalization; the caller resumes FFN immediately.

use super::*;

impl Glm5KdaLayer {
    /// The attention body over `tokens` rows. Everything here is row-local
    /// except `recurrent`, which advances whichever recurrent state(s) own the
    /// rows and returns the core output `[tokens, heads*dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_attention_rows(
        &self,
        hidden: DevicePtr,
        tokens: usize,
        decode: bool,
        capture_verify_intermediates: bool,
        ctx: &ForwardContext,
        stream: u64,
        recurrent: &mut dyn FnMut(DevicePtr, DevicePtr, DevicePtr) -> Result<DevicePtr>,
    ) -> Result<FfnPhase> {
        let m = tokens as u32;
        let h = self.hidden_size as u32;
        let p = self.heads * self.dim;
        // ATLAS_GLM_CANONICAL_VERIFY: one kernel family per projection for
        // every row count, K=5 seams off; a plain one-row decode takes the
        // verify projections too, so its row matches the same row verified.
        let canonical = crate::layers::canonical_verify::enabled();
        let verify_proj = capture_verify_intermediates || (decode && canonical);
        let bf16 = 2usize;
        let mut profile_timer = profile::start(ctx, stream)?;
        // Sequence-parallel prefill: the highway and `hidden` hold this rank's
        // rows compacted at row 0 (`layers::glm_sp`); `m_hc` is their count.
        let sp = crate::layers::glm_sp::current()
            .filter(|sp| !decode && !capture_verify_intermediates && tokens == sp.total());
        let m_hc = sp.map_or(m, |sp| sp.rows as u32);
        let local = |x: DevicePtr| sp.map_or(x, |sp| sp.local(x, self.hidden_size));

        if self.layer_idx == 0 {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                local(hidden),
                ctx.buffers.hc_streams(),
                m_hc,
                h,
                self.hc.hc_mult as u32,
                stream,
            )?;
        }
        // ATLAS_GLM_HC_PREFILL_FUSE2: a prefill's pre-site finalizers also
        // write the block input norms.
        let normed = ctx.buffers.norm_output();
        let fuse_norm = |weight: &DenseWeight| {
            (!decode
                && !capture_verify_intermediates
                && crate::layers::qwen3_attention::hc_prefill_fuse2_enabled())
            .then_some(crate::layers::qwen3_attention::HcNorm {
                weight: weight.weight,
                out: local(normed),
            })
        };
        // ATLAS_GLM_DET_TRACE stages; `det_rows` are the seam (local) rows.
        let det = crate::det_trace::on_stream(ctx.gpu, stream);
        let (det_rows, row) = (
            (sp.map_or(0, |sp| sp.row0), m_hc as usize),
            h as usize * bf16,
        );
        // ATLAS_GLM_SP_GATHER_OVERLAP: pre, norm and gather a row block at a time.
        let overlapped = self.sp_seam_overlapped(
            &self.hc.attn,
            &self.input_norm,
            (false, true),
            hidden,
            normed,
            sp,
            ctx,
            stream,
        )?;
        if !overlapped {
            let attn_normed = self.hc_pre(
                &self.hc.attn,
                hidden,
                m_hc,
                fuse_norm(&self.input_norm),
                ctx,
                stream,
            )?;
            det.tap("in", hidden, det_rows, row);
            if !attn_normed {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    hidden,
                    &self.input_norm,
                    local(normed),
                    m_hc,
                    h,
                    ctx.config.rms_norm_eps as f32,
                    stream,
                )?;
            }
            if let Some(sp) = sp {
                sp.all_gather(normed, self.hidden_size, ctx, stream)?;
            }
        }
        profile::step(ctx, stream, &mut profile_timer, "hc_attn_norm")?;

        let projected = ctx.buffers.qkv_output();
        let plane_bytes = tokens * p * bf16;
        self.forward_attention_qkv(
            normed,
            projected,
            plane_bytes,
            m,
            p,
            h,
            decode,
            verify_proj,
            &mut profile_timer,
            ctx,
            stream,
        )?;
        det.tap("x_qkv", projected, (0, tokens), 3 * p * bf16);
        let (beta, g1, g2) = self.forward_side_projections(
            normed,
            projected,
            plane_bytes,
            m,
            decode,
            verify_proj,
            capture_verify_intermediates,
            &mut profile_timer,
            ctx,
            stream,
        )?;
        det.tap("x_g", g1, (0, tokens), 2 * p * bf16);

        let core_out = recurrent(projected, g1, beta)?;
        profile::step(ctx, stream, &mut profile_timer, "pack_conv")?;
        profile::step(ctx, stream, &mut profile_timer, "recurrent")?;
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
        profile::step(ctx, stream, &mut profile_timer, "gated_norm")?;
        det.tap("x_gated", gated, (0, tokens), p * bf16);
        if verify_proj {
            self.project_hot_verify(
                gated,
                &self.weights.o_proj,
                normed,
                m,
                h,
                p as u32,
                ctx,
                stream,
            )?;
        } else {
            self.project_hot(
                gated,
                &self.weights.o_proj,
                normed,
                m,
                h,
                p as u32,
                decode,
                ctx,
                stream,
            )?;
        }
        profile::step(ctx, stream, &mut profile_timer, "o_proj")?;
        det.tap("attn", normed, (0, tokens), row);
        let fused_tp_hc = capture_verify_intermediates
            && !canonical
            && tokens == 5
            && !ctx.graph_capture
            && verify_fused_tp_hc_enabled()
            && self.hc_post_bf16_add_k.0 != 0
            && ctx.config.tp_world_size == 2
            && ctx
                .comm
                .is_some_and(|comm| comm.world_size() == 2 && comm.supports_peer_exchange_async());
        // ATLAS_GLM_SP_GATHER_OVERLAP: reduce, post + pre, norm and (MoE)
        // gather a row block at a time; all of it is timed as `tp_reduce`.
        let moe = matches!(self.ffn, crate::layers::FfnComponent::Moe(_));
        let overlapped = !fused_tp_hc
            && self.sp_seam_overlapped(
                &self.hc.ffn,
                &self.post_attn_norm,
                (true, moe),
                hidden,
                normed,
                sp,
                ctx,
                stream,
            )?;
        if overlapped {
        } else if fused_tp_hc {
            // `moe_output` is a registered, caller-owned [M,H] BF16 arena.
            // The preceding KDA work is done with it, and FFN overwrites it
            // only after this immediate mHC post consumes the peer payload.
            ctx.comm.unwrap().peer_exchange_async(
                normed.0,
                ctx.buffers.moe_output().0,
                tokens * self.hidden_size * 2,
                stream,
            )?;
        } else if let Some(sp) = sp {
            sp.reduce_scatter(normed, self.hidden_size, ctx, stream)?;
        } else if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(normed.0, tokens * self.hidden_size * 2, stream)?;
        }
        profile::step(ctx, stream, &mut profile_timer, "tp_reduce")?;
        if !overlapped {
            if !fused_tp_hc {
                det.tap("attn_red", local(normed), det_rows, row);
            }
            let mut seam = crate::layers::qwen3_attention::HcSeam::Skipped;
            if fused_tp_hc {
                ops::hc_post_bf16_add(
                    ctx.gpu,
                    self.hc_post_bf16_add_k,
                    normed,
                    ctx.buffers.moe_output(),
                    ctx.buffers.hc_streams(),
                    ctx.buffers.hc_post(),
                    ctx.buffers.hc_comb(),
                    ctx.buffers.hc_streams(),
                    m,
                    h,
                    self.hc.hc_mult as u32,
                    stream,
                )?;
            } else {
                seam = crate::layers::qwen3_attention::hc_post_pre_prefill_fused(
                    &self.hc.ffn,
                    Some(local(normed)),
                    hidden,
                    m_hc,
                    self.hc.hc_mult as u32,
                    self.hc.sinkhorn_iters as u32,
                    self.hc.hc_eps,
                    fuse_norm(&self.post_attn_norm),
                    ctx,
                    stream,
                )?;
                if !seam.ran() {
                    self.hc_post(local(normed), m_hc, ctx, stream)?;
                }
            }
            profile::step(ctx, stream, &mut profile_timer, "hc_attn_post")?;

            let ffn_normed = match seam.ran() {
                true => seam.normed(),
                false => self.hc_pre(
                    &self.hc.ffn,
                    hidden,
                    m_hc,
                    fuse_norm(&self.post_attn_norm),
                    ctx,
                    stream,
                )?,
            };
            if !ffn_normed {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    hidden,
                    &self.post_attn_norm,
                    local(normed),
                    m_hc,
                    h,
                    ctx.config.rms_norm_eps as f32,
                    stream,
                )?;
            }
            // The routed MoE needs every row; a dense FFN runs only the local ones.
            if let Some(sp) = sp.filter(|_| moe) {
                sp.all_gather(normed, self.hidden_size, ctx, stream)?;
            }
        }
        profile::step(ctx, stream, &mut profile_timer, "hc_ffn_norm")?;
        Ok(FfnPhase {
            hidden,
            normed,
            tokens,
            decode,
            capture_verify_intermediates,
            profile_timer,
        })
    }
}
