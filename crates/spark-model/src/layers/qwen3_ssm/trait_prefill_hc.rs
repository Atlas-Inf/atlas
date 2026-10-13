// SPDX-License-Identifier: AGPL-3.0-only

//! The GDN prefill under an mHC highway (Qwen3.8-Flash-Next).
//!
//! Not a wrapper around `prefill_inner`, for a structural reason.
//! `prefill_inner` FUSES its residual bookkeeping into its norms:
//!
//! ```text
//! rms_norm_residual(hidden, input_norm) -> normed, residual      # step 1
//!   prefill_block(normed)               -> out_proj_buf          # steps 2-10
//! residual_add_rms_norm(hidden, out_proj_buf, post_attn_norm)    # step 11
//! ffn.forward_prefill(norm_output)                               # step 12
//! residual_add(hidden, moe_output)                               # step 13
//! ```
//!
//! Under mHC the highway IS the residual, and the block output has to reach
//! it through `hc_post` — scaled per stream by the injection vector that
//! `hc_pre` emitted. Steps 1, 11 and 13 would each add it a second time. So
//! this path replaces them rather than running them, and the shape below
//! mirrors `qwen3_attention`'s `prefill_inner_hc` exactly:
//!
//! ```text
//! hc_expand(hidden -> streams)                    # MODEL layer 0 only
//! hc_pre(streams, attn_site) -> hidden, inj       # `hidden` is scratch here
//!   prefill_block(hidden)    -> out_proj_buf
//! hc_post(out_proj_buf, streams, inj) -> streams
//! hc_pre(streams, ffn_site)  -> hidden, inj
//!   ffn.forward_prefill(hidden) -> moe_output
//! hc_post(moe_output, streams, inj) -> streams
//! ```
//!
//! No `input_norm`, no `post_attn_norm`, no `residual_add`: on this model
//! those slots hold ones-filled placeholders because the checkpoint has no
//! per-layer norms at all — `hc_norm` inside `hc_pre` is the norm.
//!
//! ## Why `hc_expand` lives here
//!
//! It used to fire on `attn_layer_idx == 0`. With `layer_types` interleaving
//! 3:1, model layers 0-2 are GDN and attention layer 0 is model layer 3 — so
//! the highway was being seeded three layers late, on top of whatever the
//! buffer held. The seeding layer and the collapsing layer are different
//! concrete types on this model, and this is the seeding one.
//!
//! There is no `hc_head` here: the LAST model layer (47) is full attention,
//! so the final collapse stays on the attention path.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_inner_hc(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        seq_len_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill_inner_hc without mHC weights"))?;
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        // Sequence-parallel prefill (`layers::glm_sp`, qwen4_exp): the mHC
        // seams run this rank's `n` rows (compacted at row 0 of the highway
        // and of `post`; the collapses write `hidden` at the local rows,
        // which are then all-gathered); the GDN block and the MoE keep every
        // row and reduce-scatter.
        let sp = crate::layers::glm_sp::current().filter(|sp| sp.total() == num_tokens);
        let n = sp.map_or(num_tokens, |sp| sp.rows) as u32;
        let local = |x: DevicePtr| sp.map_or(x, |sp| sp.local(x, h));

        // Same counter the non-HC path bumps: one increment per SSM layer per
        // prefill, so `ATLAS_GDN_DUMP` still attributes an intermediate to the
        // right layer.
        let ssm_layer_idx =
            super::debug::SSM_LAYER_CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Stage profiler, the prefill twin of ATLAS_QWEN4EXP_DECODE_PROF:
        // serialize at stage seams and log per-stage µs for the first ~8
        // chunks (48 layer calls each). Prefill sits at ~110-130 tok/s vs a
        // 300-600 llama.cpp reference; the ranked suspects (GDN chunk path,
        // MoE grouped GEMM, PLE host work) live at different seams here.
        static PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static PROF_LEFT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(400);
        let prof = *PROF
            .get_or_init(|| std::env::var("ATLAS_QWEN4EXP_PREFILL_PROF").as_deref() == Ok("1"))
            && PROF_LEFT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed) > 0;
        let mut t = if prof {
            ctx.gpu.synchronize(stream).ok();
            Some(std::time::Instant::now())
        } else {
            None
        };
        macro_rules! stage {
            ($name:expr) => {
                if let Some(t0) = t.as_mut() {
                    ctx.gpu.synchronize(stream).ok();
                    tracing::info!(
                        "hc-prefill L{ssm_layer_idx} T={num_tokens} [{}]: {}us",
                        $name,
                        t0.elapsed().as_micros()
                    );
                    *t0 = std::time::Instant::now();
                }
            };
        }

        // ATLAS_FP32_ROUTING has the SSM's fused `residual_add_rms_norm_gatef32`
        // populate `moe_router_in_f32` for the gate GEMM to read at full
        // precision. This path does not run that kernel — `hc_norm` inside
        // `hc_pre` is the norm — so the buffer would hold the PREVIOUS layer's
        // activations and the router would route on them. Latent today (the
        // flag is off by default) and silent if it ever is not.
        anyhow::ensure!(
            !self.ffn.fp32_routing_active(),
            "qwen3_ssm mHC: ATLAS_FP32_ROUTING needs the fused gate-f32 norm, \
             which the highway path replaces. The router would read a stale \
             moe_router_in_f32. Unset it."
        );

        let streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();

        // The split starts after the layers that seed the highway and inject
        // PLE (`model::qwen4exp_prefill_sp`): PLE's causal conv runs across
        // the rows, so it needs every one of them.
        anyhow::ensure!(
            sp.is_none() || (!hc.is_first_model_layer && self.ple.is_none()),
            "qwen4_exp SP prefill reached the hc_expand or PLE layer"
        );
        if hc.is_first_model_layer {
            ops::qwen4exp_prefill_seam::clear_pending();
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                streams,
                n,
                h as u32,
                hc.hc_mult as u32,
                stream,
            )?;
        }

        // PLE injects into the highway BEFORE this layer's own
        // hyper-connection — the reference's
        // `hidden_states = hidden_states + self.ple(...)` sits above
        // `attn_hyper_connection` in `Qwen4ExpTextDecoderLayer.forward`.
        // `fresh` is a prefill starting at position 0: a new sequence, so the
        // conv state and the token history both reset.
        if let Some(ple) = self.ple.as_ref() {
            let ssm = state
                .as_any_mut()
                .downcast_mut::<crate::layer::SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("PLE host layer state is not SsmLayerState"))?;
            if ssm.ple.is_none() {
                ssm.ple = Some(ple.new_seq_state(ctx.gpu)?);
            }
            let st = ssm.ple.as_mut().expect("just created");
            // The previous layer's deferred post must land before PLE injects.
            ops::qwen4exp_prefill_seam::flush_pending(
                ctx.gpu,
                self.hc_post_k,
                hc,
                streams,
                post,
                h as u32,
                stream,
            )?;
            ple.forward(st, streams, num_tokens, seq_len_start == 0, ctx, stream)?;
        }
        stage!("ple");

        // ── GDN sublayer ──
        // `hidden` is scratch from here on: the highway carries the state
        // between layers, exactly as on the attention path. A post the
        // previous layer deferred (ATLAS_QWEN4EXP_PREFILL_HC) runs inside
        // this collapse.
        // ATLAS_QWEN4EXP_PREFILL_SP_PIPE: the gather goes out slab by slab
        // while the collapse runs (`layers::qwen4exp_sp_pipe`).
        crate::layers::qwen4exp_sp_pipe::collapse_and_gather(
            sp,
            hidden,
            None,
            n as usize,
            h,
            ctx,
            stream,
            || {
                if !ops::qwen4exp_prefill_seam::pre_with_pending(
                    ctx.gpu,
                    self.hc_post_k,
                    hc,
                    &hc.attn,
                    streams,
                    local(hidden),
                    post,
                    ctx.buffers.hc_lowrank_scratch(),
                    n,
                    h as u32,
                    eps,
                    stream,
                )? {
                    ops::hc_pre_site(
                        ctx.gpu,
                        self.hc_pre_k,
                        streams,
                        &hc.attn,
                        hc,
                        local(hidden),
                        post,
                        comb,
                        ctx.buffers.hc_lowrank_scratch(),
                        n,
                        h as u32,
                        eps,
                        stream,
                    )?;
                }
                Ok(())
            },
        )?;
        stage!("hc_pre_attn");
        // ATLAS_GLM_DET_TRACE stages (this rank's seam rows under SP).
        let det = crate::det_trace::on_stream(ctx.gpu, stream);
        let det_rows = (sp.map_or(0, |sp| sp.row0), n as usize);
        det.tap("in", local(hidden), det_rows, h * 2);
        let hc_dim = hc.hc_mult * h;
        crate::layers::ple::dump::tap_highway(
            ctx.gpu,
            streams,
            ssm_layer_idx,
            "in",
            num_tokens,
            hc_dim,
            stream,
        );
        // Tap hc_pre's BOTH outputs. `L00_post_gdn` diverges at cosine 0.80
        // with 2.6x the reference magnitude, and the sublayer is exactly
        // hc_pre -> block -> hc_post, so splitting it three ways is the whole
        // remaining search.
        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            hidden,
            ssm_layer_idx,
            "hc_pre_mixed",
            num_tokens * h,
            stream,
        );
        crate::layers::ple::dump::tap_f32(
            ctx.gpu,
            post,
            ssm_layer_idx,
            "hc_pre_inj",
            num_tokens * hc.hc_mult,
            stream,
        );
        // Under SP the block's TP reduce is a reduce-scatter
        // (`ssm_tp_all_reduce`): this rank's rows are at `local(..)`.
        let out_proj_buf =
            local(self.prefill_block(hidden, num_tokens, state, ssm_layer_idx, ctx, stream)?);
        stage!("gdn_block");
        det.tap("attn_red", out_proj_buf, det_rows, h * 2);
        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            out_proj_buf,
            ssm_layer_idx,
            "block_out",
            num_tokens * h,
            stream,
        );
        // ATLAS_QWEN4EXP_PREFILL_SP_PIPE: the MoE input's gather goes out
        // slab by slab while this collapse runs; _SP_ROUTE routes this rank's
        // rows before the join.
        let route = |s| sp.map_or(Ok(()), |sp| self.ffn.route_local_rows(hidden, sp, ctx, s));
        crate::layers::qwen4exp_sp_pipe::collapse_and_gather_then(
            sp,
            hidden,
            None,
            [n as usize, h],
            ctx,
            stream,
            || {
                // ATLAS_QWEN4EXP_PREFILL_HC: this post fused into the MoE site's
                // collapse below (same highway and `hidden` bytes, one highway read
                // fewer); `seam` says it ran, so the separate post and pre do not.
                let seam = ops::qwen4exp_prefill_hc::hc_post_pre_seam(
                    ctx.gpu,
                    hc,
                    &hc.ffn,
                    out_proj_buf,
                    streams,
                    local(hidden),
                    post,
                    ctx.buffers.hc_lowrank_scratch(),
                    n,
                    h as u32,
                    eps,
                    stream,
                )?;
                if !seam {
                    ops::hc_post_site(
                        ctx.gpu,
                        self.hc_post_k,
                        hc,
                        out_proj_buf,
                        streams,
                        post,
                        comb,
                        streams,
                        n,
                        h as u32,
                        stream,
                    )?;
                }

                stage!("hc_post_attn");
                // Tapped BEFORE the MoE on purpose: reproducing this point in the
                // reference needs only the GDN projections, not 512 experts.
                crate::layers::ple::dump::tap_highway(
                    ctx.gpu,
                    streams,
                    ssm_layer_idx,
                    "post_gdn",
                    num_tokens,
                    hc_dim,
                    stream,
                );

                // ── MoE sublayer ──
                // `prefill_block` returned `ctx.buffers.moe_output()`, which the FFN
                // is about to overwrite — safe only because the `hc_post` above has
                // already consumed it into the highway. Keep that order.
                if !seam {
                    ops::hc_pre_site(
                        ctx.gpu,
                        self.hc_pre_k,
                        streams,
                        &hc.ffn,
                        hc,
                        local(hidden),
                        post,
                        comb,
                        ctx.buffers.hc_lowrank_scratch(),
                        n,
                        h as u32,
                        eps,
                        stream,
                    )?;
                }
                Ok(())
            },
            route,
        )?;
        stage!("hc_pre_ffn");
        let moe_out = match sp {
            Some(sp) => self.ffn.forward_prefill_sp(hidden, sp, ctx, stream)?,
            None => {
                self.ffn.forward_prefill(hidden, num_tokens, ctx, stream)?;
                ctx.buffers.moe_output()
            }
        };
        stage!("moe");
        // ATLAS_QWEN4EXP_PREFILL_HC: left to the next layer's collapse.
        if !ops::qwen4exp_prefill_seam::defer_post(ctx.gpu, hc, moe_out, n, h as u32) {
            ops::hc_post_site(
                ctx.gpu,
                self.hc_post_k,
                hc,
                moe_out,
                streams,
                post,
                comb,
                streams,
                n,
                h as u32,
                stream,
            )?;
        }
        crate::layers::ple::dump::tap_highway(
            ctx.gpu,
            streams,
            ssm_layer_idx,
            "post_moe",
            num_tokens,
            hc_dim,
            stream,
        );
        stage!("hc_post_ffn");

        Ok(())
    }
}
