// SPDX-License-Identifier: AGPL-3.0-only

//! Batched (K-token, one sequence) GDN decode under the mHC highway — the
//! piece `refuse_batched_under_hc` guarded for `decode_batched` (#753 item B).
//!
//! This is the path a speculative VERIFY step needs: K consecutive tokens of a
//! single sequence in one forward. It is not the same problem as
//! `trait_decode_multi_seq/hc.rs`, which was already built — there the rows are
//! N independent sequences with independent SSM states, so the recurrence runs
//! per row. Here the K rows are sequentially dependent within one sequence, and
//! the conv/GDN body must scan them in order against one state.
//!
//! ## Why only the two ends change
//!
//! The non-hc `decode_batched_inner` is steps 1..10, and only three of them
//! touch the residual:
//!
//! ```text
//!   1.  rms_norm_residual(hidden, input_norm)      -> normed, residual
//!   2-9 the conv/GDN body                          -> out_proj_buf
//!   10. residual_add_rms_norm(hidden, out_proj_buf, post_attn_norm)
//!       ffn(normed2) -> moe_output
//!       residual_add(hidden, moe_output)
//! ```
//!
//! Under the highway those three are exactly what must NOT run: `hc_pre` is
//! the norm (the checkpoint ships no per-layer norms — the loader's
//! ones-placeholders would not make a second RMS pass an identity), and the
//! block output reaches the residual only through `hc_post`, scaled per stream
//! by the injection vector `hc_pre` emitted. Running them anyway adds each
//! block output a second time, which is the "plausible, wrong activations with
//! nothing in the log" the refusal existed to prevent.
//!
//! Steps 2-9 are untouched and shared verbatim: they read `normed` and write
//! `out_proj_buf` and never look at the residual. So this file supplies the
//! two ends and the caller keeps the body.
//!
//! Shape, mirroring `trait_prefill_hc.rs` at K tokens instead of a chunk:
//!
//! ```text
//!   hc_expand [K]                    (first model layer only)
//!   PLE [K]                          (the one layer that carries it)
//!   hc_pre(attn) -> normed           <- steps 2-9 read this
//!     ... conv/GDN body ...          -> out_proj_buf
//!   hc_post(out_proj_buf -> streams)
//!   hc_pre(ffn)  -> normed2
//!     MoE                            -> moe_output
//!   hc_post(moe_output -> streams)
//! ```
//!
//! ## Rollback is not this file's problem
//!
//! Partial acceptance rolls the SSM state back through the per-token
//! checkpoints (`conv_intermediate` / `h_checkpoint`), whose destinations the
//! MODEL stages into the layer state (`model/trait_impl/meta.rs`) and whose
//! writes happen inside the shared conv/GDN body. Since that body is reused
//! unchanged, checkpointing behaves exactly as on the non-hc path.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Qwen3SsmLayer;
use super::trait_decode_batched::GdnStates;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3SsmLayer {
    /// Sublayer input under the highway. Leaves the mixed rows in `normed`,
    /// which steps 2-9 then read exactly as they read the `rms_norm_residual`
    /// output on the non-hc path.
    pub(super) fn decode_batched_hc_in(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        normed: DevicePtr,
        gdn: &mut GdnStates<'_, '_>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_batched_hc_in without mHC weights"))?;
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;

        // Same reason `trait_prefill_hc` refuses it: this path does not run the
        // fused gate-f32 norm, so `moe_router_in_f32` would hold the PREVIOUS
        // layer's activations and the router would route on them.
        anyhow::ensure!(
            !self.ffn.fp32_routing_active(),
            "qwen3_ssm mHC batched decode: ATLAS_FP32_ROUTING needs the fused \
             gate-f32 norm, which the highway path replaces. Unset it."
        );

        let streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();

        // ATLAS_QWEN4EXP_VERIFY_PROF=1: per-stage wall clock for the SPECULATIVE
        // VERIFY, which is the one width neither the decode nor the prefill
        // profiler covers. Both of those instrument their own path; the K-row
        // verify runs through here and was, until this, unattributed -- so the
        // question "what does the third verify row actually cost" could only be
        // answered by arithmetic over end-to-end throughput. Each probe syncs,
        // so the numbers are honest and the mode is not for serving.
        let mut t = verify_prof_start(ctx, stream);
        macro_rules! stage {
            ($name:expr) => {
                if let Some(t0) = t.as_mut() {
                    ctx.gpu.synchronize(stream).ok();
                    tracing::info!(
                        "hc-verify K={num_tokens} [{}]: {}us",
                        $name,
                        t0.elapsed().as_micros()
                    );
                    *t0 = std::time::Instant::now();
                }
            };
        }

        if hc.is_first_model_layer {
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
        // hyper-connection, matching the reference's
        // `hidden_states = hidden_states + self.ple(...)` above
        // `attn_hyper_connection`. `fresh` is false: a verify step never starts
        // a sequence, so conv state and token history carry in.
        if let Some(ple) = self.ple.as_ref() {
            match gdn {
                GdnStates::Single(state) => {
                    let ssm = state
                        .as_any_mut()
                        .downcast_mut::<crate::layer::SsmLayerState>()
                        .ok_or_else(|| {
                            anyhow::anyhow!("PLE host layer state is not SsmLayerState")
                        })?;
                    let st = ssm.ple.as_mut().ok_or_else(|| {
                        anyhow::anyhow!("PLE batched decode before prefill: no seq state")
                    })?;
                    ple.forward(st, streams, num_tokens, false, ctx, stream)?;
                }
                // Concurrent verify (C>1): the rows are seq-major and RAGGED,
                // `ks[i]` of them for sequence i. Each sequence's injection has
                // to read ITS OWN token ids against ITS OWN n-gram history, so
                // this is a per-seq loop over row slices — the same shape
                // `trait_decode_multi_seq/hc.rs` uses at one row per sequence,
                // widened to `ks[i]`. Injecting the batch prefix into every
                // sequence (what a single `forward` here would do) is the
                // silent-wrongness case worth spelling out.
                GdnStates::Multi { states, ks, .. } => {
                    let host = ctx.host_token_ids.ok_or_else(|| {
                        anyhow::anyhow!("hc batched verify: PLE needs host_token_ids threaded")
                    })?;
                    anyhow::ensure!(
                        host.len() >= num_tokens,
                        "hc batched verify: PLE has {} host id(s) for {num_tokens} row(s)",
                        host.len()
                    );
                    let mut rows = Vec::with_capacity(states.len());
                    let mut row0 = 0usize;
                    for (i, state) in states.iter_mut().enumerate() {
                        let ki = *ks.get(i).ok_or_else(|| {
                            anyhow::anyhow!("hc batched verify: no k for sequence {i}")
                        })?;
                        let ssm = state
                            .as_any_mut()
                            .downcast_mut::<crate::layer::SsmLayerState>()
                            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                        // A padding row carries `ple: None` by construction and
                        // has no token id — it gets no injection, exactly as on
                        // the multi-seq decode path.
                        if let Some(st) = ssm.ple.as_mut() {
                            rows.push(crate::layers::ple::PleSeqRows {
                                st,
                                row0,
                                ids: &host[row0..row0 + ki],
                            });
                        }
                        row0 += ki;
                    }
                    ple.forward_seqs(&mut rows, streams, ctx, stream)?;
                }
            }
        }

        stage!("gdn_block");

        ops::hc_pre_site_rows(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.attn,
            hc,
            normed,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n,
            h as u32,
            eps,
            ctx.levers.qwen4exp_hc_rows(),
            stream,
        )?;
        stage!("hc_pre_attn");
        Ok(())
    }

    /// Sublayer output under the highway: land the block on the highway, then
    /// run the MoE sublayer and land that too.
    pub(super) fn decode_batched_hc_out(
        &self,
        out_proj_buf: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_batched_hc_out without mHC weights"))?;
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;

        let streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();

        let mut t = verify_prof_start(ctx, stream);
        macro_rules! stage {
            ($name:expr) => {
                if let Some(t0) = t.as_mut() {
                    ctx.gpu.synchronize(stream).ok();
                    tracing::info!(
                        "hc-verify K={num_tokens} [{}]: {}us",
                        $name,
                        t0.elapsed().as_micros()
                    );
                    *t0 = std::time::Instant::now();
                }
            };
        }

        // MUST precede the FFN: on some arms `out_proj_buf` IS
        // `ctx.buffers.moe_output()`, which the FFN below overwrites. Consuming
        // it into the highway first is what makes that safe — same ordering
        // constraint `trait_prefill_hc` documents.
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

        stage!("hc_post_attn");

        // ── MoE sublayer ──
        let normed2 = ctx.buffers.norm_output();
        ops::hc_pre_site_rows(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.ffn,
            hc,
            normed2,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n,
            h as u32,
            eps,
            ctx.levers.qwen4exp_hc_rows(),
            stream,
        )?;

        // The batched MoE arms all leave `[num_tokens, h]` in `moe_output`,
        // which is what `hc_post` needs. The non-hc path's per-token fallback
        // does NOT: `ffn.forward` reuses row 0 every call, so it would need
        // per-row staging.
        //
        // K=4..8 used to bail for MoE here, because `try_forward_km` is
        // dense-only. That capped the verify width at K=3 and therefore capped
        // speculative yield at 3 tokens/step for this model.
        //
        // Why that cap is worth removing: the A/B in `verify_max_drafts` that
        // condemned wide K was taken at **256-token completions**, where `fwd`
        // grows ~40 % from K=2 to K=3 (+29.78 ms of a 786.9 us/layer verify) and
        // that growth is what made K=3 lose — its own profiler table shows the
        // *attention/SSM* half scaling only 1.126x, BELOW the 1.354 token ratio,
        // so "on the SSM half alone K=3 would win". At long context the forward
        // is weight-read bound and flat in row count instead: measured `fwd`
        // ~= 106 ms at both 3 rows (C1) and 12 rows (C4), and the whole step
        // ~= 119 ms. The row-count penalty that decided the 256-token A/B is
        // largely absent here, which is exactly the regime change that makes
        // deeper speculation worth re-measuring.
        //
        // `forward_batched` is the generic multi-token MoE dispatcher, and it is
        // already what `forward_k2`/`forward_k3` themselves fall back to (LoRA
        // adapters, BF16-dequant experts, E8M0 scale kinds). It stages every row
        // into `moe_output` and is the same entry ordinary batched decode uses.
        // Routing the remaining widths through it beats refusing: the caller
        // answers a verify error by finishing the request, so refusing is not a
        // safe default — it is a truncated answer. Set
        // `ATLAS_HC_GENERIC_MOE_ARM=0` to restore the old refusal.
        let generic_arm = std::env::var("ATLAS_HC_GENERIC_MOE_ARM").as_deref() != Ok("0");
        stage!("hc_pre_ffn");

        // ATLAS_QWEN4EXP_EXACT_VERIFY: each row through the MoE serial decode
        // runs (`ffn.forward`, as the attention layers' verify already does in
        // `multi_seq/hc_generic.rs`), and its own hc_post. `forward_k2/k3` is
        // a different function of the row: its batched top-k breaks exact
        // logit ties the other way, and on the originals expert layout its
        // gate_up/down kernels split K across lanes differently and do not
        // clamp gate/up to +-10 as the single-row kernels do. `forward` lands
        // every row in `moe_output` row 0, so the post runs per row too.
        // ATLAS_QWEN4EXP_BATCH_FAST: the same per-row `forward` arithmetic at
        // any row count (single- or multi-sequence verify), with the EP
        // all-reduce batched over the rows and ONE elementwise post.
        let fold = ops::qwen4exp_decode_fuse::MoeHcPost::for_rows(ctx.levers, hc, streams, post);
        if ctx.levers.qwen4exp_batch_fast
            && let Some((moe_out, posted)) = self
                .ffn
                .forward_rows(normed2, num_tokens, fold, ctx, stream)?
        {
            if !posted {
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
            stage!("moe+hc_post_ffn (rows)");
            return Ok(());
        }
        if ctx.levers.qwen4exp_exact_verify && (2..=4).contains(&num_tokens) {
            for i in 0..num_tokens {
                let moe_out = self.ffn.forward(normed2.offset(i * h * 2), ctx, stream)?;
                ops::hc_post_site(
                    ctx.gpu,
                    self.hc_post_k,
                    hc,
                    moe_out,
                    streams.offset(i * hc.hc_mult * h * 4),
                    post.offset(i * hc.hc_mult * 4),
                    comb.offset(i * hc.hc_mult * hc.hc_mult * 4),
                    streams.offset(i * hc.hc_mult * h * 4),
                    1,
                    h as u32,
                    stream,
                )?;
            }
            stage!("moe+hc_post_ffn (per row)");
            return Ok(());
        }

        if num_tokens == 3 {
            self.ffn.forward_k3(normed2, ctx, stream)?;
        } else if num_tokens == 2 {
            self.ffn.forward_k2(normed2, ctx, stream)?;
        } else if (4..=8).contains(&num_tokens)
            && self
                .ffn
                .try_forward_km(normed2, num_tokens as u32, ctx, stream)
                .inspect_err(|e| tracing::error!("ffn.try_forward_km: {e:#}"))
                .unwrap_or(false)
        {
            // try_forward_km already wrote moe_output for all rows.
        } else if self.ffn.is_dense() {
            self.ffn.forward_prefill(normed2, num_tokens, ctx, stream)?;
        } else if generic_arm {
            self.ffn.forward_batched(normed2, num_tokens, ctx, stream)?;
        } else {
            anyhow::bail!(
                "qwen3_ssm mHC batched decode: no batched MoE arm for K={num_tokens}. \
                 K=2/3 use forward_k2/k3 and K=4..8 the batched GEMV; this model's \
                 512-expert MoE has no per-row staging under the highway yet. \
                 Lower --num-drafts (K = num_drafts + 1)."
            );
        }
        stage!("moe");

        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            ctx.buffers.moe_output(),
            streams,
            post,
            comb,
            streams,
            n,
            h as u32,
            stream,
        )?;
        stage!("hc_post_ffn");
        Ok(())
    }
}

/// Start the verify-path stage clock, or `None` when the probe is off.
///
/// Capped like its decode and prefill twins: a verify runs every speculative
/// step, so an uncapped probe would sync twice per stage per layer for the life
/// of the process. 600 layer calls is ~16 steps at 36 SSM layers -- enough to
/// average a K=2 against a K=3 without the syncs themselves becoming the
/// measurement.
fn verify_prof_start(ctx: &ForwardContext, stream: u64) -> Option<std::time::Instant> {
    static PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static PROF_LEFT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(600);
    let on = *PROF
        .get_or_init(|| std::env::var("ATLAS_QWEN4EXP_VERIFY_PROF").as_deref() == Ok("1"))
        && PROF_LEFT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed) > 0;
    if !on {
        return None;
    }
    ctx.gpu.synchronize(stream).ok();
    Some(std::time::Instant::now())
}
