// SPDX-License-Identifier: AGPL-3.0-only

//! Batched (N-sequence) decode for the mHC-highway GDN layer — the piece
//! `refuse_batched_under_hc` existed to guard until it was built (#753
//! item B, milestone 2: "real concurrent batches, not queued interleaving").
//!
//! Mirrors the attention side's `decode_multi_seq_inner_hc`
//! (`qwen3_attention/trait_impl/multi_seq/mod.rs`, built for DeepSeek-V4):
//! the highway REPLACES the layer's own residual bookkeeping, so the
//! non-hc path's fused `rms_norm_residual` / `residual_add_rms_norm` steps
//! must not run. Structure per layer:
//!
//!   hc_expand [N]           (first model layer only)
//!   PLE per-seq mini-loop   (each row against its own PleSeqState)
//!   hc_pre  [N]  -> mixed rows in `hidden`
//!   ssm_forward per seq     (recurrence is inherently per-seq; outputs
//!                            staged into norm_output rows — ssm_forward
//!                            reuses moe_output[0] every call)
//!   hc_post [N]  <- norm_output rows
//!   hc_pre  [N]  -> mixed rows in norm_output (the MoE input rows)
//!   MoE: forward_k2/k3 at N=2/3 (rows in moe_output), else a per-row
//!        loop staged into `hidden` rows (the mixed GDN inputs are dead
//!        by then)
//!   hc_post [N]
//!
//! The batched-GEMM projection lever (`try_decode_multi_seq_ssm_batched`)
//! is deliberately NOT engaged yet: it folds the residual into its
//! epilogues. Amortizing the QKVZ/out_proj weight reads across N rows
//! under the highway is the next increment.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3SsmLayer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn decode_multi_seq_inner_hc<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        num_seqs: usize,
        active_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_seqs;
        let bf16 = 2usize;
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_multi_seq_inner_hc without mHC weights"))?;
        let hc_mult = hc.hc_mult as u32;
        let streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let normed = ctx.buffers.norm_output();
        let moe_out = ctx.buffers.moe_output();

        if hc.is_first_model_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                streams,
                n as u32,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // ── PLE: per-seq rows against per-seq state ──
        // The highway rows are FP32 [n, hc*h]; each sequence's injection
        // reads ITS token id and ITS conv/history carry. `decode()` runs
        // this via prestage; here the forward does its own host half — the
        // per-seq token comes from `ctx.host_token_ids` via the override.
        if let Some(ple) = self.ple.as_ref() {
            let host = ctx.host_token_ids.ok_or_else(|| {
                anyhow::anyhow!("hc multi-seq decode: PLE needs host_token_ids threaded")
            })?;
            // `n` is the PADDED batch width (`padded_batch_n`), not the
            // active count: `decode_batch` rounds up so one captured CUDA
            // graph serves several batch sizes, zeroes `hidden[active..n)`,
            // and pushes dummy layer states for the padding rows — which
            // carry `ple: None` by construction and have no token id.
            //
            // `active_seqs` is the live count, so the skip below is exact
            // rather than inferred from the dummy state's shape: a row with
            // a LIVE ple state and no token id is a real defect and still
            // errors, and an ACTIVE row missing its ple carry errors too.
            for (i, state) in states.iter_mut().enumerate().take(n) {
                let ssm = state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                let Some(st) = ssm.ple.as_mut() else {
                    if i >= active_seqs {
                        continue; // padding row
                    }
                    anyhow::bail!(
                        "hc multi-seq decode: PLE layer (row {i} of {n}, \
                         {active_seqs} active) got a state with no PLE carry — \
                         every active row of a PLE-carrying model must own one"
                    );
                };
                let token = host.get(i..i + 1).ok_or_else(|| {
                    anyhow::anyhow!(
                        "hc multi-seq decode: seq {i} has a live PLE state but \
                         only {} host token id(s) were threaded",
                        host.len()
                    )
                })?;
                ple.forward_row(
                    st,
                    streams.offset(i * (hc_mult as usize) * h * 4),
                    token,
                    ctx,
                    stream,
                )?;
            }
        }

        // ── GDN sublayer ──
        // hc_pre writes the mixed+normed rows straight into norm_output —
        // both the batched-GEMM core and the per-seq fallback read there.
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.attn,
            hc,
            normed,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        // Batched-GEMM core (QKVZ/out_proj weights read ONCE for all n rows —
        // the C=N scaling lever on bandwidth-bound LPDDR5X); out rows land in
        // moe_output[0..n). `hidden`/`residual` are unused in hc mode.
        let gdn_rows = if self.try_decode_multi_seq_ssm_batched(
            hidden,
            DevicePtr::NULL,
            n,
            states,
            true,
            ctx,
            stream,
        )? {
            moe_out
        } else {
            // Fallback: per-seq recurrence reading the mixed rows; each
            // output (ssm_forward reuses moe_output[0]) staged into `hidden`
            // rows, which are dead in hc mode.
            for (i, state) in states.iter_mut().enumerate().take(n) {
                let ssm_state = state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                let ssm_out =
                    self.ssm_forward(normed.offset(i * h * bf16), ssm_state, ctx, stream, false)?;
                ctx.gpu
                    .copy_d2d_async(ssm_out, hidden.offset(i * h * bf16), h * bf16, stream)?;
            }
            hidden
        };
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            gdn_rows,
            streams,
            post,
            comb,
            streams,
            n as u32,
            h as u32,
            stream,
        )?;

        // ── MoE sublayer ──
        // hc_pre writes the mixed rows straight into norm_output — the
        // batched expert kernels' input convention.
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.ffn,
            hc,
            normed,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        let moe_rows = match n {
            2 => {
                self.ffn.forward_k2(normed, ctx, stream)?;
                moe_out
            }
            3 => {
                self.ffn.forward_k3(normed, ctx, stream)?;
                moe_out
            }
            _ => {
                // Per-row loop; stage into `hidden` rows (the mixed GDN
                // inputs there are dead once the recurrence loop above ran).
                for i in 0..n {
                    let out = self.ffn.forward(normed.offset(i * h * bf16), ctx, stream)?;
                    ctx.gpu
                        .copy_d2d_async(out, hidden.offset(i * h * bf16), h * bf16, stream)?;
                }
                hidden
            }
        };
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            moe_rows,
            streams,
            post,
            comb,
            streams,
            n as u32,
            h as u32,
            stream,
        )?;
        Ok(())
    }
}
