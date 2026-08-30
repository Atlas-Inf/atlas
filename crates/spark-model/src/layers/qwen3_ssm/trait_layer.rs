// SPDX-License-Identifier: AGPL-3.0-only

//! `TransformerLayer` impl for [`Qwen3SsmLayer`] — the trait surface that
//! forwards into the `trait_*` sibling modules holding the actual phases.
//! Split out of `mod.rs` to keep it under the 500-LoC cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::Qwen3SsmLayer;
use crate::layer::{ForwardContext, GdnPrefillBuffers, LayerState, TransformerLayer};

impl TransformerLayer for Qwen3SsmLayer {
    /// Downcast hook so the LoRA install walk can reach this layer's MoE FFN
    /// (Feature-1: routed-expert/router deltas exist on GDN layers too).
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    /// PLE's host half (hash + NVMe fault-in + slot upload), hoisted before
    /// graph replay/capture. No-op on the 47 layers without a PLE site.
    fn decode_prestage(
        &self,
        token: u32,
        state: &mut dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if let Some(ple) = self.ple.as_ref() {
            let st = ple_seq_state(ple, state, gpu)?;
            ple.prestage(st, &[token], gpu, stream)?;
        }
        Ok(())
    }

    fn verify_prestage(
        &self,
        tokens: &[u32],
        state: &mut dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if let Some(ple) = self.ple.as_ref() {
            let st = ple_seq_state(ple, state, gpu)?;
            ple.prestage(st, tokens, gpu, stream)?;
        }
        Ok(())
    }

    fn has_aux_state(&self) -> bool {
        self.ple.is_some()
    }

    /// K=2 and K=3 are the verify widths this layer's mHC batched decode
    /// actually has MoE arms for (`forward_k2`/`forward_k3`); K=4..8 goes
    /// through `try_forward_km`, which is dense-only, and a dense FFN can
    /// also fall back to `forward_prefill` at any K. So a 512-expert MoE
    /// under the highway tops out at K=3 — two drafts. Off the highway the
    /// batched path stages per row and nothing here bounds it.
    ///
    /// Reported rather than discovered: `trait_decode_batched_hc` bails on
    /// an unservable K, and that bail reaches the scheduler as a verify
    /// error, which finishes the request. `--num-drafts 3` on this model
    /// used to kill every request after one token that way.
    fn verify_max_drafts(&self) -> Option<usize> {
        if self.hc.is_none() || self.ffn.is_dense() {
            return None;
        }
        // ONE draft (K=2 verify rows), not two. Two reasons, both measured on
        // one GB10 with 256-token completions, agg tok/s:
        //
        //   arm        C=1     C=2    tok/step  greedy text vs plain decode
        //   base      17.74   26.93     1.000   (reference)
        //   K=2       21.73   25.39     1.774   IDENTICAL
        //   K=3       20.16   24.92     2.420   DIFFERS
        //
        // 1. K=3 IS SLOWER. Acceptance genuinely improves — 1.774 -> 2.420
        //    tokens per step, and 2.504 with the gate forced — but the third
        //    verify row costs more than the extra 0.65 tokens buys. This holds
        //    after the small-M MoE substitution that cut the verify's dominant
        //    term 14x, so it is not the MoE.
        //
        // 2. K=3 IS NOT OUTPUT-EXACT as it stands. At temperature 0
        //    speculation must be indistinguishable from serial decode. K=2
        //    reproduces it byte for byte; K=3 does not. Localized to ONE row
        //    of the stream-row selection (`Model::select_mtp_stream_row`) by
        //    `ATLAS_MTP_STREAM_ROW_MAX`, two prompts, sha of the completion:
        //
        //      arm                        exact   p1      tok/step
        //      rows 0,1,2 (as shipped)     NO     0.795     2.402
        //      row 1 only                  yes    0.603     2.069
        //      no selection at all         yes    0.576     2.017
        //      K=2                         yes    0.843     1.843
        //
        //    Dropping ONLY row 2 restores exactness, so this is not a general
        //    draft-invariance problem — row 2's copy specifically is wrong.
        //    (`ATLAS_QWEN4EXP_HC_SMALL_M_FFN=0` still diverges, so the MoE
        //    substitution is exonerated.) The selection is what lifted K=2
        //    acceptance 0.69 -> 0.83: right for the row it was validated on,
        //    wrong for row 2.
        //
        // AND FIXING ROW 2 WOULD NOT CHANGE THE ANSWER, which is why the clamp
        // is here rather than a TODO. The exact K=3 was benched:
        //
        //      arm                        C=1     C=2
        //      K=3 exact (row 1 only)    16.91   22.88
        //      K=3 as shipped (inexact)  20.16   24.92
        //      K=2                       21.67   25.41
        //
        // K=2 wins on throughput against BOTH, including against a K=3 with
        // strictly better tokens/step. The extra verify row costs more than
        // the extra tokens return on this model, so the acceptance gain is
        // real and irrelevant.
        //
        // HOW MUCH MORE ACCEPTANCE WOULD K=3 NEED? At its measured step cost
        // (119.1 ms vs K=2's 85.0 ms) K=3 must reach 2.582 tokens/step to TIE
        // K=2. Under 1 + p + p^2 that is a per-draft acceptance of
        //
        //     p = 0.853
        //
        // and K=2's own measured first-draft acceptance is p1 = 0.843. So a
        // perfect row-2 fix — one lifting K=3's drafting all the way to K=2's
        // quality — lands at 2.554 tok/step, 21.43 tok/s, still under K=2's
        // 21.67. K=3 would have to draft BETTER than K=2 does merely to draw.
        // And 1 + p + p^2 is optimistic: it assumes the second draft position
        // accepts at the first's rate, when later positions always accept
        // less, so the real requirement is higher still.
        //
        // WHY THE THIRD ROW COSTS WHAT IT DOES — MEASURED, and NOT the expert
        // union this comment used to blame. That story predicted the verify FFN
        // would scale with the union of activated experts,
        // 512*(1-(1-10/512)^R): 19.8 experts at R=2, 29.4 at R=3, ratio 1.485.
        // Profiling the verify (`ATLAS_QWEN4EXP_VERIFY_PROF=1`, mean us per
        // layer call) says otherwise:
        //
        //     stage          K=2      K=3     ratio
        //     moe          287.6    374.1     1.30
        //     hc_pre_attn  184.9    184.5     1.00
        //     hc_pre_ffn   183.8    184.5     1.00
        //     gdn_block     16.8     17.0     1.01
        //     TOTAL        698.9    786.9     1.126
        //
        // MoE scales 1.30x, not 1.485x, and the whole SSM verify only 1.126x —
        // BELOW the 1.354 token ratio. On the SSM half alone K=3 would win.
        //
        // WHERE IT ACTUALLY GOES (`ATLAS_MTP_TIMING=1`, per step):
        //
        //     phase        K=2       K=3       delta
        //     fwd        73.54 ms  103.32 ms  +29.78   (90% of it)
        //     propose     3.64 ms    7.27 ms   +3.63   (the 2nd draft pass)
        //     d2h         1.98 ms    1.67 ms   -0.31
        //     step_mtp   80.01 ms  113.07 ms  +33.06
        //
        // and the 36 SSM layers are only +3.17 ms of that +29.78 forward. So
        // +26.6 ms — 80% of the whole K=3 penalty — is the NON-SSM half of the
        // forward: 12 attention layers, the LM head over 3 rows instead of 2,
        // and the embedding. That half scales 1.55x where the SSM half scales
        // 1.126x, and nothing here explains why yet.
        //
        // So K=3 is NOT settled as physics, and the earlier claim that it was
        // is withdrawn. It loses today — step ratio 1.456 against a token
        // ratio 1.354, measured end to end on the same binary — but the term
        // that loses it is an unexplained 1.55x in the attention-layer forward,
        // not a bandwidth floor. If that were brought to the SSM half's 1.126x
        // the step ratio would fall under the token ratio and K=3 would pay.
        //
        // Raising this clamp therefore needs three things, in order: attribute
        // the attention-half forward at K=2 vs K=3 (it has no per-stage probe
        // yet, which is why this stops here), fix whatever it finds, and fix
        // the row-2 exactness defect. The first is the open question.
        //
        // DIAGNOSTIC (`ATLAS_MTP_MAX_DRAFTS`): raise the clamp to profile the
        // width it forbids. The arithmetic above says K=3 cannot win, but it
        // was derived from end-to-end throughput, not from a per-stage
        // attribution of the verify itself — and the verify is the one width
        // neither the decode nor the prefill profiler covers. Pair with
        // `ATLAS_QWEN4EXP_VERIFY_PROF=1` to see where a K=3 row's cost lands.
        // NOT a serving lever: K=3 is measurably slower AND not output-exact.
        Some(
            std::env::var("ATLAS_MTP_MAX_DRAFTS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|n| *n > 0)
                .unwrap_or(1),
        )
    }

    fn rollback_aux_verify(
        &self,
        state: &mut dyn LayerState,
        num_accepted: usize,
        k: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let Some(ple) = self.ple.as_ref() else {
            return Ok(());
        };
        let Some(ssm) = state
            .as_any_mut()
            .downcast_mut::<crate::layer::SsmLayerState>()
        else {
            return Ok(());
        };
        if let Some(st) = ssm.ple.as_mut() {
            ple.rollback_verify(st, num_accepted, k, gpu, stream)?;
        }
        Ok(())
    }

    /// PLE's per-seq host hash on the hc multi-seq decode path is
    /// capture-illegal (pageable reads); the single-decode path prestages
    /// around it, the batched path does not — veto batched graphs.
    fn decode_graph_unsupported(&self) -> bool {
        self.ple.is_some()
    }

    fn snapshot_aux(
        &self,
        state: &dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        let Some(ple) = self.ple.as_ref() else {
            return Ok(None);
        };
        let ssm = state
            .as_any()
            .downcast_ref::<crate::layer::SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("PLE host layer state is not SsmLayerState"))?;
        match ssm.ple.as_ref() {
            Some(st) => Ok(Some(ple.snapshot_aux(st, gpu, stream)?)),
            // Sequence never ran this layer (snapshot before first pass):
            // nothing to carry, and restore-side declines aux-less slots.
            None => Ok(None),
        }
    }

    fn restore_aux(
        &self,
        state: &mut dyn LayerState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let ple = self
            .ple
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("restore_aux: no PLE on this layer"))?;
        let st = ple_seq_state(ple, state, gpu)?;
        ple.restore_aux(st, blob, gpu, stream)
    }

    fn decode_prestage_rearm(&self, state: &mut dyn LayerState) {
        if let Some(ple) = self.ple.as_ref()
            && let Some(ssm) = state
                .as_any_mut()
                .downcast_mut::<crate::layer::SsmLayerState>()
            && let Some(st) = ssm.ple.as_mut()
        {
            ple.rearm(st);
        }
    }

    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.hc.is_some() {
            return self.decode_inner_hc(hidden, state, ctx, stream);
        }
        self.decode_inner(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    fn decode_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Under an mHC highway `decode_batched_inner` brackets the shared
        // conv/GDN body with hc_pre/hc_post instead of its own residual
        // bookkeeping (#753 item B) — no refusal needed.
        self.decode_batched_inner(
            hidden,
            residual,
            num_tokens,
            super::trait_decode_batched::GdnStates::Single(state),
            ctx,
            stream,
        )
    }

    fn decode_verify_multi<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        n_seqs: usize,
        ks: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        wy_tables: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            states.len() == n_seqs && ks.len() == n_seqs,
            "decode_verify_multi: states/ks/n mismatch"
        );
        let num_tokens: usize = ks.iter().sum();
        self.decode_batched_inner(
            hidden,
            residual,
            num_tokens,
            super::trait_decode_batched::GdnStates::Multi {
                states,
                ks,
                wy_tables,
            },
            ctx,
            stream,
        )
    }

    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.hc.is_some() {
            // #753 item B milestone 2: the highway replaces the residual the
            // non-hc path folds into its fused norm kernels; run the
            // hc-bracketed variant instead of refusing.
            return self.decode_multi_seq_inner_hc(hidden, num_seqs, states, ctx, stream);
        }
        self.decode_multi_seq_inner(
            hidden,
            residual,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Under an mHC highway the residual bookkeeping is completely
        // different — the highway IS the residual — so this is a second entry
        // path, not a flag on the first. See `trait_prefill_hc.rs`.
        if self.hc.is_some() {
            return self.prefill_inner_hc(hidden, num_tokens, state, seq_len_start, ctx, stream);
        }
        self.prefill_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            ctx,
            stream,
        )
    }

    fn is_ssm_layer(&self) -> bool {
        self.is_ssm_layer_inner()
    }

    fn prefill_phase1(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase1_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            gdn_bufs,
            token_offset,
            ctx,
            stream,
        )
    }

    fn prefill_phase1_proj_batched(
        &self,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase1_proj_batched_inner(
            hidden_stacked,
            residual_stacked,
            total_tokens,
            gdn_bufs,
            ctx,
            stream,
        )
    }

    fn prefill_phase1_conv1d_one(
        &self,
        state: &mut dyn LayerState,
        token_offset: usize,
        len: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase1_conv1d_one_inner(state, token_offset, len, gdn_bufs, ctx, stream)
    }

    fn prefill_phase1_l2_batched(
        &self,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase1_l2_batched_inner(total_tokens, gdn_bufs, ctx, stream)
    }

    fn prefill_gdn_full(
        &self,
        state: &mut dyn LayerState,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_gdn_full_inner(state, gdn_bufs, ctx, stream)
    }

    fn prefill_gdn_full_batched(
        &self,
        h_state_ptrs: DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        chunk_len: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_gdn_full_batched_inner(
            h_state_ptrs,
            gdn_bufs,
            batch_size,
            chunk_len,
            ctx,
            stream,
        )
    }

    fn prefill_gdn_full_batched_fla_varlen(
        &self,
        h_state_ptrs: DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        cu_seqlens: DevicePtr,
        max_num_chunks: u32,
        total_nt: usize,
        max_seqlen: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        self.prefill_gdn_full_batched_fla_varlen_inner(
            h_state_ptrs,
            gdn_bufs,
            batch_size,
            cu_seqlens,
            max_num_chunks,
            total_nt,
            max_seqlen,
            ctx,
            stream,
        )
    }

    fn prefill_phase3(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase3_inner(
            hidden,
            residual,
            num_tokens,
            gdn_bufs,
            token_offset,
            ctx,
            stream,
        )
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        self.alloc_state_inner(gpu)
    }

    /// Release the PLE carry. The h/conv states are pool slots that
    /// `free_sequence` releases separately; this is only the `gpu.alloc` the
    /// carry owns.
    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn LayerState) -> Result<()> {
        if self.ple.is_none() {
            return Ok(());
        }
        let Some(ssm) = state
            .as_any_mut()
            .downcast_mut::<crate::layer::SsmLayerState>()
        else {
            return Ok(());
        };
        if let Some(st) = ssm.ple.as_mut() {
            crate::layers::ple::PleLayer::free_seq_state(st, gpu)?;
        }
        Ok(())
    }
}

/// The PLE per-seq carry from a sequence's [`SsmLayerState`], lazily created
/// on first use. Errors if the state is not an `SsmLayerState`.
fn ple_seq_state<'a>(
    ple: &crate::layers::ple::PleLayer,
    state: &'a mut dyn LayerState,
    gpu: &dyn GpuBackend,
) -> Result<&'a mut crate::layers::ple::PleSeqState> {
    let ssm = state
        .as_any_mut()
        .downcast_mut::<crate::layer::SsmLayerState>()
        .ok_or_else(|| anyhow::anyhow!("PLE host layer state is not SsmLayerState"))?;
    if ssm.ple.is_none() {
        ssm.ple = Some(ple.new_seq_state(gpu)?);
    }
    Ok(ssm.ple.as_mut().expect("just created"))
}
