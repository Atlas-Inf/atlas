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
        // WHY THE THIRD ROW COSTS 1.40x. The verify FFN streams each ACTIVATED
        // expert once, so cost tracks the UNION of experts over the verify
        // rows, and this model routes top-10 of 512:
        //
        //     E[distinct experts over R rows] = 512*(1 - (1 - 10/512)^R)
        //       R=2 -> 19.8 experts -> 54.8 MB/layer
        //       R=3 -> 29.4 experts -> 81.3 MB/layer     ratio 1.485
        //
        // at 3*2560*640*0.5625 = 2.76 MB per expert. The measured step ratio
        // 1.401 sits between the token ratio 1.303 and that 1.485, which is
        // where modest routing correlation puts it. Breaking even would need
        // ~40% of the third row's experts already resident from rows 1-2.
        //
        // Stated as scope, not as a law: for THIS geometry, this kernel, C=1
        // and the measured acceptance, K=3 does not pay. Independent routing
        // is a pessimistic estimate, not a strict bound, so this is not proof
        // that no K=3 can ever win — the same experiment upstream on a 27B
        // (fewer experts, more natural overlap) came out a wash at -0.2%
        // rather than a loss. Raising this needs the row-2 defect fixed AND a
        // verify row that pays for itself; the arithmetic says the second is
        // the binding constraint.
        Some(1)
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
