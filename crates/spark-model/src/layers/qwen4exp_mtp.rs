// SPDX-License-Identifier: AGPL-3.0-only
//!
//! Ported from Enntity/atlas (AGPL-3.0-only, sparkqwen drafter series).

//! Qwen3.8-Flash-Next Multi-Token-Prediction (MTP) draft proposer.
//!
//! Implements [`DraftProposer`] over the `Qwen4ExpMtpModule` loaded by
//! `load_qwen4exp_mtp_module`. Like DeepSeek-V4's proposer (and unlike the
//! Qwen-shaped [`crate::layers::MtpHead`], a hand-rolled attention + MoE
//! block), the body here is a REUSED full model layer, so the proposer only
//! wraps it with the MTP-specific ends.
//!
//! ## Forward (`propose`, one draft position)
//!
//! ```text
//!   n_h[s] = grouped_rms_norm(streams[s], pre_fc_norm_hidden[s])   // s < hc_mult
//!   n_e    = rms_norm(embed[token], pre_fc_norm_embedding)
//!   e      = fc_embedding · n_e                                    // shared across streams
//!   streams[s] = e + fc_hidden · n_h[s]                            // per-stream, shared weight
//!   body.decode(streams, …, mtp_kv_cache)                          // MIDDLE mHC + attn + MoE
//!   h_out  = hc_head(streams)                                      // mtp.hyper_connection_mixer
//!   logits = lm_head(h_out)                                        // SHARED head
//! ```
//!
//! ## Why the input is the stream highway, not `target_hidden`
//!
//! `target_hidden` is the target's post-mixer, `hidden`-wide state. This
//! drafter cannot use it: `mtp.pre_fc_norm_hidden` is `[hc_mult * hidden]`, so
//! the block consumes the residual BEFORE the model-level mixer collapses it.
//! That state is exactly what `ctx.buffers.hc_streams()` still holds when the
//! proposer runs — the last trunk layer collapses INTO `hidden` and leaves the
//! streams intact — so no new plumbing is needed, and the `target_hidden`
//! argument is deliberately unused.
//!
//! Reconstructing the streams from the collapsed hidden (broadcast, or
//! `hc_expand`) is NOT equivalent: the mixer's collapse is lossy and the
//! checkpoint ships no MTP expand weights. It would run and draft badly.
//!
//! For draft positions after the first, the input is the drafter's OWN streams
//! from the previous position, which `body.decode` left in place — the same
//! recurrence, one buffer.
//!
//! ## Two conventions this file must match exactly
//!
//! Both are read off the shadowed `hyper_connection.cu`, not assumed:
//!
//!   1. `hc_norm` is **GROUPED**, `group_size = hidden`: each stream is
//!      normalised over its OWN slice, not once across the flattened
//!      `hc_mult * hidden` row.
//!   2. The scale is **offset from 1** (`x * rms * (1 + w)`), not `x * rms * w`.
//!
//! `rms_norm_f32` implements both (and reads the FP32 highway directly), so a
//! per-stream launch with the matching weight slice is exactly the grouped
//! form. Getting either wrong yields finite, plausible, wrong drafts — which
//! costs an acceptance run to notice.

use std::any::Any;

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::qwen4_exp::Qwen4ExpMtpModule;
use crate::weight_map::DenseWeight;

/// Per-sequence state for the qwen4_exp MTP proposer.
pub struct Qwen4ExpMtpProposerState {
    /// Block table for the drafter's OWN KV cache.
    pub block_table: Vec<u32>,
    /// Current sequence length in the drafter's KV cache.
    pub seq_len: usize,
    /// Drafts produced by the last `propose` (for `after_verify` trimming).
    pub last_num_drafted: usize,
    /// A propose wrote rows no verdict has settled yet (`qwen4exp_mtp_kv.rs`).
    pub awaiting_verdict: bool,
    /// Per-layer state for the reused body.
    pub body_state: Box<dyn LayerState>,
    /// Drafts for positions below it skip the end tokens (the target ban's
    /// min_tokens floor, `qwen4exp_mtp_ban.rs`; 0 = none).
    pub end_floor: usize,
    /// The verified rows to absorb (`qwen4exp_mtp_absorb.rs`).
    pub absorb: qwen4exp_mtp_absorb::Absorb,
}

impl ProposerState for Qwen4ExpMtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn set_end_floor(&mut self, floor: usize) {
        let on = crate::traits::EosBan::target_ban_installed();
        self.end_floor = qwen4exp_mtp_ban::drafter_floor(floor, on);
    }
}

/// Qwen3.8-Flash-Next MTP draft proposer.
pub struct Qwen4ExpMtpHead {
    module: Qwen4ExpMtpModule,
    /// Shared token embedding table (BF16), from the target.
    embed_tokens: DenseWeight,
    /// Shared LM head (BF16 on this checkpoint), from the target. Every draft
    /// is re-verified by the target's own head, so the draft head can only
    /// affect acceptance, never an emitted token.
    lm_head: DenseWeight,
    /// `--mtp-vocab`: the grammar-masked draft path's prefix of `lm_head`.
    mtp_vocab_size: u32,
    /// The unmasked draft path's head (`qwen4exp_draft_head.rs`).
    draft: crate::layers::qwen4exp_draft_head::DraftHead,
    /// `ATLAS_QWEN4EXP_MTP_CONFIDENCE`: stop the chain before a draft (after
    /// the first) whose draft-head probability is below this. 0 = off.
    conf_stop: f32,
    /// The drafter's own single-layer KV cache.
    kv_cache: Mutex<PagedKvCache>,

    // Scratch owned by the proposer rather than borrowed from `ctx.buffers`.
    // 6 small buffers, sized for `PROPOSE_BATCH_MAX` rows (~400 KB at
    // hidden=2560/hc=4; the per-sequence path uses row 0): cheap, and it
    // removes every aliasing question against the trunk's buffers, which the
    // body is simultaneously using.
    embed_buf: DevicePtr,
    normed_e: DevicePtr,
    e_branch: DevicePtr,
    normed_h: DevicePtr,
    h_streams: DevicePtr,
    h_out: DevicePtr,
    argmax_out: DevicePtr,

    // Kernel handles.
    rms_norm_k: KernelHandle,
    rms_norm_f32_k: KernelHandle,
    /// `ATLAS_QWEN4EXP_MTP_GROUPED_NORM=1`: `rms_norm_f32_grouped`, else 0.
    rms_norm_f32_grouped_k: KernelHandle,
    f32_residual_add_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    hc_expand_k: KernelHandle,
    hc_head_k: KernelHandle,
    argmax_k: KernelHandle,

    // Batched propose (`qwen4exp_mtp_batch.rs`). Optional kernels: a zero
    // handle keeps every batch on the per-sequence path.
    dense_gemv_batchm_k: KernelHandle,
    /// The lane's 16/32-row tiers for the (row, stream) projections.
    wide_rows: super::ops::Qwen4ExpWideRows,
    batched_embed_k: KernelHandle,
    argmax_batch_k: KernelHandle,
    argmax_batch_lp_k: KernelHandle,
    /// Token chain, confidences and attention metadata of one batched propose
    /// call, uploaded once and read back once (`qwen4exp_mtp_batch::Slab`).
    batch_slab: DevicePtr,
    /// Block-table entries per row that `batch_slab` holds.
    batch_slab_blocks: usize,
    /// `ATLAS_QWEN4EXP_MTP_DRAFT_TP`: the worker projects half of the draft
    /// head of each batched propose the model announces (`draft_tp`).
    tp: Option<draft_tp::DraftTp>,
    /// `ATLAS_QWEN4EXP_MTP_ABSORB` (`qwen4exp_mtp_absorb.rs`).
    absorb: qwen4exp_mtp_absorb::AbsorbBufs,
}

#[path = "qwen4exp_mtp_new.rs"]
mod qwen4exp_mtp_new;

#[path = "qwen4exp_mtp_forward.rs"]
mod qwen4exp_mtp_forward;

#[path = "qwen4exp_mtp_batch.rs"]
mod qwen4exp_mtp_batch;

#[path = "qwen4exp_mtp_kv.rs"]
mod qwen4exp_mtp_kv;
use qwen4exp_mtp_kv::drafter_kv_config;

#[path = "qwen4exp_mtp_absorb.rs"]
pub mod qwen4exp_mtp_absorb;

#[path = "qwen4exp_mtp_rows.rs"]
mod qwen4exp_mtp_rows;

#[path = "qwen4exp_mtp_prompt.rs"]
pub mod qwen4exp_mtp_prompt;

#[path = "qwen4exp_mtp_ban.rs"]
mod qwen4exp_mtp_ban;

#[path = "qwen4exp_mtp_tp.rs"]
pub mod draft_tp;

impl DraftProposer for Qwen4ExpMtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
    }

    fn propose(
        &self,
        last_token: u32,
        _target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Qwen4ExpMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid qwen4_exp MTP proposer state"))?;

        qwen4exp_mtp_kv::settle_unverified(st);
        self.absorb_before_propose(&mut [&mut *st], &[position], 1, ctx, stream)?;
        crate::layers::qsa::draft_share_set(st.body_state.as_mut(), true); // MTP_INDEX_SHARE
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_token = last_token;
        for i in 0..num_drafts {
            // Each step reads the stream highway, which the previous step's
            // `body.decode` left holding the drafter's own residual — so
            // unlike the collapsed-hidden proposers there is nothing to thread
            // between iterations.
            let want_conf = i > 0 && self.conf_stop > 0.0;
            let (draft, conf) = self.forward_one(
                current_token,
                position + i,
                st,
                ctx,
                stream,
                grammar_bitmask,
                want_conf,
            )?;
            // Confidence stop: the first draft is always kept; a later one
            // whose draft-head probability is below the threshold ends the
            // chain, unproposed. Its drafter KV row is dropped as a rejected
            // draft's is. Proposals only — the verify decides every token.
            if let Some(p) = conf
                && p < self.conf_stop
            {
                st.seq_len -= 1;
                tracing::debug!(
                    "qwen4_exp MTP confidence stop at draft {i}: p={p:.3} < {}",
                    self.conf_stop
                );
                break;
            }
            tracing::debug!(
                "qwen4_exp MTP propose[{i}]: token={current_token} pos={} mtp_seq_len={} -> draft={draft}",
                position + i,
                st.seq_len,
            );
            drafts.push(draft);
            current_token = draft;
        }
        crate::layers::qsa::draft_share_set(st.body_state.as_mut(), false);
        st.last_num_drafted = drafts.len();
        st.awaiting_verdict = true;
        Ok(drafts)
    }

    /// One drafter forward a draft position for all sequences, chained on the
    /// device (`qwen4exp_mtp_batch.rs`); `Ok(None)` outside `Self::batch_admits`.
    fn propose_batch(
        &self,
        last_tokens: &[u32],
        _target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        _expected_owners: Option<&[crate::layers::dflash_head::SequenceGeneration]>,
        ctx: &ForwardContext,
        stream: u64,
        out_conf: Option<&mut Vec<Vec<f32>>>,
        grammar_bitmasks: Option<&[Option<Vec<i32>>]>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        // An announced draft-TP propose (`draft_tp`): from here on every exit
        // issues the swaps the worker walks.
        let plan = draft_tp::Plan {
            n: last_tokens.len(),
            drafts: num_drafts,
        };
        let tp = self.draft_tp_run(ctx.comm, ctx.gpu, plan, stream)?;
        // Grammar masks are per position and host-applied: per-sequence only.
        let masked = grammar_bitmasks.is_some_and(|m| m.iter().any(Option::is_some));
        if masked || !self.batch_admits(last_tokens.len(), num_drafts, ctx) {
            return Ok(None);
        }
        let mut sts = Vec::with_capacity(states.len());
        for s in states.iter_mut() {
            match s.as_any_mut().downcast_mut::<Qwen4ExpMtpProposerState>() {
                Some(st) => sts.push(st),
                None => return Ok(None),
            }
        }
        let drafts = self.propose_batch_impl(
            last_tokens,
            positions,
            num_drafts,
            &mut sts,
            ctx,
            stream,
            out_conf,
            tp.as_ref(),
        )?;
        tp.map_or(Ok(()), draft_tp::TpRun::finish)?;
        Ok(Some(drafts))
    }

    fn draft_tp_word(
        &self,
        comm: &dyn spark_comm::CommBackend,
        n: usize,
        num_drafts: usize,
        grammar: bool,
        ctx: &ForwardContext,
    ) -> Option<u32> {
        self.tp_announce_word(comm, n, num_drafts, grammar, ctx)
    }

    fn propose_batch_max(
        &self,
        buffers: &spark_runtime::buffers::BufferArena,
        config: &atlas_core::config::ModelConfig,
    ) -> usize {
        self.batch_width(buffers, config)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Qwen4ExpMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid qwen4_exp MTP proposer state"))?;
        qwen4exp_mtp_kv::after_verdict(st, num_accepted);
        Ok(())
    }

    fn stage_absorb(
        &self,
        state: &mut dyn ProposerState,
        streams: DevicePtr,
        tokens: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.stage_absorb_impl(state, streams, tokens, gpu, stream)
    }

    fn note_prompt_rows(
        &self,
        state: &mut dyn ProposerState,
        first_pos: usize,
        rows: usize,
        hc_row: usize,
    ) {
        self.note_prompt_rows_impl(state, first_pos, rows, hc_row)
    }

    fn absorb_prompt(
        &self,
        state: &mut dyn ProposerState,
        prompt: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.absorb_prompt_impl(state, prompt, ctx, stream)
    }

    fn prompt_tail_token(&self, state: &mut dyn ProposerState, tokens: &[u32]) {
        self.prompt_tail_token_impl(state, tokens)
    }

    fn free_state(
        &self,
        _gpu: &dyn GpuBackend,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        state: &mut dyn ProposerState,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Qwen4ExpMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid qwen4_exp MTP proposer state"))?;
        if !st.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&st.block_table);
            st.block_table.clear();
        }
        self.absorb.release(&mut st.absorb);
        st.seq_len = 0;
        Ok(())
    }
}

#[cfg(test)]
#[path = "qwen4exp_mtp_kv_tests.rs"]
mod drafter_kv_tests;
