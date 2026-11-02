// SPDX-License-Identifier: AGPL-3.0-only

//! The switch and the rank protocol of the multi-sequence prefill (`multi`):
//! the head's command and the worker's side of it. Everything a rank could
//! refuse is checked on the head BEFORE the command, so no rank bails out of
//! a pass the other one runs.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::layers::ops::qwen4exp_rowinv::{MULTI_MAX_PROMPT, MULTI_MAX_SEQS};
use crate::model::impl_a2_ep_worker::EP_CMD_PREFILL_MULTI;
use crate::traits::PrefillSlice;

/// Where each prompt's metadata (positions, then slots) lies in the scratch
/// arena of a pass over prompts of `lens`, after the MoE top-k staging of
/// every row, and the end of the last region.
pub(super) fn multi_scratch_layout(
    lens: &[usize],
    topk: usize,
    mrope: bool,
) -> (Vec<usize>, usize) {
    let total: usize = lens.iter().sum();
    let mut at = (total * topk * 8 + 255) & !255;
    let starts = lens
        .iter()
        .map(|&len| {
            let start = at;
            let pos = if mrope { len * 12 } else { len * 4 };
            at = (at + ((pos + 7) & !7) + len * 8 + 255) & !255;
            start
        })
        .collect();
    (starts, at)
}

/// `ATLAS_QWEN4EXP_PREFILL_MULTI=1` (with ROWINV). Read once.
pub fn multi_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let set = matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_MULTI").as_deref(),
            Ok("1") | Ok("true")
        );
        let rowinv = crate::layers::ops::qwen4exp_rowinv::on();
        if set && !rowinv {
            tracing::warn!(
                "ATLAS_QWEN4EXP_PREFILL_MULTI=1 ignored: it needs ATLAS_QWEN4EXP_PREFILL_ROWINV=1"
            );
        }
        set && rowinv
    })
}

/// `ATLAS_QWEN4EXP_PREFILL_MULTI_CACHED=1` (default off, with `_MULTI`): a
/// prompt that hit the prefix cache rides the multi-sequence pass too (see
/// [`multi_segment_start`]). Read once.
pub fn multi_cached() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        multi_requested()
            && matches!(
                std::env::var("ATLAS_QWEN4EXP_PREFILL_MULTI_CACHED").as_deref(),
                Ok("1") | Ok("true")
            )
    })
}

/// What a prompt's prefix lookup left behind, for [`multi_segment_start`].
pub(super) struct Lookup {
    /// The lookup restored a snapshot (`marconi_skip`), at `skip_to` tokens.
    pub skip: bool,
    pub skip_to: usize,
    /// The sequence holds cached KV blocks (a radix match).
    pub shares_blocks: bool,
    /// The exact full-prompt snapshot shortcut (`marconi_exact_snap`), whose
    /// fixup only the single path's finish runs.
    pub exact_snap: bool,
    pub vision_pad: bool,
}

/// Where a prompt of `len` tokens starts its segment of the multi-sequence
/// pass, or `None` when it prefills alone after the pass. Reads only what
/// every rank agreed on (the match, the restore depth) and the tokens.
///
/// Without `cached` (the shipped behaviour) a prompt that restored a snapshot
/// or shares cached blocks prefills alone. With it, the pass computes what
/// that prompt's single prefill computes:
/// * a match with nothing restored recomputes from token 0 (the single
///   path's full recompute, same write floor);
/// * a restore starts the segment at the restored depth, its replay rows
///   under the match not written (`pc_policy::replay_floor`), as the single
///   path's uncached-portion pass does.
pub(super) fn multi_segment_start(cached: bool, len: usize, l: &Lookup) -> Option<usize> {
    if l.vision_pad {
        return None;
    }
    if !cached {
        return (!l.skip && !l.shares_blocks).then_some(0);
    }
    if !l.skip {
        return Some(0);
    }
    (!l.exact_snap && l.skip_to > 0 && l.skip_to < len).then_some(l.skip_to)
}

impl TransformerModel {
    /// Whether this model serves `prefill_multi` at all.
    /// Everything a pass would refuse mid-pass is refused here, before any
    /// prompt is deferred to it: the FP32-routing norm the highway path does
    /// not run, a KV cache that may swap layers to storage (HSS), and a QSA
    /// selection that would act inside a prompt of [`MULTI_MAX_PROMPT`].
    pub(in crate::model) fn prefill_multi_supported(&self) -> bool {
        multi_requested()
            && self.config.model_type == "qwen4_exp"
            && self.config.hc_mult > 0
            && !self.use_fp32_logits
            && self.lora.is_none()
            && std::env::var("ATLAS_FP32_ROUTING").as_deref() != Ok("1")
            && !(self.kv_cache.lock().config().cache_blocks_per_seq.is_some()
                && spark_storage::local_installed())
            && (self.config.index_topk == 0
                || self.config.index_topk + self.config.index_compress_ratio > MULTI_MAX_PROMPT)
    }

    /// Head side of `Model::prefill_multi`: send the prompts to the other
    /// ranks, then run the pass.
    pub(in crate::model) fn prefill_multi_head(
        &self,
        items: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        ensure!(
            self.prefill_multi_supported(),
            "prefill_multi is not enabled for this model"
        );
        // Everything the other ranks would refuse is refused here, before the
        // command: a rank that bails out of the pass leaves the pair desynced.
        ensure!(
            (1..=MULTI_MAX_SEQS).contains(&items.len()),
            "prefill_multi takes 1..={MULTI_MAX_SEQS} prompts, got {}",
            items.len()
        );
        for it in items.iter() {
            ensure!(
                it.chunk_start == 0
                    && it.is_last_chunk
                    && it.chunk_len == it.prompt_tokens.len()
                    && (1..=MULTI_MAX_PROMPT).contains(&it.chunk_len)
                    && it.seq.tokens.is_empty()
                    && it.seq.collect_prompt_logprobs.is_none(),
                "prefill_multi takes fresh whole prompts of 1..={MULTI_MAX_PROMPT} tokens"
            );
        }
        let total: usize = items.iter().map(|i| i.chunk_len).sum();
        ensure!(
            total <= self.buffers.max_batch_tokens(),
            "prefill_multi: {total} rows exceed the {}-row arena",
            self.buffers.max_batch_tokens()
        );
        let lens: Vec<usize> = items.iter().map(|i| i.chunk_len).collect();
        let topk = self.config.num_experts_per_tok;
        let (_, end) = multi_scratch_layout(&lens, topk, self.config.mrope_interleaved);
        ensure!(
            end <= self.buffers.scratch_bytes(),
            "prefill_multi: {total} rows need {end} scratch bytes"
        );
        // The logits rows, sized for any pass before any collective.
        self.prefill_multi_rows(MULTI_MAX_SEQS, self.gpu.default_stream())?;
        if self.multi_rank_protocol_active() {
            let slots: Vec<u32> = items.iter().map(|i| i.seq.slot_idx as u32).collect();
            let lens: Vec<u32> = items.iter().map(|i| i.chunk_len as u32).collect();
            let all: Vec<u32> = items
                .iter()
                .flat_map(|i| i.prompt_tokens.iter().copied())
                .collect();
            self.ep_broadcast_seq_and_cmd(0, EP_CMD_PREFILL_MULTI, true)?;
            self.ep_broadcast_u32(items.len() as u32)?;
            self.ep_broadcast_tokens(&slots)?;
            self.ep_broadcast_tokens(&lens)?;
            self.ep_broadcast_tokens(&all)?;
        }
        let mut seqs: Vec<(&[u32], &mut crate::traits::SequenceState)> = items
            .iter_mut()
            .map(|i| (i.prompt_tokens, &mut *i.seq))
            .collect();
        self.prefill_multi_pass(&mut seqs, stream)
    }

    /// Worker side of `EP_CMD_PREFILL_MULTI`.
    pub(in crate::model) fn ep_worker_prefill_multi(
        &self,
        slots: &mut [Option<crate::traits::SequenceState>],
    ) -> Result<bool> {
        // Sized before any collective, as the head does.
        self.prefill_multi_rows(MULTI_MAX_SEQS, self.gpu.default_stream())?;
        let n = self.ep_broadcast_u32(0)? as usize;
        ensure!(
            (1..=MULTI_MAX_SEQS).contains(&n),
            "EP prefill_multi of {n} sequences"
        );
        let ids = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let lens = self.ep_broadcast_tokens(&vec![0u32; n])?;
        ensure!(
            lens.iter()
                .all(|&l| (1..=MULTI_MAX_PROMPT).contains(&(l as usize))),
            "EP prefill_multi: a prompt length outside 1..={MULTI_MAX_PROMPT}"
        );
        let total: usize = lens.iter().map(|&l| l as usize).sum();
        let all = self.ep_broadcast_tokens(&vec![0u32; total])?;
        let mut refs = self.ep_worker_slot_refs(&ids, slots)?;
        let mut at = 0usize;
        let mut seqs: Vec<(&[u32], &mut crate::traits::SequenceState)> = Vec::with_capacity(n);
        for (seq, &len) in refs.iter_mut().zip(&lens) {
            seqs.push((&all[at..at + len as usize], &mut **seq));
            at += len as usize;
        }
        let stream = self.gpu.default_stream();
        self.prefill_multi_pass(&mut seqs, stream)?;
        // As after the worker's single prefill chunk (`0xFFFFFFF0`).
        for (_, seq) in seqs.iter() {
            if let Err(e) = self.normalize_ssm_states_dispatch(seq, stream) {
                tracing::warn!("Worker SSM state normalization failed: {e:#}");
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
#[path = "multi_ep_tests.rs"]
mod tests;
