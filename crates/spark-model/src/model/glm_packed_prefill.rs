// SPDX-License-Identifier: AGPL-3.0-only

//! Packed multi-prompt prefill (`ATLAS_GLM_PACKED_PREFILL=1`, default off).
//!
//! # Why
//!
//! A prefill pass at a few dozen rows is bound by the routed-expert weights
//! it reads: GLM-5.3-Flash holds ~83 GB of experts per rank under EP2, and 50
//! tokens x top-8 over 288 experts touch ~75% of them, ~270 ms of a ~350 ms
//! pass. Eight short prompts prefilled one after another read most of the
//! experts eight times. Packed into one traversal (400 rows touch ~100%) they
//! read them once.
//!
//! # What runs
//!
//! New prompts of at most [`MAX_TOKENS`] tokens with no restorable prefix-cache state are laid out back to
//! back in the arena (`layer::glm_packed`). Each prompt gets exactly the
//! metadata, block reservation, embedding, DFlash capture and last-row LM
//! head a cold single-prompt prefill gives it; the layers run their row-local
//! work once over all rows and each prompt's recurrence and attention over
//! its own rows. The prompts skip the prefix-cache lookup and insert
//! (a prompt that has restorable state takes the ordinary path instead).
//!
//! # Exactness
//!
//! With the switch on a prompt runs this path whether it is alone (C1) or
//! packed with others (C4/C8), and every row-count dependent choice inside a
//! packed pass is made from the prompt's own row count or pinned
//! (`packed_scope`), so its output does not depend on its company.
//! Under the determinism tracer (`ATLAS_GLM_DET_TRACE`) every packed pass
//! logs one hash line per prompt to prove it on hardware (`PACKED_CHECK`).
//!
//! # EP
//!
//! One worker command, `0xFFFFFFE8`: preamble slot of the first prompt, then
//! N, `slots[N]`, `lengths[N]` and every prompt's tokens in one bulk broadcast.
//! The head decides the packing; the worker follows it.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

mod compute;

pub(super) const EP_CMD_GLM_PACKED_PREFILL: u32 = 0xFFFF_FFE8;

/// Largest prompt the packing takes.
pub const MAX_TOKENS: usize = 512;
/// The FlashKDA recurrence starts at 2048 rows and borrows the packed
/// buffer; a packed prompt stays below it.
pub const MAX_TOKENS_CAP: usize = 2047;
const _: () = assert!(MAX_TOKENS <= MAX_TOKENS_CAP && MAX_TOKENS_CAP < 2048);
/// Most prompts in one packed pass.
pub const MAX_PROMPTS: usize = 16;

/// `ATLAS_GLM_PACKED_PREFILL=1`. Read once per process.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PACKED_PREFILL").as_deref() == Ok("1"))
}

/// Arena rows of each prompt: back to back from row 0.
pub(crate) fn row_offsets(lens: &[usize]) -> Vec<usize> {
    lens.iter()
        .scan(0usize, |next, &len| {
            let at = *next;
            *next += len;
            Some(at)
        })
        .collect()
}

/// Bytes of one prompt's metadata block (positions [x3 under MRoPE], then
/// slots), 256-aligned like the fused chunk's.
pub(crate) fn meta_block_bytes(rows: usize, mrope: bool) -> usize {
    let streams = if mrope { 3 } else { 1 };
    let slot_offset = (rows * 4 * streams + 7) & !7;
    (slot_offset + rows * 8).div_ceil(256) * 256
}

/// Scratch layout of a packed pass: the MoE top-k staging at 0 (as for one
/// chunk of `total` rows), the joint positions/slots, then one metadata block
/// per prompt. Returns `(joint_at, owners_at, end)`.
pub(crate) fn scratch_layout(
    lens: &[usize],
    top_k: usize,
    mrope: bool,
) -> (usize, Vec<usize>, usize) {
    let total: usize = lens.iter().sum();
    let align = |b: usize| b.div_ceil(256) * 256;
    let joint_at = align(total * top_k * 4 * 2);
    let mut at = joint_at + meta_block_bytes(total, mrope);
    let owners = lens
        .iter()
        .map(|&rows| {
            let here = at;
            at += meta_block_bytes(rows, mrope);
            here
        })
        .collect();
    (joint_at, owners, at)
}

impl TransformerModel {
    /// Whether this model serves packed prefill at all.
    pub(super) fn glm_packed_supported(&self) -> bool {
        enabled()
            && self.config.model_type == "glm5_next"
            // The shared MTP prompt capture reads chunk rows from 0.
            && self.mtp_prefill_hidden.is_null()
            // A packed pass does not feed the native predictor its prompt.
            && !self.native_glm_mtp
            // The FP32 logits scratch holds one row.
            && !self.use_fp32_logits
            // The row-local fused mHC mixer the pins rely on (`glm_packed_scope`).
            && crate::layers::ops::hc_bf16_for(&self.config.model_type)
    }

    /// Whether `prompt` (for adapter `adapter_id`) may be packed: short, text
    /// only, and with no restorable prefix-cache state (such a prompt takes
    /// the ordinary path, which restores it).
    pub(super) fn glm_packed_admits(&self, prompt: &[u32], adapter_id: u64) -> bool {
        self.glm_packed_supported()
            && (2..=MAX_TOKENS).contains(&prompt.len())
            && !self.tokens_have_vision_pad(prompt)
            && !self.glm_packed_restorable(prompt, adapter_id)
    }

    /// A resident snapshot at a whole-block depth the radix holds for
    /// `prompt`: what a cold lookup could restore.
    fn glm_packed_restorable(&self, prompt: &[u32], adapter_id: u64) -> bool {
        if !self.prefix_cache.is_active() {
            return false;
        }
        let bs = self.kv_cache.lock().block_size();
        let matched = self
            .prefix_cache
            .peek_matched_tokens(prompt, bs, adapter_id);
        (1..=matched / bs).any(|b| {
            self.prefix_cache
                .snapshot_at(prompt, b * bs, adapter_id)
                .is_some()
        })
    }

    /// `(rows, prompts)` one packed pass may hold.
    pub(super) fn glm_packed_capacity(&self) -> (usize, usize) {
        let v = self.config.vocab_size * 2;
        let logits_rows = self.buffers.sizes().expert_gate_out / v.max(1);
        let rows = self.buffers.max_batch_tokens();
        (rows, MAX_PROMPTS.min(logits_rows))
    }

    /// Head: announce the packing and run it. Returns each prompt's
    /// last-row logits.
    pub(super) fn prefill_packed_impl(
        &self,
        prompts: &[&[u32]],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<DevicePtr>> {
        self.glm_packed_validate(prompts, seqs)?;
        if self.multi_rank_protocol_active() {
            let slots: Vec<u32> = seqs.iter().map(|s| s.slot_idx as u32).collect();
            let lens: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
            let tokens: Vec<u32> = prompts.iter().flat_map(|p| p.iter().copied()).collect();
            self.ep_broadcast_seq_and_cmd(
                slots[0],
                EP_CMD_GLM_PACKED_PREFILL,
                self.ep_protocol_v2,
            )?;
            self.ep_broadcast_u32(prompts.len() as u32)?;
            self.ep_broadcast_tokens(&slots)?;
            self.ep_broadcast_tokens(&lens)?;
            self.ep_broadcast_tokens(&tokens)?;
        }
        self.glm_packed_compute(prompts, seqs, self.gpu.default_stream())
    }

    /// Worker side of `0xFFFFFFE8`: the same pass, then each prompt's SSM
    /// normalization (the head's scheduler normalizes after the prefill).
    pub(super) fn glm_packed_receive(&self, slots: &mut [Option<SequenceState>]) -> Result<bool> {
        let n = self.ep_broadcast_u32(0)? as usize;
        ensure!(
            (1..=MAX_PROMPTS).contains(&n) && n <= slots.len(),
            "GLM packed prefill of {n} prompts"
        );
        let ids = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let lens = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let total: usize = lens.iter().map(|&l| l as usize).sum();
        ensure!(
            total <= self.buffers.max_batch_tokens(),
            "GLM packed prefill of {total} rows"
        );
        let tokens = self.ep_broadcast_tokens(&vec![0u32; total])?;
        let mut refs: Vec<(usize, &mut SequenceState)> = slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, s)| s.as_mut().map(|s| (i, s)))
            .collect();
        let mut seqs = Vec::with_capacity(n);
        for &id in &ids {
            let at = refs
                .iter()
                .position(|(i, _)| *i == id as usize)
                .with_context(|| format!("GLM packed prefill slot {id} unallocated or repeated"))?;
            seqs.push(refs.swap_remove(at).1);
        }
        let prompts: Vec<&[u32]> =
            row_offsets(&lens.iter().map(|&l| l as usize).collect::<Vec<_>>())
                .into_iter()
                .zip(&lens)
                .map(|(at, &len)| &tokens[at..at + len as usize])
                .collect();
        self.glm_packed_validate(&prompts, &seqs)?;
        self.sync_secondary()?;
        let stream = self.gpu.default_stream();
        self.glm_packed_compute(&prompts, &mut seqs, stream)?;
        for seq in seqs.iter() {
            if let Err(e) = self.normalize_ssm_states(seq, stream) {
                tracing::warn!("Worker SSM state normalization failed: {e:#}");
            }
        }
        Ok(true)
    }

    fn glm_packed_validate(&self, prompts: &[&[u32]], seqs: &[&mut SequenceState]) -> Result<()> {
        let (row_cap, prompt_cap) = self.glm_packed_capacity();
        let total: usize = prompts.iter().map(|p| p.len()).sum();
        ensure!(
            !prompts.is_empty() && prompts.len() == seqs.len() && prompts.len() <= prompt_cap,
            "GLM packed prefill of {} prompts (cap {prompt_cap})",
            prompts.len()
        );
        ensure!(
            total <= row_cap,
            "GLM packed prefill of {total} rows (cap {row_cap})"
        );
        for (p, s) in prompts.iter().zip(seqs) {
            ensure!(
                (2..=MAX_TOKENS_CAP).contains(&p.len()),
                "GLM packed prefill prompt of {} tokens",
                p.len()
            );
            ensure!(
                s.seq_len == 0 && s.tokens.is_empty() && s.block_table.is_empty(),
                "GLM packed prefill needs a fresh sequence (slot {})",
                s.slot_idx
            );
        }
        let top_k = self.config.num_experts_per_tok;
        let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let (_, _, end) = scratch_layout(&lens, top_k, self.config.mrope_interleaved);
        ensure!(
            end <= self.buffers.scratch_bytes(),
            "GLM packed prefill metadata ({end} bytes) exceeds scratch"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "glm_packed_prefill_tests.rs"]
mod tests;
