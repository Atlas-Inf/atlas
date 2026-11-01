// SPDX-License-Identifier: AGPL-3.0-only

//! Grammar masks for the staged drafter logits, applied before selection.

use anyhow::Result;

use super::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    pub(super) fn mask_batched_draft_logits(
        &self,
        n: usize,
        grammar_bitmasks: Option<&[Option<Vec<i32>>]>,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // #102: grammar-mask each masked seq's logits rows 0/1 (anchor +
        // draft 0 — both predict pos+1). The pre-pass writes -inf in
        // place; the selector/Markov arms below stay unchanged.
        if self.kernels.grammar_bitmask.0 != 0
            && let Some(masks) = grammar_bitmasks
            && masks.iter().flatten().next().is_some()
        {
            let words = crate::layers::ops::grammar_bitmask_words(self.vocab_size as u32);
            let seq_logits_bytes = self.gamma * self.vocab_size * 2;
            for (sequence, mask) in masks.iter().take(n).enumerate() {
                let Some(mask) = mask else { continue };
                if mask.len() < words {
                    tracing::warn!(
                        "DFlash grammar bitmask short at seq {sequence} ({} < {words} words): skipping mask",
                        mask.len()
                    );
                    continue;
                }
                // #58: stage in the head's pinned mask region (slot =
                // sequence) and ship retained-async — a sync copy_h2d
                // here was one stream drain per masked sequence.
                let base = self
                    .batch_grammar_masks_host_pinned
                    .load(std::sync::atomic::Ordering::Relaxed);
                let mask_dev = self.batch_grammar_bitmask.offset(sequence * words * 4);
                let pinned_bytes = words * 4;
                if !base.is_null()
                    && sequence * pinned_bytes + pinned_bytes
                        <= self.batch_grammar_masks_pinned_bytes
                {
                    // SAFETY: slot `sequence` of the page-locked mask
                    // region is exclusively this sequence's for the
                    // whole call; the batch-token readback completes
                    // the retained copies before the slot is reused.
                    let staging = unsafe {
                        std::slice::from_raw_parts_mut(
                            base.add(sequence * pinned_bytes),
                            pinned_bytes,
                        )
                    };
                    let src: &[u8] = unsafe {
                        std::slice::from_raw_parts(mask.as_ptr() as *const u8, pinned_bytes)
                    };
                    staging.copy_from_slice(src);
                    ctx.gpu.copy_h2d_async_retained(staging, mask_dev, stream)?;
                } else {
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(mask.as_ptr() as *const u8, pinned_bytes)
                    };
                    ctx.gpu.copy_h2d(bytes, mask_dev)?;
                }
                for row in 0..2usize.min(self.gamma) {
                    crate::layers::ops::apply_grammar_bitmask(
                        ctx.gpu,
                        self.kernels.grammar_bitmask,
                        self.batch_logits
                            .offset(sequence * seq_logits_bytes + row * self.vocab_size * 2),
                        mask_dev,
                        self.vocab_size as u32,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
}
