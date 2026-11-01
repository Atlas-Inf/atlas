// SPDX-License-Identifier: AGPL-3.0-only

//! The drafter's context window: the per-sequence accumulator length and the
//! paged-attention sliding window derived from it.

use super::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    /// Rows of the per-sequence context accumulator: the drafter window,
    /// capped at `max_seq_len`.
    pub(super) fn ctx_window_len(&self) -> usize {
        self.window_size
            .unwrap_or(self.max_seq_len)
            .min(self.max_seq_len)
    }

    /// SWA window in tokens for the paged attention kernel. 0 = no window.
    ///
    /// The kernel masks `q_rope_pos - kv_slot >= window`, comparing the
    /// query's ABSOLUTE position with a context-buffer SLOT index. A
    /// bidirectional head keeps only the last `window` context slots
    /// (`max_ctx_len`, slid by `commit_ctx`), so the buffer itself is the
    /// window; masking again hides every context key once the sequence passes
    /// `window` tokens (GLM C1-16K first-draft acceptance 0.000 before this).
    /// Causal (Lightning DSpark) heads keep the kernel window.
    pub(super) fn attn_sliding_window(&self) -> u32 {
        if !self.query_causal {
            return 0;
        }
        self.window_size.unwrap_or(0) as u32
    }
}
