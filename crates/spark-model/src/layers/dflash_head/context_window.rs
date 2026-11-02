// SPDX-License-Identifier: AGPL-3.0-only

//! The drafter's context window: the per-sequence accumulator length and the
//! paged-attention sliding window derived from it.

use super::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    /// Rows of the per-sequence context accumulator: the drafter window,
    /// capped at `max_seq_len`.
    pub(super) fn ctx_window_len(&self) -> usize {
        let capacity = self.ctx_window.min(self.max_seq_len);
        if self.query_causal {
            capacity
        } else {
            self.window_size.unwrap_or(capacity).min(capacity)
        }
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

#[cfg(test)]
mod tests {
    use super::super::free_state_tests::zero_head;

    #[test]
    fn causal_context_retains_the_configured_budget() {
        let mut head = zero_head();
        head.max_seq_len = 524_288;
        head.ctx_window = 16_384;
        head.window_size = Some(2_048);
        head.query_causal = true;
        assert_eq!(head.ctx_window_len(), 16_384);
        assert_eq!(head.attn_sliding_window(), 2_048);
    }

    #[test]
    fn noncausal_window_never_exceeds_scratch_or_sequence_capacity() {
        let mut head = zero_head();
        head.max_seq_len = 524_288;
        head.ctx_window = 4_096;
        head.window_size = Some(2_048);
        head.query_causal = false;
        assert_eq!(head.ctx_window_len(), 2_048);
        assert_eq!(head.attn_sliding_window(), 0);
        head.ctx_window = 1_024;
        assert_eq!(head.ctx_window_len(), 1_024);
        head.max_seq_len = 512;
        assert_eq!(head.ctx_window_len(), 512);
    }
}
