// SPDX-License-Identifier: AGPL-3.0-only

//! What one score pass can take (`score.rs`): rows (the prefill arena and SSM
//! staging), sequences (the SSM state pool, less the slots held packs keep)
//! and whether this model scores at all.

use super::super::super::types::TransformerModel;

impl TransformerModel {
    /// Score passes run on one rank and read BF16 logits.
    pub(super) fn score_supported(&self) -> bool {
        !self.multi_rank_protocol_active() && self.comm.is_none() && !self.use_fp32_logits
    }

    /// See [`crate::traits::Model::score_capacity`].
    pub(in crate::model) fn score_capacity_impl(&self) -> (usize, usize) {
        self.score_capacity_held(self.score_packs.held())
    }

    /// [`Self::score_capacity_impl`] with `held` packs (each keeping one SSM
    /// slot), for callers that already hold the pack registry's lock.
    pub(super) fn score_capacity_held(&self, held: usize) -> (usize, usize) {
        if !self.score_supported() {
            return (0, 0);
        }
        let mut rows = self.buffers.max_batch_tokens();
        if self.ssm_pool.num_ssm_layers > 0 {
            rows = rows.min(self.gdn_buf_max_len);
        }
        let streams = if self.ssm_pool.num_ssm_layers > 0 {
            self.ssm_pool.max_slots.saturating_sub(held)
        } else {
            usize::MAX
        };
        // The metadata scratch (`score_forward`): per row, MoE top-k staging
        // (8 B per expert), this path's own per-stream metadata (24 B), the
        // staged positions (x3 with MRoPE) and slots (20 B); per stream, the
        // alignment pads and pointer, cu_seqlens, kv_lens, state-end and
        // h-state-pointer entries (256 B, generously); 64 KiB to spare.
        // Plus, per SSM layer, its pointer-table entries (`score_ssm.rs`).
        const PER_STREAM: usize = 256;
        let tables = super::score_ssm::TABLES * 8 * self.ssm_pool.num_ssm_layers;
        let per_stream = PER_STREAM + tables;
        let per_row = self.config.num_experts_per_tok.max(1) * 8 + 24 + 20;
        let fixed = 64 * 1024 + per_stream * streams.min(4096);
        let scratch_rows = self.buffers.scratch_bytes().saturating_sub(fixed) / per_row;
        (rows.min(scratch_rows), streams)
    }
}
