// SPDX-License-Identifier: AGPL-3.0-only

//! Bodies behind `impl Model for TransformerModel` entry points that grew
//! GLM/vision/distributed logic: prefill stream selection + fallible eager
//! drafter capture, the vision slice base, the native-only MTP guard and
//! GLM capability predicates.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

#[cfg(test)]
#[path = "prefill_stream_tests.rs"]
mod prefill_stream_tests;

impl TransformerModel {
    pub(super) fn set_vision_slice_base_entry(
        &self,
        row_base: usize,
        grid_base: usize,
        owned_images: usize,
        slice_rows: usize,
    ) {
        *self.vision_row_base.lock() = row_base;
        *self.vision_grid_base.lock() = grid_base;
        *self.vision_owned_images.lock() = owned_images;
        *self.vision_slice_rows.lock() = slice_rows;
    }

    pub(super) fn prefill_entry(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
    ) -> Result<DevicePtr> {
        // Full prefill computes on default; its eager consumer must follow it.
        let stream = self.gpu.default_stream();
        self.stamp_overlay_route(seq.adapter_slot);

        (|| {
            let logits = self.prefill_dispatch(tokens, seq, stream)?;
            self.try_eager_drafter_prefill(seq, true, stream)?;
            Ok(logits)
        })()
    }

    pub(super) fn prefill_chunk_entry(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        // Distributed dispatch uses default for command/collective ordering.
        // Keep capture consumption and scratch reuse on that same stream.
        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };
        self.stamp_overlay_route(seq.adapter_slot);

        (|| {
            let logits = self.prefill_chunk_dispatch(
                tokens,
                seq,
                chunk_start,
                chunk_len,
                is_last_chunk,
                stream,
            )?;
            self.try_eager_drafter_prefill(seq, is_last_chunk, stream)?;
            Ok(logits)
        })()
    }

    pub(super) fn prefill_twophase_entry(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_size: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };
        self.stamp_overlay_route(seq.adapter_slot);

        (|| {
            let logits = self.prefill_twophase_dispatch(tokens, seq, chunk_size, stream)?;
            self.try_eager_drafter_prefill(seq, true, stream)?;
            Ok(logits)
        })()
    }

    pub(super) fn has_shared_prompt_capture_impl(&self) -> bool {
        // Only the MTP drafter's prompt capture is shared; DFlash captures
        // into per-sequence proposer state.
        !self.mtp_prefill_hidden.is_null()
    }

    /// Refuse an MTP propose on a native-only (`disable_mtp`) sequence.
    pub(super) fn ensure_mtp_propose_allowed(&self, seq: &SequenceState) -> Result<()> {
        anyhow::ensure!(
            !seq.disable_mtp,
            "MTP propose invoked for native-only sequence"
        );
        Ok(())
    }

    pub(super) fn supports_chunked_mla_impl(&self) -> bool {
        self.config.model_type == "glm5_next"
            && self.config.index_kpool > 0
            && self.config.index_topk > 0
    }
}
