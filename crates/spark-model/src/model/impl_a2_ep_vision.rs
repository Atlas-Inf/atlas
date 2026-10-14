// SPDX-License-Identifier: AGPL-3.0-only

//! EP vision-state protocol: rank-0 broadcast and worker receive of the
//! encoder rows plus MRoPE metadata and item content hashes
//! (`EP_CMD_VISION_STATE`). A deferred encode sends its rows later, from the
//! chunk-0 prefix lookup (`vision_cache`).

use anyhow::Result;

use super::types::TransformerModel;
use super::vision_cache::VisionCacheState;
use super::vision_transport::{
    EP_CMD_VISION_STATE, VISION_HEADER_WORDS, VisionWireState, grid_words, parse_grid_words,
    parse_state_header, payload_bytes, validate_state,
};

impl TransformerModel {
    /// Broadcast the rank-0 vision encoder result and its MRoPE metadata before
    /// a worker enters the matching prefill command. The payload uses the
    /// encoder's own device buffer, so it never aliases the small command or
    /// token scratch buffers. `enabled=false` is an explicit text-state clear.
    pub(crate) fn ep_broadcast_vision_state_for_seq_dispatch(
        &self,
        seq_id: u32,
        enabled: bool,
        row_base: usize,
        grid_base: usize,
        owned_images: usize,
        slice_rows: usize,
    ) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            if !enabled {
                self.clear_vision_state();
            }
            return Ok(());
        }

        // Keep the worker's command stream aligned even when a malformed
        // rank-0 model has image input but no local encoder. Send an explicit
        // text state before returning the local configuration error instead of
        // leaving rank 1 blocked on the next header broadcast.
        if enabled && self.vision_encoder.is_none() {
            self.ep_broadcast_seq_and_cmd(seq_id, EP_CMD_VISION_STATE, self.ep_protocol_v2)?;
            self.ep_broadcast_tokens(&[0; VISION_HEADER_WORDS])?;
            self.clear_vision_state();
            anyhow::bail!("vision request cannot use the multi-rank protocol without an encoder");
        }

        let (state, grids, hashes, payload) = if enabled {
            let ve = self.vision_encoder.as_ref().expect("checked above");
            let grids = self.vision_image_grids.lock().clone();
            let (hashes, deferred) = {
                let st = self.vision_cache.lock();
                (st.hashes.clone(), st.deferred.is_some())
            };
            // A deferred encode has no rows yet: the header carries the rows
            // its grids describe, and the payload follows from the lookup.
            let rows = if deferred {
                grids.iter().map(|&(t, h, w)| t * h * w).sum()
            } else {
                *self.vision_embed_patches.lock()
            };
            let state = VisionWireState {
                rows,
                grid_count: grids.len(),
                row_base,
                grid_base,
                owned_images,
                slice_rows: if owned_images == 0 { rows } else { slice_rows },
                deferred,
            };
            let state = validate_state(&state, ve.output_rows())?;
            let payload = payload_bytes(&state, ve.out_hidden_size(), ve.output_rows())?;
            anyhow::ensure!(
                payload > 0,
                "vision request produced an empty encoder payload"
            );
            (state, grids, payload)
            let payload = if deferred {
                0
            } else {
                let payload = payload_bytes(&state, ve.out_hidden_size, ve.output_rows())?;
                anyhow::ensure!(
                    payload > 0,
                    "vision request produced an empty encoder payload"
                );
                payload
            };
            (state, grids, hashes, payload)
        } else {
            // Text requests must reset worker-global vision state. A stale
            // buffer is otherwise observable by the next image request after
            // slot reuse, even though this request contains no pad tokens.
            self.clear_vision_state();
            (VisionWireState::default(), Vec::new(), Vec::new(), 0)
        };

        *self.vision_slice_rows.lock() = state.slice_rows;

        if payload > 0 {
            // The encoder launches on default_stream while NCCL uses its own
            // legacy stream. Explicitly complete the producer before handing
            // that pointer to NCCL; an event on the prefill stream would not
            // order the communicator's stream.
            self.gpu.synchronize(self.gpu.default_stream())?;
        }

        // Convert every local value before sending the command preamble. A
        // wire-type failure must not leave rank 1 waiting for a header that
        // rank 0 will never send.
        let word = |v: usize, what: &str| {
            u32::try_from(v).map_err(|_| anyhow::anyhow!("vision {what} exceeds u32 wire type"))
        };
        let header = [
            word(state.rows, "rows")?,
            word(state.grid_count, "grid count")?,
            word(state.row_base, "row base")?,
            word(state.grid_base, "grid base")?,
            word(state.owned_images, "owned image count")?,
            word(state.slice_rows, "slice rows")?,
            u32::from(state.deferred),
        ];
        debug_assert_eq!(header.len(), VISION_HEADER_WORDS);
        let grid_words = grid_words(&grids, &hashes)?;
        // Apply the same bounded geometry and row-count validation locally
        // before the command preamble. Otherwise rank 0 could enter the
        // payload collective with metadata that rank 1 rejects first.
        parse_grid_words(&grid_words, &state)?;

        self.ep_broadcast_seq_and_cmd(seq_id, EP_CMD_VISION_STATE, self.ep_protocol_v2)?;
        self.ep_broadcast_tokens(&header)?;
        if state.grid_count > 0 {
            self.ep_broadcast_tokens(&grid_words)?;
        }
        if payload > 0 {
            let comm = self
                .comm
                .as_ref()
                .expect("vision payload without communicator");
            let ve = self
                .vision_encoder
                .as_ref()
                .expect("vision payload without encoder");
            comm.broadcast(ve.out_row().0, payload, 0)?;
        }
        Ok(())
    }

    /// Worker side of `EP_CMD_VISION_STATE`.
    pub(super) fn ep_worker_recv_vision_state(&self) -> Result<()> {
        let header = self.ep_broadcast_tokens(&[0; VISION_HEADER_WORDS])?;
        let max_rows = self
            .vision_encoder
            .as_ref()
            .map_or(0, |ve| ve.output_rows());
        let state = parse_state_header(&header, max_rows)?;
        let grid_words = if state.grid_count > 0 {
            self.ep_broadcast_tokens(&vec![
                0;
                state.grid_count
                    * super::vision_transport::VISION_GRID_WORDS
            ])?
        } else {
            Vec::new()
        };
        let grids = parse_grid_words(&grid_words, &state)?;
        let payload = if let Some(ve) = &self.vision_encoder {
            payload_bytes(&state, ve.out_hidden_size(), ve.output_rows())?
        let (grids, hashes) = parse_grid_words(&grid_words, &state)?;
        let payload = if state.deferred {
            0
        } else if let Some(ve) = &self.vision_encoder {
            payload_bytes(&state, ve.out_hidden_size, ve.output_rows())?
        } else {
            anyhow::ensure!(
                state.rows == 0,
                "vision payload received by a rank without an encoder"
            );
            0
        };
        if payload > 0 {
            let ve = self
                .vision_encoder
                .as_ref()
                .expect("validated vision payload without encoder");
            self.comm
                .as_ref()
                .expect("validated vision payload without communicator")
                .broadcast(ve.out_row().0, payload, 0)?;
        }
        *self.vision_embed_patches.lock() = if state.deferred { 0 } else { state.rows };
        *self.vision_image_grids.lock() = grids;
        *self.vision_cache.lock() = VisionCacheState {
            hashes,
            deferred: state.deferred.then(Vec::new),
            ..Default::default()
        };
        *self.vision_row_base.lock() = state.row_base;
        *self.vision_grid_base.lock() = state.grid_base;
        *self.vision_owned_images.lock() = state.owned_images;
        *self.vision_slice_rows.lock() = state.slice_rows;
        Ok(())
    }
}
