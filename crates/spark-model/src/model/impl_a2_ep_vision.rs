// SPDX-License-Identifier: AGPL-3.0-only

//! EP vision-state protocol: rank-0 broadcast and worker receive of the
//! encoder rows plus MRoPE metadata (`EP_CMD_VISION_STATE`).

use anyhow::Result;

use super::types::TransformerModel;
use super::vision_transport::{
    EP_CMD_VISION_STATE, VISION_HEADER_WORDS, VisionWireState, parse_grid_words,
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
                *self.vision_embed_patches.lock() = 0;
                self.vision_image_grids.lock().clear();
                *self.vision_row_base.lock() = 0;
                *self.vision_grid_base.lock() = 0;
                *self.vision_owned_images.lock() = 0;
                *self.vision_slice_rows.lock() = 0;
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
            *self.vision_embed_patches.lock() = 0;
            self.vision_image_grids.lock().clear();
            *self.vision_row_base.lock() = 0;
            *self.vision_grid_base.lock() = 0;
            *self.vision_owned_images.lock() = 0;
            *self.vision_slice_rows.lock() = 0;
            anyhow::bail!("vision request cannot use the multi-rank protocol without an encoder");
        }

        let (state, grids, payload) = if enabled {
            let ve = self.vision_encoder.as_ref().expect("checked above");
            let rows = *self.vision_embed_patches.lock();
            let grids = self.vision_image_grids.lock().clone();
            let state = VisionWireState {
                rows,
                grid_count: grids.len(),
                row_base,
                grid_base,
                owned_images,
                slice_rows: if owned_images == 0 { rows } else { slice_rows },
            };
            let state = validate_state(&state, ve.output_rows())?;
            let payload = payload_bytes(&state, ve.out_hidden_size, ve.output_rows())?;
            anyhow::ensure!(
                payload > 0,
                "vision request produced an empty encoder payload"
            );
            (state, grids, payload)
        } else {
            // Text requests must reset worker-global vision state. A stale
            // buffer is otherwise observable by the next image request after
            // slot reuse, even though this request contains no pad tokens.
            *self.vision_embed_patches.lock() = 0;
            self.vision_image_grids.lock().clear();
            *self.vision_row_base.lock() = 0;
            *self.vision_grid_base.lock() = 0;
            *self.vision_owned_images.lock() = 0;
            *self.vision_slice_rows.lock() = 0;
            (
                VisionWireState {
                    rows: 0,
                    grid_count: 0,
                    row_base: 0,
                    grid_base: 0,
                    owned_images: 0,
                    slice_rows: 0,
                },
                Vec::new(),
                0,
            )
        };

        *self.vision_slice_rows.lock() = state.slice_rows;

        if enabled {
            // The encoder launches on default_stream while NCCL uses its own
            // legacy stream. Explicitly complete the producer before handing
            // that pointer to NCCL; an event on the prefill stream would not
            // order the communicator's stream.
            self.gpu.synchronize(self.gpu.default_stream())?;
        }

        // Convert every local value before sending the command preamble. A
        // wire-type failure must not leave rank 1 waiting for a header that
        // rank 0 will never send.
        let header = [
            u32::try_from(state.rows)
                .map_err(|_| anyhow::anyhow!("vision rows exceed u32 wire type"))?,
            u32::try_from(state.grid_count)
                .map_err(|_| anyhow::anyhow!("vision grid count exceeds u32 wire type"))?,
            u32::try_from(state.row_base)
                .map_err(|_| anyhow::anyhow!("vision row base exceeds u32 wire type"))?,
            u32::try_from(state.grid_base)
                .map_err(|_| anyhow::anyhow!("vision grid base exceeds u32 wire type"))?,
            u32::try_from(state.owned_images)
                .map_err(|_| anyhow::anyhow!("vision owned image count exceeds u32 wire type"))?,
            u32::try_from(state.slice_rows)
                .map_err(|_| anyhow::anyhow!("vision slice rows exceed u32 wire type"))?,
        ];
        debug_assert_eq!(header.len(), VISION_HEADER_WORDS);
        let mut grid_words = Vec::with_capacity(state.grid_count * 3);
        for (t_len, grid_h, grid_w) in &grids {
            grid_words.extend([
                u32::try_from(*t_len)
                    .map_err(|_| anyhow::anyhow!("vision temporal length exceeds u32 wire type"))?,
                u32::try_from(*grid_h)
                    .map_err(|_| anyhow::anyhow!("vision grid height exceeds u32 wire type"))?,
                u32::try_from(*grid_w)
                    .map_err(|_| anyhow::anyhow!("vision grid width exceeds u32 wire type"))?,
            ]);
        }
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
            comm.broadcast(ve.buf_out.0, payload, 0)?;
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
            self.ep_broadcast_tokens(&vec![0; state.grid_count * 3])?
        } else {
            Vec::new()
        };
        let grids = parse_grid_words(&grid_words, &state)?;
        let payload = if let Some(ve) = &self.vision_encoder {
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
                .broadcast(ve.buf_out.0, payload, 0)?;
        }
        *self.vision_embed_patches.lock() = state.rows;
        *self.vision_image_grids.lock() = grids;
        *self.vision_row_base.lock() = state.row_base;
        *self.vision_grid_base.lock() = state.grid_base;
        *self.vision_owned_images.lock() = state.owned_images;
        *self.vision_slice_rows.lock() = state.slice_rows;
        Ok(())
    }
}
