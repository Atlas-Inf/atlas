// SPDX-License-Identifier: AGPL-3.0-only

//! Literal M1 target verification for the Lightning losslessness oracle.

use anyhow::Result;

use crate::layer::SsmLayerState;
use crate::traits::{Model, SequenceState};

use super::super::TransformerModel;

impl TransformerModel {
    pub(super) fn decode_verify_serial_m1_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.checkpoint_ssm_states_dispatch(seq)?;
        let h_bytes = self.config.ssm_h_state_bytes();
        let conv_bytes = self.config.ssm_conv_state_bytes();
        let capture_row_bytes = self.dflash_capture_layers.len() * self.config.hidden_size * 2;
        let logits_row_bytes = self.config.vocab_size * 2;
        let logits_base = self.buffers.logits();
        let mut capture_rows: Vec<Vec<u8>> = Vec::with_capacity(tokens.len());
        let mut out = Vec::with_capacity(tokens.len());

        for (t, &token) in tokens.iter().enumerate() {
            let logits = self.decode_dispatch(token, seq, stream)?;
            out.push(<Self as Model>::argmax_on_device(self, logits, stream)?);
            // Every literal M1 decode writes logits row 0. Preserve it above
            // the active row before the next token overwrites row 0. The
            // scheduler applies its canonical pipeline only after this method
            // returns and expects a contiguous [K,vocab] buffer.
            self.gpu.copy_d2d_async(
                logits,
                logits_base.offset((t + 1) * logits_row_bytes),
                logits_row_bytes,
                stream,
            )?;

            if t + 1 < tokens.len() {
                for (layer_idx, layer_state) in seq.layer_states.iter_mut().enumerate() {
                    if self.config.layer_type(layer_idx)
                        != atlas_core::config::LayerType::LinearAttention
                    {
                        continue;
                    }
                    let state = layer_state
                        .as_any_mut()
                        .downcast_mut::<SsmLayerState>()
                        .ok_or_else(|| {
                            anyhow::anyhow!("Expected SsmLayerState at layer {layer_idx}")
                        })?;
                    anyhow::ensure!(
                        t < state.h_state_intermediates.len()
                            && t < state.conv_state_intermediates.len(),
                        "serial M1 verify intermediate {t} missing at layer {layer_idx}"
                    );
                    self.gpu.copy_d2d_async(
                        state.h_state,
                        state.h_state_intermediates[t],
                        h_bytes,
                        stream,
                    )?;
                    self.gpu.copy_d2d_async(
                        state.conv_state,
                        state.conv_state_intermediates[t],
                        conv_bytes,
                        stream,
                    )?;
                }
            }

            if let Some(capture) = self.dflash_hidden_save {
                self.gpu.synchronize(stream)?;
                let mut row = vec![0u8; capture_row_bytes];
                self.gpu.copy_d2h(capture, &mut row)?;
                capture_rows.push(row);
            }
        }

        // Compact staged rows 1..=K into the canonical rows 0..K. Copies are
        // ordered on one stream and move downward, so no unread source row is
        // overwritten by an earlier copy.
        for t in 0..tokens.len() {
            self.gpu.copy_d2d_async(
                logits_base.offset((t + 1) * logits_row_bytes),
                logits_base.offset(t * logits_row_bytes),
                logits_row_bytes,
                stream,
            )?;
        }

        if let Some(capture) = self.dflash_hidden_save {
            for (t, row) in capture_rows.iter().enumerate() {
                self.gpu
                    .copy_h2d(row, capture.offset(t * capture_row_bytes))?;
            }
        }
        self.gpu.synchronize(stream)?;
        Ok(out)
    }
}
