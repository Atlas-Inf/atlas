// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::traits::SequenceState;

impl TransformerModel {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_a_compute(
        &self,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext<'_>,
        hidden: DevicePtr,
        residual: DevicePtr,
        proc_count: usize,
        seq_len_start: usize,
        layer_kv_write_start: usize,
        diag_prefill: bool,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let fp32 = 2usize;
        for (i, layer) in self.layers.iter().enumerate() {
            layer
                .prefill(
                    hidden,
                    residual,
                    proc_count,
                    seq.layer_states[i].as_mut(),
                    kv_cache,
                    seq_len_start,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    layer_kv_write_start,
                    ctx,
                    stream,
                )
                .map_err(|e| anyhow::anyhow!("Prefill layer {i} failed: {e}"))?;
            // DFlash prefill capture: writes layer i's hidden output for
            // all `proc_count` tokens into the seq's accumulator at slots
            // [layer_kv_write_start .. layer_kv_write_start + proc_count].
            // No-op when DFlash is disabled.
            self.try_dflash_prefill_capture_layer(
                seq,
                i,
                layer_kv_write_start,
                proc_count,
                stream,
            )?;

            // MLA diagnostic: dump per-layer hidden state norm (once per model).
            // Per-model latch (see `ModelStats::dumped`) rather than a static: an
            // operator who sets the flag and then swaps models must still get the
            // dump, instead of it being swallowed by the previous model's shot.
            if self.profile
                && self.config.model_type == "mistral"
                && self.stats.dumped.keyed("mla_prefill_norms")
            {
                self.gpu.synchronize(stream)?;
                // Read last token's hidden state (what goes to LM head)
                let last_offset = (proc_count - 1) * self.config.hidden_size * 4;
                let h_sz = self.config.hidden_size;
                let mut buf = vec![0u16; h_sz];
                // SAFETY: `buf` is `vec![0u16; h_sz]` on the line above, so it
                // owns exactly `h_sz * size_of::<u16>()` initialised bytes and
                // the length matches its capacity. `bytes` is the only live
                // reference to that allocation for its whole lifetime — it is
                // last used on the `copy_d2h` line below, and `buf` is not read
                // again until after that.
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, h_sz * 2)
                };
                if self.gpu.copy_d2h(hidden.offset(last_offset), bytes).is_ok() {
                    let vals: Vec<f32> = buf
                        .iter()
                        .map(|&b| f32::from_bits((b as u32) << 16))
                        .collect();
                    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    tracing::info!("LAYER_NORM L{i}: hidden_norm={norm:.4}");
                    if i == self.layers.len() - 1 {}
                }
            }

            // Diagnostic: check last token's hidden state norm at every layer.
            // This is what goes to the LM head — divergence here causes bad logits.
            if diag_prefill {
                self.gpu.synchronize(stream)?;
                let last_start = (proc_count - 1) * h;
                let (last_vals, last_norm) =
                    self.readback_bf16(hidden.offset(last_start * fp32), h.min(64))?;
                let last_nan = last_vals.iter().filter(|v| v.is_nan()).count();
                let last_inf = last_vals.iter().filter(|v| v.is_infinite()).count();
                let lt = self.config.layer_type(i);
                // Print every 4th layer + first/last to keep output manageable
                if i % 4 == 0 || i == self.layers.len() - 1 || last_nan > 0 || last_inf > 0 {
                    tracing::warn!(
                        "DIAG L{i} ({lt:?}) last_tok: norm={last_norm:.4} nan={last_nan} inf={last_inf} first4={:.4?}",
                        &last_vals[..4.min(last_vals.len())]
                    );
                }
            }
        }

        // ATLAS_MTP_DRAFTER_PREFILL: capture the processed rows' final-layer
        // hiddens for the whole-prompt drafter prefill. No-op when disabled.
        self.try_mtp_prefill_capture(seq, seq_len_start, proc_count, stream)?;

        // ── 5. Final norm on LAST token only ──
        let last_hidden = hidden.offset((proc_count - 1) * h * fp32);
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        self.final_norm_apply(last_hidden, normed, 1, h as u32, eps, stream)?;

        // ── 6. LM head on last token → logits ──
        self.lm_head(normed, stream)?;
        Ok(())
    }
}
