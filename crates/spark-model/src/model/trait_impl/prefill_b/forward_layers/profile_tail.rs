// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl TransformerModel {
    /// The per-chunk profiling tail of `prefill_b_forward_layers`: host-timing
    /// totals, the MTP drafter-prefill capture and the per-layer profile dump.
    /// Statements moved verbatim; behaviour unchanged.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_b_forward_layers_profile_tail(
        &self,
        seq: &mut SequenceState,
        effective_seq_len_start: usize,
        proc_count: usize,
        chunk_len: usize,
        use_decode_path: bool,
        t_loop: Option<std::time::Instant>,
        t_in_prefill: std::time::Duration,
        t_dflash: std::time::Duration,
        layer_times: &[u128],
        prefill_t0: Option<std::time::Instant>,
        stream: u64,
    ) -> Result<()> {
        if let Some(t) = t_loop {
            let wall = t.elapsed();
            let ffn_us = crate::layers::qwen3_attention::take_ffn_host_us();
            let ph = crate::layers::qwen3_attention::take_attn_phase_us();
            // loop_wall is the host's own elapsed time across the whole layer
            // dispatch. `in_prefill` is time inside layer.prefill() itself,
            // `dflash` is the per-layer DFlash capture (expected ~0 for models
            // with DFlash disabled), and the remainder is other per-layer
            // bookkeeping. Compare loop_wall against nsys GPU-busy time: the
            // difference is host work not overlapped with the device.
            tracing::info!(
                "PREFILL HOST TIMING layers={} tokens={}: loop_wall={:.1}ms in_prefill={:.1}ms ffn={:.1}ms attn_rest={:.1}ms dflash={:.1}ms | qkv={:.1}ms mid={:.1}ms attn_kernel={:.1}ms",
                self.layers.len(),
                proc_count,
                wall.as_secs_f64() * 1e3,
                t_in_prefill.as_secs_f64() * 1e3,
                ffn_us as f64 / 1e3,
                (t_in_prefill.as_micros() as f64 - ffn_us as f64) / 1e3,
                t_dflash.as_secs_f64() * 1e3,
                ph[0] as f64 / 1e3,
                ph[1] as f64 / 1e3,
                ph[2] as f64 / 1e3,
            );
        }

        // ATLAS_MTP_DRAFTER_PREFILL: capture this chunk's final-layer hidden
        // rows for the whole-prompt drafter prefill. No-op when disabled.
        self.try_mtp_prefill_capture(seq, effective_seq_len_start, proc_count, stream)?;
        if let Some(t0) = prefill_t0 {
            self.gpu.synchronize(stream)?;
            let total_us = t0.elapsed().as_micros();
            let mut indexed: Vec<(usize, u128)> = layer_times.iter().copied().enumerate().collect();
            indexed.sort_by_key(|x| std::cmp::Reverse(x.1));
            let top5: Vec<String> = indexed
                .iter()
                .take(5)
                .map(|(i, us)| format!("L{}={:.2}ms", i, *us as f64 / 1000.0))
                .collect();
            let path_label = if use_decode_path { "decode" } else { "prefill" };
            // Aggregate the same per-layer samples by layer type so the profile
            // attributes cost to mamba / moe / attention instead of bare indices.
            let mut by_type: std::collections::BTreeMap<String, (u128, usize)> =
                std::collections::BTreeMap::new();
            for (i, us) in layer_times.iter().copied().enumerate() {
                let e = by_type
                    .entry(format!("{:?}", self.config.layer_type(i)))
                    .or_insert((0, 0));
                e.0 += us;
                e.1 += 1;
            }
            let per_type: Vec<String> = by_type
                .iter()
                .map(|(k, (us, n))| {
                    format!(
                        "{}x{}={:.0}ms(avg {:.1})",
                        n,
                        k,
                        *us as f64 / 1000.0,
                        *us as f64 / 1000.0 / *n as f64
                    )
                })
                .collect();
            tracing::info!(
                "Prefill chunk {} tok (proc {}, {}): {:.1}ms total, by_type: {}, top5: {}",
                chunk_len,
                proc_count,
                path_label,
                total_us as f64 / 1000.0,
                per_type.join(", "),
                top5.join(", "),
            );
        }
        Ok(())
    }
}
