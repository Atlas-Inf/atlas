// SPDX-License-Identifier: AGPL-3.0-only

//! R11 prefill pass-grid queries (split from `meta.rs` for the 500-LoC cap).

use super::super::super::types::TransformerModel;

impl TransformerModel {
    /// R11 fix 1: active prefill pass grid in tokens for this model, 0 = off.
    /// Only hybrid-SSM models (whose warm path replays from an SSM anchor)
    /// use it. Off unless `--exact-prefix-cache` or `ATLAS_PREFILL_GRID`.
    pub(in super::super) fn prefill_grid(&self) -> usize {
        if self.config.num_ssm_layers() == 0 {
            return 0;
        }
        let g = spark_runtime::prefill_grid::prefill_grid();
        if g > 0 && !spark_runtime::prefill_grid::subblock_disabled() {
            spark_runtime::prefill_grid::disable_subblock();
            tracing::info!("exact prefix cache: prefill pass grid {g} tokens");
        }
        g
    }

    /// R11 fix 6: under the prefill grid a warm replay runs exactly the cold
    /// passes, so it must use the same GDN kernels (FLA) rather than the forced
    /// "exact replay" path (register-resident token-sequential by default, WY4
    /// only with `ATLAS_NO_GDN_REGRESIDENT=1`); both differ from the cold FLA
    /// pass. `ATLAS_GRID_EXACT_REPLAY=1` restores the old forcing.
    pub(in super::super) fn grid_replay_is_cold(&self) -> bool {
        static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        self.prefill_grid() > 0
            && !*FORCE
                .get_or_init(|| std::env::var("ATLAS_GRID_EXACT_REPLAY").as_deref() == Ok("1"))
    }

    /// R11 fix 1 (`cache_sequence`): under the pass grid insert only the KV of
    /// the prompt's full grid passes. Returns false (nothing done) when the
    /// grid is off.
    pub(in super::super) fn cache_grid_prefix(
        &self,
        seq: &crate::traits::SequenceState,
        bs: usize,
    ) -> bool {
        if self.prefill_grid() == 0 {
            return false;
        }
        let n = spark_runtime::prefill_grid::cuts()
            .floor(seq.prompt_len)
            .min(seq.tokens.len());
        let nb = (n / bs).min(seq.block_table.len());
        if nb > 0 {
            let acquired = self.prefix_cache.insert(
                &seq.tokens[..nb * bs],
                &seq.block_table[..nb],
                &seq.disk_block_ids[..nb.min(seq.disk_block_ids.len())],
                bs,
                nb * bs,
                seq.adapter_id,
            );
            super::super::super::block_mgmt::cache_acquires_refs(
                &acquired,
                &mut self.kv_cache.lock(),
            );
        }
        true
    }

    /// R11 fix 1 (`prefill_chunk_dispatch_with`): position-fixed pass grid
    /// (`spark_runtime::prefill_grid`). Every pass ends at a grid cut or the
    /// prompt end, with prefix caching on or off, so a warm replay from a grid
    /// anchor runs the same passes as a cold run. A chunk that crosses a cut
    /// runs as two passes; `None` when the grid is off, the prompt has vision
    /// pads, or no cut falls inside the chunk.
    pub(in super::super) fn grid_cut(
        &self,
        tokens: &[u32],
        seq: &mut crate::traits::SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Option<anyhow::Result<spark_runtime::gpu::DevicePtr>> {
        if self.prefill_grid() == 0 || self.tokens_have_vision_pad(tokens) {
            return None;
        }
        let end = chunk_start + chunk_len;
        let cut = spark_runtime::prefill_grid::cuts().cut(chunk_start, end)?;
        Some(
            self.prefill_chunk_dispatch(tokens, seq, chunk_start, cut - chunk_start, false, stream)
                .and_then(|_| {
                    self.prefill_chunk_dispatch(tokens, seq, cut, end - cut, is_last_chunk, stream)
                }),
        )
    }

    /// The tail-split checkpoint (`prefill_chunk_dispatch_with`) runs only
    /// with snapshots on and the pass grid off (the grid replaces it).
    pub(in super::super) fn tail_split_snapshots(&self) -> bool {
        self.prefill_grid() == 0 && self.ssm_snapshots.is_enabled()
    }

    /// R11 fix 1 (`finalize_last`): under the pass grid only KV written by
    /// FULL grid passes is shared (the last pass has a prompt-dependent row
    /// count) and no full-length leaf snapshot is saved (a warm replay must
    /// start on a grid point); routing through the stale-V cap branch inserts
    /// the grid-floored prefix without a snapshot. Returns
    /// `(cache_blocks, cap_applied, grid_cap)`.
    pub(in super::super) fn grid_cap_blocks(
        &self,
        cache_blocks: usize,
        full_blocks: usize,
        total: usize,
        bs: usize,
    ) -> (usize, bool, bool) {
        if self.prefill_grid() == 0 {
            return (cache_blocks, cache_blocks < full_blocks, false);
        }
        let grid_blocks = spark_runtime::prefill_grid::cuts().floor(total) / bs;
        (cache_blocks.min(grid_blocks), true, true)
    }
}
