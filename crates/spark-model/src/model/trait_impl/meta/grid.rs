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
}
