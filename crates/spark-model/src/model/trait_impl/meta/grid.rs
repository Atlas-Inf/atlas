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
    /// passes, so it must use the same GDN kernels (FLA) rather than the WY4
    /// "exact replay" path. `ATLAS_GRID_EXACT_REPLAY=1` restores the old forcing.
    pub(in super::super) fn grid_replay_is_cold(&self) -> bool {
        static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        self.prefill_grid() > 0
            && !*FORCE
                .get_or_init(|| std::env::var("ATLAS_GRID_EXACT_REPLAY").as_deref() == Ok("1"))
    }
}
