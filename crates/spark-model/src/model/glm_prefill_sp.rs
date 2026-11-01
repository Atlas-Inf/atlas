// SPDX-License-Identifier: AGPL-3.0-only
//! Eligibility of a GLM prefill chunk for sequence parallelism (`layers::glm_sp`).

use atlas_core::config::LayerType;

use super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::layers::glm_sp::{self, SpRows};

impl TransformerModel {
    /// This rank's rows when the chunk runs sequence-parallel. Every input is
    /// mirrored on both ranks, so both reach the same answer.
    pub(super) fn glm_prefill_sp_rows(
        &self,
        rows: usize,
        excluded: bool,
        ctx: &ForwardContext,
    ) -> Option<SpRows> {
        let c = &self.config;
        let layers = c.num_hidden_layers;
        let comm = self.comm.as_ref()?;
        let eligible = glm_sp::requested()
            && !excluded
            && !ctx.graph_capture
            && c.model_type == "glm5_next"
            && crate::layers::ops::hc_bf16_for(&c.model_type)
            && c.tp_world_size == 2
            && c.ep_world_size == 2
            && comm.world_size() == 2
            && rows >= 4096
            && rows.is_multiple_of(2)
            && dflash_window_stays_local(c, rows)
            && self.mtp_prefill_hidden.is_null()
            // The first and last layers (expand/contract) are KDA layers.
            && layers > 1
            && c.layer_type(0) == LayerType::LinearAttention
            && c.layer_type(layers - 1) == LayerType::LinearAttention
            && comm.supports_exchange_async(rows / 2 * c.hidden_size * 2);
        eligible.then(|| SpRows::for_rank(rows, comm.rank()))
    }
}

/// DFlash captures its context window on rank 0, and SP leaves rank 0 only
/// the upper half of the rows: the window has to fit there or the capture
/// would have holes (`try_dflash_prefill_capture_layer`).
fn dflash_window_stays_local(c: &atlas_core::config::ModelConfig, rows: usize) -> bool {
    c.dflash_capture_layers.is_empty() || c.dflash_ctx_window.is_some_and(|w| w <= rows / 2)
}

#[cfg(test)]
mod tests {
    use super::dflash_window_stays_local;
    use atlas_core::config::ModelConfig;

    #[test]
    fn dflash_keeps_sp_while_its_window_fits_rank_zero() {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        assert!(dflash_window_stays_local(&c, 4096));
        c.dflash_capture_layers = vec![40];
        c.dflash_ctx_window = None;
        assert!(!dflash_window_stays_local(&c, 8192));
        c.dflash_ctx_window = Some(2048);
        assert!(dflash_window_stays_local(&c, 4096));
        assert!(!dflash_window_stays_local(&c, 4094));
    }
}
