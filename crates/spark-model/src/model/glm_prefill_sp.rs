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
            && self.mtp_prefill_hidden.is_null()
            // The first and last layers (expand/contract) are KDA layers.
            && layers > 1
            && c.layer_type(0) == LayerType::LinearAttention
            && c.layer_type(layers - 1) == LayerType::LinearAttention
            && comm.supports_exchange_async(rows / 2 * c.hidden_size * 2);
        eligible.then(|| SpRows::for_rank(rows, comm.rank()))
    }
}
