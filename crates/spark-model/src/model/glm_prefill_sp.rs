// SPDX-License-Identifier: AGPL-3.0-only
//! Eligibility of a GLM prefill chunk for sequence parallelism (`layers::glm_sp`).

use atlas_core::config::LayerType;

use super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::layers::glm_sp::{self, SpRows};

impl TransformerModel {
    /// This rank's rows when the chunk runs sequence-parallel, over the pair
    /// (TP2/EP2) or the chain (TP4/EP4). Every input is mirrored on every
    /// rank, so all reach the same answer.
    pub(super) fn glm_prefill_sp_rows(
        &self,
        rows: usize,
        excluded: bool,
        ctx: &ForwardContext,
    ) -> Option<SpRows> {
        let c = &self.config;
        let layers = c.num_hidden_layers;
        let comm = self.comm.as_ref()?;
        let ranks = comm.world_size();
        let part = rows / ranks.max(1) * c.hidden_size * 2;
        let eligible = glm_sp::requested()
            && !excluded
            && !ctx.graph_capture
            && c.model_type == "glm5_next"
            && crate::layers::ops::hc_bf16_for(&c.model_type)
            && matches!(ranks, 2 | 4)
            && c.tp_world_size == ranks
            && c.ep_world_size == ranks
            && rows >= 4096
            && rows.is_multiple_of(ranks)
            && dflash_window_stays_local(c, rows, ranks)
            && self.mtp_prefill_hidden.is_null()
            // The first and last layers (expand/contract) are KDA layers.
            && layers > 1
            && c.layer_type(0) == LayerType::LinearAttention
            && c.layer_type(layers - 1) == LayerType::LinearAttention
            && if ranks == 2 {
                comm.supports_exchange_async(part)
            } else {
                comm.supports_gather_async(part)
            };
        eligible.then(|| SpRows::for_rank(rows, comm.rank(), ranks))
    }

    /// [`crate::traits::Model::prefill_row_multiple`]: 128 rows a rank of
    /// the chain when its SP prefill is requested, twice it with the two-half
    /// pipeline (the pair keeps its chunks). Each rank's part is then whole
    /// 128-row tiles, which the MXFP8 exchanges need (the KDA quantized
    /// gather, the all-to-all o_proj): the 8200-row arena used to cut an
    /// 8196-row first chunk (2049 rows a part) that kept both off.
    pub(super) fn glm_sp_row_multiple(&self) -> usize {
        let ranks = self.comm.as_ref().map_or(1, |c| c.world_size());
        if glm_sp::requested() && self.config.model_type == "glm5_next" && ranks > 2 {
            // The two-half pipeline splits each half over the ranks too.
            128 * ranks * if self.prefill_pipe.is_some() { 2 } else { 1 }
        } else {
            1
        }
    }

    /// After DFlash capture layer `layer` of a chain SP chunk whose drafter
    /// window may reach below rank 0's rows (`w > sp.rows`, decided alike on
    /// every rank): every rank contracts its rows of the highway into a
    /// `[total, hidden]` scratch and the chain gathers them, so rank 0's
    /// capture reads the whole chunk (the returned scope).
    pub(super) fn glm_sp_capture_gather(
        &self,
        sp: Option<SpRows>,
        layer: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> anyhow::Result<Option<glm_sp::CaptureScope>> {
        let c = &self.config;
        let Some(sp) = sp.filter(|sp| {
            sp.ranks > 2
                && c.dflash_capture_layers.contains(&layer)
                && c.dflash_ctx_window.is_none_or(|w| w > sp.rows)
        }) else {
            return Ok(None);
        };
        let h = c.hidden_size;
        let rows = self.gpu.op_cache().scratch(
            self.gpu.as_ref(),
            "glm_sp_dflash_capture",
            self.buffers.max_batch_tokens() * h * 2,
        )?;
        self.dflash_capture_rows_inner(ctx.buffers, 0, sp.rows, sp.local(rows, h), h * 2, stream)?;
        sp.all_gather(rows, h, ctx, stream)?;
        Ok(Some(glm_sp::capture_from(rows)))
    }
}

/// DFlash captures its context window on rank 0, and SP leaves rank 0 only
/// the top part of the rows: on the pair the window has to fit there or the
/// capture would have holes (`try_dflash_prefill_capture_layer`); a chain
/// gathers the capture layers' rows instead (`glm_sp_capture_gather`).
fn dflash_window_stays_local(
    c: &atlas_core::config::ModelConfig,
    rows: usize,
    ranks: usize,
) -> bool {
    c.dflash_capture_layers.is_empty()
        || ranks > 2
        || c.dflash_ctx_window.is_some_and(|w| w <= rows / ranks)
}

#[cfg(test)]
mod tests {
    use super::dflash_window_stays_local;
    use atlas_core::config::ModelConfig;

    #[test]
    fn dflash_keeps_sp_while_its_window_fits_rank_zero() {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        assert!(dflash_window_stays_local(&c, 4096, 2));
        c.dflash_capture_layers = vec![40];
        c.dflash_ctx_window = None;
        assert!(!dflash_window_stays_local(&c, 8192, 2));
        c.dflash_ctx_window = Some(2048);
        assert!(dflash_window_stays_local(&c, 4096, 2));
        assert!(!dflash_window_stays_local(&c, 4094, 2));
        // A chain gathers the capture layers' rows.
        c.dflash_ctx_window = None;
        assert!(dflash_window_stays_local(&c, 8192, 4));
    }
}
