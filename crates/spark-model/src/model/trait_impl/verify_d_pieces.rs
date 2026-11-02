// SPDX-License-Identifier: AGPL-3.0-only
//! K=γ verify: piecewise-captured KDA runs (`ATLAS_GLM_VERIFY_GRAPH=1`) and
//! the per-layer DFlash hidden capture both the eager and captured paths use.

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::traits::SequenceState;

/// Admission of the piecewise path. `eager_only` folds every reason the step
/// must stay eager or already takes a whole-step graph; all inputs are
/// rank-identical (shared profile, broadcast width, model config).
pub(super) fn admitted(requested: bool, model_type: &str, tp: usize, eager_only: bool) -> bool {
    requested && model_type == "glm5_next" && tp == 2 && !eager_only
}

/// Rank-local diagnostics that sync or read back inside a KDA run, which
/// would refuse every key: the DFlash capture trace and the MoE's
/// `tracing` DEBUG readbacks (they key on `ctx.graph_capture`, which pieces
/// leave false). Rank-local is safe: collectives are never captured, so a
/// rank that stays eager still issues the same collective calls.
pub(super) fn diagnostics_sync() -> bool {
    tracing::level_filters::LevelFilter::current() >= tracing::Level::DEBUG
        || std::env::var("ATLAS_DFLASH_CAPTURE_TRACE").as_deref() == Ok("1")
}

impl TransformerModel {
    /// Run the maximal run of KDA layers starting at `layer_idx` through the
    /// piecewise graph cache. `Ok(true)` when `layer_idx` belongs to a run
    /// (handled now, or already handled with its run's first layer).
    pub(super) fn kgamma_kda_run(
        &self,
        layer_idx: usize,
        k: usize,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let kda = |i: usize| self.config.layer_type(i) == LayerType::LinearAttention;
        if !kda(layer_idx) {
            return Ok(false);
        }
        if layer_idx > 0 && kda(layer_idx - 1) {
            return Ok(true);
        }
        let end = (layer_idx..self.layers.len())
            .find(|&i| !kda(i))
            .unwrap_or(self.layers.len());
        anyhow::ensure!(
            ctx.midchunk_capture.is_none(),
            "piecewise verify graph: verify has no mid-chunk capture"
        );
        let comm = ctx
            .comm
            .ok_or_else(|| anyhow::anyhow!("piecewise verify graph needs the TP communicator"))?;
        let (hidden, residual) = (self.buffers.hidden_states(), self.buffers.residual());
        let key = (seq.slot_idx, k, layer_idx);
        self.verify_pieces
            .run(key, self.gpu.as_ref(), comm, stream, |comm| {
                // The verify context carries no mid-chunk capture (the one
                // field that is not `Copy`).
                let ctx = ForwardContext {
                    comm: Some(comm),
                    midchunk_capture: None,
                    ..*ctx
                };
                for li in layer_idx..end {
                    self.layers[li].decode_batched(
                        hidden,
                        residual,
                        k,
                        seq.layer_states[li].as_mut(),
                        kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )?;
                    self.kgamma_dflash_capture(li, k, stream)?;
                }
                Ok(())
            })?;
        Ok(true)
    }

    /// DFlash intermediate hidden capture: snapshot each capture layer's
    /// output rows into dflash_hidden_save[slot] while hidden_states still
    /// holds this layer's activation — mirrors verify_b.rs for K=2. Must be
    /// inside the graph capture region so the per-layer intermediate (not the
    /// final-layer-only post-loop value) is recorded.
    ///
    /// Always capture every verify row: commit_ctx copies rows
    /// 0..=num_accepted, and capturing only k-1 leaves row 0 holding the WRONG
    /// token's hidden and rows 1.. stale (2026-07-09 accept-collapse root
    /// cause, which also starved EAGLE_FIX=0 under UNIFIED_CTX=1). Opt out
    /// with ATLAS_DFLASH_CAPTURE_LAST_ONLY=1 for ablation only: product
    /// Lightning serves never arm it (the admitted policy freezes the
    /// diagnostic surface).
    pub(super) fn kgamma_dflash_capture(
        &self,
        layer_idx: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        let capture_last_only = self.lightning_dspark_identity.policy().is_none()
            && std::env::var("ATLAS_DFLASH_CAPTURE_LAST_ONLY")
                .ok()
                .as_deref()
                == Some("1");
        if capture_last_only {
            self.try_dflash_capture(layer_idx, k - 1, stream)
        } else {
            self.try_dflash_capture_all(layer_idx, k, stream)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::admitted;

    #[test]
    fn opt_in_glm_tp2_only() {
        assert!(admitted(true, "glm5_next", 2, false));
        assert!(!admitted(false, "glm5_next", 2, false));
        assert!(!admitted(true, "glm5_next", 2, true));
        for tp in [1, 4] {
            assert!(!admitted(true, "glm5_next", tp, false));
        }
        for model in ["qwen3", "deepseek_v4", ""] {
            assert!(!admitted(true, model, 2, false));
        }
    }
}
