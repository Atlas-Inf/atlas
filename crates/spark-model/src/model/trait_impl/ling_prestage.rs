// SPDX-License-Identifier: AGPL-3.0-only

//! The bailing (Ling-3.2-air) family's pre-graph decode/verify metadata
//! staging. Phase 1 of `decode_dispatch` and of the K=2 verify refreshes
//! the gated-MLA layer's device metadata (rope positions, paged slots,
//! `base_len`, the block table) into its fixed scratch buffers, so a step
//! captured as a CUDA graph issues no pageable H2D inside the capture —
//! those uploads run in Phase 1 instead, and the captured kernels read the
//! refreshed buffers (`docs/ling-decode-graphs.md`). Family- and
//! switch-gated: off, or on any other model, this is dead code.

use anyhow::Result;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::types::TransformerModel;
use crate::layer::TransformerLayer;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Whether a dispatch step writes the Ling family's staged step
    /// metadata: the bailing family is loaded and the decode-graph switch
    /// or its staging-debug arm is on (both strict `== "1"`, read once —
    /// see `model::graph_flags`).
    pub(super) fn ling_meta_prestage_armed(&self) -> bool {
        self.config.model_type == "bailing_moe_v3_vl"
            && crate::model::graph_flags::ling_staging_writes()
    }

    /// Phase-1 staging for ONE sequence's step of `rows` rows (1 = the
    /// single-token decode, `k` = the MTP verify window) whose first row
    /// sits at position `seq.seq_len`. Every layer gets the call; only the
    /// Ling layers implement it (`TransformerLayer::decode_meta_prestage`).
    pub(super) fn ling_prestage_meta(
        &self,
        seq: &SequenceState,
        kv_cache: &PagedKvCache,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let bs = kv_cache.block_size();
        for l in self.layers.iter() {
            l.decode_meta_prestage(
                rows,
                seq.seq_len,
                &seq.block_table,
                bs,
                self.gpu.as_ref(),
                stream,
            )?;
        }
        Ok(())
    }

    /// Scope gate for the BATCHED decode graphs (`decode_a2`): keep the Ling
    /// family OFF them even when the family switch is on. The
    /// pool-indexed KDA lane already serves 2..=8-row batches eagerly
    /// (the measured multi-request win), and the Ling metadata staging
    /// this slice prestages is SINGLE-sequence (Phase 1 of decode_dispatch
    /// and the K=2 verify) — an a2 capture would replay MLA rows reading
    /// seq 0's staging. Phase 1 of the flip is the n=1 decode graphs and
    /// the single-sequence K=2 verify graphs; batching those is the
    /// follow-up. No effect off the family or with the switch off.
    pub(super) fn ling_batched_graphs_off(&self) -> bool {
        self.config.model_type == "bailing_moe_v3_vl"
            && crate::model::graph_flags::ling_decode_graph()
    }
}
