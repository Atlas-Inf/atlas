// SPDX-License-Identifier: AGPL-3.0-only
//! Sequence-parallel qwen4_exp prefill (`ATLAS_QWEN4EXP_PREFILL_SP=1`, TP2/EP2,
//! default off), on the `layers::glm_sp` machinery.
//!
//! Attention, the GDN block and the MoE need every row of a chunk; the mHC
//! seams (`hc_post`, the collapses, `hc_head`) are row-local and were
//! computed on both ranks. Under SP each rank runs them over its own rows:
//! the TP/EP all-reduces become reduce-scatters into those rows and the
//! collapsed block inputs are all-gathered.
//!
//! Exact by construction -- every output byte is the unsplit chunk's:
//! * **The split is slab-aligned.** The collapse runs in 2048-row slabs and
//!   the slab height picks its GEMMs (the tile kernel for full slabs,
//!   cuBLASLt below 1920 rows, cuBLASLt for the injection at every height),
//!   which is what fixes a row's bits. So rank 0 takes `[0, S)` and rank 1
//!   `[S, T)` with `S` the multiple of 2048 nearest `T/2`: each rank's slabs
//!   are exactly the unsplit chunk's slabs. The halves differ by up to 1024
//!   rows, and the exchanges are uneven (`layers::glm_sp_uneven`).
//! * **Row-count-dependent work keeps every row.** The shared expert (whose
//!   GEMMs may pick kernels by row count) runs over the whole chunk as before
//!   and only its blend takes the local rows (`SpRows::full_shared`); the
//!   router, routed experts, attention and the GDN block see every row.
//! * **A reduce-scatter sums what the all-reduce summed**: `peer + own` in
//!   BF16 through the pair's own add kernel.
//! * **The split starts after PLE.** PLE's causal conv runs across rows, so
//!   the layers up to and including the PLE layer (and layer 0, which seeds
//!   the highway) run unsplit; at the first split layer rank 1 moves its rows
//!   of the FP32 highway to row 0 and the layer before never defers its post
//!   (`qwen4exp_prefill_seam::NoDefer`).
//! * **What survives the chunk is the unsplit chunk's**: `hc_head` writes
//!   each rank's rows of `hidden`, which are all-gathered (logits, the MTP
//!   drafter's capture); the highway's row 0 (which the drafter reads) is
//!   handed to rank 1 (`glm_sp_uneven::share_row0`); GDN, conv, KV and QSA
//!   state are written from every row as before.
//!
//! `ATLAS_QWEN4EXP_PREFILL_SP_CHECK=1` logs, per eligible chunk and rank, a
//! hash of this rank's split rows of the highway after every layer and of the
//! final `hidden`, with or without the split: run a prompt once with
//! `ATLAS_QWEN4EXP_PREFILL_SP=0` and once with `=1` and the `QWEN4EXP_SP_CHECK`
//! lines must match apart from their `sp=` field.

use anyhow::Result;

use super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::layers::glm_sp::SpRows;

/// `ATLAS_QWEN4EXP_PREFILL_SP=1`.
pub fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_SP").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// `ATLAS_QWEN4EXP_PREFILL_SP_CHECK=1`.
fn check_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_SP_CHECK").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Shortest chunk that splits: both halves stay whole slabs or more.
const MIN_ROWS: usize = 4096;

/// The split row: the multiple of the collapse slab nearest `rows / 2`.
pub(crate) fn slab_split(rows: usize) -> usize {
    let slab = crate::layers::ops::HC_PREFILL_SLAB as usize;
    ((rows / 2 + slab / 2) / slab).max(1) * slab
}

/// One chunk's split.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QwenSpPlan {
    pub sp: SpRows,
    /// First layer that runs split.
    pub first_layer: usize,
    /// The split runs (else it is only hashed, `_SP_CHECK`).
    pub active: bool,
    /// `_SP_CHECK` hashes.
    pub check: bool,
}

impl TransformerModel {
    /// This chunk's split, when it is eligible and SP or its check is
    /// requested. Every input is mirrored on both ranks, so both reach the
    /// same answer.
    pub(super) fn qwen4exp_prefill_sp_plan(
        &self,
        rows: usize,
        excluded: bool,
        ctx: &ForwardContext,
    ) -> Option<QwenSpPlan> {
        let (active, check) = (requested(), check_requested());
        if !(active || check) {
            return None;
        }
        let c = &self.config;
        let comm = self.comm.as_ref()?;
        // `ple_layer_ids` is 1-indexed: id k is model layer k - 1, so the
        // first layer after it is k. Layer 0 seeds the highway.
        let first_layer = c.ple_layer_ids.iter().copied().max().unwrap_or(0).max(1);
        let split = slab_split(rows);
        let slab = crate::layers::ops::HC_PREFILL_SLAB as usize;
        let eligible = !excluded
            && !ctx.graph_capture
            && c.model_type == "qwen4_exp"
            && c.tp_world_size == 2
            && c.ep_world_size == 2
            && comm.world_size() == 2
            && rows >= MIN_ROWS
            // Both halves at least one full slab: the collapse sizes its
            // scratch layout by `min(rows, slab)` (always `slab` then).
            && split >= slab
            && rows - split >= slab
            && c.dflash_capture_layers.is_empty()
            && first_layer + 1 < c.num_hidden_layers
            && !crate::layers::ple::dump::tapping()
            && comm.supports_exchange_async(split.max(rows - split) * c.hidden_size * 2);
        eligible.then(|| QwenSpPlan {
            sp: SpRows::split_at(rows, split, comm.rank()),
            first_layer,
            active,
            check,
        })
    }

    /// Enter the split: rank 1 moves its rows of the FP32 highway to row 0.
    /// Front to back in blocks no longer than the gap, so no block reads
    /// rows an earlier one wrote.
    pub(super) fn qwen4exp_sp_compact(&self, sp: SpRows, stream: u64) -> Result<()> {
        let row = self.qwen4exp_hc_row();
        let streams = self.buffers.hc_streams();
        let mut done = 0;
        while sp.row0 > 0 && done < sp.rows {
            let k = (sp.rows - done).min(sp.row0);
            self.gpu.copy_d2d_async(
                streams.offset((sp.row0 + done) * row),
                streams.offset(done * row),
                k * row,
                stream,
            )?;
            done += k;
        }
        Ok(())
    }

    /// Bytes of one highway row.
    pub(super) fn qwen4exp_hc_row(&self) -> usize {
        self.config.hc_mult
            * self.config.hidden_size
            * crate::layers::ops::hc_elem_bytes(&self.config.model_type)
    }

    /// `_SP_CHECK`: hash this rank's split rows of the highway after `layer`
    /// (`split`: compacted at row 0), or of the final `hidden` (`layer` =
    /// None, every row).
    pub(super) fn qwen4exp_sp_check(
        &self,
        plan: QwenSpPlan,
        chunk_start: usize,
        layer: Option<usize>,
        split: bool,
        stream: u64,
    ) {
        let sp = plan.sp;
        let (what, ptr, bytes) = match layer {
            Some(i) => {
                let row = self.qwen4exp_hc_row();
                let from = if split { 0 } else { sp.row0 };
                (
                    format!("L={i} r0={} n={}", sp.row0, sp.rows),
                    self.buffers.hc_streams().offset(from * row),
                    sp.rows * row,
                )
            }
            None => (
                format!("L=final r0=0 n={}", sp.total()),
                self.buffers.hidden_states(),
                sp.total() * self.config.hidden_size * 2,
            ),
        };
        let hash = crate::det_trace::device_hash(self.gpu.as_ref(), stream, ptr, bytes, 32 << 20);
        tracing::info!(
            "QWEN4EXP_SP_CHECK r={} c={chunk_start} T={} sp={} {what} h={}",
            self.config.ep_rank,
            sp.total(),
            u8::from(plan.active),
            hash.map_or("unread".to_string(), |h| format!("{h:016x}")),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::slab_split;

    #[test]
    fn split_is_the_nearest_slab_multiple() {
        assert_eq!(slab_split(4096), 2048);
        assert_eq!(slab_split(5000), 2048);
        assert_eq!(slab_split(13000), 6144);
        assert_eq!(slab_split(16046), 8192);
        assert_eq!(slab_split(29000), 14336);
        for rows in 4096..40000 {
            let s = slab_split(rows);
            assert_eq!(s % 2048, 0);
            assert!(s.abs_diff(rows / 2) <= 1024, "{rows} -> {s}");
            assert!(s >= 2048 && rows - s >= 2048, "{rows} -> {s}");
        }
    }
}
