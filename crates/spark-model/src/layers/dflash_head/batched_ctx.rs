// SPDX-License-Identifier: AGPL-3.0-only
//! #58 follow-up: batched ctx-precompute projection across sequences.
//!
//! `precompute_ctx_kv` used to run its fc/hidden_norm/fused-KV GEMMs once
//! per sequence inside the batched-propose prepare loop — at bs16 that is
//! ~2×16 weight re-reads (~100 MB each) plus ~16× the launches. This
//! module splits the flow: the prepare loop only *records* each
//! sequence's new ctx chunks (`PendingCtxChunk`), then one pass gathers
//! all chunks into `batch_ctx_in`, runs a single `ctx_kv_project` over
//! the Σn_i rows, and scatters per sequence through `ctx_kv_scatter`.
//!
//! **Row identity** (why this is byte-identical per row):
//! `ctx_kv_project`'s arms are row-independent — rms_norm reads only the
//! row it writes, and `drafter_dense_gemm` resolves to
//! `dense_gemm_bf16_pipelined` (mma accumulation over K only) for every M
//! above the small-m GEMV bound, so row i's bytes equal those of the
//! n_i-row launch. A chunk only joins the batch when its own per-sequence
//! arm would ALSO be the pipelined kernel — `plan_ctx_chunks` excludes
//! chunks with `n_i ≤ DENSE_GEMV_BATCHM_MAX_M` while the GEMV lever is
//! on (ATLAS_DFLASH_SMALL_M_GEMV / atlas_scale), because those would have
//! taken `dense_gemv_batchm` serially. Excluded or overflowing chunks run
//! the original per-sequence `precompute_ctx_kv` unchanged.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{BlockDiffusionDraftHead, DflashScratch, small_m_gemm};
use crate::layer::ForwardContext;
use crate::layers::ops::{self, DENSE_GEMV_BATCHM_MAX_M};

/// Per-sequence row cap inside the batched ctx staging. Steady-state
/// tails are `accepted+1` rows (≤ γ+1 ≈ 9); a chunk larger than this is
/// a post-prefill commit or a serial-append stretch — rare enough that
/// falling back to per-sequence GEMMs is the right trade against the
/// scratch size (`batch_capacity × ROWS_PER_SEQ` rows ≈ 15 MB at 27B).
pub(super) const BATCH_CTX_ROWS_PER_SEQ: usize = 64;

/// One sequence's uncommitted ctx tail slice, recorded by
/// `propose_drafts_on_lane` in collect mode (it still ran the lifecycle,
/// decode-append and block-table bookkeeping — only the projection and
/// scatter are deferred).
pub(super) struct PendingCtxChunk {
    /// This sequence's `ctx_hidden_acc` base; rows
    /// `[start_slot, start_slot+count)` are the projection input.
    pub ctx_base: DevicePtr,
    /// Block-table device pointer — scatter rebuilds slot_mapping from it.
    pub block_table: DevicePtr,
    pub start_slot: usize,
    pub count: usize,
    /// `ctx_positions[start_slot..start_slot+count]`, copied at record
    /// time (the vec can grow later; this slice's values are fixed).
    pub positions: Vec<i32>,
}

/// Row-offset plan: `Some(row_off)` per chunk that joins the batched
/// projection, `None` for chunks that must take the per-sequence path.
/// `None` overall (total == 0) means "no batching this step".
pub(super) fn plan_ctx_chunks(
    counts: &[usize],
    small_m_gemv_active: bool,
    total_cap: usize,
) -> Option<Vec<Option<usize>>> {
    let mut offsets = Vec::with_capacity(counts.len());
    let mut total = 0usize;
    for &count in counts {
        // Eligible only when the serial path resolves to the pipelined
        // GEMM for this count — see the module doc for row identity.
        let pipelined_per_seq = !(small_m_gemv_active && count <= DENSE_GEMV_BATCHM_MAX_M as usize);
        if !pipelined_per_seq || count == 0 || total + count > total_cap {
            offsets.push(None);
            continue;
        }
        offsets.push(Some(total));
        total += count;
    }
    (total > 0).then_some(offsets)
}

impl BlockDiffusionDraftHead {
    /// Second half of the batched prepare loop: gather every recorded
    /// chunk into `batch_ctx_in`, run ONE `ctx_kv_project` over Σn_i rows,
    /// then per-chunk `ctx_kv_scatter` (or the original
    /// `precompute_ctx_kv` for chunks the planner excluded / couldn't
    /// fit). `scratch` is the lane-0 scratch every seq prepared on.
    pub(super) fn run_batched_ctx_stage(
        &self,
        pending: &[PendingCtxChunk],
        ctx: &ForwardContext,
        stream: u64,
        scratch: &DflashScratch,
    ) -> Result<()> {
        if pending.is_empty() {
            return Ok(());
        }
        let gpu = ctx.gpu;
        let slot_mapping = scratch.slot_mapping_dev;
        let h = self.hidden_size as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let target_hidden_dim = (self.target_layer_ids.len() * self.target_hidden_size) as u32;
        let ctx_slot_bytes = (target_hidden_dim as usize) * 2;
        let row_stride = self.num_layers * 2 * (kv_dim as usize) * 2;
        let small_m_gemv_active =
            small_m_gemm::small_m_gemv_enabled() && self.kernels.dense_gemv_batchm.0 != 0;
        let counts: Vec<usize> = pending.iter().map(|c| c.count).collect();
        let offsets = plan_ctx_chunks(&counts, small_m_gemv_active, self.batch_ctx_rows);

        // Serial per-chunk path: the original precompute_ctx_kv — the
        // fallback for excluded (GEMV-arm) chunks and cap overflow.
        let run_serial = |chunk: &PendingCtxChunk| -> Result<()> {
            ops::fill_slots_from_block_table(
                gpu,
                self.kernels.fill_slots,
                slot_mapping,
                chunk.block_table,
                chunk.start_slot as u32,
                chunk.count as u32,
                16u32, // block_size — matches propose.rs BLOCK_SIZE
                stream,
            )?;
            self.precompute_ctx_kv(
                chunk.ctx_base,
                chunk.start_slot,
                chunk.count,
                &chunk.positions,
                slot_mapping,
                ctx,
                stream,
                true,
                scratch,
            )
        };

        let Some(offsets) = offsets else {
            for chunk in pending {
                run_serial(chunk)?;
            }
            return Ok(());
        };

        // Gather: each chunk's rows are contiguous — one async copy each.
        for (chunk, off) in pending.iter().zip(&offsets) {
            if let Some(off) = off {
                gpu.copy_d2d_async(
                    chunk.ctx_base.offset(chunk.start_slot * ctx_slot_bytes),
                    self.batch_ctx_in.offset(off * ctx_slot_bytes),
                    chunk.count * ctx_slot_bytes,
                    stream,
                )?;
            }
        }
        // One projection over the gathered rows; `drafter_dense_gemm`
        // resolves the same kernel the included chunks' per-seq calls
        // would take (planner guarantee — module doc).
        // Σn_i = the highest used offset + its chunk's count (offsets are
        // assigned in order, so the last batched chunk ends the range).
        let total_rows = offsets
            .iter()
            .zip(&counts)
            .filter_map(|(o, &c)| o.map(|off| off + c))
            .max()
            .unwrap_or(0) as u32;
        self.ctx_kv_project(
            gpu,
            self.batch_ctx_in,
            total_rows,
            h,
            target_hidden_dim,
            self.batch_ctx_fc,
            self.batch_ctx_fused,
            stream,
            None,
        )?;
        // Scatter per chunk at its fused-KV row offset.
        for (chunk, off) in pending.iter().zip(&offsets) {
            match off {
                Some(off) => {
                    ops::fill_slots_from_block_table(
                        gpu,
                        self.kernels.fill_slots,
                        slot_mapping,
                        chunk.block_table,
                        chunk.start_slot as u32,
                        chunk.count as u32,
                        16u32, // block_size — matches propose.rs BLOCK_SIZE
                        stream,
                    )?;
                    self.ctx_kv_scatter(
                        gpu,
                        self.batch_ctx_fused.offset(off * row_stride),
                        chunk.count,
                        &chunk.positions,
                        slot_mapping,
                        ctx,
                        stream,
                        true,
                        scratch,
                        None,
                    )?;
                }
                None => run_serial(chunk)?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::precompute_ctx_kv::carve_region;
    use super::{BATCH_CTX_ROWS_PER_SEQ, plan_ctx_chunks};
    use crate::layers::ops::DENSE_GEMV_BATCHM_MAX_M;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn carve_region_layout_and_overflow() {
        let cursor = AtomicUsize::new(0);
        // Two equal regions pack adjacently, like per-sequence carves.
        assert_eq!(carve_region(&cursor, 64, 32), Some(0));
        assert_eq!(carve_region(&cursor, 64, 32), Some(32));
        // Third overflows -> None (callers take the sync-copy fallback).
        assert_eq!(carve_region(&cursor, 64, 32), None);
        assert_eq!(carve_region(&cursor, 64, 1), None);
        // Unaligned is fine — byte offsets, not slots.
        let cursor = AtomicUsize::new(0);
        assert_eq!(carve_region(&cursor, 10, 4), Some(0));
        assert_eq!(carve_region(&cursor, 10, 4), Some(4));
        assert_eq!(carve_region(&cursor, 10, 4), None);
        // Zero bytes always carves at the cursor.
        assert_eq!(carve_region(&cursor, 10, 0), Some(8));
    }

    #[test]
    fn precompute_uses_no_sync_row_copies() {
        // Structural pin for #58: the K/V row-compaction loops and the
        // position upload must never again pay a per-row pipeline drain.
        let src = include_str!("precompute_ctx_kv.rs");
        assert_eq!(src.matches("gpu.copy_d2d(").count(), 0);
        // The only sync copy left is the overflow fallback in Step 4.
        assert_eq!(src.matches("gpu.copy_h2d(").count(), 1);
    }

    #[test]
    fn plan_ctx_chunks_offsets_and_exclusion() {
        let cap = 4 * BATCH_CTX_ROWS_PER_SEQ;
        // All pipelined-eligible chunks pack contiguously.
        let offs = plan_ctx_chunks(&[3, 5, 2], false, cap).unwrap();
        assert_eq!(offs, vec![Some(0), Some(3), Some(8)]);
        // GEMV-active: chunks ≤ BATCHM_MAX_M keep their per-seq GEMV arm
        // and are excluded; bigger ones batch.
        let m = DENSE_GEMV_BATCHM_MAX_M as usize;
        let offs = plan_ctx_chunks(&[3, m + 2, 1], true, cap).unwrap();
        assert_eq!(offs, vec![None, Some(0), None]);
        // Same counts with the lever off: everything batches.
        let offs = plan_ctx_chunks(&[3, m + 2, 1], false, cap).unwrap();
        assert_eq!(offs, vec![Some(0), Some(3), Some(m + 5)]);
        // Capacity overflow: chunks past the cap go serial, earlier
        // chunks stay batched (partial batching beats none).
        let offs = plan_ctx_chunks(&[10, 10, 4], false, 16).unwrap();
        assert_eq!(offs, vec![Some(0), None, None]);
        // Nothing eligible → no batching at all.
        assert!(plan_ctx_chunks(&[1, 2, 3], true, cap).is_none());
        assert!(plan_ctx_chunks(&[], false, cap).is_none());
        assert!(plan_ctx_chunks(&[0, 0], false, cap).is_none());
    }

    #[test]
    fn batched_prepare_issues_one_projection() {
        // Structural pin: the batched stage runs ONE ctx_kv_project per
        // step (fc + fused inside), not one per sequence.
        let src = include_str!("batched_ctx.rs");
        assert_eq!(src.matches("self.ctx_kv_project(").count(), 1);
        assert_eq!(src.matches("self.precompute_ctx_kv(").count(), 1); // serial fallback only
    }
}
