// SPDX-License-Identifier: AGPL-3.0-only

//! QSA on the batched multi-row path: the per-row indexer walk.
//!
//! A multi-row step (a speculative verify: K drafts + the committed token,
//! all owned by ONE sequence) attends every row in one forward. Past the
//! inert bound each row needs its OWN selection, cut at its own position:
//! row `i` must see exactly what a serial decode step at that position would
//! — the drafts before it, none after it, and any block those drafts closed.
//! One selection shared by the window is a different function
//! (`atlas-core` `qsa_tests::one_shared_selection_cannot_serve_a_verify_window`).
//!
//! So this phase IS R serial steps of the indexer, in row order, through the
//! same `decode_select` and the same bs=1 selected attention the serial path
//! uses. Indexer state after the step therefore equals serial state by
//! construction, and `QsaIndexer::rewind_verify` stays exact.
//!
//! THE INVARIANT: select -> gather -> attend is completed (enqueued) for row
//! `i` BEFORE `decode_select` runs for row `i + 1`. A `QsaSelection` points
//! at LAYER-OWNED scratch that the next `decode_select` overwrites; selecting
//! for all rows first and attending afterwards would make every row attend
//! over the last row's set — plausible-looking and silent. All launches go to
//! one stream, so enqueue order is execution order.
//!
//! Cost: indexer scoring, top-k, gather and selected attention are paid once
//! per row, exactly as R serial steps would pay them; the step still batches
//! everything outside this phase. Rows still INSIDE the bound (a window that
//! straddles it) take the dense bs=1 attention, as serial decode does.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::MultiSeqCtx;
use crate::layer::{AttnMetadataDev, LayerState};
use crate::layers::ops;
use crate::layers::qsa::{QsaIndexer, QsaSelection};
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// Ingest row `i` into its owner's indexer state and return its selection
    /// (`None` inside the inert bound).
    ///
    /// Without `row_owner` the rows ARE the sequences (plain concurrent
    /// decode). With it, several rows share one sequence's state and advance
    /// it in row order — `decode_select` asserts `pos == ingested`, so the
    /// ordering is the invariant, not an optimization.
    #[allow(clippy::too_many_arguments)]
    fn qsa_select_row(
        &self,
        qsa: &QsaIndexer,
        c: &MultiSeqCtx<'_>,
        states: &mut [&mut (dyn LayerState + 'static)],
        row_owner: Option<&[usize]>,
        seq_lens: &[usize],
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        i: usize,
    ) -> Result<Option<QsaSelection>> {
        let owner = match row_owner {
            Some(map) => *map
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("QSA row_owner has no entry for row {i}"))?,
            None => i,
        };
        let state = states.get_mut(owner).ok_or_else(|| {
            anyhow::anyhow!("QSA row {i} owned by seq {owner}, which has no state")
        })?;
        let st = crate::layers::qwen3_attention::helpers::qsa_seq_state(qsa, *state, c.fwd.gpu)?;
        qsa.decode_select(
            st,
            c.normed.offset(i * c.h * c.bf16),
            seq_lens[i],
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            meta.block_table
                .offset(i * meta.max_blocks_per_seq as usize * 4),
            c.bs,
            c.fwd.gpu,
            c.stream,
        )
    }

    /// Ingest continuity for a step whose rows are ALL inert: the batched
    /// dense attention already ran, the indexer only has to keep its raw keys
    /// contiguous. A `Some` here means the pre-mutation plan (`guard.rs`) and
    /// this path disagree — refuse loudly rather than serve dense-past-budget,
    /// which is not the reference model.
    pub(super) fn ms_qsa_ingest_rows(
        &self,
        c: &MultiSeqCtx<'_>,
        states: &mut [&mut (dyn LayerState + 'static)],
        row_owner: Option<&[usize]>,
        seq_lens: &[usize],
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<()> {
        let Some(qsa) = self.qsa.as_ref() else {
            return Ok(());
        };
        for i in 0..c.n {
            let sel =
                self.qsa_select_row(qsa, c, states, row_owner, seq_lens, kv_cache, meta, i)?;
            anyhow::ensure!(
                sel.is_none(),
                "QSA selection active for row {i} on the batched ms path; \
                 the pre-mutation plan should have routed this step per-row"
            );
        }
        Ok(())
    }

    /// Phase 5 for a step with at least one ACTIVE row: per row, in order,
    /// `decode_select` then bs=1 attention over that row's selection (dense
    /// for a row still inside the bound). Replaces BOTH the batched dense
    /// attention — whose output would be discarded — and the ingest loop.
    /// Returns the attn_out buffer in the batched layout (`[n, nq*hd]`).
    pub(super) fn ms_phase_attn_qsa_rows(
        &self,
        c: &MultiSeqCtx<'_>,
        states: &mut [&mut (dyn LayerState + 'static)],
        row_owner: Option<&[usize]>,
        seq_lens: &[usize],
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<DevicePtr> {
        let qsa = self
            .qsa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("QSA per-row phase on a layer without an indexer"))?;
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bs,
            bf16,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        let attn_out = fwd.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let out_row = (nq * hd) as usize * bf16;
        let table_row = meta.max_blocks_per_seq as usize * 4;
        for i in 0..n {
            // Q leads each row's interleaved [Q|K|V|gate] block; with one
            // sequence per launch the kernel never applies the stride.
            let q_i = qkv_buf.offset(i * per_seq_qkv);
            let out_i = attn_out.offset(i * out_row);
            // Consumed inside this iteration — see THE INVARIANT above.
            match self.qsa_select_row(qsa, c, states, row_owner, seq_lens, kv_cache, meta, i)? {
                Some(sel) => ops::paged_decode_attn_bf16(
                    fwd.gpu,
                    self.paged_decode_k,
                    q_i,
                    sel.k_scratch,
                    sel.v_scratch,
                    out_i,
                    sel.table_dev,
                    sel.seq_len_dev,
                    sel.max_blocks,
                    1,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    nq * hd,
                    0,
                    stream,
                )?,
                None => self.run_paged_decode(
                    fwd.gpu,
                    q_i,
                    kv_cache,
                    out_i,
                    meta.block_table.offset(i * table_row),
                    meta.seq_len.offset(i * 4),
                    meta.max_blocks_per_seq,
                    1,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    nq * hd,
                    fwd.buffers.splitk_workspace(),
                    fwd.levers.max_decode_seqs,
                    stream,
                )?,
            }
        }
        Ok(attn_out)
    }
}
