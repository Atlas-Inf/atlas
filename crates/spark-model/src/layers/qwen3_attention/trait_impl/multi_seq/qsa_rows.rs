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
use crate::layers::qsa::{QSA_ROWS_MAX, QsaIndexer, QsaSelection};
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
    ///
    /// Live rows only: the padding rows of a batched decode (`c.active..c.n`)
    /// sit on dummy states, and ingesting them allocated a fresh indexer
    /// carry (raw + pooled key buffers) per padding row, layer and step that
    /// nothing ever freed. Inside a staged run (`qsa_staged.rs`) each live
    /// row only stages; the model commits after the run.
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
        if crate::layers::qsa::staged_ingest() {
            for i in 0..c.active {
                qsa.stage_row(c.normed.offset(i * c.h * c.bf16), i, c.fwd.gpu, c.stream)?;
            }
            return Ok(());
        }
        for i in 0..c.active {
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
        let owner_of = |i: usize| row_owner.map_or(i, |m| m.get(i).copied().unwrap_or(usize::MAX));
        let mut i = 0;
        while i < n {
            // ATLAS_QWEN4EXP_QSA_DECODE_ROWS: a run of one sequence's
            // consecutive active rows selects and attends in one launch per
            // stage (`qsa_decode_rows.rs`), bit-identical to the loop below.
            let run = if qsa.decode_rows_on() && i < c.active {
                batched_run_len(i, c.active, QSA_ROWS_MAX, &owner_of, seq_lens, |p| {
                    qsa.is_active_at(p)
                })
            } else {
                0
            };
            if run > 0 {
                let owner = owner_of(i);
                let state = states.get_mut(owner).ok_or_else(|| {
                    anyhow::anyhow!("QSA row {i} owned by seq {owner}, which has no state")
                })?;
                let st =
                    crate::layers::qwen3_attention::helpers::qsa_seq_state(qsa, *state, fwd.gpu)?;
                let sel = qsa.decode_select_rows(
                    st,
                    c.normed.offset(i * c.h * bf16),
                    c.h * bf16,
                    seq_lens[i],
                    run,
                    fwd.gpu,
                    stream,
                )?;
                // The run's last row maps every token any of its rows reads.
                qsa.attend_rows(
                    &sel,
                    qkv_buf.offset(i * per_seq_qkv),
                    (per_seq_qkv / bf16) as u32,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    meta.block_table.offset((i + run - 1) * table_row),
                    attn_out.offset(i * out_row),
                    nq,
                    nkv,
                    bs,
                    inv_sqrt_d,
                    fwd.gpu,
                    stream,
                )?;
                i += run;
                continue;
            }
            // Q leads each row's interleaved [Q|K|V|gate] block; with one
            // sequence per launch the kernel never applies the stride.
            let q_i = qkv_buf.offset(i * per_seq_qkv);
            let out_i = attn_out.offset(i * out_row);
            // Consumed inside this iteration — see THE INVARIANT above. A
            // padding row (`c.active..n`) owns no indexer carry: dense.
            let sel = if i < c.active {
                self.qsa_select_row(qsa, c, states, row_owner, seq_lens, kv_cache, meta, i)?
            } else {
                None
            };
            match sel {
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
            i += 1;
        }
        Ok(attn_out)
    }
}

/// Rows from `i` that one batched selection serves: the same owner, one
/// position apart, every one ACTIVE (activity is monotone in position, so the
/// first row decides), at most `max`, never past `active`. 0: row `i` takes
/// the per-row path.
pub(super) fn batched_run_len(
    i: usize,
    active: usize,
    max: usize,
    owner_of: &dyn Fn(usize) -> usize,
    seq_lens: &[usize],
    is_active_at: impl Fn(usize) -> bool,
) -> usize {
    if i >= active || i >= seq_lens.len() || !is_active_at(seq_lens[i]) {
        return 0;
    }
    let mut len = 1;
    while len < max
        && i + len < active
        && i + len < seq_lens.len()
        && owner_of(i + len) == owner_of(i)
        && seq_lens[i + len] == seq_lens[i] + len
    {
        len += 1;
    }
    len
}

#[cfg(test)]
mod run_tests {
    use super::batched_run_len;

    fn run(i: usize, active: usize, owners: &[usize], lens: &[usize]) -> usize {
        batched_run_len(i, active, 4, &|r| owners[r], lens, |p| p >= 100)
    }

    #[test]
    fn a_verify_window_is_one_run() {
        assert_eq!(run(0, 4, &[0, 0, 0, 0], &[200, 201, 202, 203]), 4);
        assert_eq!(run(1, 4, &[0, 0, 0, 0], &[200, 201, 202, 203]), 3);
    }

    #[test]
    fn runs_split_at_owner_position_cap_and_padding() {
        // Two sequences' windows: the run stops at the owner change.
        assert_eq!(run(0, 4, &[0, 0, 1, 1], &[200, 201, 500, 501]), 2);
        assert_eq!(run(2, 4, &[0, 0, 1, 1], &[200, 201, 500, 501]), 2);
        // Same owner, a position gap: not one window.
        assert_eq!(run(0, 3, &[0, 0, 0], &[200, 201, 205]), 2);
        // Capped at `max` (4 here); padding rows (`active..`) never join.
        assert_eq!(run(0, 6, &[0; 6], &[200, 201, 202, 203, 204, 205]), 4);
        assert_eq!(run(0, 2, &[0; 4], &[200, 201, 202, 203]), 2);
        assert_eq!(run(2, 2, &[0; 4], &[200, 201, 202, 203]), 0);
    }

    #[test]
    fn an_inert_first_row_takes_the_per_row_path() {
        // A window that straddles the bound: row 0 inert, the rest per row.
        assert_eq!(run(0, 4, &[0; 4], &[98, 99, 100, 101]), 0);
        assert_eq!(run(2, 4, &[0; 4], &[98, 99, 100, 101]), 2);
    }
}
