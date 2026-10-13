// SPDX-License-Identifier: AGPL-3.0-only

//! MTP speculative draft proposal step.

use super::*;

mod bootstrap;
mod depth;
mod glm_repair;
mod one_forward;
mod owner_batch;
use depth::{deep_ladder, ladder_truncate, lone_dflash_width, single_depth_ladder};
use glm_repair::glm_repaired_narrow;
use owner_batch::verify_owner_batch;

/// MTP-aware step: bootstrap sequences without drafts, then verify via CUDA graph.
/// Supports K=2 (num_drafts=1) and K=3 (num_drafts=2).
///
/// `verify_ctx` carries the tokenizer special-token IDs the verify
/// pipeline needs (`<think>` / `</think>` / `<tool_call>` /
/// `</tool_call>`). Threaded down to every verify call site so the
/// 8-stage [`crate::scheduler::logit_processors`] pipeline can run on
/// each verify-position's logits — the fix for MTP-emitted tokens
/// bypassing all pre-sample masks. See `verify_pipeline_helper`.
pub fn step_mtp(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    // ATLAS_MTP_TIMING outer bracket: `step_mtp` minus the per-chunk verify
    // guard's TOTAL is the driver's own host prep/tail (classification,
    // bootstrap, D-Cut plan, chunk sort) — one component of the out-of-step
    // GAP. One Instant::now() when disarmed, same cost note as StepTimer.
    let t_step_outer = std::time::Instant::now();
    // Model-capability clamp, applied to `num_drafts` itself rather than to
    // the ladder below, because it has to be TOTAL: the per-sequence verify
    // dispatch near the end of this function branches on the raw
    // `num_drafts` (`num_drafts >= 3` picks the K=4 arm), and the bootstrap
    // propose passes it straight through. Clamping only the ladder left both
    // of those reaching for a width the model cannot serve.
    //
    // The batched verify runs K = drafts + 1 rows, and a model can cap K in a
    // way no pool capacity lifts (the mHC highway has MoE arms for K=2/3
    // only). Asking past it bails INSIDE the step, and `verify_k*_step`
    // answers a verify error with `a.finished = true` — `--num-drafts 3` on
    // qwen4_exp killed every request after one token that way. Clamped, those
    // steps just speculate less deeply.
    let num_drafts = match model.verify_max_drafts() {
        Some(max_nd) => num_drafts.min(max_nd),
        None => num_drafts,
    };
    let glm_repaired_narrow = glm_repaired_narrow(num_drafts);
    let mut bootstrap_idxs: Vec<usize> = Vec::new();
    let mut verify_idxs: Vec<usize> = Vec::new();
    for (i, a) in active.iter().enumerate() {
        if !a.pending_drafts.is_empty() {
            verify_idxs.push(i);
        } else {
            bootstrap_idxs.push(i);
        }
    }

    // K-vs-batch ladder (task #35): the per-step draft count is a function
    // of the CURRENT concurrency. Default ladder `4:3,8:3,16:1,32:1` holds
    // 3 drafts through n=8 and drops to 1 through the cap (32), so R =
    // Σ(drafts+1) tops out at the 64-row buffer bound at n=32 (32x2); n=8
    // (8x4) and n=16 (16x2) sit at 32 rows. The depth step-down that used
    // to sit at n>4 was an artifact of the chunk cap below, not of GDN
    // depth cost (see that comment); the one at n>8 is real (16:2 -> 94.1).
    // SSOT + overrides (`ATLAS_MTP_K_LADDER`, `ATLAS_NO_MTP_K_LADDER`):
    // `spark_model::speculative::ladder`. DFlash keeps its own γ economics.
    // Wave 28: at the n=16 rung the draft count is ACCEPT-RATE-AWARE — the
    // static rung cannot win both regimes (prose wants k=1, tool-shaped
    // wants k=2, same binary, same boot). `adaptive_rung::drafts_for`
    // returns the static ladder at every other width.
    let ladder_nd = if dflash_verify_raw_argmax {
        num_drafts
    } else {
        crate::scheduler::adaptive_rung::drafts_for(active.len(), num_drafts)
    };
    let ladder_nd = single_depth_ladder(active, sched, ladder_nd, dflash_verify_raw_argmax);
    // ATLAS_MTP_DYNAMIC_DEPTH: each request steers its own ceiling up to
    // `num_drafts` (`mtp_deep_depth`); the step drafts to the deepest and
    // `plan_verify` trims every sequence to its own and the row budget.
    let deep = !dflash_verify_raw_argmax
        && !glm_repaired_narrow
        && super::mtp_deep_depth::deep_step(sched.levers.depth(), num_drafts, active.len());
    let ladder_nd = if deep {
        deep_ladder(active, sched, num_drafts)
    } else {
        ladder_nd
    };
    super::mtp_step_trace::note_step(ladder_nd, deep);
    // Tiered verify-pool capacity clamp (2026-08-16): the step's draft
    // count must respect the MINIMUM slot capacity across the active
    // sequences — a sequence in a K=2-sized slot must never receive K=4
    // drafts. Capacities are the model's ACTUAL pool geometry
    // (`mtp_slot_draft_capacity`); full-width pools (kill switch, DFlash-γ,
    // pure-attention) report usize::MAX and leave the ladder untouched.
    // See `spec_capacity` for the invariant and its two trigger shapes.
    let ladder_nd = crate::scheduler::spec_capacity::clamp_drafts_to_slot_capacity(
        ladder_nd,
        active
            .iter()
            .map(|a| model.mtp_slot_draft_capacity(a.seq.slot_idx)),
    );

    // Plan the batched verify FIRST outside DFlash: a draftless sequence it
    // can carry as a decode row skips the bootstrap forward below
    // (`one_forward`). DFlash plans after Phase A, whose bootstrap stashes
    // drafts for Phase B (`late_dflash`).
    let early_plan = (!dflash_verify_raw_argmax).then(|| {
        one_forward::plan_verify(
            model,
            sched,
            active,
            &verify_idxs,
            &bootstrap_idxs,
            ladder_nd,
            deep.then_some(num_drafts),
            false,
        )
    });
    if let Some(p) = &early_plan {
        bootstrap_idxs.retain(|i| !p.decode_rows.contains(i));
    }

    // ── Phase A: Bootstrap decode for sequences without a draft ──
    if !bootstrap_idxs.is_empty() {
        // The previous verify commit's live-state restore runs async on the
        // secondary stream; order it before the bootstrap decode reads
        // h_state/conv_state (and before start_checkpoint_async snapshots
        // the live state). GPU-side event wait, zero CPU cost.
        if let Err(e) = model.sync_secondary() {
            tracing::error!("bootstrap sync_secondary: {e:#}");
        }
    }
    // Batched form: ONE `decode_batch` for every draftless sequence plus a
    // batched cross-sequence propose, replacing n M=1 weight sweeps of the
    // target and n of the drafter. Falls back to the per-sequence loop below
    // whenever the envelope does not hold (`mtp_bootstrap_step`); kill switch
    // ATLAS_NO_MTP_BATCH_BOOTSTRAP.
    if !spark_model::speculative::glm_repair_policy::enabled()
        && can_batch_bootstrap(model, sched, bootstrap_idxs.len(), dflash_verify_raw_argmax)
    {
        for &i in &bootstrap_idxs {
            super::mtp_step_trace::note_seq(&active[i], b'B', 0);
        }
        super::mtp_step_trace::note_forward(bootstrap_idxs.len());
        step_mtp_bootstrap_batched(model, active, sched, &bootstrap_idxs, ladder_nd, verify_ctx);
        bootstrap_idxs.clear();
    }
    let mut late_dflash: Vec<usize> = Vec::new();
    let n_active = active.len();
    for &idx in &bootstrap_idxs {
        super::mtp_step_trace::note_seq(&active[idx], b'b', 0);
        super::mtp_step_trace::note_forward(1);
        if bootstrap::bootstrap_one(
            model,
            &mut active[idx],
            sched,
            num_drafts,
            ladder_nd,
            n_active,
            glm_repaired_narrow,
            verify_ctx,
            dflash_verify_raw_argmax,
        ) {
            late_dflash.push(idx);
        }
    }
    verify_idxs.extend(late_dflash);

    // ── Phase B: Verify with pipelined checkpoint ──
    //
    // Batched multi-seq K-row verify (batched-MTP E11 + the ladder). Only
    // reachable when `ATLAS_MTP_MAX_SEQS > 1` puts >= 2 sequences in one
    // step (`ATLAS_MTP_MAX_SEQS=1` ⇒ every seq takes the per-seq loop below,
    // byte-identical to the pre-batched HEAD). Members, rows and order:
    // `one_forward::plan_verify` (grammarless sequences at their own depth,
    // decode rows where the model verifies them, D-Cut).
    if active.len() > 1 {
        tracing::info!(
            "DFLASH WIDTH n_active={} verify={} boot={} ladder_nd={}",
            active.len(),
            verify_idxs.len(),
            bootstrap_idxs.len(),
            ladder_nd
        );
    }
    let plan = early_plan.unwrap_or_else(|| {
        one_forward::plan_verify(
            model,
            sched,
            active,
            &verify_idxs,
            &[],
            ladder_nd,
            deep.then_some(num_drafts),
            dflash_verify_raw_argmax,
        )
    });
    let mut serial_idxs = plan.serial;
    let batchable_idxs = plan.batch;
    let ks = plan.ks;

    // Chunking: the 128-row buffer bound `can_batch_verify` enforces, with
    // the per-chunk sequence cap DERIVED from it (`VERIFY_ROW_BUDGET /
    // widest rows` — chunk_ranges; n is separately bounded at 32 by
    // VERIFY_WY_TABLE_SEQS, so the 32:2 shape at 96 rows still fits one
    // chunk). Default-ladder shapes chunk exactly as before (every default
    // rung's row total already fit the old 64-row budget in one chunk).
    //
    // History — the SAME stale-cap artifact, twice: rows=4 was once capped at
    // 4 seqs, which split 8 batchable sequences into TWO serialized 4-wide
    // verify forwards (2x the weight reads per step) and is what the
    // "8:3 collapses" measurements (57.9, 62.6 on 2026-07-28) actually
    // recorded — NOT depth-3 at width 8. Then rows=3/4 stayed hardcoded at
    // 8 seqs after the budget widened 32→64, which serialized every depth
    // shape above n=8 the same way (fixer r2 2026-07-30: a 16:2 env-ladder
    // leg read `n=8 k_drafts=2` in its accept telemetry — two chunks). The
    // cap is now anchored to the row budget so the artifact class is closed.
    for (lo, hi) in mtp_dcut::chunk_ranges(&ks) {
        let chunk = &batchable_idxs[lo..hi];
        let chunk_ks = &ks[lo..hi];
        // A lone chunk batches only where the model verifies one sequence's
        // window wider than its own verify serves (`can_batch_verify`:
        // qwen4_exp's exact lane past 4 rows).
        if model.can_batch_verify(chunk_ks) {
            // Collect disjoint &mut refs — the iterator walk requires ASCENDING
            // indices, so sort a copy of the chunk before walking and restore
            // the batch order (with each sequence's k) immediately after.
            let mut asc: Vec<(usize, usize)> = chunk
                .iter()
                .copied()
                .zip(chunk_ks.iter().copied())
                .collect();
            asc.sort_unstable();
            let mut refs: Vec<(&mut ActiveSeq, usize)> = Vec::with_capacity(chunk.len());
            let mut it = active.iter_mut();
            let mut consumed = 0usize;
            for &(i, k) in &asc {
                let a = it.nth(i - consumed).expect("chunk index within active");
                consumed = i + 1;
                refs.push((a, k));
            }
            // Batch order, from the ONE ordering rule shared with the graph
            // key (`verify_key`), asked with the arm `plan` chose so order
            // and assignment can never disagree. Canonical (n >=
            // `CANONICAL_KEY_MIN_WIDTH` = 8, every depth within the drafts
            // its sequence holds): ssm slots ascending = also deepest-first
            // under the canonical assignment, so the key is a function of the
            // depth MULTISET not its arrangement (266 keys → 3 at n=8) and
            // each depth run owns a consecutive slot block for the
            // batched-GDN precondition; otherwise deepest-first then slot —
            // the pre-canonical order byte for byte. Idempotent on `plan`'s
            // ordered batch under both arms; it still runs because `plan`
            // returns the batch UNORDERED whenever D-Cut declines (with
            // uniform `k`, the sort by slot). PERMUTATION ONLY — depths
            // stay attached to the sequence
            // `plan` truncated for (`verify_k4_batch_step` pins
            // `drafts + 1 == ks[i]`). Verdicts are index-mapped inside the
            // step, so batch order is free to the caller.
            let chunk_slots: Vec<usize> = refs
                .iter()
                .map(|(a, _)| a.seq.ssm_slot_idx().unwrap_or(usize::MAX))
                .collect();
            let chunk_depths: Vec<usize> = refs.iter().map(|&(_, k)| k).collect();
            let order = spark_model::speculative::verify_key::verify_batch_permutation(
                &chunk_slots,
                &chunk_depths,
                plan.canonical,
            );
            let sorted_ks: Vec<usize> = order.iter().map(|&p| chunk_depths[p]).collect();
            let mut slotted: Vec<Option<&mut ActiveSeq>> =
                refs.into_iter().map(|(a, _)| Some(a)).collect();
            let mut batch: Vec<&mut ActiveSeq> = order
                .iter()
                .map(|&p| {
                    slotted[p]
                        .take()
                        .expect("verify_batch_permutation is a permutation")
                })
                .collect();
            for (a, &k) in batch.iter().zip(&sorted_ks) {
                let path = if k > 1 { b'v' } else { b'd' };
                super::mtp_step_trace::note_seq(a, path, k - 1);
            }
            super::mtp_step_trace::note_forward(sorted_ks.iter().sum());
            if dflash_verify_raw_argmax {
                step_verify_dflash_batched(
                    model,
                    &mut batch,
                    sched,
                    &sorted_ks,
                    ladder_nd,
                    verify_ctx,
                    dflash_verify_raw_argmax,
                );
            } else {
                step_verify_k4_batched(
                    model, &mut batch, sched, &sorted_ks, ladder_nd, num_drafts, verify_ctx,
                );
            }
        } else {
            // Model can't batch this width (or a lone leftover): fall back
            // to the existing per-seq dispatch for these sequences.
            serial_idxs.extend_from_slice(chunk);
        }
    }
    verify_owner_batch(
        model,
        active,
        sched,
        &mut serial_idxs,
        num_drafts,
        ladder_nd,
        glm_repaired_narrow,
        verify_ctx,
        dflash_verify_raw_argmax,
    );
    for &idx in &serial_idxs {
        let a = &mut active[idx];
        // Confidences travel with the taken drafts and are cut with them.
        let (mut drafts, mut conf) = a.take_drafts();
        if drafts.is_empty() {
            continue;
        }
        lone_dflash_width(a, &conf, &mut drafts, dflash_verify_raw_argmax);

        // Spec-decode boundary awareness (arXiv:2512.15834): when a
        // grammar is active, validate the draft sequence against the
        // matcher and truncate at the first token that crosses a
        // grammar transition. Without this, a draft span that crosses
        // `</function>` (or any other structural boundary) gets
        // accepted by the verifier and emitted, but the post-emit
        // `accept_token` silently fails — desync'ing the grammar
        // from the output stream. Truncating here downgrades K=4 →
        // K=3 → K=2 cleanly.
        // Strict grammars trim at the verify, thinking-aware (`strict_spec`).
        if let Some(gs) = a.grammar_state.as_mut().filter(|gs| !gs.is_strict()) {
            let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
            if kept < drafts.len() {
                drafts.truncate(kept);
            }
            if drafts.is_empty() {
                continue;
            }
        }
        ladder_truncate(a, &mut drafts, ladder_nd);
        // A window only the batched verify serves (qwen4_exp's exact lane
        // past 4 rows) that lands here (grammar) takes the 4-row verify; so
        // does a grammar sequence's deep window (`ladder_truncate` leaves it
        // whole), whichever widths the model batches.
        if !dflash_verify_raw_argmax
            && drafts.len() > 3
            && ((deep && a.grammar_state.is_some()) || model.can_batch_verify(&[drafts.len() + 1]))
        {
            drafts.truncate(3);
        }
        conf.truncate(drafts.len());
        super::mtp_step_trace::note_seq(a, b's', drafts.len());
        super::mtp_step_trace::note_forward(drafts.len() + 1);

        // DFlash/DSpark verify: route by proposer, not draft count.
        // `--dflash` sets dflash_verify_raw_argmax. The old `drafts.len()>=4`
        // ladder sent K=3 (`--dflash-gamma 4`) into MTP K=3 verify.
        if dflash_verify_raw_argmax || glm_repaired_narrow || drafts.len() >= 4 {
            step_verify_dflash(
                model,
                a,
                sched,
                &drafts,
                &conf,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        } else if num_drafts >= 3 && drafts.len() >= 3 {
            step_verify_k4(
                model,
                a,
                sched,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        } else if num_drafts >= 2 && drafts.len() >= 2 {
            step_verify_k3(
                model,
                a,
                sched,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        } else {
            step_verify_k2(
                model,
                a,
                sched,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        }
    }
    sched
        .timing
        .record(crate::scheduler::mtp_timing::Phase::StepOuter, t_step_outer);
}
