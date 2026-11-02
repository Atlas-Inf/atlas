// SPDX-License-Identifier: AGPL-3.0-only

//! D-Cut: adaptive verification-depth pruning (arXiv 2607.14647).
//!
//! # What it does
//!
//! The drafter emits `k_drafts` tokens per sequence and each one costs a VERIFY
//! ROW. Rows are the verify step's price: the batched forward reads every weight
//! once for `R = Σ rows_i` rows, and R is capped at 128 by the row buffers
//! (`VERIFY_ROW_BUDGET`; 64 before the wave-11 depth-at-width widening, 32
//! before the 32:1 ladder rung). D-Cut spends that budget where it will be
//! accepted instead of spreading it evenly — but only in the <= 8-sequence
//! regime it was measured to win ([`dcut_width_cap`]).
//!
//! The drafter reports a per-position top-1 log-probability `ln c_{i,t}`
//! (`argmax_bf16_batch_lp`). A draft at depth `j` is only reachable if every
//! draft before it was accepted, so its SURVIVAL score is the prefix product
//! `s_{i,j} = Π_{t<=j} c_{i,t}` — in log space the prefix SUM. All prunable
//! positions across the WHOLE batch are ranked by that score and the top
//! `ratio` fraction is retained.
//!
//! ★ Because log-probabilities are <= 0, `s_{i,j}` is non-increasing in `j`, so
//! the retained set is automatically a per-sequence contiguous PREFIX — no tree,
//! no gaps, just a per-sequence draft count. That is the whole reason this needs
//! ragged row counts and nothing else.
//!
//! # v1 scope (deliberate)
//!
//! * Every sequence keeps AT LEAST ONE of its drafts. That holds `rows_i` in
//!   2..=4 (2..=8 on qwen4_exp's exact lane under `ATLAS_MTP_DYNAMIC_DEPTH`)
//!   for a sequence that drafted — the envelope `can_batch_verify`, the
//!   `gdn_decode_wy{2,3,4}` handles and the SSM intermediates pools were built
//!   and audited for. D-Cut never prunes a sequence to zero drafts; a
//!   sequence that arrives WITHOUT drafts (a decode row riding the verify,
//!   `mtp_step/one_forward.rs`) is planned at one row and is not prunable.
//! * A sequence may hold fewer drafts than the ladder depth (the drafter's
//!   confidence stop). Its drafts cap its depth: positions it did not draft
//!   are not rankable, and no assignment may deepen it past them.
//! * The budget is a FIXED ratio from the discrete bucket set, not a profiled
//!   cost table. `ATLAS_MTP_DCUT_RATIO` picks it; values snap to the nearest
//!   bucket so the search space stays the paper's four points.
//! * Pruning changes only the VERIFY width. The propose already ran at full
//!   width when this is called, so v1 banks the row saving, not a drafter
//!   saving.

use super::types::ActiveSeq;

/// The paper's discrete retention buckets.
const BUCKETS: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

/// Verify row-buffer capacity — the exact bound `can_batch_verify` enforces
/// as `Σ rows_i <= 128` (logits rows / meta gaps / bt staging, `sizes.rs`;
/// model twin: `VERIFY_ROW_CAP`, verify_e2.rs — keep in lock-step). 128
/// since the DFlash2 C=16 sweep (job 376): γ=8 → 8 rows/seq, n=16 ⇒ 128
/// rows hit the new budget dead on — at 96 it split 12+4 = two verify
/// passes per step (C16 measured 66.2 < C8's 69.2); at 128 one pass. 96 since wave 11 (depth at
/// width: 32:2 = n=32 × k=3 rows hits 96 dead on, 24:2 = 72); previously 64
/// (the 32:1 rung, n=32 × k=2), 32 before that. Raising the budget is
/// behavior-neutral for every shape that already fit: row totals ≤ 96
/// chunk and prune identically — only shapes with 97..=128 rows (32:2's
/// 96 still one chunk) or DFlash γ=8 at n≥12 change.
pub(super) const VERIFY_ROW_BUDGET: usize = 128;

/// Widest verify batch (SEQUENCES, not rows) D-Cut may prune —
/// the D-Cut-at-depth policy. Value-parsed from `ATLAS_MTP_DCUT_MAX_SEQS`
/// once per process (0 disables pruning entirely; `ATLAS_NO_MTP_DCUT` also
/// does).
///
/// Default 8, anchored to two measurements on the same binary class:
/// * D-Cut's win is a C=8 result — ratio 0.75 pooled 108.57 vs 105.56 off
///   (+2.6%, binary `296b9674`), measured at n<=8 where `ladder_nd = 3`.
/// * At depth-at-width (the 16:2 rung, n=16 × nd=2) pruning is NEGATIVE:
///   fixer r2 leg D read 176.6-179.4 at C=16 vs 194.4-196.0 for the same
///   ladder with pruning off (-9%) — ragged nd=2 pruning fragments the
///   contiguous GDN depth runs and sheds winning drafts. The wave-11 grid
///   confirmed the winner (195.0) with pruning off at n=16.
/// So pruning engages only at batch width <= 8 — exactly the regime it was
/// measured to win — and the 16:2 default rung always verifies the uniform
/// single-chunk `[3; n]` shape that measured +5.7%.
pub(super) fn dcut_width_cap() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_MTP_DCUT_MAX_SEQS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    })
}

/// Default ON, kill switch `ATLAS_NO_MTP_DCUT` — PRESENCE check (house
/// convention: `=0` is NOT off).
///
/// Measured at C=8 on binary `296b9674` (one fresh serve per leg, warmup
/// discarded, 5 scored reps): D-Cut off 105.56, ratio 1.0 (wiring live, zero
/// rows pruned) 105.80 — statistically identical, so the plumbing is inert
/// when it prunes nothing — and ratio 0.75 pools to 108.57 over two serves,
/// **+2.6%**. Pruning is additionally a no-op at `ladder_nd < 2` and above
/// [`dcut_width_cap`] sequences (the D-Cut-at-depth policy — see that fn:
/// pruning at the 16:2 rung's n=16 measured -9%).
pub(super) fn dcut_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_MTP_DCUT").is_none())
}

/// Retention ratio, VALUE-parsed once and snapped to the nearest
/// [`BUCKETS`] entry.
///
/// Default 0.75 — the ONLY bucket that wins. The whole bucket set was swept at
/// C=8 on binary `296b9674`, one fresh serve each: 1.0 (control) 105.80 ·
/// **0.75 108.57** · 0.5 107.43 · 0.25 101.56. Telemetry shows exactly why —
/// tok_step degrades monotonically as rows are pruned (2.52 / 2.54 / 2.43 /
/// 2.16) while kept_frac falls (1.000 / 0.876 / 0.750 / 0.626), and 0.75 is
/// the one point where the row saving outruns the token loss.
/// 1.0 remains the natural A/B control (wiring live, zero rows pruned).
pub(super) fn dcut_ratio() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let raw = std::env::var("ATLAS_MTP_DCUT_RATIO")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.75)
            .clamp(0.0, 1.0);
        *BUCKETS
            .iter()
            .min_by(|a, b| {
                (*a - raw)
                    .abs()
                    .partial_cmp(&(*b - raw).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("BUCKETS is non-empty")
    })
}

/// Retained draft count per sequence for one verify batch.
///
/// `confs[i]` is sequence i's per-draft top-1 LOG-probability, in draft order.
/// A short or empty row means "not measured": those positions score 0.0 (=
/// certainty) and therefore survive, so a drafter that cannot report confidence
/// is never pruned on a number nobody produced.
///
/// Returns `retained[i]` in `1..=k_drafts`. `row_budget` caps `Σ (retained+1)`
/// so the result can never exceed the verify row buffer.
pub(super) fn select(
    confs: &[&[f32]],
    k_drafts: usize,
    row_budget: usize,
    ratio: f32,
) -> Vec<usize> {
    select_capped(confs, &vec![k_drafts; confs.len()], row_budget, ratio)
}

/// [`select`] over sequences holding `caps[i]` drafts each (`caps[i] <=` the
/// ladder depth): `retained[i]` is in `1..=caps[i]`, and `0` for a sequence
/// with no drafts. With every cap equal this is [`select`] exactly.
pub(super) fn select_capped(
    confs: &[&[f32]],
    caps: &[usize],
    row_budget: usize,
    ratio: f32,
) -> Vec<usize> {
    debug_assert_eq!(confs.len(), caps.len());
    // Depth 1 is mandatory (see module docs), so only depths 2..=caps[i] are
    // rankable.
    let mut retained: Vec<usize> = caps.iter().map(|&c| c.min(1)).collect();
    let prunable: usize = caps.iter().map(|&c| c.saturating_sub(1)).sum();
    if prunable == 0 {
        return retained;
    }

    // Score every prunable position by its log survival (prefix sum).
    let mut ranked: Vec<(f32, usize, usize)> = Vec::with_capacity(prunable);
    for (i, c) in confs.iter().enumerate() {
        let mut acc = 0.0f32;
        for j in 0..caps[i] {
            // Missing measurement -> 0.0 (certain), which sorts to the top.
            acc += c.get(j).copied().unwrap_or(0.0);
            if j >= 1 {
                ranked.push((acc, i, j));
            }
        }
    }
    // Descending by score; ties break on (sequence, depth) so the selection is
    // a deterministic function of the batch — a graph key derived from the
    // resulting shape must not depend on sort instability.
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });

    let by_ratio = ((prunable as f32) * ratio).round() as usize;
    // Rows already committed: one base row + the mandatory draft per sequence.
    let committed: usize = retained.iter().map(|r| r + 1).sum();
    let by_budget = row_budget.saturating_sub(committed);
    let keep = by_ratio.min(by_budget).min(prunable);

    for &(_, i, j) in ranked.iter().take(keep) {
        // Scores are non-increasing in depth, so the top-`keep` set is already
        // prefix-closed; `max` records the deepest retained position.
        retained[i] = retained[i].max(j + 1);
    }
    retained
}

/// One planned verify batch: per-sequence ROW counts in dispatch order, and
/// whether the depths were assigned canonically (the order `mtp_step` must
/// re-apply per chunk — see [`assign`]).
pub(super) struct Planned {
    pub ks: Vec<usize>,
    pub canonical: bool,
}

/// Plan one verify batch: choose the retained draft-count MULTISET from the
/// drafter's confidences, assign it to the batch's ssm slots in the canonical
/// order, truncate each sequence's drafts to its assigned prefix, and return
/// the resulting per-sequence ROW count (`retained + 1`) in dispatch order.
///
/// Each sequence's depth is capped by the drafts it holds (at most
/// `ladder_nd`; fewer after the drafter's confidence stop, none for a decode
/// row). With every sequence at the ladder depth this is the pre-cap plan
/// byte for byte.
///
/// ★ The depth→slot ASSIGNMENT is canonical (`verify_key::verify_batch_order`
/// — depths descending paired with slots ascending), not confidence-ordered,
/// AT BATCH WIDTHS >= `verify_key::CANONICAL_KEY_MIN_WIDTH` (8). D-Cut's row
/// saving comes from the multiset, which stays confidence-chosen; WHO gets
/// which depth was the half that multiplied the batched-verify CUDA-graph key
/// space (266 arrangements at n=8 against a 32-entry cache → 89% of steps
/// re-capturing, 23.2 ms/step). It also RECONCILES the batch's two ordering
/// demands — contiguous equal-depth runs and ascending consecutive ssm slots
/// — which the confidence-ordered arrangement put in direct conflict.
///
/// Below that width the key space is 2 (n=2) or 10 (n=4) keys against a
/// 32-entry cache — nothing to collapse — and the forced assignment measured
/// NET NEGATIVE there (-2.4% at C=2, -3.7% at C=4), so this falls back to the
/// pre-canonical assignment byte for byte. `verify_key::canonical_assignment`
/// is the single gate (threshold + `ATLAS_CANONICAL_KEY_MIN_WIDTH` override +
/// the `ATLAS_NO_CANONICAL_VERIFY_KEY` kill switch); see `verify_key`'s module
/// docs and `CANONICAL_KEY_MIN_WIDTH` for the A/B table.
///
/// With `ATLAS_NO_MTP_DCUT` set — or the batch wider than [`dcut_width_cap`]
/// sequences (the D-Cut-at-depth policy: pruning at the 16:2 rung's n=16
/// measured -9%, so depth-at-width always verifies the uniform shape that
/// won) — every sequence verifies all the drafts it holds and the batch order
/// is untouched: with uniform drafts the caller's downstream path is then
/// byte-identical to the pre-D-Cut one.
pub(super) fn plan(
    active: &mut [ActiveSeq],
    batchable: &mut Vec<usize>,
    ladder_nd: usize,
) -> Planned {
    let caps: Vec<usize> = batchable
        .iter()
        .map(|&i| active[i].pending_drafts.len().min(ladder_nd))
        .collect();
    if !dcut_enabled()
        || ladder_nd < 2
        || batchable.is_empty()
        || batchable.len() > dcut_width_cap()
    {
        // Ordered by `mtp_step` (deepest first): with uniform caps that is
        // the slot order either arm produces.
        return Planned {
            ks: caps.iter().map(|c| c + 1).collect(),
            canonical: false,
        };
    }
    // Length-matched or nothing: a stale or absent confidence vector must
    // read as "not measured" (full depth), never as a score.
    let confs: Vec<&[f32]> = batchable.iter().map(|&i| active[i].draft_conf()).collect();
    let retained = select_capped(&confs, &caps, VERIFY_ROW_BUDGET, dcut_ratio());
    let slots: Vec<usize> = batchable
        .iter()
        .map(|&idx| active[idx].seq.ssm_slot_idx().unwrap_or(usize::MAX))
        .collect();
    // Dispatch order + depth assignment, from the ONE ordering rule shared
    // with the graph key. This is the ONE place the assignment is decided, so
    // it is the ONE place the width gate is asked — `mtp_step` re-applies the
    // ORDER with the `canonical` this returns.
    let (order, ks_out, canonical) = assign(
        &slots,
        &caps,
        &retained,
        spark_model::speculative::verify_key::canonical_assignment(batchable.len()),
    );
    // Truncate to the ASSIGNED depth — a prefix of what the drafter produced
    // by construction (`assign` never deepens a sequence past its cap).
    let reordered: Vec<usize> = order.iter().map(|&p| batchable[p]).collect();
    for (idx, &k) in reordered.iter().zip(&ks_out) {
        let a = &mut active[*idx];
        debug_assert!(
            k >= 1 && k - 1 <= a.pending_drafts.len(),
            "assigned depth {k} exceeds the {} drafts proposed",
            a.pending_drafts.len()
        );
        a.pending_drafts.truncate(k - 1);
        a.pending_draft_conf.truncate(k - 1);
    }
    record(
        caps.iter().map(|c| c + 1).sum(),
        ks_out.iter().sum(),
        &ks_out,
    );
    *batchable = reordered;
    Planned {
        ks: ks_out,
        canonical,
    }
}

/// Order a planned batch and pair its row counts (`retained + 1`) with it:
/// `(order, ks, canonical)`. Canonical when `canonical_allowed` and the
/// canonical pairing (depths descending onto slots ascending) gives no
/// sequence more rows than its `caps` (drafts held) + 1; otherwise each
/// sequence keeps its own depth, deepest first — the pre-canonical
/// arrangement, and the only one that is always a prefix of every
/// sequence's drafts.
pub(super) fn assign(
    slots: &[usize],
    caps: &[usize],
    retained: &[usize],
    canonical_allowed: bool,
) -> (Vec<usize>, Vec<usize>, bool) {
    use spark_model::speculative::verify_key::verify_batch_order;
    let ks: Vec<usize> = retained.iter().map(|r| r + 1).collect();
    if canonical_allowed {
        let (order, depths) = verify_batch_order(slots, &ks, true);
        if order.iter().zip(&depths).all(|(&p, &k)| k <= caps[p] + 1) {
            return (order, depths, true);
        }
    }
    let (order, depths) = verify_batch_order(slots, &ks, false);
    (order, depths, false)
}

/// Split a batch into verify chunks: `[lo, hi)` index ranges over `ks`.
///
/// ONE cap: the row-buffer bound (`VERIFY_ROW_BUDGET` = 128) — the audited
/// verify envelope (meta gaps / logits rows / bt staging, sizes.rs). The
/// sequence-count cap is DERIVED from it per chunk (`budget / widest rows`:
/// rows=4 → 32 seqs, rows=3 → 42, rows=2 → 64 (sequence count is separately
/// bounded at 32 by VERIFY_WY_TABLE_SEQS — the 32:2 shape still fits one
/// chunk at 96 rows);
/// `can_batch_verify` separately bounds n at 32 = `VERIFY_WY_TABLE_SEQS`), no
/// longer a hardcoded 8 for the deep widths. The old 8 was stale from the
/// 32-row budget era and SILENTLY SERIALIZED any depth shape above n=8 into
/// 8-wide verify chunks (double weight reads per step): a 2026-07-30 fixer-r2
/// leg with `ATLAS_MTP_K_LADDER=..,16:2,..` measured 127-135 tok/s at C=16 vs
/// a 184-185 same-session 16:1 control, and its accept telemetry read
/// `n=8 k_drafts=2` — the chunk cap, not depth economics (the exact artifact
/// class the ladder history documents for "8:3 collapses" / "16:2 → 94.1").
/// Every default-ladder shape (`4:3,8:3,16:2,32:1` — widest totals 32/48/64
/// rows) is a SINGLE chunk under both the old 64 and this 96 budget, so the
/// widening only opens the explicit 24:2/32:2 env rungs (72/96 rows). With
/// uniform `ks` this still reproduces `chunks(budget/rows)` exactly, so
/// D-Cut-off stays byte-identical per chunk. `ks` is deepest-first, so the
/// widest row count is the chunk's first element and the cap never changes
/// mid-chunk.
pub(super) fn chunk_ranges(ks: &[usize]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut lo = 0usize;
    while lo < ks.len() {
        // Derived, not hardcoded: rows <= 8 (the exact lane's deep windows,
        // `mtp_deep_depth`; 4 elsewhere), so the division is >= 16.
        let seq_cap = VERIFY_ROW_BUDGET / ks[lo].max(1);
        let mut hi = lo;
        let mut r = 0usize;
        while hi < ks.len() && hi - lo < seq_cap && r + ks[hi] <= VERIFY_ROW_BUDGET {
            r += ks[hi];
            hi += 1;
        }
        // A single sequence wider than the whole budget cannot happen (rows <=
        // 4), but never emit an empty range.
        if hi == lo {
            hi = lo + 1;
        }
        out.push((lo, hi));
        lo = hi;
    }
    out
}

/// Per-step retained-rows telemetry, under the existing
/// `ATLAS_MTP_ACCEPT_DEBUG` gate. Counters only; one line per `PERIOD` steps.
fn record(rows_full: usize, rows_kept: usize, ks: &[usize]) {
    use std::sync::atomic::{AtomicU64, Ordering};
    const PERIOD: u64 = 200;
    static STEPS: AtomicU64 = AtomicU64::new(0);
    static FULL: AtomicU64 = AtomicU64::new(0);
    static KEPT: AtomicU64 = AtomicU64::new(0);
    if !spark_model::speculative::mtp_accept_debug() {
        return;
    }
    FULL.fetch_add(rows_full as u64, Ordering::Relaxed);
    KEPT.fetch_add(rows_kept as u64, Ordering::Relaxed);
    if STEPS.fetch_add(1, Ordering::Relaxed) + 1 >= PERIOD {
        let steps = STEPS.swap(0, Ordering::Relaxed).max(1);
        let full = FULL.swap(0, Ordering::Relaxed).max(1);
        let kept = KEPT.swap(0, Ordering::Relaxed);
        tracing::info!(
            "MTP D-Cut ratio={:.2} steps={steps} rows_full={full} rows_kept={kept} \
             kept_frac={:.3} last_ks={ks:?}",
            dcut_ratio(),
            kept as f64 / full as f64,
        );
    }
}
#[cfg(test)]
#[path = "mtp_dcut_tests.rs"]
mod tests;
