// SPDX-License-Identifier: AGPL-3.0-only

//! Dense-ctx-window capture planner for DFlash prefill.
//!
//! The ctx accumulator is a compacted array of `acc_rows` rows: carried rows
//! live in `[0..base)`, this prefill's captures land at `base + (chunk_start -
//! origin)`. When a chunk's writes would overflow the window, the newest
//! `keep = max_ctx_len / 2` rows are kept via a single in-place slide — this
//! module computes that slide (and how many leading chunk tokens fall outside
//! the retained window) as pure arithmetic so it stays provably in-bounds.
//!
//! The old inline version silently truncated when a single chunk exceeded
//! `keep` (GB10's 8193-token prefill chunks against a 4096-row window): the
//! slide was skipped, the next chunk then slid with a source range beyond the
//! accumulator — a ~100-200 MB out-of-bounds D2D read. Planned arithmetic here
//! guarantees the slide's source range is always inside `[0, acc_rows)` —
//! including the `move_rows == 0` case where the slide is a pure drop.

/// In-place slide applied before a chunk's writes: rows
/// `[drop_n .. drop_n + move_rows)` move to `[0 .. move_rows)`.
/// `carried_drop` is how many of the dropped rows were carried (<= base) —
/// the rest come off the prefill's own captured prefix (`origin` advances by
/// `drop_n - carried_drop`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Slide {
    pub drop_n: usize,
    pub move_rows: usize,
    pub carried_drop: usize,
}

/// Where one prefill chunk's captures land in the dense ctx window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapturePlan {
    /// Slide to apply first (None = none needed).
    pub slide: Option<Slide>,
    /// Leading tokens of this chunk that fall outside the retained window.
    pub skip: usize,
    /// Accumulator row of token `skip`; token t (t >= skip) goes to
    /// `first_row + (t - skip)`.
    pub first_row: usize,
    /// `ctx_prefill_base` / `ctx_prefill_origin` after the slide.
    pub base: usize,
    pub origin: usize,
}

/// Plan one chunk's captures. `base`/`origin` are the current
/// `ctx_prefill_base`/`ctx_prefill_origin` (carry bookkeeping); `acc_rows`
/// the accumulator's row capacity (== `ctx_acc_rows`); `max_ctx_len` the
/// retention window.
///
/// Invariants the caller relies on:
/// - `slide.source = [drop_n, drop_n + move_rows)` is always inside
///   `[0, acc_rows)` and non-overlapping with `[0, move_rows)` (debug-asserted).
/// - Re-planning the same chunk after applying `base`/`origin` yields
///   `slide = None` with the same `skip`/`first_row` (the per-layer calls are
///   idempotent).
/// - `first_row + (proc_count - skip)` never exceeds `acc_rows`: after a slide
///   `first_row + count == keep <= acc_rows / 2`; without one, `needed <=
///   acc_rows` by definition.
pub(crate) fn plan_capture(
    base: usize,
    origin: usize,
    chunk_start: usize,
    proc_count: usize,
    acc_rows: usize,
    max_ctx_len: usize,
) -> CapturePlan {
    // Tokens already captured before `origin` (e.g. a carry that adopted a
    // window the new prompt extends) must not be written again — they own
    // their rows already.
    let skip0 = origin.saturating_sub(chunk_start).min(proc_count);
    let cursor0 = base + chunk_start.saturating_sub(origin);
    let needed = cursor0 + (proc_count - skip0);
    debug_assert!(
        cursor0 <= acc_rows,
        "cursor {cursor0} > acc_rows {acc_rows}: a prior chunk already wrote past the accumulator"
    );
    if needed > acc_rows {
        let keep = max_ctx_len / 2;
        // Enough rows are dropped to leave keep rows after this chunk. The
        // drop can exceed cursor0 (an oversized chunk) — then move_rows = 0
        // and the leading `skip` chunk tokens are simply never captured.
        let drop_n = needed - keep;
        let move_rows = cursor0.saturating_sub(drop_n);
        debug_assert!(
            move_rows <= drop_n,
            "slide overlaps: {move_rows} > {drop_n}"
        );
        debug_assert!(
            move_rows == 0 || drop_n + move_rows <= acc_rows,
            "slide source [{drop_n},{}) out of {acc_rows} bounds",
            drop_n + move_rows
        );
        let carried_drop = drop_n.min(base);
        let base = base - carried_drop;
        let origin = origin + (drop_n - carried_drop);
        let skip = origin.saturating_sub(chunk_start).min(proc_count);
        let first_row = base + chunk_start.saturating_sub(origin);
        return CapturePlan {
            slide: Some(Slide {
                drop_n,
                move_rows,
                carried_drop,
            }),
            skip,
            first_row,
            base,
            origin,
        };
    }
    CapturePlan {
        slide: None,
        skip: skip0,
        first_row: cursor0,
        base,
        origin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // (a) The GB10 crash case: an 8193-token chunk against a 4096-row window.
    #[test]
    fn gb10_oversized_chunk_stays_in_bounds() {
        let (acc_rows, win) = (4096usize, 4096usize);
        // chunk 1: start 11984, 8193 tokens
        let p = plan_capture(0, 11984, 11984, 8193, acc_rows, win);
        assert_eq!(
            p.slide,
            Some(Slide {
                drop_n: 6145,
                move_rows: 0,
                carried_drop: 0
            })
        );
        assert_eq!(p.skip, 6145);
        assert_eq!(p.first_row, 0);
        assert_eq!((p.base, p.origin), (0, 18129));
        // writes land in [0, 2048)
        assert!(p.first_row + (8193 - p.skip) <= acc_rows);
        // chunk 2 (re-plan with the updated base/origin): no slide, append
        let p2 = plan_capture(p.base, p.origin, 20177, 353, acc_rows, win);
        assert_eq!(p2.slide, None);
        assert_eq!(p2.skip, 0);
        assert_eq!(p2.first_row, 2048);
        // final ctx_len = base + (chunk_end - origin') = 2401 < acc_rows
        let chunk_end = 20177 + 353;
        assert_eq!(p2.base + (chunk_end - p2.origin), 2401);
    }

    // (b) Re-planning after applying a plan is a no-op: later capture layers
    // must see slide=None with the same skip/first_row.
    #[test]
    fn replan_is_idempotent() {
        let (acc_rows, win) = (4096usize, 4096usize);
        let p = plan_capture(0, 11984, 11984, 8193, acc_rows, win);
        let p2 = plan_capture(p.base, p.origin, 11984, 8193, acc_rows, win);
        assert_eq!(p2.slide, None);
        assert_eq!((p2.skip, p2.first_row), (p.skip, p.first_row));
        // small-chunk slide case: base 3500/4096, a 1024 chunk at origin 3500
        let q = plan_capture(3500, 12000, 12000, 1024, acc_rows, win);
        assert!(q.slide.is_some());
        let q2 = plan_capture(q.base, q.origin, 12000, 1024, acc_rows, win);
        assert_eq!(q2.slide, None);
        assert_eq!((q2.skip, q2.first_row), (q.skip, q.first_row));
    }

    // (c) Old-path equivalence in the Strix regime (chunks <= keep, origin <=
    // chunk_start): plan == the old drop_n/move_rows/cursor formula.
    #[test]
    fn strix_regime_matches_the_old_formula() {
        for win in [8192usize, 12288] {
            let keep = win / 2;
            for mut base in [0usize, 3000] {
                let mut origin = 0usize;
                for chunk_start in (0..25000).step_by(1024) {
                    if origin == 0 {
                        origin = chunk_start;
                    }
                    let proc = 1024usize.min(25000 - chunk_start);
                    let p = plan_capture(base, origin, chunk_start, proc, win, win);
                    // old formula
                    let cursor = base + chunk_start.saturating_sub(origin);
                    let needed = cursor + proc;
                    if needed <= win {
                        assert_eq!(p.slide, None);
                        assert_eq!(p.first_row, cursor);
                        assert_eq!(p.skip, 0);
                    } else {
                        let drop_n = needed - keep;
                        assert_eq!(
                            p.slide.unwrap(),
                            Slide {
                                drop_n,
                                move_rows: cursor - drop_n,
                                carried_drop: drop_n.min(base)
                            }
                        );
                        assert_eq!(p.first_row, cursor - drop_n);
                        assert_eq!(p.skip, 0);
                        base = p.base;
                        origin = p.origin;
                    }
                }
            }
        }
    }

    // (d) Carried base + oversized chunk: carried rows get dropped first,
    // then the captured prefix (origin advances past them).
    #[test]
    fn carried_base_with_oversized_chunk() {
        let p = plan_capture(2147, 12000, 12000, 8193, 4096, 4096);
        assert_eq!(
            p.slide,
            Some(Slide {
                drop_n: 8292,
                move_rows: 0,
                carried_drop: 2147
            })
        );
        assert_eq!((p.base, p.origin), (0, 18145));
        assert_eq!(p.skip, 6145);
        assert_eq!(p.first_row, 0);
    }

    // (e) Exhaustive grid simulation: every write row in-bounds, slides
    // in-bounds and non-overlapping, rows are a contiguous prefix, positions
    // consecutive, and n == base + (chunk_end - origin) (the update formula).
    #[test]
    fn grid_simulation() {
        for win in [2048usize, 4096, 8192, 12288] {
            for chunk in [512usize, 1024, 2048, 4096, 8193] {
                for base0 in [0usize, 1000, win / 2] {
                    for replay in [100usize, 3000, 9000, 25000] {
                        simulate(win, chunk, base0, replay);
                    }
                }
            }
        }
    }

    fn simulate(win: usize, chunk: usize, base0: usize, replay: usize) {
        // rows[i] = absolute position captured in row i (None = empty)
        let mut rows: Vec<Option<usize>> = vec![None; win];
        // seed `base0` carried rows with positions < origin
        let mut origin = 10000usize;
        for i in 0..base0.min(win) {
            rows[i] = Some(origin - base0 + i);
        }
        let mut base = base0.min(win);
        let mut chunk_start = origin;
        let end = chunk_start + replay;
        while chunk_start < end {
            let proc = chunk.min(end - chunk_start);
            let p = plan_capture(base, origin, chunk_start, proc, win, win);
            if let Some(s) = p.slide {
                assert!(
                    s.move_rows == 0 || s.drop_n + s.move_rows <= win,
                    "slide src oob"
                );
                assert!(s.move_rows <= s.drop_n, "slide overlaps");
                if s.move_rows > 0 {
                    rows.copy_within(s.drop_n..s.drop_n + s.move_rows, 0);
                }
                // rows above move_rows are dead after the slide
                for r in &mut rows[s.move_rows..] {
                    *r = None;
                }
                base = p.base;
                origin = p.origin;
            }
            for t in p.skip..proc {
                let row = p.first_row + (t - p.skip);
                assert!(row < win, "write row {row} >= {win}");
                rows[row] = Some(chunk_start + t);
            }
            chunk_start += proc;
        }
        let chunk_end = end;
        // contiguous prefix
        let n = rows.iter().take_while(|r| r.is_some()).count();
        assert!(rows[n..].iter().all(|r| r.is_none()), "holes in rows");
        // positions strictly consecutive, last == chunk_end - 1
        let mut prev = None;
        for (i, r) in rows.iter().take(n).enumerate() {
            if let Some(pos) = r {
                if let Some(pv) = prev {
                    assert_eq!(*pos, pv + 1, "row {i} position gap");
                }
                prev = Some(*pos);
            }
        }
        if n > 0 {
            assert_eq!(rows[n - 1].unwrap(), chunk_end - 1);
        }
        // update formula: n == base + (chunk_end - origin) — the same
        // ctx_len `update_dflash_ctx_len_after_prefill` computes.
        assert_eq!(
            n,
            base + (chunk_end - origin),
            "chunk_end={chunk_end} base={base} origin={origin}"
        );
        // the populated window's first position is exactly origin - base
        if n > 0 {
            assert_eq!(rows[0].unwrap(), origin - base);
        }
    }
}
