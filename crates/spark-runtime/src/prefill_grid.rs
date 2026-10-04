// SPDX-License-Identifier: AGPL-3.0-only

//! Position-fixed prefill pass layout (R11 prefix-reuse fix 1).
//!
//! With prefix caching on, a warm turn used to restore SSM state at a
//! snapshot and replay from there, while a cold run of the same prompt cut
//! its passes somewhere else (one pass with caching off, a tail split one
//! block below the prompt end with caching on). BF16/FP32 sums are not
//! associative, so the two layouts give slightly different hidden states
//! and flip near-tied argmaxes at temperature 0 (measured: a tool roundtrip
//! answered wrongly only with caching on).
//!
//! The grid makes the layout a function of absolute token position only:
//! every prefill pass ends at a multiple of `grid` (or at the prompt end),
//! with caching on or off. SSM snapshots are saved only at grid points and
//! only KV that was written by a full grid pass is inserted into the prefix
//! cache, so a warm replay from grid point `g` runs exactly the passes a
//! cold run runs from `g`.
//!
//! `ATLAS_PREFILL_GRID=0` restores the old layout (tail split, leaf /
//! decode / finish-leaf snapshots). Default 2048 tokens. The grid must be a
//! multiple of 64 (GDN sub-chunk, KV block) and should divide
//! `--max-prefill-tokens` so scheduler chunk ends land on grid points.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

static SUBBLOCK_OFF: AtomicBool = AtomicBool::new(false);

/// Turn radix sub-block matches off for this process (set by a model that
/// runs the grid; default stays on so non-grid models are unchanged).
pub fn disable_subblock() {
    SUBBLOCK_OFF.store(true, Ordering::Relaxed);
}

/// Whether a grid model turned sub-block matching off.
pub fn subblock_disabled() -> bool {
    SUBBLOCK_OFF.load(Ordering::Relaxed)
}

/// Default grid in tokens.
pub const DEFAULT_PREFILL_GRID: usize = 2048;

/// The active grid (tokens), or 0 when the grid is off. Read once.
pub fn prefill_grid() -> usize {
    static G: OnceLock<usize> = OnceLock::new();
    *G.get_or_init(|| {
        parse_grid(std::env::var("ATLAS_PREFILL_GRID").ok().as_deref())
    })
}

/// Parse the env value: unset -> default, "0" -> off, otherwise a positive
/// multiple of 64 (anything else falls back to the default).
pub fn parse_grid(v: Option<&str>) -> usize {
    match v {
        None => DEFAULT_PREFILL_GRID,
        Some(s) => match s.trim().parse::<usize>() {
            Ok(0) => 0,
            Ok(g) if g % 64 == 0 => g,
            _ => DEFAULT_PREFILL_GRID,
        },
    }
}

/// Largest multiple of `grid` that is `<= n` (0 when `grid == 0`).
pub fn grid_floor(n: usize, grid: usize) -> usize {
    if grid == 0 { 0 } else { (n / grid) * grid }
}

/// First grid point strictly inside `(start, end)`, if any.
pub fn grid_cut(start: usize, end: usize, grid: usize) -> Option<usize> {
    if grid == 0 || end <= start {
        return None;
    }
    let c = (start / grid + 1) * grid;
    (c < end).then_some(c)
}

/// The passes `[s, e)` that cover `[start, end)` under the grid.
pub fn pass_layout(start: usize, end: usize, grid: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut s = start;
    while s < end {
        let e = grid_cut(s, end, grid).unwrap_or(end);
        out.push((s, e));
        s = e;
    }
    out
}

/// Whether a pass ending at `end_token` of a `total`-token prompt should
/// save an SSM checkpoint: a grid point, strictly below the prompt end, and
/// one of the last two grid points below it (the anchors a next turn needs).
pub fn is_grid_checkpoint(end_token: usize, total: usize, grid: usize) -> bool {
    if grid == 0 || end_token == 0 || end_token >= total || end_token % grid != 0 {
        return false;
    }
    let last = grid_floor(total - 1, grid);
    end_token == last || end_token + grid == last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        assert_eq!(parse_grid(None), DEFAULT_PREFILL_GRID);
        assert_eq!(parse_grid(Some("0")), 0);
        assert_eq!(parse_grid(Some("1024")), 1024);
        assert_eq!(parse_grid(Some("100")), DEFAULT_PREFILL_GRID);
        assert_eq!(parse_grid(Some("x")), DEFAULT_PREFILL_GRID);
    }

    #[test]
    fn cut_and_floor() {
        assert_eq!(grid_cut(0, 1081, 1024), Some(1024));
        assert_eq!(grid_cut(1024, 1081, 1024), None);
        assert_eq!(grid_cut(0, 1024, 1024), None);
        assert_eq!(grid_cut(5, 10, 0), None);
        assert_eq!(grid_floor(2047, 1024), 1024);
        assert_eq!(grid_floor(5, 0), 0);
    }

    /// Cold layout from 0 restricted to `[g, n)` equals the warm layout from
    /// any grid point g, for many prompt lengths and grids. Also holds when
    /// the cold run is first cut into scheduler chunks that are multiples of
    /// the grid.
    #[test]
    fn warm_layout_equals_cold_suffix() {
        for &grid in &[64usize, 1024, 2048, 4096] {
            for n in [1usize, 63, 64, 65, 1081, 1146, 2048, 2049, 8192, 8193, 30001] {
                let cold = pass_layout(0, n, grid);
                // Scheduler chunks of 8192 (a multiple of every grid here).
                let mut chunked = Vec::new();
                let mut s = 0;
                while s < n {
                    let e = (s + 8192).min(n);
                    chunked.extend(pass_layout(s, e, grid));
                    s = e;
                }
                assert_eq!(cold, chunked, "grid {grid} n {n}");
                let mut g = grid;
                while g < n {
                    let warm = pass_layout(g, n, grid);
                    let suffix: Vec<_> = cold.iter().copied().filter(|p| p.0 >= g).collect();
                    assert_eq!(warm, suffix, "grid {grid} n {n} g {g}");
                    g += grid;
                }
                for p in &cold {
                    assert!(p.1 == n || p.1 % grid == 0);
                    assert!(p.1 - p.0 <= grid);
                }
            }
        }
    }

    #[test]
    fn checkpoints_are_grid_points() {
        for &grid in &[1024usize, 2048] {
            for n in 1..(5 * grid) {
                for end in (0..=n).step_by(16) {
                    if is_grid_checkpoint(end, n, grid) {
                        assert_eq!(end % grid, 0);
                        assert!(end < n && end > 0);
                        assert!(end + 2 * grid > n - 1);
                    }
                }
            }
        }
        assert!(is_grid_checkpoint(1024, 1081, 1024));
        assert!(!is_grid_checkpoint(1024, 1024, 1024));
        assert!(is_grid_checkpoint(2048, 4000, 1024));
        assert!(is_grid_checkpoint(3072, 4000, 1024));
        assert!(!is_grid_checkpoint(1024, 4000, 1024));
        assert!(!is_grid_checkpoint(1024, 4000, 0));
    }
}
