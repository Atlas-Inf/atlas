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
//! The cut set is `{k*G} ∪ {k*F : k*F < G}`: a coarse grid `G` plus an
//! optional fine grid `F` below the first coarse point, so short prompts
//! (agent tool preambles) still get reuse anchors. Any FIXED cut set works:
//! warm from cut `a` runs the passes between consecutive cuts above `a`,
//! exactly as a cold run does.
//!
//! Off by default: the grid makes a warm turn re-prefill up to G tokens past
//! its last cut (main resumes at the end-of-turn snapshot), which costs ~50%
//! wall time on long agentic replays at G = 4096. `--exact-prefix-cache`
//! turns it on at G = 4096; `ATLAS_PREFILL_GRID` = G sets it explicitly
//! (overrides the flag; 0 = off, i.e. main's layout: tail split, leaf /
//! decode / finish-leaf snapshots). `ATLAS_PREFILL_GRID_FINE`
//! = F (default 0 = none). Both must be multiples of 64 (GDN sub-chunk, KV
//! block), F must divide G, and G should divide `--max-prefill-tokens` so
//! scheduler chunk ends land on cut points.

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

static EXACT: AtomicBool = AtomicBool::new(false);
static CUTS: OnceLock<Cuts> = OnceLock::new();

/// `--exact-prefix-cache`: turn the grid on at the default G when
/// `ATLAS_PREFILL_GRID` is unset. Call before the model is built; the cut
/// set is read once.
pub fn set_exact_prefix_cache(on: bool) {
    if CUTS.get().is_some() {
        tracing::warn!("--exact-prefix-cache set after the prefill grid was read; ignored");
    }
    EXACT.store(on, Ordering::Relaxed);
}

/// Coarse grid used by `--exact-prefix-cache`, in tokens.
pub const DEFAULT_PREFILL_GRID: usize = 4096;

/// A fixed cut set: multiples of `g`, plus multiples of `f` below `g`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cuts {
    pub g: usize,
    pub f: usize,
}

impl Cuts {
    pub fn new(g: usize, f: usize) -> Self {
        // A fine grid that does not divide the coarse one is ignored.
        let f = if g > 0 && f > 0 && f < g && g.is_multiple_of(f) {
            f
        } else {
            0
        };
        Cuts { g, f }
    }
    pub fn is_on(&self) -> bool {
        self.g > 0
    }
    pub fn is_cut(&self, p: usize) -> bool {
        self.g > 0
            && p > 0
            && (p.is_multiple_of(self.g) || (self.f > 0 && p < self.g && p.is_multiple_of(self.f)))
    }
    /// Smallest cut strictly greater than `p`.
    pub fn next_after(&self, p: usize) -> usize {
        if self.f > 0 && p < self.g {
            ((p / self.f + 1) * self.f).min(self.g)
        } else {
            (p / self.g + 1) * self.g
        }
    }
    /// Largest cut `<= n` (0 when none or off).
    pub fn floor(&self, n: usize) -> usize {
        if self.g == 0 {
            return 0;
        }
        if n >= self.g {
            return (n / self.g) * self.g;
        }
        if self.f > 0 { (n / self.f) * self.f } else { 0 }
    }
    /// First cut strictly inside `(start, end)`.
    pub fn cut(&self, start: usize, end: usize) -> Option<usize> {
        if self.g == 0 || end <= start {
            return None;
        }
        let c = self.next_after(start);
        (c < end).then_some(c)
    }
    /// Passes `[s, e)` covering `[start, end)`.
    pub fn layout(&self, start: usize, end: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut s = start;
        while s < end {
            let e = self.cut(s, end).unwrap_or(end);
            out.push((s, e));
            s = e;
        }
        out
    }
    /// R11 fix 5: end a non-last scheduler chunk `[offset, offset+len)` of a
    /// `total`-token prompt on the deepest cut inside it; keep `len` when no
    /// cut lies above `offset` (e.g. `--max-prefill-tokens` below the grid).
    /// Depends only on `(offset, len, total)`, so a warm run, which chunks
    /// from 0 like a cold one, gets the same chunk ends.
    pub fn align_chunk(&self, offset: usize, len: usize, total: usize) -> usize {
        if self.g == 0 || offset + len >= total {
            return len;
        }
        let end = self.floor(offset + len);
        if end > offset { end - offset } else { len }
    }
    /// Save an SSM checkpoint at a pass ending at `end` of a `total`-token
    /// prompt: one of the last two cuts strictly below the prompt end.
    pub fn is_checkpoint(&self, end: usize, total: usize) -> bool {
        if !self.is_cut(end) || end >= total {
            return false;
        }
        let last = self.floor(total - 1);
        end == last || (last > 0 && end == self.floor(last - 1))
    }
}

fn parse_size(v: Option<&str>, default: usize) -> usize {
    match v {
        None => default,
        Some(s) => match s.trim().parse::<usize>() {
            Ok(0) => 0,
            Ok(g) if g % 64 == 0 => g,
            _ => default,
        },
    }
}

/// Grid size from `ATLAS_PREFILL_GRID` and `--exact-prefix-cache`: an
/// explicit value wins ("0" = off, invalid = the flag's choice); unset ->
/// `DEFAULT_PREFILL_GRID` with the flag, off without it.
pub fn parse_grid(v: Option<&str>, exact: bool) -> usize {
    let default = if exact { DEFAULT_PREFILL_GRID } else { 0 };
    parse_size(v, default)
}

/// The active cut set (read once from the environment).
pub fn cuts() -> Cuts {
    *CUTS.get_or_init(|| {
        Cuts::new(
            parse_grid(
                std::env::var("ATLAS_PREFILL_GRID").ok().as_deref(),
                EXACT.load(Ordering::Relaxed),
            ),
            parse_size(std::env::var("ATLAS_PREFILL_GRID_FINE").ok().as_deref(), 0),
        )
    })
}

/// The active coarse grid (tokens), or 0 when the grid is off.
pub fn prefill_grid() -> usize {
    cuts().g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        // Off by default; --exact-prefix-cache turns on the default grid.
        assert_eq!(parse_grid(None, false), 0);
        assert_eq!(parse_grid(None, true), DEFAULT_PREFILL_GRID);
        // An explicit ATLAS_PREFILL_GRID wins either way.
        assert_eq!(parse_grid(Some("0"), true), 0);
        assert_eq!(parse_grid(Some("1024"), false), 1024);
        assert_eq!(parse_grid(Some("1024"), true), 1024);
        // Invalid values fall back to the flag's choice.
        assert_eq!(parse_grid(Some("100"), false), 0);
        assert_eq!(parse_grid(Some("100"), true), DEFAULT_PREFILL_GRID);
        assert_eq!(Cuts::new(4096, 1000).f, 0);
        assert_eq!(Cuts::new(4096, 1024).f, 1024);
    }

    #[test]
    fn cut_and_floor() {
        let c = Cuts::new(1024, 0);
        assert_eq!(c.cut(0, 1081), Some(1024));
        assert_eq!(c.cut(1024, 1081), None);
        assert_eq!(c.floor(2047), 1024);
        assert_eq!(c.floor(1000), 0);
        let c = Cuts::new(4096, 1024);
        assert_eq!(c.cut(0, 1081), Some(1024));
        assert_eq!(c.cut(3072, 5000), Some(4096));
        assert_eq!(c.cut(4096, 9000), Some(8192));
        assert_eq!(c.floor(1081), 1024);
        assert_eq!(c.floor(4095), 3072);
        assert_eq!(c.floor(5000), 4096);
        assert_eq!(Cuts::new(0, 0).cut(5, 10), None);
    }

    /// Proof obligation as a test: for every cut set, the cold layout from 0
    /// restricted to `[a, n)` equals the warm layout from any cut `a < n`,
    /// also when the cold run is first split into 8192-token scheduler chunks.
    /// Every pass ends at a cut or at `n`.
    #[test]
    fn warm_layout_equals_cold_suffix() {
        for c in [
            Cuts::new(64, 0),
            Cuts::new(1024, 0),
            Cuts::new(2048, 0),
            Cuts::new(4096, 0),
            Cuts::new(4096, 512),
            Cuts::new(4096, 1024),
            Cuts::new(8192, 1024),
        ] {
            for n in [
                1usize, 63, 64, 65, 1081, 1146, 2048, 2049, 4095, 4097, 8192, 8193, 30001, 100_003,
            ] {
                let cold = c.layout(0, n);
                let mut chunked = Vec::new();
                let mut s = 0;
                while s < n {
                    let e = (s + 8192).min(n);
                    chunked.extend(c.layout(s, e));
                    s = e;
                }
                assert_eq!(cold, chunked, "{c:?} n {n}");
                let mut a = c.next_after(0);
                while a < n {
                    let warm = c.layout(a, n);
                    let suffix: Vec<_> = cold.iter().copied().filter(|p| p.0 >= a).collect();
                    assert_eq!(warm, suffix, "{c:?} n {n} a {a}");
                    a = c.next_after(a);
                }
                for p in &cold {
                    assert!(p.1 == n || c.is_cut(p.1), "{c:?} n {n} pass {p:?}");
                }
            }
        }
    }

    /// Scheduler budget below the grid (gfx1151: `--max-prefill-tokens 2048`,
    /// G = 4096). Chunk ends alternate on/off grid, but `align_chunk` depends
    /// only on `(offset, len, total)` and a warm run chunks from 0 like a cold
    /// one, so both see the same chunk ends and (fix 4) the same normalize
    /// points at or above the anchor. What must hold is that every cut is a
    /// pass boundary of the cold run, so a snapshot saved at a cut is the
    /// cold state there, and that every pass ends at a cut, a chunk end, or n.
    #[test]
    fn chunk_ends_below_grid_match_warm_and_cold() {
        for c in [
            Cuts::new(4096, 0),
            Cuts::new(4096, 1024),
            Cuts::new(2048, 0),
        ] {
            for budget in [1024usize, 1536, 2048, 3072, 4096, 8192] {
                for n in [2049usize, 4097, 6144, 6300, 12_289, 30_001] {
                    let mut ends = Vec::new();
                    let mut s = 0;
                    while s < n {
                        let e = s + c.align_chunk(s, budget.min(n - s), n);
                        assert!(e > s && e <= n, "{c:?} b {budget} n {n}");
                        ends.push(e);
                        s = e;
                    }
                    let mut passes = Vec::new();
                    let mut s = 0;
                    for &e in &ends {
                        passes.extend(c.layout(s, e));
                        s = e;
                    }
                    for p in &passes {
                        assert!(p.1 == n || c.is_cut(p.1) || ends.contains(&p.1));
                    }
                    let mut a = c.next_after(0);
                    while a < n {
                        assert!(
                            passes.iter().any(|p| p.1 == a),
                            "{c:?} b {budget} n {n}: cut {a} is not a pass boundary"
                        );
                        a = c.next_after(a);
                    }
                }
            }
        }
    }

    /// Checkpoints are cuts, strictly below the prompt end, and the anchor
    /// the next turn needs (the deepest cut below the end) is always one.
    #[test]
    fn checkpoints_are_cuts() {
        for c in [
            Cuts::new(1024, 0),
            Cuts::new(4096, 1024),
            Cuts::new(2048, 512),
        ] {
            for n in 2..(3 * c.g) {
                let mut count = 0;
                for end in 1..n {
                    if c.is_checkpoint(end, n) {
                        assert!(c.is_cut(end) && end < n);
                        count += 1;
                    }
                }
                assert!(count <= 2);
                let last = c.floor(n - 1);
                if last > 0 {
                    assert!(c.is_checkpoint(last, n), "{c:?} n {n}");
                }
            }
        }
        assert!(!Cuts::new(0, 0).is_checkpoint(1024, 4000));
    }
}
