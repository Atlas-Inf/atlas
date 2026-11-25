// SPDX-License-Identifier: AGPL-3.0-only

//! A launch's tiles in spans: the runs of tiles a batch's routed experts go
//! in, each span's hidden activations gathered beside the next span's
//! launches (`block_moe_big.rs`).

use std::ops::Range;

use super::Runs;

/// Eighths of a batch's pairs in the first of the two spans the ranks
/// gather their hidden activations in (`block_moe_big.rs`), unless
/// `ATLAS_DSV41_MOE_SPAN` names another; 0 for one span, gathered after
/// every pair is in. The first span is the larger: its gather has the
/// second's launches to hide under, and the second's only the first's way
/// out.
const FIRST_SPAN: usize = 5;

pub(super) fn first_span() -> usize {
    std::env::var("ATLAS_DSV41_MOE_SPAN")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&eighths| eighths < 8)
        .unwrap_or(FIRST_SPAN)
}

/// A span starts on a pair whose scales start on a 16-byte boundary of a
/// part (`Sent`), which the ranks' copies want: a pair's scales are a
/// multiple of two bytes.
const SPAN_PAIRS: usize = 8;

/// A run of a launch's tiles, and the pairs in them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub tiles: Range<usize>,
    pub pairs: Range<usize>,
}

impl Runs {
    /// Every tile, and every pair.
    pub fn whole(&self) -> Span {
        Span {
            tiles: 0..self.tile_count(),
            pairs: 0..self.pairs,
        }
    }

    /// The tiles in two spans, the first of about `eighths` eighths of the
    /// pairs: cut at the tile nearest that which starts on a pair that is
    /// a multiple of [`SPAN_PAIRS`]. None where no tile inside does.
    pub fn halves(&self, eighths: usize) -> Option<[Span; 2]> {
        let want = self.pairs * eighths / 8;
        let (tile, pair) = (1..self.tile_count())
            .map(|tile| (tile, self.tiles[tile * 3 + 1] as usize))
            .filter(|&(_, pair)| pair.is_multiple_of(SPAN_PAIRS) && pair < self.pairs)
            .min_by_key(|&(_, pair)| pair.abs_diff(want))?;
        Some([
            Span {
                tiles: 0..tile,
                pairs: 0..pair,
            },
            Span {
                tiles: tile..self.tile_count(),
                pairs: pair..self.pairs,
            },
        ])
    }
}
