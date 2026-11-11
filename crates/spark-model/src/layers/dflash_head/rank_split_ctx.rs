// SPDX-License-Identifier: AGPL-3.0-only

//! Rank-split context append (`ATLAS_GLM_DRAFT_TP_CTX`, default off): the
//! plan side.
//!
//! Before its block forward a propose appends the context rows the verify
//! accepted: `fc` over the rows' target hidden stack, `hidden_norm`, and one
//! fused K/V projection for every drafter layer (`precompute_ctx_kv`). Those
//! two projections are the drafter's largest reads after the MLP and the
//! head, and the head ran them alone while the worker waited for the first
//! layer's swap. With the switch each is split by output rows exactly as the
//! layer projections are (`rank_split`): the head sends the input rows, both
//! ranks run the unchanged tensor-core GEMV over a row range of the NVFP4
//! twin at the same row count, the `fc` halves are joined on both ranks
//! (each runs `hidden_norm` over the same bytes for its half of the K/V
//! rows) and the K/V halves on the head, which continues the append as
//! before. Every value is the one the unsplit launch writes.
//!
//! The worker must size these swaps before the head reaches them, so the
//! head announces each append's row count with the propose, in the preamble
//! word (`announce_word`): the row count it will append, computed from the
//! state the propose starts from. A propose whose append turns out
//! different (it never should) drains the announced swaps and appends
//! unsplit. Only appends of 1..=[`CTX_MAX_ROWS`] rows the NVFP4 tiers take
//! are split; a batched propose announces up to [`CTX_SLOTS`] sequences, the
//! first [`PREAMBLE_SLOTS`] in the preamble word and, for a propose of more
//! sequences, the rest in one more word after the command
//! ([`announce_extra`]).

use anyhow::{Result, bail, ensure};

use super::{Step, Swap};

/// Context appends a propose announces: an eight-sequence batch's.
pub(crate) const CTX_SLOTS: usize = 8;
/// The appends the preamble word carries; the rest ride one more word.
const PREAMBLE_SLOTS: usize = 4;
/// Bits a slot's row count takes in the preamble word.
const CTX_BITS: u32 = 5;
/// The largest split context append.
pub(crate) const CTX_MAX_ROWS: usize = (1 << CTX_BITS) - 1;
/// Bits a batched propose's rows take below the slots.
const ROWS_BITS: u32 = 12;
/// The largest batched propose the preamble can announce with the slots.
pub(crate) const ROWS_MAX: usize = (1 << ROWS_BITS) - 1;

/// The rows of a propose's context appends; zero = not split.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CtxRows(pub(crate) [u8; CTX_SLOTS]);

impl CtxRows {
    /// One append of `n` rows (at most [`CTX_MAX_ROWS`]).
    pub(crate) fn single(n: usize) -> Self {
        let mut rows = Self::default();
        rows.0[0] = n.min(CTX_MAX_ROWS) as u8;
        rows
    }

    /// Rows of append `i` (zero past the slots).
    pub(crate) fn get(&self, i: usize) -> usize {
        self.0.get(i).map_or(0, |&n| n as usize)
    }

    /// The appends that split, in order.
    pub(crate) fn appends(&self) -> impl Iterator<Item = usize> + '_ {
        (0..CTX_SLOTS).filter(|&i| self.get(i) > 0)
    }

    pub(crate) fn pack(&self) -> u64 {
        self.0
            .iter()
            .enumerate()
            .fold(0, |w, (i, &n)| w | (n as u64) << (i as u32 * CTX_BITS))
    }

    pub(crate) fn unpack(word: u64) -> Self {
        Self(std::array::from_fn(|i| {
            (word >> (i as u32 * CTX_BITS) & CTX_MAX_ROWS as u64) as u8
        }))
    }
}

/// Bits of the packed context rows the preamble word carries.
const PREAMBLE_BITS: u32 = PREAMBLE_SLOTS as u32 * CTX_BITS;

/// The preamble word of a split propose: a batched propose's rows (0 for a
/// single sequence) and, above them, the first `PREAMBLE_SLOTS` packed
/// context rows. Without context rows it is the rows alone, as before the
/// switch.
pub fn announce_word(rows: usize, ctx: u64) -> Result<u32> {
    ensure!(
        ctx == 0 || rows <= ROWS_MAX,
        "rank-split propose of {rows} rows cannot announce its context rows"
    );
    Ok(u32::try_from(rows)? | ((ctx & ((1 << PREAMBLE_BITS) - 1)) as u32) << ROWS_BITS)
}

/// Whether a split propose of `sequences` sequences sends a second
/// announce word after its command: with the context split, when it has
/// more appends than the preamble carries. Both ends decide it from what
/// they share (the head counts its sequences, a worker reads them off the
/// preamble's rows), so they agree.
pub fn extra_follows(sequences: usize, ctx_split: bool) -> bool {
    ctx_split && sequences > PREAMBLE_SLOTS
}

/// The second announce word: the packed context rows past the preamble's.
pub fn announce_extra(ctx: u64) -> u32 {
    (ctx >> PREAMBLE_BITS) as u32
}

/// The worker's reading of [`announce_word`] (and of [`announce_extra`]
/// when [`extra_follows`]): `(rows, context rows)`. Only ranks that agreed
/// on the context split read the slots.
pub(crate) fn read_announce(word: u32, extra: Option<u32>, ctx_split: bool) -> (usize, CtxRows) {
    if !ctx_split {
        return (word as usize, CtxRows::default());
    }
    let high = extra.map_or(0, |w| (w as u64) << PREAMBLE_BITS);
    (
        (word & ROWS_MAX as u32) as usize,
        CtxRows::unpack((word >> ROWS_BITS) as u64 | high),
    )
}

/// `ATLAS_GLM_DRAFT_TP_CTX`: `1` splits the context append too; unset,
/// empty or `0` = off; anything else is refused.
pub(crate) fn parse_ctx(value: Option<&str>) -> Result<bool> {
    match value.map(str::trim) {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("ATLAS_GLM_DRAFT_TP_CTX must be 0 or 1, got '{other}'"),
    }
}

/// `ATLAS_GLM_DRAFT_TP_CTX`, from the profile both ranks share.
pub fn ctx_requested() -> Result<bool> {
    parse_ctx(std::env::var("ATLAS_GLM_DRAFT_TP_CTX").ok().as_deref())
}

/// Context append `i`'s walk, the same on both ranks: the head's input rows
/// out, each rank's `fc` half, the joined rows normed on both, each rank's
/// K/V half, the K/V halves back to the head.
pub(super) fn ctx_steps(i: usize) -> [Step; 6] {
    use super::Piece::*;
    [
        Step::Swap(Swap::CtxInput(i)),
        Step::Run(CtxFc(i)),
        Step::Swap(Swap::CtxHidden(i)),
        Step::Run(CtxNorm(i)),
        Step::Run(CtxKv(i)),
        Step::Swap(Swap::CtxKv(i)),
    ]
}
