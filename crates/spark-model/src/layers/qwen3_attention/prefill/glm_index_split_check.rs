// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_INDEX_SPLIT_CHECK=1`: compare an owner's split selection with
//! the replicated one on both ranks, and fail the request on both on any
//! difference, naming what went wrong (`cause=`):
//!
//! * `launch`: a rank's own split passes differ from its own full pass, so a
//!   row's selection depends on the rows launched with it;
//! * `transport`: a quarter differs between the rank that selected it and the
//!   rank that received it, so the pair exchange changed or delayed rows;
//! * `inputs`: the ranks' replicated selections differ, so the queries,
//!   weights or pooled keys they select from are not the same on both ranks
//!   (the split is then exact per rank but not bit-identical to either).
//!
//! A clean check swaps one verdict; only a difference pays for the evidence.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::{IndexSplit, OwnerRows};
use crate::layer::ForwardContext;
use crate::layers::glm_sp;

/// The four quarters and the rows past them.
const REGIONS: [&str; 5] = ["Q0", "Q1", "Q2", "Q3", "tail"];

/// What one rank's compare of its split rows with its replicated rows found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Verdict {
    /// First differing row.
    pub(super) first: Option<usize>,
    /// Differing rows this rank selected itself.
    pub(super) own: usize,
    /// Differing rows it received from its peer.
    pub(super) received: usize,
}

impl Verdict {
    pub(super) const WORDS: usize = 3;

    fn of(differing: &[usize], split: &IndexSplit) -> Self {
        let received = differing.iter().filter(|&&r| split.receives(r)).count();
        Self {
            first: differing.first().copied(),
            own: differing.len() - received,
            received,
        }
    }

    pub(super) fn words(&self) -> [u64; Self::WORDS] {
        let first = self.first.map_or(0, |row| row as u64 + 1);
        [first, self.own as u64, self.received as u64]
    }

    pub(super) fn from_words(w: &[u64]) -> Self {
        Self {
            first: w[0].checked_sub(1).map(|row| row as usize),
            own: w[1] as usize,
            received: w[2] as usize,
        }
    }
}

/// One rank's fingerprints of an owner, swapped after a difference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Evidence {
    /// Each quarter of the split rows, as selected or as received.
    pub(super) split: [u64; 4],
    /// Each region of the replicated rows.
    pub(super) replicated: [u64; 5],
    /// The index queries and the head weights of every row.
    pub(super) inputs: [u64; 2],
}

impl Evidence {
    pub(super) const WORDS: usize = 11;

    /// `split` and `replicated` hold `rows` rows of `row_bytes`.
    pub(super) fn of(
        split: &[u8],
        replicated: &[u8],
        inputs: [&[u8]; 2],
        rows: usize,
        row_bytes: usize,
    ) -> Self {
        let q = rows / 4 * row_bytes;
        let region = |bytes: &[u8], k: usize| {
            let end = if k == 4 { bytes.len() } else { (k + 1) * q };
            fingerprint(&bytes[k * q..end])
        };
        Self {
            split: std::array::from_fn(|k| region(split, k)),
            replicated: std::array::from_fn(|k| region(replicated, k)),
            inputs: inputs.map(fingerprint),
        }
    }

    pub(super) fn words(&self) -> [u64; Self::WORDS] {
        let mut w = [0; Self::WORDS];
        w[..4].copy_from_slice(&self.split);
        w[4..9].copy_from_slice(&self.replicated);
        w[9..].copy_from_slice(&self.inputs);
        w
    }

    pub(super) fn from_words(w: &[u64]) -> Self {
        Self {
            split: w[..4].try_into().expect("four quarters"),
            replicated: w[4..9].try_into().expect("five regions"),
            inputs: w[9..11].try_into().expect("two inputs"),
        }
    }
}

/// A 64-bit fingerprint for comparing a buffer across the ranks.
pub(super) fn fingerprint(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mix = |h: u64, w: u64| (h.rotate_left(5) ^ w).wrapping_mul(K);
    let mut words = bytes.chunks_exact(8);
    let h = words
        .by_ref()
        .map(|w| u64::from_le_bytes(w.try_into().expect("8-byte words")))
        .fold(bytes.len() as u64, mix);
    words.remainder().iter().fold(h, |h, &b| mix(h, b as u64))
}

/// The rows of `a` and `b` (`row_bytes` each) that differ.
pub(super) fn differing_rows(a: &[u8], b: &[u8], row_bytes: usize) -> Vec<usize> {
    a.chunks(row_bytes)
        .zip(b.chunks(row_bytes))
        .enumerate()
        .filter_map(|(row, (x, y))| (x != y).then_some(row))
        .collect()
}

/// Where `a` and `b` (`rows` rows of `row_bytes`) first differ, as a region
/// and the byte offset in it: a stale transfer starts on a transport
/// boundary (a segment, a rail's share, a page), a selection does not.
pub(super) fn first_byte(a: &[u8], b: &[u8], rows: usize, row_bytes: usize) -> String {
    let Some(byte) = a.iter().zip(b).position(|(x, y)| x != y) else {
        return "none".to_owned();
    };
    let q = rows / 4 * row_bytes;
    let region = byte.checked_div(q).map_or(4, |k| k.min(4));
    format!("{}+{}", REGIONS[region], byte - region * q)
}

/// Ascending `rows` as at most `max` runs, e.g. `151..=255 300 (+2 runs)`.
pub(super) fn runs(rows: &[usize], max: usize) -> String {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for &row in rows {
        match out.last_mut() {
            Some(last) if last.1 + 1 == row => last.1 = row,
            _ => out.push((row, row)),
        }
    }
    let mut text: Vec<String> = out
        .iter()
        .take(max)
        .map(|&(a, b)| match a == b {
            true => a.to_string(),
            false => format!("{a}..={b}"),
        })
        .collect();
    if out.len() > max {
        text.push(format!("(+{} runs)", out.len() - max));
    }
    text.join(" ")
}

/// Why the pair's selections differ, from both ranks' verdicts and evidence,
/// then the regions whose exchanged and replicated rows differ across them.
pub(super) fn causes(
    here: Verdict,
    peer: Verdict,
    mine: &Evidence,
    theirs: &Evidence,
) -> (String, [Vec<&'static str>; 2]) {
    let unlike = |a: &[u64], b: &[u64]| -> Vec<&'static str> {
        let differs = a.iter().zip(b).zip(REGIONS).filter(|((x, y), _)| x != y);
        differs.map(|(_, name)| name).collect()
    };
    let exchanged = unlike(&mine.split, &theirs.split);
    let replicated = unlike(&mine.replicated, &theirs.replicated);
    let found = [
        ("launch", here.own + peer.own > 0),
        ("transport", !exchanged.is_empty()),
        ("inputs", !replicated.is_empty()),
    ];
    let names: Vec<_> = found.iter().filter(|c| c.1).map(|c| c.0).collect();
    let cause = match names.is_empty() {
        true => "unexplained".to_owned(),
        false => names.join("+"),
    };
    (cause, [exchanged, replicated])
}

impl IndexSplit {
    /// Whether this rank received `row` from its peer.
    fn receives(&self, row: usize) -> bool {
        let quarter = |s: &super::Swap| (s.peer..s.peer + s.rows).contains(&row);
        self.swaps.iter().any(quarter)
    }

    /// Compare every row of `rows.selected` with the replicated rows in
    /// `rows.scratch`, swap the verdicts through `scratch`, and fail on both
    /// ranks if either saw a difference, so the pair stops at the same
    /// collective instead of one rank running on.
    pub(super) fn check(
        &self,
        rows: &OwnerRows,
        layer: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (n, row_bytes) = (self.rows, rows.row_bytes);
        let read = |ptr: DevicePtr, bytes: usize| -> Result<Vec<u8>> {
            let mut host = vec![0u8; bytes];
            ctx.gpu.copy_d2h_on_stream(ptr, &mut host, stream)?;
            Ok(host)
        };
        let swap = |words: &[u64]| -> Result<Vec<u64>> {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let theirs = rows.scratch.offset(bytes.len());
            ctx.gpu.copy_h2d_async(&bytes, rows.scratch, stream)?;
            glm_sp::exchange_rows(rows.scratch, theirs, 1, bytes.len(), false, ctx, stream)?;
            let words = read(theirs, bytes.len())?;
            let word = |w: &[u8]| u64::from_le_bytes(w.try_into().expect("8-byte words"));
            Ok(words.chunks(8).map(word).collect())
        };
        let split = read(rows.selected, n * row_bytes)?;
        let replicated = read(rows.scratch, n * row_bytes)?;
        let differing = differing_rows(&split, &replicated, row_bytes);
        let here = Verdict::of(&differing, self);
        let peer = Verdict::from_words(&swap(&here.words())?);
        if here.first.is_none() && peer.first.is_none() {
            tracing::info!("ATLAS_GLM_INDEX_SPLIT_CHECK ok layer={layer} rows={n}");
            return Ok(());
        }
        // A landing that only finished late reads clean the second time.
        let again = read(rows.selected, n * row_bytes)?;
        let reread = differing_rows(&again, &replicated, row_bytes).len();
        let [query, weights] = rows.inputs;
        let inputs = [read(query.0, query.1)?, read(weights.0, weights.1)?];
        let mine = Evidence::of(&split, &replicated, [&inputs[0], &inputs[1]], n, row_bytes);
        let theirs = Evidence::from_words(&swap(&mine.words())?);
        let (cause, [exchanged, selections]) = causes(here, peer, &mine, &theirs);
        let same = |k: usize| match mine.inputs[k] == theirs.inputs[k] {
            true => "same",
            false => "differ",
        };
        bail!(
            "ATLAS_GLM_INDEX_SPLIT_CHECK: layer {layer}: the split selection of {n} rows differs from the replicated selection (first differing row: here {:?}, peer {:?}); cause={cause}; here {} own + {} received rows differ [{}] from byte {}, {reread} after a re-read; peer {} own + {} received; across the pair: exchanged rows differ in {exchanged:?}, replicated rows in {selections:?}, queries {}, weights {}",
            here.first,
            peer.first,
            here.own,
            here.received,
            runs(&differing, 8),
            first_byte(&split, &replicated, n, row_bytes),
            peer.own,
            peer.received,
            same(0),
            same(1),
        );
    }
}

#[cfg(test)]
#[path = "glm_index_split_check_tests.rs"]
mod tests;
