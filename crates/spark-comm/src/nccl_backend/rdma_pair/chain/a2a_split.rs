// SPDX-License-Identifier: AGPL-3.0-only

//! How the four-rank all-to-all's planes go in one-shot ops.
//!
//! An op carries a run of every plane's rows: its four segments within the
//! one-shot maximum, and a segment of whole 128-byte units. Equal shares of
//! the rows serve where the rows divide well. Where they do not (3,902
//! rows are 2 x 1,951, and nothing between 2 and 1,951 divides them) equal
//! shares mean an op for every two rows, and a 40 MB plane went in 1,951
//! ops where six carry it. Planes of the same row count are then cut in
//! runs as long as fit, the last one shorter.

use super::MAX_RANKS;

/// The rows of each plane one op takes: `(first row, rows)` a plane.
pub(super) type Op = [(usize, usize); 2];

/// Equal shares are kept while they take no more than this many times the
/// fewest ops the bytes need.
const EQUAL_WITHIN: usize = 2;

/// The ops of an all-to-all of `planes` (`(rows, row_bytes)` each, one or
/// two) under a one-shot maximum of `max` bytes, or none where no split
/// into whole rows and 128-byte segments serves.
pub(super) fn ops(planes: &[(usize, usize)], max: usize) -> Option<Vec<Op>> {
    if planes.is_empty() || planes.len() > 2 || max == 0 {
        return None;
    }
    let seg = |n: usize| -> usize { planes.iter().map(|&(rows, bytes)| rows / n * bytes).sum() };
    let fits = |n: usize| {
        planes.iter().all(|&(rows, _)| rows.is_multiple_of(n))
            && seg(n).is_multiple_of(128)
            && MAX_RANKS * seg(n) <= max
    };
    let most = planes.iter().map(|&(rows, _)| rows).max().unwrap_or(0);
    let first = (MAX_RANKS * seg(1)).div_ceil(max).max(1);
    let equal = (first..=most).find(|&n| fits(n));
    let shares = |n: usize| -> Vec<Op> {
        (0..n)
            .map(|i| {
                let mut op = Op::default();
                for (slot, &(rows, _)) in op.iter_mut().zip(planes) {
                    *slot = (i * (rows / n), rows / n);
                }
                op
            })
            .collect()
    };
    if let Some(n) = equal.filter(|&n| n <= EQUAL_WITHIN * first) {
        return Some(shares(n));
    }
    // Runs of the same rows of every plane: as many as fit, in whole units
    // of rows whose bytes are whole 128.
    let rows = planes[0].0;
    let row: usize = planes.iter().map(|&(_, bytes)| bytes).sum();
    if rows > 0 && row > 0 && planes.iter().all(|&(r, _)| r == rows) {
        let unit = (1..=128).find(|u| (u * row).is_multiple_of(128))?;
        let run = max / (MAX_RANKS * row) / unit * unit;
        if run > 0 && rows.is_multiple_of(unit) {
            return Some(
                (0..rows)
                    .step_by(run)
                    .map(|at| {
                        let mut op = Op::default();
                        for slot in op.iter_mut().take(planes.len()) {
                            *slot = (at, run.min(rows - at));
                        }
                        op
                    })
                    .collect(),
            );
        }
    }
    equal.map(shares)
}

#[cfg(test)]
mod tests {
    use super::ops;

    /// The chain's one-shot maximum as the Sparks run it.
    const MAX: usize = 8_388_544;

    #[test]
    fn rows_that_divide_go_in_equal_shares() {
        // 2,048 rows of 2,560 bytes: 21 MB over four ranks, three ops' worth,
        // and four divide the rows.
        let got = ops(&[(2048, 2560)], MAX).expect("ops");
        assert_eq!(got.len(), 4);
        assert!(
            got.iter()
                .enumerate()
                .all(|(i, op)| op[0] == (i * 512, 512))
        );
        // Two planes of different row counts keep their shares.
        let got = ops(&[(4096, 64), (32, 512)], MAX).expect("ops");
        assert_eq!(got, vec![[(0, 4096), (0, 32)]]);
    }

    #[test]
    fn rows_that_do_not_divide_go_in_runs() {
        // 3,902 = 2 x 1,951: equal shares would be 1,951 ops of two rows.
        let got = ops(&[(3902, 2560)], MAX).expect("ops");
        assert_eq!(got.len(), 5);
        assert_eq!(got[0][0], (0, 819));
        assert_eq!(got[4][0], (3276, 626));
        let (mut next, mut most) = (0, 0);
        for op in &got {
            assert_eq!(op[0].0, next);
            next += op[0].1;
            most = most.max(4 * op[0].1 * 2560);
        }
        assert!(next == 3902 && most <= MAX);
        // Codes and their scales, 1,056 bytes a row: a run is whole fours.
        let got = ops(&[(1852, 1024), (1852, 32)], MAX).expect("ops");
        assert!(got.len() <= 2 && got.iter().all(|op| op[0] == op[1] && op[0].1 % 4 == 0));
    }

    #[test]
    fn no_ops_where_a_segment_cannot_be_whole() {
        // 3,902 rows of 1,056 bytes: no run of them is whole 128 bytes but
        // in fours, and the rows are not.
        assert_eq!(ops(&[(3902, 1024), (3902, 32)], MAX), None);
        assert_eq!(ops(&[], MAX), None);
        assert_eq!(ops(&[(8, 128)], 0), None);
    }
}
