// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail, ensure};

/// `(first row, rows)` of each of `seqs` sequences in `row_owner`, which must
/// list every sequence's rows together and in order.
pub(super) fn owner_runs(row_owner: &[usize], seqs: usize) -> Result<Vec<(usize, usize)>> {
    let mut runs: Vec<(usize, usize)> = Vec::with_capacity(seqs);
    for (r, &o) in row_owner.iter().enumerate() {
        let seen = runs.len();
        if seen > 0 && o + 1 == seen {
            runs[seen - 1].1 += 1;
        } else if o == seen {
            runs.push((r, 1));
        } else {
            bail!("inkling2: verify row {r} of sequence {o} out of order");
        }
    }
    ensure!(
        runs.len() == seqs && seqs > 0,
        "inkling2: verify rows of {} sequences for {seqs} states",
        runs.len()
    );
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::owner_runs;

    #[test]
    fn rows_group_by_sequence_in_order() {
        assert_eq!(owner_runs(&[0, 0, 0], 1).unwrap(), vec![(0, 3)]);
        assert_eq!(
            owner_runs(&[0, 0, 1, 1, 1], 2).unwrap(),
            vec![(0, 2), (2, 3)]
        );
    }

    #[test]
    fn out_of_order_or_missing_rows_fail() {
        assert!(owner_runs(&[0, 1, 0], 2).is_err());
        assert!(owner_runs(&[1, 1], 2).is_err());
        assert!(owner_runs(&[0, 0], 2).is_err());
        assert!(owner_runs(&[], 1).is_err());
    }
}
