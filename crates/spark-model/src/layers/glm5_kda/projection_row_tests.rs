// SPDX-License-Identifier: AGPL-3.0-only

use super::verify_row_chunks;

#[test]
fn verify_row_chunks_cover_rows_within_the_kernel_cap() {
    for m in 9..=96u32 {
        let chunks: Vec<_> = verify_row_chunks(m).collect();
        let mut next = 0;
        for &(start, rows) in &chunks {
            assert_eq!(start, next, "m={m} chunks must be contiguous");
            assert!((2..=8).contains(&rows), "m={m} chunk of {rows} rows");
            next += rows;
        }
        assert_eq!(next, m, "m={m} chunks must cover every row");
    }
    assert_eq!(verify_row_chunks(12).collect::<Vec<_>>(), [(0, 6), (6, 6)]);
}
