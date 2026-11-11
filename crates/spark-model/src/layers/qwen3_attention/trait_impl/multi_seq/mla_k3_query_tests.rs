// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn flags_are_strict_and_compare_requires_batchm() {
    assert!(!parse_flag(None).unwrap());
    assert!(!parse_flag(Some("0")).unwrap());
    assert!(parse_flag(Some("1")).unwrap());
    for value in ["", "2", "true", " 1"] {
        assert!(parse_flag(Some(value)).is_err());
    }
    assert!(!parse_compare(None, false).unwrap());
    assert!(parse_compare(Some("1"), true).unwrap());
    assert!(parse_compare(Some("1"), false).is_err());
    for value in ["", "2", "true", " 1"] {
        assert!(parse_compare(Some(value), true).is_err());
    }
}

const ROW_COUNTS: [usize; 3] = [2, 3, MAX_ROWS];

/// Arena bases far enough apart for MAX_ROWS rows of every partition.
const INPUT: u64 = 0x1000;
const LATENT: u64 = 0x2_0000;
const QUERY: u64 = 0x4_0000;
const DIAGNOSTIC: u64 = 0x10_0000;

fn plan(
    rows: usize,
    input_capacity: usize,
    latent_capacity: usize,
    q_capacity: usize,
    diagnostic: Option<(DevicePtr, usize)>,
    live: &[(DevicePtr, usize)],
) -> Result<QueryPlan> {
    QueryPlan::new(
        rows,
        DevicePtr(INPUT),
        input_capacity,
        DevicePtr(LATENT),
        latent_capacity,
        DevicePtr(QUERY),
        q_capacity,
        diagnostic,
        live,
    )
}

fn exact(rows: usize, live: &[(DevicePtr, usize)]) -> Result<QueryPlan> {
    plan(
        rows,
        input_bytes(rows),
        latent_bytes(rows),
        query_bytes(rows),
        None,
        live,
    )
}

fn with_diagnostic(
    rows: usize,
    diagnostic: (DevicePtr, usize),
    live: &[(DevicePtr, usize)],
) -> Result<QueryPlan> {
    plan(
        rows,
        input_bytes(rows),
        latent_bytes(rows),
        query_bytes(rows),
        Some(diagnostic),
        live,
    )
}

#[test]
fn partitions_scale_with_rows_and_keep_k3_offsets() {
    // The repaired K3 lane keeps its original byte partition.
    assert_eq!(latent_bytes(3), 9216);
    assert_eq!(index_offset(3), 49152);
    assert_eq!(query_bytes(3), 73728);
    assert_eq!(diagnostic_bytes(3), 73728);
    assert_eq!(input_bytes(3), 24576);
    for rows in ROW_COUNTS {
        assert_eq!(latent_bytes(rows), rows * LATENT_ROW);
        assert_eq!(index_offset(rows), rows * Q_ROW);
        assert_eq!(query_bytes(rows), rows * (Q_ROW + INDEX_ROW));
    }
    // The widest block still fits the test's arena spacing.
    assert!(LATENT + latent_bytes(MAX_ROWS) as u64 <= QUERY);
    assert!(QUERY + query_bytes(MAX_ROWS) as u64 <= DIAGNOSTIC);
}

#[test]
fn rows_use_contiguous_latent_q_and_padded_index_tail() {
    for rows in ROW_COUNTS {
        let plan = exact(rows, &[]).unwrap();
        let index = QUERY + index_offset(rows) as u64;
        for row in 0..rows {
            let actual = plan.row(row).unwrap();
            assert_eq!(actual.q_latent.0, LATENT + (row * LATENT_ROW) as u64);
            assert_eq!(actual.q_full.0, QUERY + (row * Q_ROW) as u64);
            assert_eq!(actual.index_query.0, index + (row * INDEX_ROW) as u64);
        }
        assert!(plan.row(rows).is_err());
        // The last index-Q row ends exactly at the staged capacity.
        let last = plan.row(rows - 1).unwrap().index_query.0 + INDEX_ROW as u64;
        assert_eq!(last, QUERY + query_bytes(rows) as u64);
    }
}

#[test]
fn plan_rejects_unsupported_row_counts() {
    for rows in [0, 1, MAX_ROWS + 1] {
        let bytes = MAX_ROWS + 1;
        assert!(
            plan(
                rows,
                input_bytes(bytes),
                latent_bytes(bytes),
                query_bytes(bytes),
                None,
                &[]
            )
            .is_err()
        );
    }
}

#[test]
fn plan_rejects_short_or_aliasing_ranges() {
    for rows in ROW_COUNTS {
        let (input, latent, query) = (input_bytes(rows), latent_bytes(rows), query_bytes(rows));
        assert!(exact(rows, &[]).is_ok());
        assert!(plan(rows, input - 1, latent, query, None, &[]).is_err());
        assert!(plan(rows, input, latent - 1, query, None, &[]).is_err());
        assert!(plan(rows, input, latent, query - 1, None, &[]).is_err());
        // Capacity for fewer rows cannot host a wider block.
        if rows > 2 {
            let fewer = rows - 1;
            assert!(plan(rows, input_bytes(fewer), latent, query, None, &[]).is_err());
            assert!(plan(rows, input, latent_bytes(fewer), query, None, &[]).is_err());
            assert!(plan(rows, input, latent, query_bytes(fewer), None, &[]).is_err());
        }
        let last_latent = LATENT + ((rows - 1) * LATENT_ROW) as u64;
        assert!(exact(rows, &[(DevicePtr(last_latent), 16)]).is_err());
        let first_index = QUERY + index_offset(rows) as u64;
        assert!(exact(rows, &[(DevicePtr(first_index), 16)]).is_err());
        let last_index = QUERY + (query_bytes(rows) - INDEX_ROW) as u64;
        assert!(exact(rows, &[(DevicePtr(last_index), 16)]).is_err());
        // Exactly adjacent live ranges do not overlap.
        let end = QUERY + query_bytes(rows) as u64;
        assert!(exact(rows, &[(DevicePtr(end), 16)]).is_ok());
    }
}

#[test]
fn diagnostic_scratch_uses_dead_qkvz_arena() {
    for rows in ROW_COUNTS {
        let bytes = diagnostic_bytes(rows);
        let plan = with_diagnostic(rows, (DevicePtr(DIAGNOSTIC), bytes), &[]).unwrap();
        assert_eq!(plan.diagnostic.unwrap().0, DIAGNOSTIC);
        assert!(with_diagnostic(rows, (DevicePtr(DIAGNOSTIC), bytes - 1), &[]).is_err());
        // The exact qkvz base is the intentional dead-scratch borrow; any
        // distinct live range overlapping the diagnostic span is rejected.
        assert!(
            with_diagnostic(
                rows,
                (DevicePtr(DIAGNOSTIC), bytes),
                &[(DevicePtr(DIAGNOSTIC + bytes as u64 - 16), 32)],
            )
            .is_err()
        );
    }
}

#[test]
fn diagnostic_scratch_does_not_hide_equal_base_live_alias() {
    for rows in ROW_COUNTS {
        assert!(
            with_diagnostic(
                rows,
                (DevicePtr(DIAGNOSTIC), diagnostic_bytes(rows)),
                // Same base, different logical span: this is an accidental
                // weight/live alias and must not be mistaken for qkvz owner.
                &[(DevicePtr(DIAGNOSTIC), 16)],
            )
            .is_err()
        );
    }
}

#[test]
fn diagnostic_scratch_rejects_duplicate_qkvz_owner_tuple() {
    for rows in ROW_COUNTS {
        let qkvz = (DevicePtr(DIAGNOSTIC), diagnostic_bytes(rows));
        assert!(with_diagnostic(rows, qkvz, &[qkvz]).is_ok());
        assert!(with_diagnostic(rows, qkvz, &[qkvz, qkvz]).is_err());
    }
}
