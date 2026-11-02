// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded shared-expert prefill slabs; the M64 kernel stays within 1024 rows.
use anyhow::{Result, ensure};

pub(crate) fn validate_budget(rows: usize, limit: usize) -> Result<()> {
    ensure!(
        matches!(limit, 1024 | 2048 | 4096),
        "unknown shared FP8 prefill profile"
    );
    // The 4096 profile's 1024-row slabs cover any wider chunk too.
    ensure!(
        rows >= 1 && (limit == 4096 || rows <= limit),
        "shared FP8 cache prefill must be 1..{limit}"
    );
    Ok(())
}

#[derive(Debug, PartialEq)]
pub(crate) struct Slab {
    pub rows: u32,
    pub input: u64,
    pub output: u64,
}

pub(crate) fn plan(
    rows: u32,
    arena_rows: usize,
    n: u32,
    k: u32,
    input: u64,
    output: u64,
    budget: usize,
) -> Result<Vec<Slab>> {
    ensure!(
        matches!(budget, 2048 | 4096),
        "unknown shared FP8 slab profile"
    );
    // A configured chunk may include four aligned scheduling rows; the
    // 4096 profile admits any arena-bounded chunk.
    ensure!(
        rows > 0 && (budget == 4096 || rows as usize <= budget + 4) && rows as usize <= arena_rows,
        "shared FP8 slab capacity"
    );
    ensure!(
        matches!((n, k), (2048, 4096) | (4096, 2048)),
        "shared FP8 slab geometry"
    );
    ensure!(
        input > 0 && output > 0 && input.is_multiple_of(16) && output.is_multiple_of(16),
        "shared FP8 slab pointers"
    );
    for (ptr, width) in [(input, k), (output, n)] {
        ensure!(
            ptr.checked_add(u64::from(rows) * u64::from(width) * 2)
                .is_some(),
            "shared FP8 slab span overflow"
        );
    }
    Ok((0..rows)
        .step_by(1024)
        .map(|start| Slab {
            rows: (rows - start).min(1024),
            input: input + u64::from(start) * u64::from(k) * 2,
            output: output + u64::from(start) * u64::from(n) * 2,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_profile_admits_only_bounded_configured_budget() {
        for rows in [1, 1024] {
            assert!(validate_budget(rows, 1024).is_ok());
            assert!(validate_budget(rows, 2048).is_ok());
        }
        for rows in [1025, 2048] {
            assert!(validate_budget(rows, 1024).is_err());
            assert!(validate_budget(rows, 2048).is_ok());
        }
        for rows in [0, 2049, 2052, usize::MAX] {
            assert!(validate_budget(rows, 2048).is_err());
        }
    }

    #[test]
    fn actual_rows_cover_full_chunks_and_alignment_tail_without_gaps() {
        for (n, k) in [(2048, 4096), (4096, 2048)] {
            for rows in [1, 5, 1024, 1025, 2048, 2049, 2052] {
                let input = 0x1_0000_0000;
                let output = 0x2_0000_0000;
                let slabs = plan(rows, 2052, n, k, input, output, 2048).unwrap();
                let mut start = 0;
                for slab in &slabs {
                    assert_eq!(slab.rows, (rows - start).min(1024));
                    assert_eq!(slab.input, input + u64::from(start) * u64::from(k) * 2);
                    assert_eq!(slab.output, output + u64::from(start) * u64::from(n) * 2);
                    start += slab.rows;
                }
                assert_eq!(start, rows);
                assert_eq!(slabs.len(), rows.div_ceil(1024) as usize);
            }
        }
    }

    #[test]
    fn rejects_unknown_arena_geometry_or_pointer_span_before_dispatch() {
        for (rows, arena) in [
            (0, 2052),
            (2053, 4096),
            (2048, 2047),
            (u32::MAX, usize::MAX),
        ] {
            assert!(plan(rows, arena, 2048, 4096, 16, 32, 2048).is_err());
        }
        for (input, output) in [
            (0, 32),
            (16, 0),
            (17, 32),
            (16, 33),
            (u64::MAX - 15, 32),
            (16, u64::MAX - 15),
        ] {
            assert!(plan(2052, 2052, 2048, 4096, input, output, 2048).is_err());
            assert!(plan(4100, 4100, 2048, 4096, input, output, 4096).is_err());
        }
        assert!(plan(2048, 2052, 4096, 4096, 16, 32, 2048).is_err());
    }

    #[test]
    fn explicit_4096_profile_preserves_1024_slabs_and_checks_tail_and_arena() {
        assert!(validate_budget(4096, 4096).is_ok());
        assert!(validate_budget(8192, 4096).is_ok());
        assert!(validate_budget(4096, 2048).is_err());
        for (n, k) in [(2048, 4096), (4096, 2048)] {
            let slabs = plan(4100, 4100, n, k, 0x100000, 0x10000000, 4096).unwrap();
            assert_eq!(
                slabs.iter().map(|s| s.rows).collect::<Vec<_>>(),
                [1024, 1024, 1024, 1024, 4]
            );
            assert_eq!(slabs[4].input, 0x100000 + 4096 * u64::from(k) * 2);
            assert_eq!(slabs[4].output, 0x10000000 + 4096 * u64::from(n) * 2);
        }
        assert!(plan(4100, 4099, 2048, 4096, 16, 32, 4096).is_err());
        assert_eq!(plan(8196, 8196, 2048, 4096, 16, 32, 4096).unwrap().len(), 9);
        assert!(plan(2053, 4100, 2048, 4096, 16, 32, 2048).is_err());
    }
}
