// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn geometry(rows_per_cta: u32, pools_per_cta: u32) -> Result<IndexLogitsLaunch> {
    index_logits_launch(
        256,
        9,
        4096,
        33,
        32,
        128,
        4,
        32,
        rows_per_cta,
        pools_per_cta,
    )
}

#[test]
fn scorer_tiles_cover_partial_rows_and_pools() {
    let scalar = geometry(1, 8).unwrap();
    let row8 = geometry(8, 8).unwrap();
    let wmma = geometry(8, 32).unwrap();
    assert_eq!(scalar.grid, [5, 9, 1]);
    assert_eq!(row8.grid, [5, 2, 1]);
    assert_eq!(wmma.grid, [2, 2, 1]);
    for launch in [scalar, row8, wmma] {
        assert_eq!(launch.block, [256, 1, 1]);
    }
}

#[test]
fn only_scalar_row8_requests_dynamic_shared_memory() {
    assert_eq!(geometry(1, 8).unwrap().shared_mem, 0);
    assert_eq!(geometry(8, 8).unwrap().shared_mem, 2048);
    // WMMA owns 25,600 static bytes; adding those again as dynamic memory
    // would exceed its intended per-CTA footprint and hurt occupancy.
    assert_eq!(geometry(8, 32).unwrap().shared_mem, 0);
}

#[test]
fn single_row_wmma_and_exact_tiles_use_nonzero_grids() {
    let single = index_logits_launch(256, 1, 4096, 1, 32, 128, 4, 32, 8, 32).unwrap();
    let exact = index_logits_launch(256, 16, 4096, 64, 32, 128, 4, 32, 8, 32).unwrap();
    assert_eq!(single.grid, [1, 1, 1]);
    assert_eq!(exact.grid, [2, 2, 1]);
}

#[test]
fn unsupported_tiles_are_rejected() {
    let v2 = GLM_INDEX_LOGITS_V2_POOLS;
    for (rows, pools) in [(0, 8), (4, 8), (1, 32), (8, v2), (8, 0), (8, 16), (8, 64)] {
        assert!(
            geometry(rows, pools).is_err(),
            "accepted tile {rows}x{pools}"
        );
    }
}

#[test]
fn wmma_requires_exact_shape_and_query_alignment() {
    for (query, heads, dim, pool) in [
        (258, 32, 128, 4),
        (256, 16, 128, 4),
        (256, 32, 64, 4),
        (256, 32, 128, 2),
    ] {
        for (rows, pools) in [(8, 32), (4, GLM_INDEX_LOGITS_V2_POOLS)] {
            let launch = index_logits_launch(query, 9, 4096, 33, heads, dim, pool, 32, rows, pools);
            assert!(launch.is_err());
        }
    }
    // WMMA's stricter alignment and specialization must not accidentally
    // remove shapes supported by the two scalar implementations.
    for rows_per_cta in [1, 8] {
        assert!(index_logits_launch(258, 9, 4096, 33, 16, 64, 2, 32, rows_per_cta, 8).is_ok());
    }
}

#[test]
fn empty_dimensions_and_incomplete_cache_pools_are_rejected() {
    for (rows_per_cta, pools_per_cta) in [(1, 8), (8, 8), (8, 32), (4, GLM_INDEX_LOGITS_V2_POOLS)] {
        for (rows, stride, heads, dim, pool, block) in [
            (0, 33, 32, 128, 4, 32),
            (9, 0, 32, 128, 4, 32),
            (9, 33, 0, 128, 4, 32),
            (9, 33, 32, 0, 4, 32),
            (9, 33, 32, 128, 0, 32),
            (9, 33, 32, 128, 4, 0),
            (9, 33, 32, 128, 4, 2),
            (9, 33, 32, 128, 4, 34),
        ] {
            assert!(
                index_logits_launch(
                    256,
                    rows,
                    4096,
                    stride,
                    heads,
                    dim,
                    pool,
                    block,
                    rows_per_cta,
                    pools_per_cta
                )
                .is_err()
            );
        }
    }
}

#[test]
fn arithmetic_and_shared_memory_limits_are_checked() {
    assert!(index_logits_launch(256, 9, u32::MAX, 33, 32, 128, 4, 32, 8, 32).is_err());
    assert!(index_logits_launch(256, 9, 4096, 33, u32::MAX, 128, 4, 32, 1, 8).is_err());
    assert!(index_logits_launch(256, 9, 4096, 33, 1, 4096, 4, 32, 8, 8).is_err());
    assert!(index_logits_launch(256, 9, 4096, 33, 1, u32::MAX, 4, 32, 8, 8).is_err());
}
