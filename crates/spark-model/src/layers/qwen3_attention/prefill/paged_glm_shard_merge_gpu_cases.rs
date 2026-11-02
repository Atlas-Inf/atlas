// SPDX-License-Identifier: AGPL-3.0-only

use super::runner::run_case;
use super::*;

#[test]
#[ignore = "requires a GB10 and a glm-5.3-flash kernel build"]
fn canonical_matches_the_shard_bitwise() -> Result<()> {
    let gpu: &dyn GpuBackend = crate::layers::glm_gpu_test_backend::backend()?;
    let sparse = |skew, holes| Selection::Sparse { skew, holes };
    let mut cases = vec![];
    for dtype in [KvCacheDtype::Fp8G128, KvCacheDtype::Bf16] {
        let case = |rows, ctx, selection, owners| Case {
            dtype,
            rows,
            ctx,
            selection,
            owners,
            pinned_row: false,
        };
        for rows in 1..=8 {
            cases.push(case(rows, 65536, sparse(50, false), 1));
        }
        cases.extend([
            case(8, 4352, sparse(50, false), 4),
            case(8, 131072, sparse(50, true), 1),
            case(4, 16384, sparse(70, false), 2),
            case(2, 8192, sparse(100, false), 1),
            case(3, 8192, sparse(0, true), 1),
            case(16, 32768, sparse(50, false), 1),
            case(24, 32768, sparse(50, true), 1),
            case(48, 32768, sparse(50, false), 1),
            case(64, 32768, sparse(50, false), 1),
            case(8, 64, Selection::Causal(0), 1),
            case(26, 64, Selection::Causal(0), 1),
            case(3, 64, Selection::Causal(2), 1),
            case(8, 2048, Selection::Causal(1000), 2),
            case(8, 2048, Selection::Causal(2040), 1),
            case(1, 2048, Selection::Causal(1500), 1),
            case(64, 2048, Selection::Causal(500), 1),
        ]);
    }
    let mut failed = 0;
    for (i, case) in cases.iter().enumerate() {
        let seed = 0x9e37_79b9_7f4a_7c15 ^ i as u64;
        let differing = run_case(gpu, *case, seed, Placement::fresh())?.differing;
        println!(
            "{} {case:?} splits={}",
            if differing == 0 { "BITWISE" } else { "DIFFERS" },
            shard::MERGE_SPLITS
        );
        failed += (differing != 0) as usize;
    }
    ensure!(failed == 0, "{failed} of {} cases differ", cases.len());
    Ok(())
}

#[test]
#[ignore = "requires a GB10 and a glm-5.3-flash kernel build"]
fn canonical_and_shard_are_history_free() -> Result<()> {
    let gpu: &dyn GpuBackend = crate::layers::glm_gpu_test_backend::backend()?;
    // One canonical region for every case: each owner finds the previous
    // owner's partitions, IDs and counts (another row count, another
    // context), or the first time, poison.
    let shared_bytes = ops::CanonicalLayout::bytes(shard::MERGE_MAX_ROWS as u32);
    let shared_base = gpu.alloc(shared_bytes)?;
    gpu.memset(shared_base, 0x7f, shared_bytes)?;
    let shared = Some((shared_base, shared_bytes));
    let sparse = |skew, holes| Selection::Sparse { skew, holes };
    let mut cases = vec![];
    for dtype in [KvCacheDtype::Fp8G128, KvCacheDtype::Bf16] {
        let case = |rows, ctx, selection| Case {
            dtype,
            rows,
            ctx,
            selection,
            owners: 2,
            pinned_row: false,
        };
        for rows in (1..=8).chain([16, 64]) {
            cases.push(case(rows, 25_947, Selection::Indexer));
        }
        cases.extend([
            case(8, 131_072, Selection::Indexer),
            case(2, 131_072, sparse(50, true)),
            case(16, 65_536, sparse(50, false)),
            case(8, 2048, Selection::Causal(1000)),
            case(64, 2048, Selection::Causal(500)),
            case(1, 2048, Selection::Causal(2040)),
        ]);
    }
    let mut failed = 0;
    for (i, case) in cases.iter().enumerate() {
        // Production-sized pools: ~58K blocks resident, 7,500 capped.
        let big = match case.dtype {
            KvCacheDtype::Fp8G128 => 60_000,
            _ => 40_000,
        };
        let placements = [
            Placement {
                seed: 2,
                pool_blocks: big,
                poison: 0xff,
                scratch: None,
            },
            Placement {
                seed: 3,
                pool_blocks: big,
                poison: 0x7f,
                scratch: shared,
            },
            Placement {
                seed: 4,
                pool_blocks: 7_500,
                poison: 0xff,
                scratch: shared,
            },
        ];
        let seed = 0x51ed_270b_27a3_4e8d ^ i as u64;
        let base = run_case(gpu, *case, seed, Placement::fresh())?;
        let mut bad = vec![];
        if base.differing != 0 {
            bad.push(format!("shard vs canonical {} bytes", base.differing));
        }
        for place in placements {
            let got = run_case(gpu, *case, seed, place)?;
            let moved: usize = got
                .bits
                .iter()
                .zip(&base.bits)
                .map(|(x, y)| x.iter().zip(y).filter(|(a, b)| a != b).count())
                .sum();
            if moved != 0 || got.differing != 0 {
                bad.push(format!(
                    "seed {} pool {} poison {:#x} shared {}: {moved} bytes moved, \
                     shard vs canonical {}",
                    place.seed,
                    place.pool_blocks,
                    place.poison,
                    place.scratch.is_some(),
                    got.differing
                ));
            }
        }
        let verdict = if bad.is_empty() {
            "HISTORY-FREE"
        } else {
            "DEPENDS"
        };
        println!("{verdict} {case:?} splits={}", shard::MERGE_SPLITS);
        for b in &bad {
            println!("  {b}");
        }
        failed += usize::from(!bad.is_empty());
    }
    gpu.free(shared_base)?;
    ensure!(
        failed == 0,
        "{failed} of {} cases depend on placement or history",
        cases.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires a GB10 and a glm-5.3-flash kernel build"]
fn a_row_does_not_depend_on_the_rows_beside_it() -> Result<()> {
    let gpu: &dyn GpuBackend = crate::layers::glm_gpu_test_backend::backend()?;
    let mut failed = 0;
    for dtype in [KvCacheDtype::Fp8G128, KvCacheDtype::Bf16] {
        for (ctx, selection) in [
            (25_947, Selection::Indexer),
            (
                65_536,
                Selection::Sparse {
                    skew: 50,
                    holes: true,
                },
            ),
            (2048, Selection::Causal(1000)),
        ] {
            let mut first: Option<Vec<Vec<u8>>> = None;
            for rows in [1, 2, 3, 5, 8, 16, 64] {
                let case = Case {
                    dtype,
                    rows,
                    ctx,
                    selection,
                    owners: 1,
                    pinned_row: true,
                };
                let got = run_case(gpu, case, 0x7e57_c0de, Placement::fresh())?;
                // Row 0 of every output (32 x 512 BF16) and LSE (32 FP32),
                // canonical then shard, both ranks' heads.
                let row0: Vec<Vec<u8>> = got
                    .bits
                    .iter()
                    .enumerate()
                    .map(|(i, b)| b[..if i % 2 == 0 { 32 * 512 * 2 } else { 32 * 4 }].to_vec())
                    .collect();
                let base = first.get_or_insert_with(|| row0.clone());
                let moved: usize = row0
                    .iter()
                    .zip(base.iter())
                    .map(|(x, y)| x.iter().zip(y).filter(|(a, b)| a != b).count())
                    .sum();
                let ok = moved == 0 && got.differing == 0;
                println!(
                    "{} {dtype:?} ctx={ctx} {selection:?} rows={rows} splits={}: row 0 {moved} \
                     bytes from rows=1, shard vs canonical {}",
                    if ok { "SAME" } else { "MOVED" },
                    shard::MERGE_SPLITS,
                    got.differing
                );
                failed += usize::from(!ok);
            }
        }
    }
    ensure!(failed == 0, "{failed} widths change a row's bits");
    Ok(())
}
