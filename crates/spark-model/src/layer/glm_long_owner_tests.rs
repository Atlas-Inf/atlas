// SPDX-License-Identifier: AGPL-3.0-only
//! Owner-batch width, arena capacity and stage row placement (no numerics).
use super::*;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::mock::MockGpuBackend;

const WIDTHS: [usize; 3] = [2, K3_ROWS, MAX_OWNER_ROWS];

fn small_rows() -> RowBytes {
    // hidden 4, hc 2, vocab 3: every span a distinct, small row size.
    RowBytes::new(4, 2, 3)
}

#[test]
fn widths_admit_up_to_eight_owners_within_the_row_budget() {
    assert_eq!(
        (K3_ROWS, MAX_OWNER_ROWS, MAX_OWNERS, MAX_ROWS),
        (3, 8, 8, 32)
    );
    for owners in 1..=MAX_OWNERS {
        for rows in 2..=MAX_OWNER_ROWS {
            assert_eq!(
                width_supported(owners, rows),
                owners * rows <= MAX_ROWS,
                "{owners}x{rows}"
            );
        }
        for rows in [0, 1, MAX_OWNER_ROWS + 1] {
            assert!(!width_supported(owners, rows), "{owners}x{rows}");
        }
    }
    for (owners, rows) in [(8, 4), (6, 5), (5, 6), (4, 8), (1, 8)] {
        assert!(width_supported(owners, rows), "{owners}x{rows}");
    }
    for (owners, rows) in [(8, 5), (5, 7), (0, K3_ROWS), (MAX_OWNERS + 1, 2)] {
        assert!(!width_supported(owners, rows), "{owners}x{rows}");
    }
    assert_eq!(max_rows_per_owner(8), 4);
    assert_eq!(max_rows_per_owner(3), 8);
    assert_eq!(max_rows_per_owner(9), 0);
}

#[test]
fn arena_capacity_is_checked_per_row_count() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 4096;
    config.hc_mult = 4;
    let rows = RowBytes::new(config.hidden_size, config.hc_mult, config.vocab_size);
    let mut sizes = BufferSizes::from_config(&config, 1024, 2048, 16, 4);
    assert!(rows.arena_fits(&sizes, 4 * K3_ROWS));
    assert!(rows.arena_fits(&sizes, MAX_ROWS));
    sizes.logits = MAX_ROWS * rows.logits - 1;
    assert!(rows.arena_fits(&sizes, 4 * K3_ROWS));
    assert!(!rows.arena_fits(&sizes, MAX_ROWS));
    assert!(!rows.arena_fits(&sizes, usize::MAX));
}

/// Arena rows are tagged `row+1`, stage rows `0x80+row`, per span.
fn fill(gpu: &MockGpuBackend, ptr: DevicePtr, rows: usize, bytes: usize, tag: u8) {
    let data: Vec<u8> = (0..rows)
        .flat_map(|r| std::iter::repeat_n(tag + r as u8, bytes))
        .collect();
    gpu.copy_h2d(&data, ptr).unwrap();
}

fn row(gpu: &MockGpuBackend, ptr: DevicePtr, r: usize, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    gpu.copy_d2h(ptr.offset(r * bytes), &mut out).unwrap();
    out
}

#[test]
fn stage_spans_are_disjoint_and_hold_max_rows() {
    let gpu = MockGpuBackend::new();
    let r = small_rows();
    let stage = GlmLongStage::alloc(&gpu, r).unwrap();
    let spans = [
        (stage.hidden, r.hidden),
        (stage.norm, r.hidden),
        (stage.highway, r.highway),
        (stage.post, r.post),
        (stage.comb, r.comb),
        (stage.logits, r.logits),
        (stage.ffn, r.hidden),
    ];
    for pair in spans.windows(2) {
        assert_eq!(pair[1].0.0, pair[0].0.0 + (MAX_ROWS * pair[0].1) as u64);
    }
    let (last, bytes) = spans[spans.len() - 1];
    assert_eq!(
        last.0 + (MAX_ROWS * bytes) as u64,
        stage.hidden.0 + r.total() as u64
    );
    // The last stage row of every span is addressable.
    for (ptr, bytes) in spans {
        assert_eq!(row(&gpu, ptr, MAX_ROWS - 1, bytes), vec![0; bytes]);
    }
}

#[test]
fn stage_places_each_owner_at_owner_major_rows() {
    let gpu = MockGpuBackend::new();
    let r = small_rows();
    let stage = GlmLongStage::alloc(&gpu, r).unwrap();
    let arena = [
        (
            gpu.alloc(MAX_ROWS * r.hidden).unwrap(),
            stage.hidden,
            r.hidden,
        ),
        (
            gpu.alloc(MAX_ROWS * r.highway).unwrap(),
            stage.highway,
            r.highway,
        ),
        (
            gpu.alloc(MAX_ROWS * r.logits).unwrap(),
            stage.logits,
            r.logits,
        ),
    ];
    for rows in WIDTHS {
        for owners in (1..=MAX_OWNERS).filter(|&o| width_supported(o, rows)) {
            let total = owners * rows;
            for &(a, s, bytes) in &arena {
                fill(&gpu, a, MAX_ROWS, bytes, 1);
                fill(&gpu, s, MAX_ROWS, bytes, 0x80);
            }
            // Every owner's joint rows to the stage, then owner by owner back
            // to arena rows [0, rows), as the MLA and restore paths do.
            stage.copy(&gpu, &arena, 0, 0, total, true, 0).unwrap();
            for owner in 0..owners {
                stage
                    .copy(&gpu, &arena, 0, owner * rows, rows, false, 0)
                    .unwrap();
                for &(a, s, bytes) in &arena {
                    for t in 0..rows {
                        let joint = (owner * rows + t) as u8 + 1;
                        assert_eq!(row(&gpu, a, t, bytes), vec![joint; bytes]);
                        assert_eq!(row(&gpu, s, owner * rows + t, bytes), vec![joint; bytes]);
                    }
                    // Stage rows past this call's width are untouched.
                    if total < MAX_ROWS {
                        assert_eq!(row(&gpu, s, total, bytes), vec![0x80 + total as u8; bytes]);
                    }
                }
            }
        }
    }
    let rows = MAX_OWNER_ROWS;
    assert!(
        stage
            .copy(&gpu, &arena, 0, MAX_ROWS - rows + 1, rows, true, 0)
            .is_err()
    );
    assert!(
        stage
            .copy(&gpu, &arena, MAX_ROWS - rows + 1, 0, rows, true, 0)
            .is_err()
    );
    assert!(
        stage
            .copy(&gpu, &arena, 0, MAX_ROWS - rows, rows, true, 0)
            .is_ok()
    );
}
