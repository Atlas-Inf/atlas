// SPDX-License-Identifier: AGPL-3.0-only

//! Where the qwen4_exp dense and branch-point checkpoints land.

use super::{PointAsk, grid_spacing, num_lcm, plan_points};

fn ask(start: usize, count: usize) -> PointAsk {
    PointAsk {
        start,
        count,
        tail: None,
        branch: None,
        every: 0,
        align: 64,
    }
}

#[test]
fn nothing_asked_nothing_planned() {
    assert!(plan_points(ask(0, 16_384), |_| false).is_empty());
}

#[test]
fn dense_rows_are_the_multiples_strictly_inside_the_pass() {
    let a = PointAsk {
        every: 4096,
        ..ask(16_384, 16_384)
    };
    let rows: Vec<_> = plan_points(a, |_| false).into_iter().map(|p| p.0).collect();
    assert_eq!(rows, vec![20_480, 24_576, 28_672]);
    // A cold first chunk.
    let a = PointAsk {
        every: 4096,
        ..ask(0, 16_384)
    };
    let rows: Vec<_> = plan_points(a, |_| false).into_iter().map(|p| p.0).collect();
    assert_eq!(rows, vec![4096, 8192, 12_288]);
}

#[test]
fn dense_rows_stop_below_the_tail_and_skip_what_the_index_has() {
    // Last pass of a 33,017-token prompt: tail row 32,960.
    let a = PointAsk {
        tail: Some(32_960),
        every: 4096,
        ..ask(32_768, 249)
    };
    assert_eq!(plan_points(a, |_| false), vec![(32_960, false)]);
    let a = PointAsk {
        tail: Some(16_000),
        every: 4096,
        ..ask(0, 16_046)
    };
    let got = plan_points(a, |r| r == 8192);
    assert_eq!(got, vec![(4096, false), (12_288, false), (16_000, false)]);
}

#[test]
fn the_branch_row_is_floored_to_the_pass_grid_and_marked() {
    // A 3,700-token shared preamble, recomputed from 0.
    let a = PointAsk {
        tail: Some(5_952),
        branch: Some(3_700),
        ..ask(0, 6_000)
    };
    assert_eq!(
        plan_points(a, |_| false),
        vec![(3_648, true), (5_952, false)]
    );
    // Coinciding with a dense row: one point, a branch.
    let a = PointAsk {
        branch: Some(8_200),
        every: 4096,
        ..ask(0, 16_384)
    };
    assert_eq!(
        plan_points(a, |_| false),
        vec![(4096, false), (8192, true), (12_288, false)]
    );
    // On the tail row: the tail, marked.
    let a = PointAsk {
        tail: Some(5_952),
        branch: Some(5_960),
        ..ask(0, 6_000)
    };
    assert_eq!(plan_points(a, |_| true), vec![(5_952, true)]);
    // Past this pass (a later chunk takes it), or at its start.
    let a = PointAsk {
        branch: Some(20_000),
        ..ask(0, 16_384)
    };
    assert!(plan_points(a, |_| false).is_empty());
    let a = PointAsk {
        branch: Some(16_384),
        ..ask(16_384, 16_384)
    };
    assert!(plan_points(a, |_| false).is_empty());
    // A replay pass from a restore below it.
    let a = PointAsk {
        branch: Some(30_000),
        ..ask(16_384, 20_000)
    };
    assert_eq!(plan_points(a, |_| false), vec![(29_952, true)]);
}

#[test]
fn rows_respect_the_pass_grid_and_the_alignment() {
    // A pass off the 64-row grid (no ROWINV): rows are 64-row boundaries of
    // the pass that are also block (16) multiples.
    let a = PointAsk {
        every: 1024,
        align: 16,
        ..ask(16_388, 4_000)
    };
    for (r, _) in plan_points(a, |_| false) {
        assert_eq!((r - 16_388) % 64, 0);
        assert_eq!(r % 16, 0);
    }
    let a = PointAsk {
        every: 1024,
        align: 64,
        ..ask(16_388, 4_000)
    };
    assert!(
        plan_points(a, |_| false).is_empty(),
        "no row is on both grids"
    );
}

#[test]
fn at_most_the_kernel_list_deepest_dense_first() {
    let a = PointAsk {
        tail: Some(16_320),
        branch: Some(100),
        every: 64,
        ..ask(0, 16_384)
    };
    let got = plan_points(a, |_| false);
    assert_eq!(got.len(), crate::layers::qwen4exp_ckpt::MAX_POINTS);
    assert!(got.contains(&(16_320, false)) && got.contains(&(64, true)));
    assert_eq!(got[got.len() - 2].0, 16_256, "the deepest dense rows stay");
    assert!(got.windows(2).all(|w| w[0].0 < w[1].0));
}

#[test]
fn spacing_and_alignment_helpers() {
    assert_eq!(grid_spacing(0), 0);
    assert_eq!(grid_spacing(10), 64);
    assert_eq!(grid_spacing(4100), 4096);
    assert_eq!(num_lcm(4, 64), 64);
    assert_eq!(num_lcm(4, 16), 16);
    assert_eq!(num_lcm(3, 16), 48);
}
