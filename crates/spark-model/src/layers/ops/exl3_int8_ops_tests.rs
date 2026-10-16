// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free tests for the sq launch geometry: the plan formulas of
//! `exl3_int8_ops` on the model's decode shapes, the refusals, and one plan
//! computed by hand.

use super::*;

/// Every dense decode shape of qwen3.8-flash-next (in, out), plus the MTP fc
/// at bits 5 — checked at grid 96 (2 blocks/SM on a 48-SM part).
const SHAPES: [(usize, usize); 8] = [
    (2560, 12288),
    (2560, 512),
    (4096, 2560),
    (2560, 10240),
    (2560, 6144),
    (6144, 2560),
    (2560, 248320),
    (5120, 2560),
];

#[test]
fn sq_plan_ok_on_model_shapes() {
    for (k, n) in SHAPES {
        for m in [1, 2] {
            for bits in [5, 6] {
                let p = sq_plan(k, n, m, bits, 96)
                    .unwrap_or_else(|e| panic!("k={k} n={n} m={m} bits={bits}: {e}"));
                assert!(p.smem <= 99 * 1024, "k={k} n={n} m={m} bits={bits}: {p:?}");
                assert!(p.ksplit <= 64, "k={k} n={n} m={m} bits={bits}: {p:?}");
                assert_eq!(p.rows_per % 8, 0, "k={k} n={n} m={m} bits={bits}: {p:?}");
                assert!(p.rows_per >= 16, "k={k} n={n} m={m} bits={bits}: {p:?}");
            }
        }
    }
}

#[test]
fn sq_plan_refusals() {
    // n not a multiple of 256, k not a multiple of 128, m > 2, bits outside 4..=6.
    assert!(sq_plan(2560, 640, 1, 6, 96).is_err());
    assert!(sq_plan(2600, 2560, 1, 6, 96).is_err());
    assert!(sq_plan(2560, 2560, 3, 6, 96).is_err());
    assert!(sq_plan(2560, 2560, 1, 3, 96).is_err());
    assert!(sq_plan(2560, 2560, 1, 7, 96).is_err());
}

#[test]
fn sq_stage_bytes_only_for_odd_rates() {
    assert!(sq_stage_bytes(5) > 0);
    assert_eq!(sq_stage_bytes(4), 0);
    assert_eq!(sq_stage_bytes(6), 0);
}

/// k = 2560, n = 2560, m = 1, bits = 6, grid = 96, by hand from the formulas:
///
/// rows_total = 2560/16 = 160, nb256 = 2560/256 = 10
/// r          = ceil(160*10 / 96) = ceil(1600/96) = ceil(16.67) = 17
/// rows_per   = (max(17, min(34, 32)) + 7) & !7 = (32 + 7) & !7 = 39 & !7 = 32
/// rows_max(1) = min((81920 / 96) & !7, 512) = min(853 & !7, 512) = min(848, 512) = 512
/// rows_per   = min(32, 512) = 32;  min(32, ceil(160/8)*8 = 160) = 32
/// ksplit     = ceil(160 / 32) = 5
/// smem       = 32*16*2 + 32*16*4*1 + stage_bytes(6)=0 + 2*1*128*4
///            = 1024 + 2048 + 0 + 1024 = 4096
/// ws_ints    = SQ_WS_RESERVED + 5*1*2560 = (4096 + 4*64*4) + 12800 = 5120 + 12800 = 17920
#[test]
fn sq_plan_hand_computed() {
    let p = sq_plan(2560, 2560, 1, 6, 96).unwrap();
    assert_eq!(
        p,
        SqPlan {
            grid: 96,
            rows_per: 32,
            ksplit: 5,
            smem: 4096,
            ws_ints: 17920,
        }
    );
}
