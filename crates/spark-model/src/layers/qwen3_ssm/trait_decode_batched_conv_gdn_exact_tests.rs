// SPDX-License-Identifier: AGPL-3.0-only

use super::exact_row;

// Real 27B GDN shapes: nk=16, nv=48, kd=vd=128, d_conv=4.
const QKVZ: usize = 16384; // conv_dim + value_dim
const CONV_DIM: usize = 10240; // 2*2048 + 6144
const VALUE_DIM: usize = 6144; // 48*128
const NV: usize = 48;

/// POSITIVE: token offsets stride by the ROW sizes of each buffer — qkvz
/// rows in `deinterleaved`/`ssm_conv_out_f32`, value_dim rows in the
/// normed output, 2*nv FP32 in gates — on the real 27B shapes.
#[test]
fn exact_row_strides_match_27b_shapes() {
    let r2 = exact_row(2, 4, QKVZ, CONV_DIM, VALUE_DIM, NV);
    assert_eq!(r2.qkv_in, 2 * QKVZ * 2, "deinterleaved rows are qkvz BF16");
    assert_eq!(r2.conv_out_f32, 2 * QKVZ * 4, "f32 conv rows are qkvz FP32");
    assert_eq!(
        r2.gdn_out_f32,
        2 * QKVZ * 4 + CONV_DIM * 4,
        "f32 gdn scratch is the conv row's Z-region tail"
    );
    assert_eq!(r2.z, 2 * QKVZ * 2 + CONV_DIM * 2, "Z sits after [Q|K|V]");
    assert_eq!(r2.gate, 2 * NV * 2 * 4, "gates rows are [gate|beta] FP32");
    assert_eq!(r2.beta, r2.gate + NV * 4);
    assert_eq!(
        r2.normed_out,
        2 * VALUE_DIM * 2,
        "normed rows are value_dim BF16"
    );
    // The FP32 GDN scratch must fit inside the row's tail: exactly
    // value_dim FP32 elements between conv_dim and qkvz_size.
    assert_eq!(QKVZ - CONV_DIM, VALUE_DIM);
}

/// POSITIVE + NEGATIVE: rollback intermediates are written for tokens
/// 0..K-2 and NOT for K-1 (no reader exists for that index — writing it
/// is the dead d2d the K=4 arm removed; skipping earlier ones would break
/// rollback).
#[test]
fn exact_row_snapshot_skips_only_last_token() {
    for k in 2..=8usize {
        for t in 0..k {
            let row = exact_row(t, k, QKVZ, CONV_DIM, VALUE_DIM, NV);
            assert_eq!(
                row.snapshot,
                t + 1 < k,
                "snapshot policy wrong at t={t}, k={k}"
            );
        }
    }
}
