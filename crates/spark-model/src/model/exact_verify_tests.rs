// SPDX-License-Identifier: AGPL-3.0-only
//! Tests for [`super`] (the dense exact-verify lever's plumbing).

use super::*;

fn levers(qwen4exp: bool, generic: bool) -> ModelLevers {
    ModelLevers {
        qwen4exp_exact_verify: qwen4exp,
        exact_verify: generic,
        ..ModelLevers::defaults()
    }
}

#[test]
fn the_switch_is_exactly_one() {
    assert!(requested_from(Some("1")));
    for off in [
        None,
        Some("0"),
        Some(""),
        Some("true"),
        Some("yes"),
        Some(" 1"),
    ] {
        assert!(!requested_from(off), "{off:?}");
    }
}

#[test]
fn lever_is_qwen3_5_only() {
    let none = Conflicts::default();
    assert!(lever_from(true, "qwen3_5", none).unwrap());
    // Off: nothing changes for anyone who does not set it.
    assert!(!lever_from(false, "qwen3_5", none).unwrap());
    // Other models ignore it (qwen4_exp has its own switch).
    for other in ["qwen4_exp", "qwen3_5_moe", "qwen3_next", "glm5_next"] {
        assert!(!lever_from(true, other, none).unwrap(), "{other}");
    }
}

#[test]
fn lever_refuses_each_conflict_and_ignores_them_when_off() {
    let cases = [
        Conflicts {
            h_f16: true,
            ..Default::default()
        },
        Conflicts {
            ffn_via_gemm: true,
            ..Default::default()
        },
        Conflicts {
            no_split_silu: true,
            ..Default::default()
        },
    ];
    for c in cases {
        let err = lever_from(true, "qwen3_5", c).unwrap_err().to_string();
        assert!(err.contains("ATLAS_EXACT_VERIFY"), "{err}");
        // Off, or on another model, the conflicts are none of its business.
        assert!(!lever_from(false, "qwen3_5", c).unwrap());
        assert!(!lever_from(true, "qwen3_5_moe", c).unwrap());
    }
}

#[test]
fn the_check_is_qwen3_5_only() {
    assert!(check_from(true, "qwen3_5"));
    assert!(!check_from(false, "qwen3_5"));
    assert!(!check_from(true, "qwen3_5_moe"));
}

#[test]
fn verify_bit_exact_needs_a_lever_and_no_dflash() {
    // Off: false, as the trait default.
    assert!(!verify_bit_exact(&levers(false, false), false));
    // Dense lever: true for MTP, false beside a DFlash drafter.
    assert!(verify_bit_exact(&levers(false, true), false));
    assert!(!verify_bit_exact(&levers(false, true), true));
    // qwen4_exp keeps its own answer (both fields set by its lever).
    assert!(verify_bit_exact(&levers(true, true), false));
    assert!(levers(false, true).dense_exact_verify());
    assert!(!levers(true, true).dense_exact_verify());
}

#[test]
fn max_drafts_caps_the_dense_lever_at_the_widest_exact_verify() {
    let dense = levers(false, true);
    assert_eq!(max_drafts(&dense, false, None), Some(MAX_DRAFTS));
    assert_eq!(max_drafts(&dense, false, Some(7)), Some(MAX_DRAFTS));
    assert_eq!(max_drafts(&dense, false, Some(1)), Some(1));
    // DFlash, the lever off, or qwen4_exp: the layers' ceiling as before.
    assert_eq!(max_drafts(&dense, true, None), None);
    assert_eq!(max_drafts(&levers(false, false), false, Some(7)), Some(7));
    assert_eq!(max_drafts(&levers(true, true), false, None), None);
}

#[test]
fn attention_split_pin_covers_the_widest_verify_under_the_dense_lever() {
    let mut l = levers(false, true);
    l.max_decode_seqs = 1;
    assert_eq!(l.attn_split_ref_seqs(), MAX_ROWS as u32);
    l.max_decode_seqs = 8;
    assert_eq!(l.attn_split_ref_seqs(), 8);
    // Off, and for qwen4_exp: the configured max batch, unchanged.
    for mut other in [levers(false, false), levers(true, true)] {
        other.max_decode_seqs = 1;
        assert_eq!(other.attn_split_ref_seqs(), 1);
    }
}

/// Native FP8 attention (`ATLAS_DENSE_NATIVE_FP8_ATTN`): every exact verify
/// width is ONE `w8a16_gemv_batch4` launch (rows = serial `w8a16_gemv`), on
/// NVIDIA and HIP alike, whatever the vl2 lever says; a missing batch4 leaves
/// the per-row `w8a16_gemv` loop, which is serial decode itself.
#[test]
fn native_fp8_attention_verify_rows_are_the_serial_twin() {
    use spark_runtime::gpu::KernelHandle;
    let k = crate::layers::ops::Fp8RowsKernels {
        batch4: KernelHandle(4),
        batch8: KernelHandle(8),
        batch8_dyn_vl2: KernelHandle(9),
        batch8_dyn_vl2_n2: KernelHandle(10),
        batch16: KernelHandle(16),
    };
    for rows in 2..=MAX_ROWS {
        for (wide, vl2) in [(true, true), (true, false), (false, true), (false, false)] {
            let plan = crate::layers::ops::fp8_rows_plan(rows, wide, vl2, k).unwrap();
            assert_eq!(plan.len(), 1, "rows {rows}");
            assert_eq!((plan[0].kernel.0, plan[0].div), (4, 4), "rows {rows}");
            assert_eq!((plan[0].first, plan[0].rows), (0, rows));
        }
        let no_b4 = crate::layers::ops::Fp8RowsKernels {
            batch4: KernelHandle(0),
            ..k
        };
        assert!(crate::layers::ops::fp8_rows_plan(rows, true, true, no_b4).is_none());
    }
}
