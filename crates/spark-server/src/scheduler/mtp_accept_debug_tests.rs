// SPDX-License-Identifier: AGPL-3.0-only

// The bucket table must cover the MTP dispatch cap, or distinct widths
// alias onto one bucket and `adaptive_rung` steers the n=16 rung from a
// mixture of n=16 and n>16 statistics (see the MAX_N doc). Regression
// guard for the 2026-07-30 cap raise that MAX_N never followed.
#[test]
fn bucket_table_covers_the_mtp_dispatch_cap() {
    assert_eq!(BUCKETS.len(), MAX_N);
    // The compiled default cap is 32, so 32 must be individually tracked.
    const { assert!(MAX_N > 32) };
    // And it must cover whatever cap THIS process is configured for
    // (CI does not set the override; an operator who raises it past the
    // table re-introduces the documented fold).
    if std::env::var_os("ATLAS_MTP_MAX_SEQS").is_none() {
        assert!(
            MAX_N > spark_model::speculative::mtp_max_seqs(),
            "MAX_N {MAX_N} does not cover dispatch cap {}",
            spark_model::speculative::mtp_max_seqs()
        );
    }
}

// Distinct widths must not share a bucket anywhere the scheduler can
// dispatch MTP. Asserted on `bucket_idx`, the SSOT `record` indexes
// with, so it fails for exactly the widths that aliased before the fix
// (17..=32 onto 16). Env-independent: CI sets neither override.
#[test]
fn widths_up_to_the_cap_do_not_alias() {
    if std::env::var_os("ATLAS_MTP_ACCEPT_FOLD_AT_16").is_some()
        || std::env::var_os("ATLAS_MTP_MAX_SEQS").is_some()
    {
        return; // the kill switch deliberately restores the fold
    }
    for n in 0..=spark_model::speculative::mtp_max_seqs() {
        assert_eq!(bucket_idx(n), n, "width {n} aliases onto another bucket");
    }
    // Beyond the table the documented fold still applies, and it must
    // stay inside the array.
    assert_eq!(bucket_idx(1_000), MAX_N - 1);
    assert!(bucket_idx(usize::MAX) < BUCKETS.len());
}

use super::*;

#[test]
fn empty_suffix_is_zeros() {
    let a = RequestAccept::default();
    assert_eq!(
        a.done_suffix(),
        "serial=0.00 mtp=0.00 p1=0.000 mean_na=0.000 tok_step=1.000 regime_reprobes=0 depth=k5 depth_switches=0 surv=0.00,0.00,0.00,0.00,0.00,0.00,0.00"
    );
}

#[test]
fn survival_curve_counts_steps_accepting_at_least_each_position() {
    let mut a = RequestAccept::default();
    for emitted in [1, 2, 4, 8, 8] {
        a.record_verify_emitted(emitted); // accepted 0, 1, 3, 7, 7
    }
    assert!(
        a.done_suffix()
            .ends_with("surv=0.80,0.60,0.60,0.40,0.40,0.40,0.40")
    );
}

#[test]
fn thinking_serial_run_is_all_serial() {
    let mut a = RequestAccept::default();
    for _ in 0..300 {
        a.record_serial();
    }
    assert!((a.serial_frac() - 1.0).abs() < 1e-9);
    assert_eq!(a.mean_na(), 0.0);
    assert_eq!(a.tok_step(), 1.0);
    assert!(a.done_suffix().contains("serial=1.00"));
    assert!(a.done_suffix().contains("mtp=0.00"));
}

#[test]
fn mtp_run_reports_p1_mean_na_tok_step() {
    let mut a = RequestAccept::default();
    for _ in 0..7 {
        a.record_verify_emitted(2); // 1 draft, d1 match
    }
    for _ in 0..3 {
        a.record_verify_emitted(1); // reject
    }
    assert!((a.mtp_frac() - 1.0).abs() < 1e-9);
    assert!((a.p1() - 0.7).abs() < 1e-9);
    assert!((a.mean_na() - 0.7).abs() < 1e-9);
    // 7 verifies each accepting 1 draft: the per-request total the usage
    // field reports is the raw sum, not a rate.
    assert_eq!(a.accepted_total(), 7);
    assert!((a.tok_step() - 1.7).abs() < 1e-9);
    a.note_regime_reprobe();
    assert!(a.done_suffix().contains("mean_na=0.700"));
    assert!(a.done_suffix().contains("tok_step=1.700"));
    assert!(a.done_suffix().contains("regime_reprobes=1"));
}

#[test]
fn low_yield_k5_moves_to_k3() {
    let mut a = RequestAccept::default();
    // K5 yield = 2.75, projected K3 yield = 2.50, ratio 1.10.
    for accepted in [3, 3, 3, 2, 2, 2, 2, 2, 2, 0, 0, 0] {
        a.record_depth_verify_inner(4, accepted);
    }
    assert_eq!(a.depth_mode, DEPTH_SHALLOW);
    assert_eq!(a.depth_switches, 1);
}

#[test]
fn high_yield_k5_stays_deep() {
    let mut a = RequestAccept::default();
    for _ in 0..DEPTH_WINDOW {
        a.record_depth_verify_inner(4, 4);
    }
    assert_eq!(a.depth_mode, DEPTH_DEEP);
    assert_eq!(a.depth_switches, 0);
}

#[test]
fn saturated_k3_triggers_a_deep_probe() {
    let mut a = RequestAccept {
        depth_mode: DEPTH_SHALLOW,
        ..Default::default()
    };
    for i in 0..DEPTH_WINDOW {
        a.record_depth_verify_inner(2, if i % 4 == 0 { 1 } else { 2 });
    }
    assert_eq!(a.depth_mode, DEPTH_DEEP);
    assert_eq!(a.depth_switches, 1);
}

#[test]
fn fixed_single_draft_reports_k2_without_adaptive_depth() {
    let mut a = RequestAccept::default();
    for accepted in [1, 0, 1] {
        a.record_depth_verify(1, accepted, false);
        a.record_verify_emitted(accepted + 1);
        assert!(a.done_suffix().contains("depth=k2 "));
        assert!(a.tok_step() <= 2.0);
        assert_eq!(a.depth_drafts(1, false), 1);
    }
    a.record_depth_verify(4, 4, false);
    assert!(a.done_suffix().contains("depth=k5 "));
}

#[test]
fn fixed_two_drafts_report_k3_without_adaptation_or_depth_lift() {
    let mut a = RequestAccept::default();
    for accepted in [2, 1, 0, 2] {
        a.record_depth_verify(2, accepted, false);
        a.record_verify_emitted(accepted + 1);
        assert!(a.done_suffix().contains("depth=k3 "));
        assert!(a.tok_step() <= 3.0);
        assert_eq!(a.depth_drafts(2, false), 2);
    }
}

#[test]
fn a_two_or_three_draft_ceiling_is_steered_by_the_depth_ladder_only_when_armed() {
    let mut a = RequestAccept::default();
    // Unarmed: the ladder is never fed and never consulted.
    for _ in 0..64 {
        a.record_depth_verify(3, 0, false);
    }
    assert_eq!(a.depth_drafts(3, false), 3);
    assert_eq!(
        a.depth_drafts(3, true),
        3,
        "an unfed ladder starts at the ceiling"
    );
    // Armed at a 2/3-draft ceiling the ladder answers, never above the ceiling;
    // the GLM K5/K3 controller keeps 4+.
    for _ in 0..64 {
        a.record_depth_verify(3, 0, true);
    }
    assert!((1..=3).contains(&a.depth_drafts(3, true)));
    assert!((1..=2).contains(&a.depth_drafts(2, true)));
    assert_eq!(a.depth_drafts(4, true), 4);
}
