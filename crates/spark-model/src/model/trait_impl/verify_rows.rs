// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence row counts of the batched MTP verify (`verify_e`).
//!
//! A sequence verifies `drafts + 1` rows: 2..=4 under the MTP ladder. On the
//! qwen4_exp exact lane (`ATLAS_QWEN4EXP_BATCH_FAST=1` +
//! `ATLAS_QWEN4EXP_EXACT_VERIFY=1`) a sequence holding NO drafts may ride the
//! same forward at one row, a DECODE ROW: the scheduler puts every active
//! sequence in one forward a step instead of a batched verify plus a
//! batched bootstrap decode (`spark-server` `mtp_step/one_forward.rs`).
//!
//! Why one row is the serial decode row, bit for bit, on that lane: every op
//! of the verify forward computes each row with serial decode's arithmetic
//! (`qwen4exp_exact_verify.rs`'s table; the GDN chain is the per-token exact
//! arm, which runs token 0 first and needs nothing after it), and the
//! batch-fast lane makes each row independent of the rows beside it
//! (`qwen4exp_batch_fast.rs`). Row 0 of a verify window never reads the
//! window's later rows (causal attention, sequential GDN chain, per-row MoE
//! and head), so a one-row window is row 0 of any wider one. Its commit is a
//! full accept of one row: the live GDN state the chain leaves IS the state
//! after the token (`commit_accepted_prefix(1, 1)`), the QSA and PLE carries
//! advance by exactly that row, and there is no drafter row to trim.
//!
//! Every other lane keeps 2..=4: their WY verify kernels start at K=2, and
//! their verify rows are not decode's arithmetic.
//!
//! The same argument carries the lane past 4 rows a sequence, up to
//! [`EXACT_LANE_MAX_ROWS`] (7 drafts, `qwen4exp_mtp_depth.rs`). Per op,
//! nothing a row computes depends on how many rows its window holds:
//!
//! * projections, LM head, MoE, mHC: the lane's per-row arithmetic is a
//!   function of the step's TOTAL row count only through which byte-identical
//!   tier carries the rows (`ops::Qwen4ExpWideRows`, <= 32 rows a launch);
//!   the per-sequence split is invisible to them;
//! * GDN: the exact chain is the per-token loop of serial decode
//!   (`decode_batched_conv_gdn_exact`, its strided multi-sequence twin, or
//!   `qwen4exp_gdn_verify_fused_rows` up to `GDN_VERIFY_KMAX` = 8 tokens), token
//!   `t` reading only the state token `t - 1` left; one rollback slot per
//!   token, `K - 1` H snapshots (the pools are sized from `--num-drafts`);
//! * attention: per row (`qsa_rows.rs`, paged decode with no split-K), row
//!   `t` reading the KV the rows before it wrote — causal, as in decode; the
//!   QSA raw-key window keeps `REWIND_MARGIN` (512) rows for a rewind;
//! * PLE: one conv-carry snapshot per row, `VERIFY_SNAP_SLOTS` = 9 covers
//!   the 8 rows plus the pre-window carry.
//!
//! A LONE sequence verifies its 2..=4 rows on its own single-sequence path
//! (`verify_b/c/c2`, graphed); past [`SINGLE_SEQ_MAX_ROWS`] it rides the
//! batched verify as a batch of one, the only verify serving 5..8 rows.

/// Rows a sequence may verify on the qwen4_exp exact lane (7 drafts).
pub(in crate::model) const EXACT_LANE_MAX_ROWS: usize = 8;
/// Rows the single-sequence verifies cover; a lone sequence verifies a wider
/// window on the batched verify.
pub(in crate::model) const SINGLE_SEQ_MAX_ROWS: usize = 4;

/// Whether a batched MTP verify (not DFlash) admits `ks`: every count in
/// 2..=4, or on the exact lane (`exact_lane`) 1..=8; a batch of decode rows
/// only is a batched decode (`decode_batch`), never a verify. A batch of ONE
/// sequence is admitted only on the lane and only past the single-sequence
/// verify's 4 rows.
pub(in crate::model) fn mtp_verify_rows_ok(ks: &[usize], exact_lane: bool) -> bool {
    let (min, max) = if exact_lane {
        (1, EXACT_LANE_MAX_ROWS)
    } else {
        (2, SINGLE_SEQ_MAX_ROWS)
    };
    let width_ok = match ks {
        [k] => exact_lane && *k > SINGLE_SEQ_MAX_ROWS,
        _ => true,
    };
    width_ok && ks.iter().all(|k| (min..=max).contains(k)) && ks.iter().any(|&k| k >= 2)
}

/// Whether the worker trims its drafter after a verified window of `k`
/// rows. A decode row (`k == 1`) drafted nothing, as the head's verdict
/// skips the trim for it (`k4_apply_verdict`).
pub(in crate::model) fn verdict_trims_drafter(k: usize) -> bool {
    k > 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_rows_only_on_the_lane_that_verifies_them() {
        assert!(mtp_verify_rows_ok(&[4, 3, 2], false));
        assert!(!mtp_verify_rows_ok(&[4, 1], false));
        assert!(mtp_verify_rows_ok(&[4, 2, 1], true));
        assert!(
            !mtp_verify_rows_ok(&[5, 1], false),
            "off the lane the ladder caps at 4"
        );
        assert!(!mtp_verify_rows_ok(&[0, 2], true));
    }

    #[test]
    fn the_exact_lane_verifies_up_to_eight_rows_a_sequence() {
        assert!(mtp_verify_rows_ok(&[8, 5, 1], true));
        assert!(mtp_verify_rows_ok(&[8; 16], true));
        assert!(!mtp_verify_rows_ok(&[9, 2], true));
        assert!(!mtp_verify_rows_ok(&[8, 2], false));
    }

    #[test]
    fn a_lone_sequence_batches_only_past_its_own_verify() {
        for k in 5..=8 {
            assert!(mtp_verify_rows_ok(&[k], true), "k={k}");
            assert!(!mtp_verify_rows_ok(&[k], false), "k={k}");
        }
        for k in 1..=4 {
            assert!(
                !mtp_verify_rows_ok(&[k], true),
                "k={k}: verify_b/c/c2 serve it"
            );
        }
        assert!(!mtp_verify_rows_ok(&[9], true));
    }

    #[test]
    fn decode_rows_alone_are_not_a_verify() {
        assert!(!mtp_verify_rows_ok(&[1, 1, 1], true));
        assert!(!mtp_verify_rows_ok(&[], true));
    }

    #[test]
    fn only_a_window_with_drafts_trims_the_drafter() {
        assert!(!verdict_trims_drafter(1));
        assert!(verdict_trims_drafter(2) && verdict_trims_drafter(4));
    }
}
