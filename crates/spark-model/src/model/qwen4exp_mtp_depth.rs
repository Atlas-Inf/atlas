// SPDX-License-Identifier: AGPL-3.0-only
//! qwen4_exp (Qwen3.8-Flash-Next) MTP depth ceiling
//! (`ATLAS_QWEN4EXP_MTP_DEPTH=1..7`, default unset = 1 draft).
//!
//! What capped this model at one draft: the GDN layer's
//! `verify_max_drafts` (`qwen3_ssm/trait_layer.rs`) answers 1 unless
//! `ATLAS_MTP_MAX_DRAFTS` says otherwise, and the scheduler clamps every
//! step's draft count to the model's ceiling — so `--num-drafts 2|3` served
//! exactly what `--num-drafts 1` served (second-draft survival 0).
//!
//! With the switch the ceiling is `D` drafts (K = D + 1 verify rows). The
//! scheduler still drafts `min(--num-drafts, D)`, so serve with
//! `--num-drafts D` (the startup pools are sized from `--num-drafts`). The
//! drafter itself always chained: `Qwen4ExpMtpHead::propose` feeds draft
//! `i`'s token and the drafter's own stream highway to draft `i + 1`.
//!
//! Two widths:
//!
//! * D <= 3 (K <= 4): the single-sequence verifies (`verify_b/c/c2`), whose
//!   K=3/4 rows are output-exact only under `ATLAS_QWEN4EXP_EXACT_VERIFY=1`;
//!   without it the switch still serves (and says so), since speculation
//!   that is merely not bit-exact is a measurement setting, not a fault.
//! * D = 4..=7 (K = 5..=8, [`DEEP_MAX_DEPTH`]): only on the exact lane
//!   (`ATLAS_QWEN4EXP_EXACT_VERIFY=1` + `ATLAS_QWEN4EXP_BATCH_FAST=1`). A
//!   verify wider than 4 rows always runs the batched verify (`verify_e`,
//!   one sequence or several), whose every row is serial decode's
//!   arithmetic at any row count (`trait_impl/verify_rows.rs`). Refused
//!   elsewhere: no other verify serves 5..8 rows exactly, and the default
//!   lanes' wide arms are not decode's arithmetic.
//!
//! Both ranks must agree (`startup_parity`): the head dispatches the K the
//! ceiling allows and the worker mirrors it.

use std::sync::OnceLock;

use anyhow::{Result, bail};

/// Deepest draft count a single-sequence verify of its own covers (K=4).
pub(crate) const MAX_DEPTH: usize = 3;
/// Deepest draft count the exact lane's batched verify covers (K=8).
pub(crate) const DEEP_MAX_DEPTH: usize = 7;

/// The raw `ATLAS_QWEN4EXP_MTP_DEPTH` value, read once (0 = unset).
pub(crate) fn requested() -> usize {
    static D: OnceLock<usize> = OnceLock::new();
    *D.get_or_init(|| {
        std::env::var("ATLAS_QWEN4EXP_MTP_DEPTH")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// The model's verify ceiling override: `Some(D)` on a qwen4_exp model with
/// the switch set, else `None` (the layers' own ceiling stands).
pub(crate) fn lever(
    model_type: &str,
    exact_verify: bool,
    batch_fast: bool,
) -> Result<Option<usize>> {
    lever_from(requested(), model_type, exact_verify, batch_fast)
}

fn lever_from(
    depth: usize,
    model_type: &str,
    exact_verify: bool,
    batch_fast: bool,
) -> Result<Option<usize>> {
    if depth == 0 || model_type != "qwen4_exp" {
        return Ok(None);
    }
    if depth > DEEP_MAX_DEPTH {
        bail!(
            "ATLAS_QWEN4EXP_MTP_DEPTH={depth}: qwen4_exp verifies at most {DEEP_MAX_DEPTH} \
             drafts (K={} rows)",
            DEEP_MAX_DEPTH + 1
        );
    }
    if depth > MAX_DEPTH && !(exact_verify && batch_fast) {
        bail!(
            "ATLAS_QWEN4EXP_MTP_DEPTH={depth}: past {MAX_DEPTH} drafts the verify runs on \
             the exact lane only — set ATLAS_QWEN4EXP_EXACT_VERIFY=1 and \
             ATLAS_QWEN4EXP_BATCH_FAST=1"
        );
    }
    if depth >= 2 && !exact_verify {
        tracing::warn!(
            "ATLAS_QWEN4EXP_MTP_DEPTH={depth} without ATLAS_QWEN4EXP_EXACT_VERIFY=1: \
             K={} verify rows are NOT bit-exact against serial decode",
            depth + 1
        );
    }
    tracing::info!(
        "qwen4_exp MTP depth ceiling {depth} draft(s) (ATLAS_QWEN4EXP_MTP_DEPTH); \
         serve with --num-drafts {depth}"
    );
    Ok(Some(depth))
}

#[cfg(test)]
mod tests {
    use super::lever_from;

    #[test]
    fn depth_is_qwen4exp_only_and_bounded_by_the_exact_verify_width() {
        assert_eq!(lever_from(0, "qwen4_exp", true, true).unwrap(), None);
        assert_eq!(lever_from(3, "qwen3_next", true, true).unwrap(), None);
        for d in 1..=3 {
            assert_eq!(lever_from(d, "qwen4_exp", true, false).unwrap(), Some(d));
        }
        // Not exact: still served, with a warning.
        assert_eq!(lever_from(3, "qwen4_exp", false, false).unwrap(), Some(3));
        assert!(lever_from(8, "qwen4_exp", true, true).is_err());
    }

    #[test]
    fn deep_depth_needs_the_exact_lane() {
        for d in 4..=7 {
            assert_eq!(lever_from(d, "qwen4_exp", true, true).unwrap(), Some(d));
            assert!(lever_from(d, "qwen4_exp", true, false).is_err());
            assert!(lever_from(d, "qwen4_exp", false, true).is_err());
        }
    }
}
