// SPDX-License-Identifier: AGPL-3.0-only
//! qwen4_exp (Qwen3.8-Flash-Next) MTP depth ceiling
//! (`ATLAS_QWEN4EXP_MTP_DEPTH=1|2|3`, default unset = 1 draft).
//!
//! What capped this model at one draft: the GDN layer's
//! `verify_max_drafts` (`qwen3_ssm/trait_layer.rs`) answers 1 unless
//! `ATLAS_MTP_MAX_DRAFTS` says otherwise, and the scheduler clamps every
//! step's draft count to the model's ceiling — so `--num-drafts 2|3` served
//! exactly what `--num-drafts 1` served (second-draft survival 0).
//!
//! With the switch the ceiling is `D` drafts (K = D + 1 verify rows), up to
//! the K=4 verify the exact path covers (`model/qwen4exp_exact_verify.rs`).
//! The scheduler still drafts `min(--num-drafts, D)`, so serve with
//! `--num-drafts D` (the startup pools are sized from `--num-drafts`). The
//! drafter itself always chained: `Qwen4ExpMtpHead::propose` feeds draft
//! `i`'s token and the drafter's own stream highway to draft `i + 1`.
//!
//! K=3/4 rows are output-exact only under `ATLAS_QWEN4EXP_EXACT_VERIFY=1`;
//! without it the switch still serves (and says so), since speculation that
//! is merely not bit-exact is a measurement setting, not a fault.
//!
//! Both ranks must agree (`startup_parity`): the head dispatches the K the
//! ceiling allows and the worker mirrors it.

use std::sync::OnceLock;

use anyhow::{Result, bail};

/// Deepest draft count the exact K-row verify covers (K=4).
pub(crate) const MAX_DEPTH: usize = 3;

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
pub(crate) fn lever(model_type: &str, exact_verify: bool) -> Result<Option<usize>> {
    lever_from(requested(), model_type, exact_verify)
}

fn lever_from(depth: usize, model_type: &str, exact_verify: bool) -> Result<Option<usize>> {
    if depth == 0 || model_type != "qwen4_exp" {
        return Ok(None);
    }
    if depth > MAX_DEPTH {
        bail!(
            "ATLAS_QWEN4EXP_MTP_DEPTH={depth}: qwen4_exp verifies at most {MAX_DEPTH} drafts \
             (K={} rows)",
            MAX_DEPTH + 1
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
        assert_eq!(lever_from(0, "qwen4_exp", true).unwrap(), None);
        assert_eq!(lever_from(3, "qwen3_next", true).unwrap(), None);
        for d in 1..=3 {
            assert_eq!(lever_from(d, "qwen4_exp", true).unwrap(), Some(d));
        }
        // Not exact: still served, with a warning.
        assert_eq!(lever_from(3, "qwen4_exp", false).unwrap(), Some(3));
        assert!(lever_from(4, "qwen4_exp", true).is_err());
    }
}
