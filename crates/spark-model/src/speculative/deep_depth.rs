// SPDX-License-Identifier: AGPL-3.0-only

//! Dynamic MTP draft depth past the K-vs-batch ladder
//! (`ATLAS_MTP_DYNAMIC_DEPTH=1`, default off).
//!
//! The ladder ([`super::ladder`]) fixes one draft count per concurrency
//! rung (3 through n=8). With this lever each request steers its own draft
//! ceiling between a floor (`ATLAS_MTP_DEEP_MIN`, default 3) and the
//! configured `--num-drafts` (up to 7 on qwen4_exp's exact lane,
//! `ATLAS_QWEN4EXP_MTP_DEPTH`), from how often its deepest verified
//! position is accepted (`spark-server` `scheduler::mtp_deep_depth`); the
//! drafter's confidence stop still cuts each step's drafts short where it
//! is unsure, so a deep ceiling costs drafter and verify rows only on
//! confident runs.
//!
//! This module is the half both crates must agree on:
//!
//! * which widths may go deep ([`deep_max_seqs`], default 8 — the regime
//!   where the batched verify is weight-read bound and D-Cut prunes), and
//!   the per-slot ceiling the SSM verify pools are sized for
//!   ([`slot_ceiling`], read by `ssm_reserve::verify_slot_drafts` and the
//!   preflight reserve through it): every slot a <= 8-wide batch can occupy
//!   holds `--num-drafts` H snapshots, wider rungs keep the ladder's;
//! * the verify row budget at width `n` ([`row_budget`]): one sequence may
//!   verify its whole window; a batch spends at most
//!   `ATLAS_MTP_DEEP_ROWS_PER_SEQ` (default 5) rows a sequence on average,
//!   so depth goes to the requests that accept it instead of multiplying
//!   every row of the step (the MoE's expert reads and the GDN chain grow
//!   with rows at small n).

use std::sync::OnceLock;

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

/// `ATLAS_MTP_DYNAMIC_DEPTH=1`, read once.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MTP_DYNAMIC_DEPTH").as_deref() == Ok("1"))
}

/// Widest batch that may draft past the ladder (`ATLAS_MTP_DEEP_MAX_SEQS`,
/// default 8).
pub fn deep_max_seqs() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| env_usize("ATLAS_MTP_DEEP_MAX_SEQS").unwrap_or(8))
}

/// Shallowest ceiling the controller steers to (`ATLAS_MTP_DEEP_MIN`,
/// default 3), clamped into `1..=num_drafts`.
pub fn deep_floor(num_drafts: usize) -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    let f = *N.get_or_init(|| env_usize("ATLAS_MTP_DEEP_MIN").unwrap_or(3));
    f.clamp(1, num_drafts.max(1))
}

/// Average verify rows a sequence may spend in a batch
/// (`ATLAS_MTP_DEEP_ROWS_PER_SEQ`, default 5, at least 2).
pub fn rows_per_seq() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| env_usize("ATLAS_MTP_DEEP_ROWS_PER_SEQ").unwrap_or(5).max(2))
}

/// Pure core of [`slot_ceiling`]: the deepest draft count a step at width
/// `n` may hand a sequence.
pub fn ceiling_with(
    on: bool,
    max_seqs: usize,
    n: usize,
    num_drafts: usize,
    ladder: usize,
) -> usize {
    if on && n <= max_seqs {
        num_drafts.max(ladder)
    } else {
        ladder
    }
}

/// The deepest draft count a step at width `n` may hand a sequence: the
/// configured ceiling at `n <= deep_max_seqs()` under the lever, else the
/// ladder's count. SSOT for the per-slot pool tiers and the scheduler.
pub fn slot_ceiling(n: usize, num_drafts: usize) -> usize {
    ceiling_with(
        enabled(),
        deep_max_seqs(),
        n,
        num_drafts,
        super::mtp_ladder_drafts(n, num_drafts),
    )
}

/// Pure core of [`row_budget`].
pub fn row_budget_with(n: usize, max_rows: usize, per_seq: usize, cap: usize) -> usize {
    if n <= 1 {
        max_rows.min(cap)
    } else {
        (n * per_seq).min(cap)
    }
}

/// Verify rows a step of `n` deep-drafting sequences may spend, each
/// verifying at most `max_rows` (`drafts + 1`); `cap` is the verify's row
/// buffer (`VERIFY_ROW_BUDGET`).
pub fn row_budget(n: usize, max_rows: usize, cap: usize) -> usize {
    row_budget_with(n, max_rows, rows_per_seq(), cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ceiling_goes_deep_only_under_the_lever_and_inside_the_width() {
        assert_eq!(ceiling_with(false, 8, 1, 7, 3), 3);
        assert_eq!(ceiling_with(true, 8, 1, 7, 3), 7);
        assert_eq!(ceiling_with(true, 8, 8, 7, 3), 7);
        assert_eq!(ceiling_with(true, 8, 9, 7, 2), 2);
        // Never below the ladder.
        assert_eq!(ceiling_with(true, 8, 4, 2, 3), 3);
    }

    #[test]
    fn a_lone_sequence_verifies_its_window_and_a_batch_shares_a_budget() {
        assert_eq!(row_budget_with(1, 8, 5, 128), 8);
        assert_eq!(row_budget_with(2, 8, 5, 128), 10);
        assert_eq!(row_budget_with(8, 8, 5, 128), 40);
        assert_eq!(row_budget_with(32, 8, 5, 128), 128);
    }
}
