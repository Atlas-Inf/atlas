// SPDX-License-Identifier: AGPL-3.0-only

//! Pure decision functions for generic-DFlash runtime levers. The
//! environment boundary lives in `product_policy::from_env_lenient`; these
//! take already-read `Option<&str>` values so tests never touch the
//! process environment.

/// Pure decision: generic-DFlash Option B (paged drafter context with
/// incremental ctx precompute, graph-eligible). DEFAULT-ON: it is the only
/// measured GB10 configuration and is required by batched propose. The
/// legacy contiguous path stays correct on the h128 drafter (from_weights
/// hard-requires `inferspark_prefill_h128`), but it is eager-only and
/// rebuilds ctx K/V every propose. `ATLAS_DFLASH_OPTION_B=0` rolls back;
/// any other malformed value keeps the default with a startup WARN at the
/// call site.
pub(super) fn option_b_decision(env: Option<&str>) -> bool {
    !matches!(env, Some("0"))
}

/// Pure decision: generic-DFlash authoritative Bxgamma propose. It is
/// DEFAULT-ON when the staged seam is reachable (Option B paged context,
/// exactly one proposal lane — multi-lane pins per-lane scratch and graph
/// state that the staged path does not carry). `ATLAS_DFLASH_BATCHED_PROPOSE`
/// is the rollback: `"0"` disables, `"1"` forces on when the preconditions
/// hold (the only spelling that warns when they do not), and any other value
/// is malformed — off, matching the legacy lenient convention that a
/// malformed opt-in lever is treated as unset.
pub(super) fn generic_batch_authoritative_decision(
    env: Option<&str>,
    option_b: bool,
    lanes: usize,
) -> bool {
    let reachable = option_b && lanes == 1;
    match env {
        None | Some("1") => reachable,
        Some(_) => false,
    }
}
