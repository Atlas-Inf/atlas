// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_PC_FINISH_LEAF` and its save span: what the flag needs before
//! it takes effect (see "Preconditions" in the parent module).

use std::sync::OnceLock;

/// The flag with its preconditions `(what it needs, whether it holds)`.
pub(super) fn resolve(flag: bool, needs: &[(&str, bool)]) -> bool {
    let missing: Vec<&str> = needs.iter().filter(|n| !n.1).map(|n| n.0).collect();
    if flag && !missing.is_empty() {
        tracing::warn!(
            "ATLAS_GLM_PC_FINISH_LEAF=1 ignored: it also needs {}",
            missing.join(", ")
        );
    }
    flag && missing.is_empty()
}

/// `ATLAS_GLM_PC_FINISH_LEAF=1` with its preconditions. Read once.
pub(in crate::model) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        resolve(
            std::env::var("ATLAS_GLM_PC_FINISH_LEAF").as_deref() == Ok("1"),
            &[
                // The radix walk's own read (`RadixTreeInner::walk`). opt/audit-fixes
                // exports it as `radix_tree::subblock_matching()`: use that once
                // it lands (adding the helper here too merges as a duplicate).
                (
                    "ATLAS_PREFIX_SUBBLOCK=0",
                    std::env::var("ATLAS_PREFIX_SUBBLOCK").as_deref() == Ok("0"),
                ),
                (
                    "ATLAS_MARCONI_PREFILL_ONLY=1",
                    crate::model::mtp_carry::marconi_prefill_only(),
                ),
                (
                    "ATLAS_GLM_PC_EVICT=1",
                    spark_runtime::radix_tree::glm_pc_evict_enabled(),
                ),
            ],
        )
    })
}

/// Blocks between rolling saves when `ATLAS_GLM_PC_FINISH_LEAF_BLOCKS` is
/// unset. The leaf lands on the last boundary that is a multiple of `n`
/// blocks, so `n` trades copies (`1/n` of them) and reach (a next prompt that
/// stops `u` tokens short of the end misses the leaf about `u / (16 n)` of
/// the time) against up to `16 (n - 1)` more replayed tokens.
const DEFAULT_SPAN_BLOCKS: usize = 4;

/// The save span in blocks (`ATLAS_GLM_PC_FINISH_LEAF_BLOCKS`). Read once.
pub(in crate::model) fn span_blocks() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_GLM_PC_FINISH_LEAF_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_SPAN_BLOCKS)
    })
}
