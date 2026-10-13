// SPDX-License-Identifier: AGPL-3.0-only

//! GLM repair-policy helpers for the MTP step: the narrow-lane predicate and
//! the deferred terminal-error marker.

use super::*;

/// Mark an MTP failure for the normal retirement pass. Sending the error
/// immediately would free the live owner while it is still present in the
/// active set; leaving only `finished=true` would instead synthesize a
/// successful `stop` response and cache the partial prefix. `finish_sequence`
/// consumes this marker and uses the terminal error/free path exactly once.
pub(super) fn mark_engine_error(a: &mut ActiveSeq, error: impl Into<String>) {
    a.engine_error = Some(error.into());
    a.finished = true;
}

/// Repair owns an explicit verdict on both ranks; legacy K2/K3 have no record hook.
pub(super) fn glm_repaired_narrow(num_drafts: usize) -> bool {
    matches!(num_drafts, 1 | 2) && spark_model::speculative::glm_repair_policy::enabled()
}
