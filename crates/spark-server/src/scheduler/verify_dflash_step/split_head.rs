// SPDX-License-Identifier: AGPL-3.0-only

//! Admission and exchange failures on the vocabulary-split verify head.
use super::*;

/// Greedy requests retain the raw argmax fast path. A sampled request needs
/// the split exchange protocol; refuse it before starting a verify rather
/// than silently sampling greedily.
pub(super) fn ensure_available(model: &dyn Model, a: &ActiveSeq) -> anyhow::Result<()> {
    anyhow::ensure!(
        !model.verify_logits_argmax_only() || a.temperature <= 0.0 || model.verify_split_sampling(),
        "split-head sampling requires the complete verify exchange"
    );
    Ok(())
}

/// A failed read after the forward leaves the worker protocol uncertain.
/// Abort the request; EP ranks follow the existing lifeline fatal-step policy.
pub(super) fn abort_exchange(
    model: &dyn Model,
    a: &mut ActiveSeq,
    operation: &str,
    error: &anyhow::Error,
) {
    let reason = format!(
        "verify {operation} failed (slot={}, seq_len={}): {error:#}",
        a.seq.slot_idx, a.seq.seq_len
    );
    a.abort_on_engine_error(&reason);
    if model.is_ep() {
        tracing::error!("{reason}; EP verdict/exchange state uncertain; terminating rank");
        crate::ep_peer_lifeline::terminate();
    }
}

#[cfg(test)]
#[path = "split_head_tests.rs"]
mod tests;
