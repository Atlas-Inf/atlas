// SPDX-License-Identifier: AGPL-3.0-only

use super::{DeltaVec, StreamDelta, StreamState, resolve_wire_finish_reason};

fn output_rejected(state: &StreamState) -> bool {
    state.guard_stop.is_some()
        || state.tool_loop_capped
        || (state.stop_string_triggered && !state.stop_string_matched)
}

/// Actual Done output finalization, separate from metrics/refund/dump I/O.
/// The callback is the existing detector/sanitizer flush body.
pub(super) fn finalize_done<'a>(
    state: &mut StreamState,
    finish_reason: &'a str,
    usage: crate::ir::Usage,
    return_token_ids: bool,
    flush: impl FnOnce(&mut StreamState) -> DeltaVec,
) -> (DeltaVec, &'a str) {
    // A genuine client stop may leave a known-safe pre-stop prefix in the
    // sanitizer; it is not a rejected-output guard. Keep that original flush.
    let mut deltas = if output_rejected(state) {
        Vec::new()
    } else {
        flush(state)
    };
    // Flush-time tool handlers can themselves trip a guard. Their queued
    // deltas have not reached the wire yet, so discard them as one batch.
    let rejected = output_rejected(state);
    if rejected {
        deltas.clear();
        state.pending_token_ids.clear();
    }
    let fr = resolve_wire_finish_reason(
        finish_reason,
        state.tool_loop_capped,
        state.detector.as_ref().is_some_and(|d| d.has_tool_calls()) || state.salvaged_tool_call,
        state.stop_string_matched,
        state.guard_stop,
    );

    // Refusal classification.
    let refusal_signal = if !rejected && state.detector.as_ref().is_none_or(|d| !d.has_tool_calls())
    {
        crate::refusal::detect(&state.refusal_scan_buf)
    } else {
        None
    };
    if let Some(ref r) = refusal_signal {
        deltas.push(StreamDelta::Refusal { text: r.clone() });
    }

    // Terminal delta: finish reason + usage. The `include_usage`
    // two-chunk framing (usage-only chunk before a usage-less finish
    // chunk) is the OpenAI encoder's decision
    // (`openai::delta_to_chunk_events`), not the core's. Residual
    // token ids ride Finish only for accepted output. IDs for guard-rejected
    // buffers were cleared above. IDs need not sum to completion usage: the
    // scheduler also counts EOS tokens that do not enter the token stream.
    deltas.push(StreamDelta::Finish {
        reason: crate::ir::FinishReason::from(fr),
        usage,
        token_ids: state.take_ids_if(return_token_ids),
    });

    (deltas, fr)
}

#[cfg(test)]
#[path = "done_finalization_tests.rs"]
mod tests;
