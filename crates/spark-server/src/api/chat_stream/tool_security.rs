// SPDX-License-Identifier: AGPL-3.0-only

// Ported from Enntity/atlas fa5331b2 et seq. (AGPL-3.0-only).
use super::{ctx::StreamCtx, state::StreamState};
use atlas_tool_guard::BlockReason;

pub(super) fn reject(state: &mut StreamState, reason: BlockReason) {
    if state.guard_stop == Some(atlas_tool_guard::GUARD_STOP) {
        return;
    }
    tracing::warn!(reason = reason.code(), "tool-call guard blocked a response");
    state.buffered_tool_chunks.clear();
    state.streaming_tool_args.clear();
    state.pending_token_ids.clear();
    state.guard_stop = Some(atlas_tool_guard::GUARD_STOP);
    state.stop_string_triggered = true;
    state
        .cancel_flag
        .store(true, std::sync::atomic::Ordering::Release);
}

pub(super) fn check(state: &mut StreamState, ctx: &StreamCtx, arguments: &str) -> bool {
    if let Err(reason) = ctx.state.tool_guard.inspect(arguments) {
        reject(state, reason);
        true
    } else {
        false
    }
}

pub(super) fn check_size(state: &mut StreamState, ctx: &StreamCtx, additional: usize) -> bool {
    if !ctx.state.tool_guard.enabled {
        return false;
    }
    let bytes = state
        .streaming_tool_args
        .values()
        .fold(additional, |n, (name, args)| {
            n.saturating_add(name.len()).saturating_add(args.len())
        });
    if bytes > ctx.state.tool_guard.max_bytes {
        reject(state, BlockReason::ArgumentsTooLarge);
        true
    } else {
        false
    }
}

/// Unreleased raw tool syntax that `additional` more bytes would take past the
/// guard's bound: refuse it before the detector buffers more.
pub(super) fn check_pending(state: &mut StreamState, ctx: &StreamCtx, additional: usize) -> bool {
    let bound = ctx.state.tool_guard.max_bytes;
    let over = ctx.state.tool_guard.enabled
        && state.detector.as_ref().is_some_and(|d| {
            d.pending_tool_bytes() > 0 && d.pending_tool_bytes().saturating_add(additional) > bound
        });
    if over {
        reject(state, BlockReason::ArgumentsTooLarge);
    }
    over
}
