// SPDX-License-Identifier: AGPL-3.0-only
//! Drain a token's detector-output events into delta events (extracted;
//! identical content). Returns `true` when the caller must return `deltas`
//! immediately (a guard cancelled the response or a stop string fired).

use super::*;

pub(super) fn drain_detector_outputs(
    state: &mut StreamState,
    ctx: &StreamCtx,
    outputs: Vec<tool_parser::DetectorOutput>,
    deltas: &mut DeltaVec,
) -> bool {
    for output in outputs {
        match output {
            tool_parser::DetectorOutput::Content(text) => {
                if let Some(events_out) = detector_content_arm(state, ctx, &text) {
                    deltas.extend(events_out);
                    // One token can yield text before, inside and after a
                    // literal marker (Enntity/atlas fbda04d1); drain every
                    // event unless a guard actually cancelled this response.
                    if state.stop_string_triggered {
                        return true;
                    }
                }
            }
            tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                handle_complete_tool_call(state, ctx, &mut tc, tc_idx, deltas);
            }
            tool_parser::DetectorOutput::ToolCallStart {
                id: tc_id,
                name,
                idx,
            } => {
                handle_tool_call_start(state, ctx, tc_id, name, idx, deltas);
            }
            tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                handle_tool_call_delta(state, ctx, args, idx, deltas);
            }
            tool_parser::DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                handle_tool_call_args_fragment(state, ctx, fragment, idx, deltas);
            }
            tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                handle_tool_call_end(state, ctx, idx, deltas);
            }
        }
        if state.guard_stop == Some(atlas_tool_guard::GUARD_STOP) {
            deltas.clear();
            return true;
        }
    }
    false
}

/// Emit the pre-stop prefix once on a genuine client stop match, then end
/// the token (identical content; extracted). Returns `true` when the
/// caller must return `deltas` immediately.
pub(super) fn finish_on_stop_string(
    state: &mut StreamState,
    ctx: &StreamCtx,
    delta: &str,
    deltas: &mut DeltaVec,
) -> bool {
    if !state.stop_string_triggered {
        return false;
    }
    // Only the current token's genuine client stop match can reach this
    // branch. Emit its pre-stop prefix once, preserving markup sanitization.
    // Tokens queued AFTER a guard or stop match are rejected at handle_token
    // entry, before detokenization, content accumulation or token-ID tracking.
    if !delta.is_empty() {
        let cleaned = sanitize_content_chunk_with_literal_state(
            delta,
            &mut state.tag_scan_buf,
            &mut state.suppressing_param_leak,
            &mut state.inside_envelope,
            &mut state.literal_xml_state,
            &ctx.leak_markers,
        );
        if !cleaned.is_empty() {
            deltas.push(StreamDelta::Content {
                text: cleaned,
                token_ids: state.take_ids_if(ctx.req_return_token_ids),
            });
        }
    }
    true
}
