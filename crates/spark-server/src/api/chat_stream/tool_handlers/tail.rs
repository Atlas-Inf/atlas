// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Advance the same-name run counter for a completed call, returning the new
/// run length. Pure over the `Option` state so the #192 parallel fan-out
/// contract (N same-name calls with distinct args below the cap must NOT be
/// treated as a doom loop) is unit-testable without a `StreamState`.
pub(crate) fn advance_name_run(name_run: &mut Option<(String, u32)>, name: &str) -> u32 {
    let run_len = match name_run {
        Some((prev, n)) if prev == name => *n + 1,
        _ => 1,
    };
    *name_run = Some((name.to_string(), run_len));
    run_len
}

/// `DetectorOutput::ToolCallEnd` — F11 within-response dedup +
/// F44 cross-turn permanent-failure check + Bug-2 name-run cap.
///
/// Bug-2 cap (`MAX_CONSEC_SAME_NAME_CALLS`): trips when the same tool
/// name fires N times in a row regardless of args. F11 keys on
/// `(name, canonical_args)` and is defeated by runaway loops where
/// the model rolls a fresh timestamp / sequence number / id into the
/// payload each iteration; the F12 total cap (default 12) is the
/// only other server-side circuit, but a runaway can already have
/// flooded the SSE channel before F12 fires. The name-run cap is
/// strictly tighter than F11 and F12 for the runaway pattern.
///
/// A3 (2026-05-26): tightened from 6 → 3 to match opencode's
/// `DOOM_LOOP_THRESHOLD = 3`. Live Wave-1/3 traces showed the model
/// emitting 4-6 same-name bash calls with drifted args before any
/// guard tripped, by which point ~MB-long degenerate commands had
/// already flooded the stream and the .git/ artifact pollution was
/// already created. Three same-name calls is the empirical threshold
/// at which opencode itself bails to the user for permission. Atlas
/// matching this means we end the response slightly before opencode
/// would surrender, giving the outer retry loop a clean signal.
///
/// #192 (2026-07-02): relaxed 3 → 8. A3's premise predates parallel tool
/// calls: back then one response could not legitimately contain 3 same-name
/// calls (generation hard-stopped at the first `</tool_call>`), so a run of
/// 3 was proof of degeneration. Post-#192 the same-name run IS the designed
/// shape of a parallel fan-out (BFCL `parallel`: get_weather × 3 cities —
/// live 2026-07-02 the cap flipped that clean turn to finish="length").
/// 8 mirrors the scheduler's own parallel bound
/// (`MAX_POST_COMPLETION_TOOL_OPENS = 8`, decode_logits_step.rs): a real
/// runaway still trips it an order of magnitude below the F12 total cap,
/// while any plausible legit fan-out stays under it. Identical-args loops
/// are still caught earlier by the F11 within-response dedup.
pub(crate) const MAX_CONSEC_SAME_NAME_CALLS: u32 = 8;

pub(crate) fn handle_tool_call_end(
    state: &mut StreamState,
    ctx: &StreamCtx,
    idx: usize,
    deltas: &mut DeltaVec,
) {
    if state.guard_stop == Some(atlas_tool_guard::GUARD_STOP) {
        return;
    }
    if let Some((name, args_json)) = state.streaming_tool_args.remove(&idx) {
        if super::super::tool_security::check(state, ctx, &args_json) {
            return;
        }
        if !ctx.state.tool_guard.enabled {
            flush_buffered_tool_chunks(state, idx, deltas);
        }
        let reference = ctx.state.reference_mode;
        if !reference && state.tool_arg_dedup_within.check(&name, &args_json) {
            tracing::warn!(
                tool = %name,
                "F11 within-response dedup tripped: 2+ identical streaming tool calls; ending response"
            );
            state.stop_string_triggered = true;
            state.tool_loop_capped = true;
        }
        let run_len = advance_name_run(&mut state.name_run, &name);
        if run_len >= MAX_CONSEC_SAME_NAME_CALLS && !state.stop_string_triggered && !reference {
            tracing::warn!(
                tool = %name,
                run = run_len,
                "Bug-2 name-run cap tripped: {run_len} successive `{name}` tool calls; ending response (F11 missed because args drift)"
            );
            state.stop_string_triggered = true;
            state.tool_loop_capped = true;
        }
        if ctx.state.tool_guard.enabled {
            if state.stop_string_triggered {
                drop_buffered_tool_chunks(state, idx);
                state
                    .cancel_flag
                    .store(true, std::sync::atomic::Ordering::Release);
            } else {
                flush_buffered_tool_chunks(state, idx, deltas);
            }
        }
        if !state.stop_string_triggered {
            // Successful streaming tool call — log + metric to match the
            // blocking and complete-call paths.
            let preview: String = args_json.chars().take(120).collect();
            let s = if args_json.len() > preview.len() {
                "…"
            } else {
                ""
            };
            tracing::info!("Tool call: {name}({preview}{s})");
            crate::metrics::TOOL_CALLS_TOTAL.inc();
        }
    }
}

#[cfg(test)]
mod name_run_cap_tests {
    //! #192: the Bug-2 same-name run cap vs parallel fan-outs. A BFCL
    //! `parallel`-shape response (get_weather x 3 cities) must NOT be
    //! classified as a doom loop (live 2026-07-02: the old cap of 3 flipped a
    //! clean 3-call hermes turn to finish_reason="length"); a genuine
    //! same-name runaway must still trip the cap before the F12 total cap.
    use super::super::{MAX_CONSEC_SAME_NAME_CALLS, advance_name_run};

    #[test]
    fn three_call_parallel_fanout_stays_under_cap() {
        let mut run = None;
        for i in 1..=3u32 {
            let len = advance_name_run(&mut run, "get_weather");
            assert_eq!(len, i);
            assert!(
                len < MAX_CONSEC_SAME_NAME_CALLS,
                "a 3-call same-name parallel fan-out must not trip the doom-loop cap"
            );
        }
    }

    #[test]
    fn runaway_same_name_run_still_trips() {
        let mut run = None;
        let mut tripped_at = None;
        for i in 1..=12u32 {
            if advance_name_run(&mut run, "bash") >= MAX_CONSEC_SAME_NAME_CALLS {
                tripped_at = Some(i);
                break;
            }
        }
        assert_eq!(
            tripped_at,
            Some(MAX_CONSEC_SAME_NAME_CALLS),
            "cap must fire below the F12 total cap (12)"
        );
    }

    #[test]
    fn different_name_resets_run() {
        let mut run = None;
        assert_eq!(advance_name_run(&mut run, "get_weather"), 1);
        assert_eq!(advance_name_run(&mut run, "get_weather"), 2);
        assert_eq!(advance_name_run(&mut run, "get_time"), 1);
        assert_eq!(advance_name_run(&mut run, "get_weather"), 1);
    }
}
