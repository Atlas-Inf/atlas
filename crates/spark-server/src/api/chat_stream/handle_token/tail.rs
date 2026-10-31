// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

pub(crate) fn process_detector_content(
    state: &mut StreamState,
    ctx: &StreamCtx,
    sanitized_or_raw: &str,
) -> Option<DeltaVec> {
    // From the detector-active branch the input is the Content(text)
    // payload that still needs sanitization. From the no-detector
    // branch the input is already sanitized. Distinguish via a thin
    // wrapper: detector branch ALSO sanitizes; non-detector branch
    // skips by passing the already-sanitized text. To keep the call
    // site simple, we sanitize here only when the input contains the
    // hallmark of an unfiltered Content payload — which we can't
    // reliably detect. Solution: split into two paths.
    //
    // Inlining: this helper is only called once per branch with the
    // correct input type; it never re-sanitizes. The parameter is the
    // post-sanitizer text in both call sites.
    let sanitized = sanitized_or_raw;

    // F4 SimHash guard — fence-aware sentence-loop state machine
    // (`simhash_step` owns the rules; this call site only threads the
    // per-stream state through). Armed by the model's
    // `enable_stream_loop_guards`, past the min_tokens floor.
    let guards_armed = ctx.state.chat.stream_loop_guards
        && crate::api::stream_guards::watchdog_floor_reached(
            state.generated_tokens,
            ctx.min_tokens,
        );
    let semantic_trip = if guards_armed && !state.loop_watchdog_triggered {
        crate::loop_simhash::simhash_step(
            &mut state.simhash_pending,
            &mut state.simhash_in_fence,
            &mut state.simhash_fence_scan,
            &mut state.simhash_guard,
            sanitized,
        )
    } else {
        false
    };

    let token_trip = guards_armed
        && check_loop_watchdog(
            sanitized,
            &mut state.loop_scan_buf,
            state.loop_watchdog_triggered,
        );

    if (semantic_trip || token_trip) && !ctx.state.reference_mode {
        if semantic_trip {
            tracing::warn!(
                ring_len = state.simhash_guard.len(),
                "SimHash semantic-loop watchdog fired (paraphrased sentence repeat)"
            );
        }
        state.loop_watchdog_triggered = true;
        state.stop_string_triggered = true;
        state.guard_stop = Some(if semantic_trip {
            "simhash_semantic_loop"
        } else {
            "token_loop_watchdog"
        });
        state
            .cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);

        // Watchdog fired: short-circuit the stream with no further
        // content. The model emitted a degenerate loop; we end the
        // response here rather than salvaging a synthetic tool call.
        return Some(DeltaVec::new());
    }

    if !sanitized.is_empty() {
        if state.refusal_scan_buf.len() < 16_384 {
            state.refusal_scan_buf.push_str(sanitized);
        }
        let out: DeltaVec = vec![StreamDelta::Content {
            text: sanitized.to_string(),
            token_ids: state.take_ids_if(ctx.req_return_token_ids),
        }];
        return Some(out);
    }
    None
}

/// Detector-active branch's `Content(text)` arm: sanitize first,
/// then run the shared semantic/token watchdog + emit pipeline.
pub(crate) fn detector_content_arm(
    state: &mut StreamState,
    ctx: &StreamCtx,
    text: &str,
) -> Option<DeltaVec> {
    let sanitized = sanitize_content_chunk_with_literal_state(
        text,
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &mut state.inside_envelope,
        &mut state.literal_xml_state,
        &ctx.leak_markers,
    );
    process_detector_content(state, ctx, &sanitized)
}

/// Pure stop-string accumulator + hold-back algorithm. Returns the
/// bytes that should be forwarded to the client this delta; the
/// remainder (≤ `buffer_len` bytes) stays withheld inside
/// `accumulated_content` until the next call or until `handle_done`
/// flushes the tail at stream close.
///
/// Mirrors vLLM's `IncrementalDetokenizer.update`
/// (`vllm/v1/engine/detokenizer.py`):
/// 1. Append `new_chars` to the accumulator.
/// 2. Search the accumulator for any stop string.
/// 3a. On hit, truncate the accumulator AND the emittable delta at
///     the match position (Atlas never echoes the stop literal).
/// 3b. On miss, hold back the last `buffer_len` bytes; emit
///     everything between the previously emitted offset and the
///     hold-back boundary, snapped to a valid UTF-8 char boundary.
///
/// Pre/postconditions:
/// - `*triggered` must be `false` on entry (callers gate on this).
/// - On match, `*triggered` is flipped to `true` and the accumulator
///   is truncated to the prefix that precedes the stop string.
/// - On miss, `*triggered` stays `false`.
pub(crate) fn apply_stop_string_holdback(
    new_chars: &str,
    stop_strings: &[String],
    buffer_len: usize,
    accumulated_content: &mut String,
    emitted_len: &mut usize,
    triggered: &mut bool,
) -> String {
    debug_assert!(!*triggered, "caller must gate on !triggered");
    accumulated_content.push_str(new_chars);

    // Bounded search window: only the suffix that could contain a stop
    // string straddling the newly appended chars can hold a *new* match —
    // every prior call already full-scanned (and found nothing in) the
    // content before this window. A match can begin at earliest
    // `max_stop_len - 1` bytes before the new chars; we also back up over
    // the held-back `buffer_len` bytes for margin. This keeps per-token
    // cost O(new + buffer + max_stop) instead of O(total), turning the
    // whole-response scan from O(n²) into O(n).
    let max_stop_len = stop_strings.iter().map(String::len).max().unwrap_or(0);
    let search_start = {
        let raw = accumulated_content
            .len()
            .saturating_sub(new_chars.len() + buffer_len + max_stop_len);
        accumulated_content.floor_char_boundary(raw)
    };
    let matched_pos = stop_strings
        .iter()
        .filter_map(|s| accumulated_content[search_start..].find(s.as_str()))
        .min()
        .map(|rel| rel + search_start);

    if let Some(pos) = matched_pos {
        accumulated_content.truncate(pos);
        let emit_start = (*emitted_len).min(pos);
        let out = accumulated_content[emit_start..pos].to_string();
        *emitted_len = pos;
        *triggered = true;
        return out;
    }

    // No match: hold back the last `buffer_len` bytes. Snap to a UTF-8
    // boundary so the emitted prefix is always valid Rust `str` and
    // the held-back tail never contains a partial codepoint.
    let acc_len = accumulated_content.len();
    let raw_emit_end = acc_len.saturating_sub(buffer_len);
    let emit_end = accumulated_content.floor_char_boundary(raw_emit_end);
    let emit_start = (*emitted_len).min(emit_end);
    let out = accumulated_content[emit_start..emit_end].to_string();
    *emitted_len = emit_end;
    out
}

#[cfg(test)]
mod stop_string_holdback_tests {
    use super::super::apply_stop_string_holdback;

    /// Stop string spanning a chunk boundary must not leak the
    /// partial prefix in the first delta. When the suffix arrives in
    /// the next chunk the full output up to (but excluding) the stop
    /// string is emitted; the stop literal itself is consumed.
    #[test]
    fn stop_string_spanning_chunk_boundary_does_not_leak() {
        let stops = vec!["<|im_start|>".to_string()];
        let buffer_len = "<|im_start|>".len() - 1; // 11
        let mut acc = String::new();
        let mut emitted = 0usize;
        let mut triggered = false;

        // Delta 1: "hello " — entirely inside the hold-back window
        // (6 bytes < buffer_len=11). Nothing emitted.
        let out = apply_stop_string_holdback(
            "hello ",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "");
        assert_eq!(acc, "hello ");
        assert!(!triggered);

        // Delta 2: "<|im_st" — partial stop string. acc="hello <|im_st"
        // (len=13). raw_emit_end=13-11=2, so we emit "he".
        // Crucially, "<|im_st" is HELD BACK — never sent to client.
        let out = apply_stop_string_holdback(
            "<|im_st",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "he");
        assert!(!out.contains("<|im_st"), "partial stop leaked to client");
        assert!(!triggered);

        // Delta 3: "art|>" completes the stop string. We match at
        // pos=6, truncate acc to "hello ", and emit "llo " (bytes
        // 2..6 of acc).
        let out = apply_stop_string_holdback(
            "art|>",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "llo ");
        assert_eq!(acc, "hello ");
        assert!(triggered);

        // Concatenating all emitted deltas yields the pre-stop output
        // ("hello ") with the stop literal consumed. No partial leak.
        let total = String::new() + "" + "he" + "llo ";
        assert_eq!(total, "hello ");
        assert!(!total.contains("<|im_st"));
        assert!(!total.contains("<|im_start|>"));
    }

    /// When no stop strings are configured, `buffer_len=0` and the
    /// hold-back collapses to a pass-through: every byte of every
    /// delta is emitted immediately.
    #[test]
    fn no_stop_strings_is_zero_behavior_change() {
        let stops: Vec<String> = Vec::new();
        let buffer_len = 0usize;
        let mut acc = String::new();
        let mut emitted = 0usize;
        let mut triggered = false;

        let out = apply_stop_string_holdback(
            "hello ",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "hello ");
        assert!(!triggered);

        let out = apply_stop_string_holdback(
            "world",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "world");
        assert!(!triggered);

        // Even a string that LOOKS like a stop marker is forwarded
        // verbatim because no stop strings are configured.
        let out = apply_stop_string_holdback(
            "<|im_start|>",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "<|im_start|>");
        assert!(!triggered);
    }

    /// Multi-byte UTF-8 inside the hold-back window must never be
    /// sliced mid-codepoint. `floor_char_boundary` snaps the cut to a
    /// valid boundary so the emitted prefix is always valid `str`.
    #[test]
    fn utf8_boundary_safety_in_holdback() {
        // "é" is 2 bytes (0xC3 0xA9). Build an accumulator whose
        // raw cut would land inside the codepoint and verify
        // floor_char_boundary saves us.
        let stops = vec!["STOP".to_string()];
        let buffer_len = 3usize; // > 0 to exercise the hold-back
        let mut acc = String::new();
        let mut emitted = 0usize;
        let mut triggered = false;

        // acc becomes "aébc" (5 bytes). raw_emit_end = 5-3 = 2 lands
        // mid-codepoint of 'é' (1..3). floor_char_boundary snaps to
        // 1, so we emit "a" only.
        let out = apply_stop_string_holdback(
            "aébc",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "a");
        assert!(out.is_char_boundary(out.len()));
        assert!(!triggered);
    }
}

#[cfg(test)]
mod role_literal_strip_tests {
    use super::super::strip_bare_role_literal;
    use crate::tool_parser::{DetectorOutput, StreamingToolDetector};

    /// Faithfully mirror the `handle_token` content-phase pipeline for the
    /// two steps under test: the bare role-literal strip (guarded by the
    /// detector's `inside_tool_call()`) feeding the surviving delta into the
    /// streaming tool-call detector. Returns every `ToolCallStart` name.
    fn stream_names(chunks: &[&str]) -> Vec<String> {
        let mut det = StreamingToolDetector::new();
        let mut names = Vec::new();
        for &c in chunks {
            let mut delta = c.to_string();
            // Same call the real content phase makes, in the same order:
            // read the detector's in-body flag BEFORE feeding this delta.
            let inside_tool_call = det.inside_tool_call();
            strip_bare_role_literal(&mut delta, inside_tool_call);
            if delta.is_empty() {
                continue;
            }
            for o in det.process(&delta) {
                if let DetectorOutput::ToolCallStart { name, .. } = o {
                    names.push(name);
                }
            }
        }
        for o in det.flush() {
            if let DetectorOutput::ToolCallStart { name, .. } = o {
                names.push(name);
            }
        }
        names
    }

    /// Issue #222: a `tool_search` NAME whose leading `tool` arrives as a
    /// standalone BPE fragment must stream intact, not truncate to `_search`.
    #[test]
    fn tool_search_name_split_after_tool_streams_intact() {
        let names = stream_names(&[
            "<tool_call>\n{\"name\": \"",
            "tool", // standalone BPE fragment of the NAME
            "_search\", \"arguments\": {\"query\": \"CRM\"}}",
            "\n</tool_call>",
        ]);
        assert_eq!(names, vec!["tool_search".to_string()]);
    }

    /// `tool_call` — the name that collides most directly with the markup.
    #[test]
    fn tool_call_name_split_after_tool_streams_intact() {
        let names = stream_names(&[
            "<tool_call>\n{\"name\": \"",
            "tool",
            "_call\", \"arguments\": {}}",
            "\n</tool_call>",
        ]);
        assert_eq!(names, vec!["tool_call".to_string()]);
    }

    /// `tool_describe` — same class, different suffix.
    #[test]
    fn tool_describe_name_split_after_tool_streams_intact() {
        let names = stream_names(&[
            "<tool_call>\n{\"name\": \"",
            "tool",
            "_describe\", \"arguments\": {\"id\": 7}}",
            "\n</tool_call>",
        ]);
        assert_eq!(names, vec!["tool_describe".to_string()]);
    }

    /// Non-`tool_*` name fed in fragments must remain intact (control).
    #[test]
    fn ordinary_name_streams_intact() {
        let names = stream_names(&[
            "<tool_call>\n{\"name\": \"get",
            "_weather\", \"arguments\": {\"city\": \"NYC\"}}",
            "\n</tool_call>",
        ]);
        assert_eq!(names, vec!["get_weather".to_string()]);
    }

    /// The #204/#205/#206 leak suppression the guard must NOT regress: a bare
    /// role literal in genuine content (no tool call in flight) is still
    /// cleared. Verified for all three literals and for a `tool` fragment
    /// that is only a real leak when `inside_tool_call == false`.
    #[test]
    fn bare_role_literal_still_stripped_outside_tool_call() {
        for lit in ["user", "assistant", "tool", "  tool  "] {
            let mut d = lit.to_string();
            strip_bare_role_literal(&mut d, false);
            assert!(
                d.is_empty(),
                "bare role literal {lit:?} must be stripped in content"
            );
        }
    }

    /// Inside a tool-call body the same fragments must survive untouched.
    #[test]
    fn bare_role_literal_preserved_inside_tool_call() {
        for lit in ["user", "assistant", "tool"] {
            let mut d = lit.to_string();
            strip_bare_role_literal(&mut d, true);
            assert_eq!(d, lit, "fragment {lit:?} must survive inside a tool call");
        }
    }

    #[test]
    fn dsml_split_opener_preserves_tool_fragment_and_streams_call() {
        let mut det = StreamingToolDetector::new();
        let chunks = [
            "\n\n",
            "<｜DSM",
            "L",
            "｜",
            "tool",
            "_calls>\n<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        ];
        let mut outputs = Vec::new();
        for chunk in chunks {
            let mut delta = chunk.to_string();
            strip_bare_role_literal(&mut delta, det.inside_tool_call());
            if !delta.is_empty() {
                outputs.extend(det.process(&delta));
            }
        }
        outputs.extend(det.flush());

        let mut content = String::new();
        let mut calls = Vec::new();
        for output in outputs {
            match output {
                DetectorOutput::Content(text) => content.push_str(&text),
                DetectorOutput::ToolCall(call, index) => calls.push((index, call)),
                _ => {}
            }
        }
        assert!(content.is_empty(), "DSML envelope leaked: {content:?}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, 0);
        assert_eq!(calls[0].1.function.name, "get_weather");
        assert_eq!(calls[0].1.function.arguments, r#"{"city":"Paris"}"#);
    }

    /// Ordinary content (not a bare role literal) is never touched, in or out
    /// of a tool call.
    #[test]
    fn ordinary_content_untouched() {
        for inside in [false, true] {
            let mut d = "the tool ran".to_string();
            strip_bare_role_literal(&mut d, inside);
            assert_eq!(d, "the tool ran");
        }
    }
}
