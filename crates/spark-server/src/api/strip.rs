// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

// Sibling-cluster items hoisted from the original `api.rs`. These uses
// give every sub-file access to helpers that the un-split file took for
// granted via single-module visibility.
use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::inference_impl::{extract_thinking, strip_stop_sequences, tokenize_stop_sequences};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};

// Re-export sibling helpers via crate::api::* for short paths.
use super::inference_types::*;
use super::sanitizer::*;

pub(crate) fn strip_thinking_tags(text: &str) -> String {
    let default_parser = crate::reasoning_parser::ReasoningFormat::Qwen.into_parser();
    extract_thinking(text, false, false, Some(&*default_parser)).1
}

/// Residual thinking-marker scrub for the assistant `content` channel.
///
/// The model's chat template opens `<think>` on the generation turn, and the
/// model closes it by emitting `</think>` (token id 19 for Laguna). It may emit
/// the close MORE THAN ONCE — observed live: `[19, …answer…, 19, …answer…]` for
/// "What is 17*23?", where the model declines to reason, closes immediately,
/// answers, then closes and answers again. The split only consumes the FIRST
/// close, so any later one lands in `content` verbatim.
///
/// The streaming path already removed these inline
/// (`chat_stream/handle_token.rs`); this is that logic hoisted so the blocking
/// path shares one implementation. Semantics are preserved exactly: drop every
/// occurrence and `trim_start` what follows, so `"…391.</think>17 × 23"` becomes
/// `"…391.17 × 23"` rather than leaving a ragged gap.
///
/// NOTE this only removes COMPLETE markers. Splitting reasoning from content is
/// the caller's job; this is defense-in-depth for markers that survive it.
pub(crate) fn scrub_think_markers(text: &str) -> String {
    const MARKERS: [&str; 5] = [
        "</think>",
        "</thinking>",
        "<thinking>",
        "</analysis>",
        "<analysis>",
    ];
    let mut out = text.to_string();
    for tag in MARKERS {
        while let Some(pos) = out.find(tag) {
            out = format!("{}{}", &out[..pos], out[pos + tag.len()..].trim_start());
        }
    }
    out
}

#[cfg(test)]
mod scrub_think_tests {
    use super::scrub_think_markers;

    #[test]
    fn removes_second_close_left_by_the_split() {
        // The exact live regression: model emits close, answers, closes, answers.
        assert_eq!(
            scrub_think_markers("17 × 23 = 391.</think>17 × 23 = 391."),
            "17 × 23 = 391.17 × 23 = 391."
        );
    }

    #[test]
    fn trims_whitespace_after_a_removed_marker() {
        assert_eq!(scrub_think_markers("</think>\n\n  hello"), "hello");
    }

    #[test]
    fn removes_every_occurrence_not_just_the_first() {
        assert_eq!(scrub_think_markers("a</think>b</think>c"), "abc");
    }

    #[test]
    fn leaves_ordinary_content_untouched() {
        let s = "Merge sort splits [38, 27] then merges. O(n log n).";
        assert_eq!(scrub_think_markers(s), s);
    }

    #[test]
    fn does_not_eat_partial_or_lookalike_markers() {
        // An incomplete marker is not a complete one; leave it for the caller.
        assert_eq!(
            scrub_think_markers("think about </thin"),
            "think about </thin"
        );
    }
}

/// Truncate assistant `content` at the first tool-call opener.
///
/// This model (and other tool-trained models) sometimes emit a
/// `<tool_call>…` block after a normal answer even when the request defined NO
/// tools — observed live: `…the story you keep telling yourself."<tool_call>catch_error({…})`.
/// When tools ARE active the tool parser extracts and strips these; when they
/// are NOT active nothing runs, so the raw markup lands in `content`.
///
/// `<tool_call>` (token id 25) and `<function=` are control markers, not prose,
/// so their presence in content is always spurious here. Everything from the
/// first one onward is the orphan tool block; cut it and trim. Apply ONLY when
/// no real tool call was produced (otherwise the tool parser already handled
/// the content).
///
/// `literal_xml`: the served model spells its tool envelope as added tokens
/// that also occur in prose (GLM-5.3, Qwen4_exp; `glm_tool_markup_as_text`).
/// There a bare opener is not enough: the cut needs call syntax after it, so
/// "The delimiter <tool_call> opens a tool invocation." stays text.
pub(crate) fn strip_orphan_tool_markup(text: &str, literal_xml: bool) -> String {
    strip_orphan_tool_markup_with_literal_state(
        text,
        &crate::literal_xml::LiteralXmlState::default(),
        literal_xml,
    )
}

/// Streaming variant of [`strip_orphan_tool_markup`]. The caller owns the
/// lexical state and feeds the accepted text through the normal sanitizer
/// afterward; this helper only classifies opener positions and never mutates
/// the state itself.
pub(crate) fn strip_orphan_tool_markup_with_literal_state(
    text: &str,
    literal_state: &crate::literal_xml::LiteralXmlState,
    literal_xml: bool,
) -> String {
    const OPENERS: [&str; 2] = ["<tool_call>", "<function="];
    // Trim ONLY when an orphan opener was actually found and cut. The no-match
    // path must pass the text through byte-identical: this function runs
    // per-delta on the streaming content path (handle_token.rs), and a bare
    // `.trim_end()` there silently deletes any delta that is entirely trailing
    // whitespace — e.g. Qwen's standalone `Ġ` space token before a digit
    // (" 7" -> ""), collapsing "ACK 7741-C" to "ACK7741-C" and "1.\n2.\n3." to
    // one line. The blocking path calls this once over the whole message where
    // the trim was harmless; per-delta it is destructive. Regression from
    // #473 (680b3a568). Pinned by orphan_tool_tests below.
    let mut state = *literal_state;
    let mut remaining = text;
    let mut offset = 0;
    while let Some((position, len)) =
        crate::literal_xml::first_marker_outside_literal(remaining, state, &OPENERS)
    {
        if !literal_xml || opens_call_syntax(&remaining[position..], len) {
            return text[..offset + position].trim_end().to_string();
        }
        let consumed = position + len;
        state.feed(&remaining[..consumed]);
        remaining = &remaining[consumed..];
        offset += consumed;
    }
    text.to_string()
}

/// Whether the opener (`len` bytes) starting `text` is followed by call
/// syntax: a function marker, JSON, or a name with arguments or a close.
fn opens_call_syntax(text: &str, len: usize) -> bool {
    let tail = text[len..].trim_start();
    let name_end = tail
        .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':')))
        .unwrap_or(tail.len());
    let after_name = tail[name_end..].trim_start();
    text.starts_with("<function=")
        || tail.starts_with("<function")
        || tail.starts_with("<invoke ")
        || tail.starts_with('{')
        || (name_end > 0
            && ["<arg_key>", "</tool_call>", "(", "{"]
                .iter()
                .any(|suffix| after_name.starts_with(suffix)))
}

/// Remove structural XML fragments left around a validated call while
/// preserving the same markers when they occur in quoted or fenced examples.
pub(crate) fn strip_tool_markup_preserving_literals(text: &str) -> String {
    let masked = crate::literal_xml::mask_literal_markers(
        text,
        &[
            "<tool_call>",
            "</tool_call>",
            "<function=",
            "<function ",
            "<|function=",
            "<|function ",
            "<parameter=",
            "</function>",
            "</parameter>",
        ],
    );
    let mut cleaned = masked;
    for tag in ["</parameter>", "</function>", "</tool_call>", "<tool_call>"] {
        cleaned = cleaned.replace(tag, "");
    }
    for opener in [
        "<function=",
        "<function ",
        "<|function=",
        "<|function ",
        "<parameter=",
    ] {
        while let Some(start) = cleaned.find(opener) {
            let end = cleaned[start..]
                .find('>')
                .map(|position| start + position + 1)
                .unwrap_or(cleaned.len());
            cleaned = format!("{}{}", &cleaned[..start], &cleaned[end..]);
        }
    }
    crate::literal_xml::unmask_literal_markers(&cleaned)
}

#[cfg(test)]
mod orphan_tool_tests {
    use super::strip_orphan_tool_markup;

    #[test]
    fn cuts_the_live_leak() {
        let s = "You are the story you keep telling yourself.\
                 <tool_call>catch_error({'error': {'message': \"nope\"}})";
        assert_eq!(
            strip_orphan_tool_markup(s, false),
            "You are the story you keep telling yourself."
        );
    }

    #[test]
    fn cuts_at_function_opener() {
        assert_eq!(
            strip_orphan_tool_markup("Here you go.<function=foo>{}</function>", false),
            "Here you go."
        );
    }

    #[test]
    fn leaves_quoted_and_fenced_markers_untouched() {
        let quoted = "The tag is `<tool_call>` and this is prose.";
        assert_eq!(strip_orphan_tool_markup(quoted, false), quoted);
        let fenced = "```xml\n<tool_call>\n```";
        assert_eq!(strip_orphan_tool_markup(fenced, false), fenced);
    }

    #[test]
    fn finds_real_opener_after_literal_marker() {
        let text = "The tag is `<tool_call>`; <tool_call>";
        assert_eq!(
            strip_orphan_tool_markup(text, false),
            "The tag is `<tool_call>`;"
        );
        let text = "The tag is `<tool_call>`; <tool_call>real</tool_call>";
        assert_eq!(
            strip_orphan_tool_markup(text, true),
            "The tag is `<tool_call>`;"
        );
    }

    /// Literal-XML carriers (GLM): a bare opener in prose stays text; the cut
    /// needs call syntax after it. Other models keep cutting at the opener.
    #[test]
    fn preserves_unquoted_bare_marker_and_later_prose() {
        let text = "The delimiter <tool_call> opens a tool invocation. END.";
        assert_eq!(strip_orphan_tool_markup(text, true), text);
        assert_eq!(strip_orphan_tool_markup(text, false), "The delimiter");
        assert_eq!(strip_orphan_tool_markup("<tool_call>", true), "<tool_call>");
        assert_eq!(
            strip_orphan_tool_markup(
                "Mention <tool_call> here; <tool_call>real</tool_call>",
                true
            ),
            "Mention <tool_call> here;"
        );
        for call in [
            "x<tool_call>get<arg_key>a</arg_key>",
            "x<tool_call>{\"name\":\"f\"}",
            "x<tool_call>f(1)",
            "x<function=f>",
        ] {
            assert_eq!(strip_orphan_tool_markup(call, true), "x", "{call}");
        }
    }

    #[test]
    fn structured_scrub_keeps_literal_xml_when_real_call_is_present() {
        let text = "The example is `<tool_call>`; <tool_call><function=real>\
                    <parameter=x>ok</parameter></function></tool_call>";
        assert_eq!(
            super::strip_tool_markup_preserving_literals(text),
            "The example is `<tool_call>`; ok"
        );
    }

    #[test]
    fn leaves_clean_content_untouched() {
        let s = "The sky is blue due to Rayleigh scattering.";
        assert_eq!(strip_orphan_tool_markup(s, false), s);
    }

    #[test]
    fn does_not_trip_on_the_word_function() {
        // Only the control markers "<tool_call>" / "<function=" cut; prose is safe.
        let s = "You can call a function to do that.";
        assert_eq!(strip_orphan_tool_markup(s, false), s);
    }

    /// PINS the streaming-detok space-drop bug (wt/repair-479).
    ///
    /// `chat_stream/handle_token.rs` applies this function PER STREAM DELTA
    /// on no-tools requests. Qwen's ByteLevel BPE pretokenizer (`\p{N}` with
    /// no leading-space alternative) emits the space before any digit as a
    /// standalone `Ġ` token, so the whole delta is `" "`. The unconditional
    /// `trim_end()` on the no-opener path deletes it, streaming
    /// `"ACK7741-C and7 sections"` where non-streaming decode yields
    /// `"ACK 7741-C and 7 sections"` (reproduced live 2026-08-15,
    /// Qwen3.6-35B-A3B-FP8, temp 0). Same mechanism eats standalone `ĊĊ`
    /// paragraph-break deltas and CJK inter-word `Ġ` deltas. When no opener
    /// is present the input must pass through byte-identical.
    ///
    /// EXPECTED TO FAIL until the no-match arm stops trimming.
    #[test]
    fn no_opener_passes_through_byte_identical_including_whitespace() {
        // The standalone space token before a digit — the live repro.
        assert_eq!(strip_orphan_tool_markup(" ", false), " ");
        // Standalone paragraph-break token (`ĊĊ`).
        assert_eq!(strip_orphan_tool_markup("\n\n", false), "\n\n");
        // A delta that merely ENDS in whitespace must keep it: the next
        // delta ("7", "3", …) concatenates directly after these bytes.
        assert_eq!(strip_orphan_tool_markup("found ", false), "found ");
    }
}
