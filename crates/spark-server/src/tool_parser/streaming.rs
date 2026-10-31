// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Buffers streaming text and detects `<tool_call>` tags in real-time.
/// Emits incremental tool call events as tokens arrive:
/// - `ToolCallStart` when the function name is extracted
/// - `ToolCallDelta` for each argument fragment
/// - `ToolCallEnd` when `</tool_call>` is found
pub struct StreamingToolDetector {
    pub(super) buffer: String,
    pub(super) inside_tag: bool,
    /// Exact opener retained so a non-call XML mention can be returned as
    /// content byte-for-byte, including namespaced/native opener variants.
    pub(super) current_tag_open: Option<String>,
    pub(super) inside_dsml: bool,
    /// Whether a bare identifier inside `<tool_call>` is a legitimate
    /// zero-argument call (Poolside v1 only — see
    /// `ToolCallParser::promotes_bare_call_names`). Default FALSE: every
    /// other format drops it as malformed, matching the blocking parser.
    pub(super) promote_bare_names: bool,
    pub(super) call_counter: u32,
    /// Track if any tool calls were emitted during process() to prevent
    /// flush() from re-emitting them (causes duplicate arguments in stream).
    pub(super) emitted_tool_calls: bool,
    /// For incremental streaming: name already emitted for current tool call.
    pub(super) current_tc_name: Option<String>,
    /// For incremental streaming: ID of the current in-progress tool call.
    pub(super) current_tc_id: Option<String>,
    /// Bytes of the in-progress tool call's body already consumed for
    /// incremental emission. For XML formats this is a byte offset into
    /// `buffer` past the last `</parameter>` already streamed; for JSON
    /// formats it is the number of argument-object bytes already streamed.
    pub(super) current_tc_emitted: usize,
    /// Tool schemas for per-parameter coercion (`coerce_all`) during live
    /// argument streaming. Empty when the detector was built without
    /// schemas (e.g. unit tests) — coercion is then a no-op and values
    /// stream as raw strings.
    pub(super) tools: Vec<ToolDefinition>,
    /// When true, restore the legacy buffer-until-`</tool_call>` behaviour
    /// (a single `ToolCallDelta` with the full args at close). Set from the
    /// `ATLAS_BUFFER_TOOL_ARGS` env kill-switch. Default false = live stream.
    pub(super) buffer_args: bool,
    /// Live-streaming per-call state: whether the opening `{` of the current
    /// XML tool call's argument object has been emitted yet.
    pub(super) args_open: bool,
    /// Live-streaming per-call state: parameter keys already streamed for the
    /// current XML tool call (so close-time backfill doesn't duplicate them).
    pub(super) emitted_keys: Vec<String>,
    /// True once any incremental `ToolCallArgsFragment` has been emitted for
    /// the current call — tells the close branch to emit only the residual
    /// (closing `}` / backfill / JSON tail) instead of the full args.
    pub(super) incremental_emitted: bool,
    /// Lexical context for content already emitted by the detector. XML-looking
    /// markers inside code or quotes remain content and update this state.
    pub(super) literal_xml_state: crate::literal_xml::LiteralXmlState,
}

pub enum DetectorOutput {
    /// Plain text content (not a tool call).
    Content(String),
    /// Complete tool call (used by flush/blocking path).
    ToolCall(ToolCall, usize),
    /// Incremental: tool call header (name + id). Emitted once when name is known.
    ToolCallStart {
        id: String,
        name: String,
        idx: usize,
    },
    /// Incremental: argument fragment. The detector emits this for the
    /// legacy/buffered/fallback paths carrying the FULL canonical args once
    /// at close — the handler runs backfill+coerce+validate on it.
    ToolCallDelta { args: String, idx: usize },
    /// Live-streaming: a ready-to-forward slice of `function.arguments`.
    /// The detector has already coerced (XML) or sliced (JSON) it, so the
    /// handler appends it verbatim to the accumulated args and emits it as
    /// an OpenAI `tool_calls[idx].function.arguments` fragment WITHOUT any
    /// further coercion/validation. Concatenating all fragments for a given
    /// `idx` yields the complete JSON arguments object.
    ToolCallArgsFragment { fragment: String, idx: usize },
    /// Incremental: tool call complete. `</tool_call>` seen.
    ToolCallEnd { idx: usize },
}

impl StreamingToolDetector {
    /// Unreleased raw tool syntax, for the API's bounded tool-call guard.
    pub fn pending_tool_bytes(&self) -> usize {
        if self.inside_tag || self.inside_dsml || self.current_tc_name.is_some() {
            self.buffer.len()
        } else {
            0
        }
    }

    pub(super) fn push_content(&mut self, outputs: &mut Vec<DetectorOutput>, text: String) {
        if text.is_empty() {
            return;
        }
        self.literal_xml_state.feed(&text);
        outputs.push(DetectorOutput::Content(text));
    }

    pub(super) fn marker_is_literal(&self, position: usize) -> bool {
        let mut state = self.literal_xml_state;
        state.feed(&self.buffer[..position]);
        state.is_literal()
    }

    pub(super) fn has_partial_tool_opener(&self) -> bool {
        // Whitespace is retained until an ambiguous opener is identified.
        // It must not hide the pending protocol prefix from role-word guards.
        let pending = self.buffer.trim_start();
        !pending.is_empty()
            && [
                "<tool_call>",
                "<|tool_call>",
                "<minimax:tool_call>",
                "<minimax:_call>",
                DSML_OPEN,
            ]
            .iter()
            .any(|tag| tag.starts_with(pending))
    }

    pub(super) fn find_tool_opener(&self) -> Option<(usize, usize)> {
        self.buffer
            .find("<tool_call>")
            .map(|p| (p, "<tool_call>".len()))
            .or_else(|| {
                self.buffer
                    .find("<|tool_call>")
                    .map(|p| (p, "<|tool_call>".len()))
            })
            .or_else(|| {
                self.buffer
                    .find("<minimax:tool_call>")
                    .map(|p| (p, "<minimax:tool_call>".len()))
            })
            .or_else(|| {
                self.buffer
                    .find("<minimax:_call>")
                    .map(|p| (p, "<minimax:_call>".len()))
            })
    }

    pub(super) fn is_registered_tool(&self, name: &str) -> bool {
        self.tools.is_empty() || self.tools.iter().any(|tool| tool.function.name == name)
    }
}

/// Extract function name from partial tool call buffer for incremental streaming.
/// Handles Poolside v1, Hermes JSON, Qwen3-Coder XML, Gemma-4, and Mistral native.
pub(super) fn extract_streaming_name(buffer: &str, poolside: bool) -> Option<String> {
    // The detector has already removed <tool_call>. In Poolside mode the
    // leading name is complete only when <arg_key> arrives. Never scan a
    // nested JSON value for a Hermes "name" field in this mode.
    if poolside {
        let end = buffer.find("<arg_key>")?;
        return poolside_v1_name(&buffer[..end]);
    }
    // Mistral native: [TOOL_CALLS]NAME[ARGS]
    if let Some(start) = buffer.find(MISTRAL_TOOL_CALLS_TAG) {
        let after = &buffer[start + MISTRAL_TOOL_CALLS_TAG.len()..];
        if let Some(end) = after.find(MISTRAL_ARGS_TAG) {
            let name = after[..end].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    // Gemma-4 native: call:NAME{
    if let Some(start) = buffer.find("call:") {
        let after = &buffer[start + 5..]; // len("call:") = 5
        if let Some(end) = after.find('{') {
            let name = after[..end].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    // Qwen3-Coder: <function=NAME>
    if let Some(start) = buffer.find("<function=") {
        let after = &buffer[start + "<function=".len()..];
        if let Some(end) = after.find(['>', '\n', '<']) {
            let mut name = after[..end].trim().to_string();
            // Sanitize: model may generate "Bash=bashash" or "Bash=Bash" at long context.
            if let Some(eq_pos) = name.find('=') {
                name = name[..eq_pos].trim().to_string();
            }
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    // Hermes JSON: "name":"X"
    if let Some(start) = buffer.find("\"name\"") {
        let after = &buffer[start + "\"name\"".len()..];
        // Skip optional whitespace and colon
        let after = after
            .trim_start()
            .strip_prefix(':')
            .unwrap_or(after)
            .trim_start();
        if let Some(after) = after.strip_prefix('"')
            && let Some(end) = after.find('"')
        {
            let name = &after[..end];
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

/// Find where arguments start in the buffer, after the name header.
/// Returns byte offset into buffer where argument content begins.
pub(super) fn find_args_start(buffer: &str) -> usize {
    // Gemma-4: after call:NAME{
    if let Some(pos) = buffer.find("call:")
        && let Some(brace) = buffer[pos..].find('{')
    {
        return pos + brace + 1;
    }
    // Qwen3-Coder: after <function=NAME>\n
    if let Some(pos) = buffer.find("<function=")
        && let Some(gt) = buffer[pos..].find('>')
    {
        let after_gt = pos + gt + 1;
        // Skip leading newline after >
        if after_gt < buffer.len() && buffer.as_bytes().get(after_gt) == Some(&b'\n') {
            return after_gt + 1;
        }
        return after_gt;
    }
    // Hermes JSON: after "arguments":
    if let Some(pos) = buffer.find("\"arguments\"") {
        let after = &buffer[pos + "\"arguments\"".len()..];
        let after = after.trim_start();
        if let Some(rest) = after.strip_prefix(':') {
            return buffer.len() - rest.len();
        }
    }
    buffer.len()
}

impl Default for StreamingToolDetector {
    fn default() -> Self {
        Self::new()
    }
}
