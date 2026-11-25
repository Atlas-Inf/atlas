// SPDX-License-Identifier: AGPL-3.0-only

//! Ling-3.2 tool-call format (models: bailing_moe_v3_vl, Ling-3.2-air).
//!
//! Ported from SGLang's ling3 detector (Ling3Detector, a thin relaxation of
//! the GLM-4.5 XML detector). The tag layout is the Poolside v1 one
//! (tool_call / arg_key / arg_value, angle-bracket delimited), but the
//! model emits NEWLINES between the structural tags: the checkpoint chat
//! template renders multi-turn assistant calls with a newline after the
//! function name and between the key close and the value open, and its
//! tools header documents the same layout. The strict Poolside parser
//! requires adjacency (value open directly after the key close), so a
//! template-shaped Ling emission is prose to it; parse_ling3_call is the
//! whitespace-tolerant relaxation, which accepts every adjacent Poolside
//! emission as a side effect (the separators are optional everywhere).
//!
//! SGLang ling3 accepts, additionally (docstring + regexes):
//! - a literal backslash-n OR a real newline after the function name,
//! - the first arg-key opener immediately after the name (no separator),
//! - the call close immediately after the name: a zero-argument call, which
//!   is why this format sets promotes_bare_call_names() (Poolside-style
//!   bare-name zero-argument calls, not fabricated ones; see the trait doc).
//!
//! Divergences from the letter of SGLang, both inherited from Atlas's
//! shared poolside-family pipeline rather than reinvented here:
//! - values are kept as raw text and typed afterwards by coerce_all
//!   (SGLang types each value at parse time from the tool's JSON schema);
//! - inter-call and trailing prose is preserved as response content (SGLang
//!   keeps only the text before the first call).

use super::*;

/// Ling-3.2 format: Poolside tag layout with optional newline separators
/// inside a tool_call envelope.
///
/// The checkpoint chat template (jinja-templates/ling-3.2-air.jinja) renders
/// tool definitions itself, so unlike hermes/qwen3_coder this parser
/// injects no system prompt; Atlas's API layer skips empty system prompts.
pub struct Ling3Parser;

impl ToolCallParser for Ling3Parser {
    fn name(&self) -> &str {
        "ling3"
    }

    fn system_prompt(
        &self,
        _tools: &[ToolDefinition],
        _tool_choice: &ToolChoice,
        _levers: &PromptLevers,
    ) -> String {
        // The checkpoint's chat template owns the tools block: definitions,
        // the XML layout instruction, and the relevance guidance all render
        // from it. Injecting a second contract here doubles the block (the
        // ATLAS_NO_TOOL_INJECT experiment measures +1019 tokens) with no
        // behavioral gain.
        String::new()
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        // Byte-shape of the template's own multi-turn render: the opener
        // glued to the name, then a newline, the key open/close, a newline,
        // the value open/close per argument, and a newline before the
        // closer. A string-typed value renders raw, anything else as JSON -
        // matching the template's `{%- if v is string %}`.
        let mut output = String::new();
        for call in calls {
            output.push_str("<tool_call>");
            output.push_str(&call.function.name);
            if let Ok(serde_json::Value::Object(args)) =
                serde_json::from_str::<serde_json::Value>(&call.function.arguments)
            {
                for (key, value) in args {
                    output.push_str("\n<arg_key>");
                    output.push_str(&key);
                    output.push_str("</arg_key>");
                    output.push_str("\n<arg_value>");
                    match value {
                        serde_json::Value::String(value) => output.push_str(&value),
                        value => {
                            output.push_str(&serde_json::to_string(&value).unwrap_or_default())
                        }
                    }
                    output.push_str("</arg_value>");
                }
            }
            output.push_str("\n</tool_call>");
        }
        output
    }

    fn wants_typed_arguments(&self) -> bool {
        // Values render as free text between value tags, same as the
        // poolside family; coerce_all applies the tool schema afterwards.
        true
    }

    fn promotes_bare_call_names(&self) -> bool {
        // A zero-argument call is the bare name inside the envelope - a name
        // immediately closed by the call close, SGLang Ling3Detector's
        // "close a no-argument tool call immediately" case.
        true
    }

    fn leak_markers(&self) -> LeakMarkers {
        PoolsideV1Parser.leak_markers()
    }
}

/// Trim the separator noise Ling may place between structural tags: plain
/// whitespace and runs of the literal two characters backslash + n (SGLang
/// matches either; real newlines are covered by the whitespace trim).
pub(super) fn trim_ling3_separators(s: &str) -> &str {
    let mut out = s.trim();
    while let Some(stripped) = out.strip_suffix("\\n") {
        out = stripped.trim();
    }
    while let Some(stripped) = out.strip_prefix("\\n") {
        out = stripped.trim();
    }
    out
}

/// Parse a Ling-3.2 call body (envelope already stripped). Tolerates real
/// newlines and literal backslash-n between every structural tag. Duplicate
/// keys overwrite, as in SGLang's detect_and_parse (dict assignment), not
/// reject like the strict Poolside path.
pub(super) fn parse_ling3_call(text: &str) -> Option<ToolCall> {
    let name_end = text.find("<arg_key>").unwrap_or(text.len());
    let name = poolside_v1_name(trim_ling3_separators(&text[..name_end]))?;

    let mut args = serde_json::Map::new();
    let mut rest = trim_ling3_separators(&text[name_end..]);
    while let Some(after_key_open) = rest.strip_prefix("<arg_key>") {
        let key_end = after_key_open.find("</arg_key>")?;
        let key = trim_ling3_separators(&after_key_open[..key_end]);
        if key.is_empty() {
            return None;
        }
        let after_key = trim_ling3_separators(&after_key_open[key_end + "</arg_key>".len()..]);
        let after_value_open = after_key.strip_prefix("<arg_value>")?;
        // A value may contain any bytes; the first value close tag ends it.
        let value_end = after_value_open.find("</arg_value>")?;
        let raw_value = &after_value_open[..value_end];
        args.insert(
            key.to_string(),
            serde_json::Value::String(raw_value.to_string()),
        );
        rest = trim_ling3_separators(&after_value_open[value_end + "</arg_value>".len()..]);
    }
    // Anything left after the pairs is drifting prose inside the envelope;
    // SGLang's group-2 regex is equally permissive about inter-pair bytes.

    Some(ToolCall {
        id: next_tool_call_id(),
        call_type: "function".into(),
        function: FunctionCall {
            name,
            arguments: serde_json::to_string(&serde_json::Value::Object(args))
                .unwrap_or_else(|_| "{}".into()),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_of(call: &ToolCall) -> serde_json::Value {
        serde_json::from_str(&call.function.arguments).unwrap()
    }

    #[test]
    fn parses_newlined_template_shape() {
        let call = parse_ling3_call(
            "get_weather\n<arg_key>city</arg_key>\n<arg_value>Tokyo</arg_value>\n<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n",
        )
        .expect("ling3 call");
        assert_eq!(call.function.name, "get_weather");
        let args = args_of(&call);
        assert_eq!(args["city"], "Tokyo");
        assert_eq!(args["date"], "2024-06-27");
    }

    #[test]
    fn parses_glued_poolside_shape_without_newlines() {
        let call = parse_ling3_call("Bash<arg_key>command</arg_key><arg_value>pwd</arg_value>").expect("ling3 call");
        assert_eq!(call.function.name, "Bash");
        assert_eq!(args_of(&call)["command"], "pwd");
    }

    #[test]
    fn parses_literal_backslash_n_separators() {
        let call =
            parse_ling3_call("get_weather\\n<arg_key>city</arg_key>\\n<arg_value>Tokyo</arg_value>\\n").expect("ling3 call");
        assert_eq!(call.function.name, "get_weather");
        assert_eq!(args_of(&call)["city"], "Tokyo");
    }

    #[test]
    fn parses_name_glued_to_first_argument_tag() {
        let call = parse_ling3_call("ping<arg_key>host</arg_key><arg_value>localhost</arg_value>").expect("ling3 call");
        assert_eq!(call.function.name, "ping");
        assert_eq!(args_of(&call)["host"], "localhost");
    }

    #[test]
    fn duplicate_keys_overwrite_like_sglang() {
        let call = parse_ling3_call(
            "f\n<arg_key>k</arg_key>\n<arg_value>1</arg_value>\n<arg_key>k</arg_key>\n<arg_value>2</arg_value>",
        )
        .expect("ling3 call");
        assert_eq!(args_of(&call)["k"], "2");
    }

    #[test]
    fn zero_argument_body_is_just_the_name() {
        let call = parse_ling3_call("get_time").expect("ling3 call");
        assert_eq!(call.function.name, "get_time");
        assert_eq!(call.function.arguments, "{}");
    }

    #[test]
    fn name_must_be_a_tool_name_component() {
        assert!(parse_ling3_call("not a name!").is_none());
    }

    #[test]
    fn trim_keeps_names_and_keys_ending_in_n() {
        assert_eq!(trim_ling3_separators("open"), "open");
        assert_eq!(trim_ling3_separators(" open\n"), "open");
        assert_eq!(trim_ling3_separators("open\\n"), "open");
        assert_eq!(trim_ling3_separators("\n spawn\n"), "spawn");
    }

    #[test]
    fn values_keep_their_bytes_until_the_close_tag() {
        let call = parse_ling3_call(
            "edit\n<arg_key>content</arg_key>\n<arg_value>if (a < b) { return; }\n...and more</arg_value>\n",
        )
        .expect("ling3 call");
        assert_eq!(
            args_of(&call)["content"],
            "if (a < b) { return; }\n...and more"
        );
    }

    #[test]
    fn template_render_round_trips_through_the_parser() {
        let calls = [IncomingToolCall {
            id: None,
            function: IncomingFunction {
                name: "get_weather".into(),
                arguments: r#"{"city":"Tokyo","days":3}"#.into(),
            },
        }];
        let rendered = Ling3Parser.format_tool_calls(&calls);
        let body = &rendered["<tool_call>".len()..rendered.rfind("</tool_call>").unwrap()];
        let call = parse_ling3_call(body).expect("round trip");
        assert_eq!(call.function.name, "get_weather");
        let args = args_of(&call);
        assert_eq!(args["city"], "Tokyo");
        // Non-string template values render as JSON and come back as raw
        // text; wants_typed_arguments + coerce_all restore the type.
        assert_eq!(args["days"], "3");
    }

    #[test]
    fn envelope_scan_finds_tool_calls_in_newlined_output() {
        let raw =
            "Let me check.\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Tokyo</arg_value>\n</tool_call>";
        let (content, calls) = parse_tool_calls_promoting_bare_names(raw);
        assert_eq!(content.as_deref(), Some("Let me check."));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"city\":\"Tokyo\"}");
    }

    #[test]
    fn multiple_calls_in_separate_envelopes() {
        let raw =
            "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Tokyo</arg_value>\n</tool_call>\n<tool_call>get_time\n</tool_call>";
        let (_content, calls) = parse_tool_calls_promoting_bare_names(raw);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
        assert_eq!(calls[1].function.arguments, "{}");
    }
}
