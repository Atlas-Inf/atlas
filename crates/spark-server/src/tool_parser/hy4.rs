// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
pub(super) struct Hy4Parser;
impl ToolCallParser for Hy4Parser {
    fn name(&self) -> &str {
        "hy4"
    }
    fn system_prompt(&self, _t: &[ToolDefinition], _c: &ToolChoice, _l: &PromptLevers) -> String {
        String::new()
    }
    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        let value_close = self
            .param_value_close_delim()
            .expect("hy4 declares a parameter-value close delimiter");
        Some(engine.compile_poolside_family_tool_grammar(
            tools,
            use_triggers,
            &crate::grammar::HY4_TAGS,
            value_close,
        ))
    }
    fn has_tool_grammar(&self) -> bool {
        true
    }
    fn param_value_close_delim(&self) -> Option<&'static str> {
        Some("</arg_value:opensource>")
    }
    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut text = PoolsideV1Parser.format_tool_calls(calls);
        for tag in ["tool_call", "arg_key", "arg_value"] {
            for slash in ["", "/"] {
                text = text.replace(
                    &format!("<{slash}{tag}>"),
                    &format!("<{slash}{tag}:opensource>"),
                );
            }
        }
        format!("<tool_calls:opensource>{text}</tool_calls:opensource>")
    }
    fn format_tool_response(&self, content: &str) -> String {
        format!("<tool_response:opensource>{content}</tool_response:opensource>")
    }
    fn wants_typed_arguments(&self) -> bool {
        true
    }
    fn promotes_bare_call_names(&self) -> bool {
        true
    }
    fn leak_markers(&self) -> LeakMarkers {
        // `canonical` strips the native `<tool_calls:opensource>` envelope
        // token, but a model that spells it with ordinary BPE pieces reaches
        // the stream as literal text no single decode window can strip.
        LeakMarkers {
            close: &[
                "</arg_key>",
                "</arg_value>",
                "</tool_call>",
                "<tool_calls:opensource>",
                "</tool_calls:opensource>",
            ],
            ..PoolsideV1Parser.leak_markers()
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::hy4::canonical;
    #[test]
    fn native_tools_parse_complete_empty_and_malformed_argument_blocks() {
        assert!(Hy4Parser.has_tool_grammar());
        for (body, count) in [
            ("ping", 1),
            (
                "get<arg_key:opensource>x</arg_key:opensource><arg_value:opensource>42</arg_value:opensource>",
                1,
            ),
            (
                "get<arg_key:opensource>x</arg_key:opensource><arg_value:opensource>42",
                0,
            ),
            ("", 0),
        ] {
            let raw = format!(
                "<tool_calls:opensource><tool_call:opensource>{body}</tool_call:opensource></tool_calls:opensource>"
            );
            let (_, calls) = parse_tool_calls_promoting_bare_names(&canonical(raw, true));
            assert_eq!(calls.len(), count, "{body}");
        }
    }
}
