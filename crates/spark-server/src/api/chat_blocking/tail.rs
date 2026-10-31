// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

pub(crate) fn stop_match_corrected(fr: ir::FinishReason, stop_matched: bool) -> ir::FinishReason {
    if stop_matched && fr == ir::FinishReason::Length {
        ir::FinishReason::Stop
    } else {
        fr
    }
}

#[cfg(test)]
mod stop_match_corrected_tests {
    use super::stop_match_corrected;
    use crate::ir::{FINISH_REASON_TIMEOUT, FinishReason};

    #[test]
    fn matched_stop_corrects_length_to_stop() {
        assert_eq!(
            stop_match_corrected(FinishReason::Length, true),
            FinishReason::Stop
        );
    }

    #[test]
    fn everything_else_passes_through() {
        // No match ⇒ a real budget stop stays "length".
        assert_eq!(
            stop_match_corrected(FinishReason::Length, false),
            FinishReason::Length
        );
        // A match never rewrites tool_calls or the timeout contract.
        assert_eq!(
            stop_match_corrected(FinishReason::ToolCalls, true),
            FinishReason::ToolCalls
        );
        assert_eq!(
            stop_match_corrected(FinishReason::Other(FINISH_REASON_TIMEOUT.into()), true),
            FinishReason::Other(FINISH_REASON_TIMEOUT.into())
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        extract_hoisted_tool_calls, merge_hoisted_tool_calls, output_tokens_without_stop,
        split_at_first_think_end,
    };
    use crate::tool_parser::{FunctionCall, ToolCall};

    fn tool_call(id: &str, city: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: format!(r#"{{"city":"{city}"}}"#),
            },
        }
    }

    #[test]
    fn duplicate_call_across_reasoning_and_content_is_emitted_once() {
        let merged = merge_hoisted_tool_calls(
            vec![tool_call("reasoning", "Boston")],
            vec![tool_call("content", "Boston")],
        );

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].id, "content");
    }

    #[test]
    fn distinct_calls_across_reasoning_and_content_are_preserved() {
        let merged = merge_hoisted_tool_calls(
            vec![tool_call("reasoning", "Boston")],
            vec![tool_call("content", "Seattle")],
        );

        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn blocking_decode_excludes_terminal_stop_token() {
        let tokens = [10, 11, 24];

        assert_eq!(output_tokens_without_stop(&tokens, "stop"), &[10, 11]);
        assert_eq!(output_tokens_without_stop(&tokens, "length"), &tokens);
    }

    #[test]
    fn blocking_thinking_split_uses_first_end_token() {
        let tokens = [10, 19, 20, 19, 30];

        let split = split_at_first_think_end(&tokens, 19, true);

        assert_eq!(split, Some((&[10][..], &[20, 19, 30][..])));
        assert_eq!(split_at_first_think_end(&tokens, 19, false), None);
    }

    #[test]
    fn poolside_tool_call_in_reasoning_is_not_hoisted() {
        let reasoning = "plan <tool_call>write_file<arg_key>path</arg_key>\
            <arg_value>/tmp/x</arg_value></tool_call> more";

        let (preserved, calls) = extract_hoisted_tool_calls(Some(reasoning), Some("poolside_v1"));

        assert_eq!(preserved.as_deref(), Some(reasoning));
        assert!(calls.is_empty());
    }

    #[test]
    fn non_poolside_tool_call_in_reasoning_is_still_hoisted() {
        let reasoning =
            r#"plan <tool_call>{"name":"get_weather","arguments":{"city":"Boston"}}</tool_call>"#;

        let (_scrubbed, calls) = extract_hoisted_tool_calls(Some(reasoning), Some("hermes"));

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }
}
