// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for `sampling_setup`: tool-choice/parser coupling and the
//! grammar escape hatch. A sibling file (the `msg_entry_tests.rs`
//! pattern) so the parent stays under the 500-LoC cap.

#[cfg(test)]
mod tests {
    // `super::super::` — this file is a sibling module of `sampling_setup`,
    // so one `super` only reaches `sampling_setup_tests` itself.
    use super::super::{
        GrammarChoice, grammar_choice, preset_penalties_apply, tool_choice_required_for_parser,
        tool_grammar_escape_applies,
    };

    use crate::tool_parser::{ToolChoice, ToolChoiceFunction};

    /// Core requests inherit preset penalties only with the model's opt-in;
    /// tool-calling requests always keep the `[sampling.tools]` penalties.
    #[test]
    fn preset_penalties_follow_the_core_opt_in_but_always_apply_to_tools() {
        assert!(preset_penalties_apply(false, true));
        assert!(!preset_penalties_apply(false, false));
        assert!(preset_penalties_apply(true, true));
        assert!(preset_penalties_apply(true, false));
    }

    /// The escape hatch's own help text scopes it to `tool_choice="auto"`.
    #[test]
    fn the_escape_hatch_applies_in_auto_mode() {
        let required = tool_choice_required_for_parser(true, None, Some("qwen3_coder"));

        assert!(!required);
        assert!(tool_grammar_escape_applies(true, required));
    }

    /// `required` is *implemented by* the grammar, so the hatch must not
    /// remove it — the escape hatch is a model property, `required` is a
    /// request property, and the request wins.
    #[test]
    fn required_mode_keeps_the_grammar_despite_the_escape_hatch() {
        let choice = ToolChoice::Mode("required".to_string());
        let required = tool_choice_required_for_parser(true, Some(&choice), Some("qwen3_coder"));

        assert!(required);
        assert!(!tool_grammar_escape_applies(true, required));
    }

    #[test]
    fn specific_function_keeps_the_grammar_despite_the_escape_hatch() {
        let choice = ToolChoice::Specific {
            function: ToolChoiceFunction {
                name: "memory".to_string(),
            },
        };
        let required = tool_choice_required_for_parser(true, Some(&choice), Some("qwen3_coder"));

        assert!(required);
        assert!(!tool_grammar_escape_applies(true, required));
    }

    /// minimax_xml's grammar is the anti-corruption frame, not just a
    /// tool-choice enforcer, so the hatch must not remove it either.
    #[test]
    fn minimax_xml_keeps_the_grammar_despite_the_escape_hatch() {
        let required = tool_choice_required_for_parser(true, None, Some("minimax_xml"));

        assert!(required);
        assert!(!tool_grammar_escape_applies(true, required));
    }

    /// The fix must not switch the grammar ON for models that never asked
    /// for the hatch — with the hatch off, nothing about this changes.
    #[test]
    fn the_hatch_being_off_is_unaffected_by_tool_choice() {
        assert!(!tool_grammar_escape_applies(false, false));
        assert!(!tool_grammar_escape_applies(false, true));
    }

    #[test]
    fn bare_json_auto_uses_triggered_grammar() {
        assert!(!tool_choice_required_for_parser(
            true,
            None,
            Some("bare_json")
        ));
    }

    #[test]
    fn bare_json_required_mode_enforces_from_first_token() {
        let choice = ToolChoice::Mode("required".to_string());

        assert!(tool_choice_required_for_parser(
            true,
            Some(&choice),
            Some("bare_json")
        ));
    }

    #[test]
    fn specific_function_enforces_from_first_token() {
        let choice = ToolChoice::Specific {
            function: ToolChoiceFunction {
                name: "memory".to_string(),
            },
        };

        assert!(tool_choice_required_for_parser(
            true,
            Some(&choice),
            Some("bare_json")
        ));
    }

    #[test]
    fn minimax_xml_remains_parser_required() {
        assert!(tool_choice_required_for_parser(
            true,
            None,
            Some("minimax_xml")
        ));
    }

    #[test]
    fn inactive_tools_are_not_required() {
        assert!(!tool_choice_required_for_parser(
            false,
            None,
            Some("bare_json")
        ));
    }

    /// A tool-armed request gets the `<tool_call>` +3 bias unless the model
    /// turns tool-call steering off (MODEL.toml `[behavior]`).
    #[test]
    fn tool_call_steering_gates_the_tool_call_bias() {
        let dir = tempfile::tempdir().unwrap();
        let vocab = [("[UNK]".to_string(), 0), ("<tool_call>".to_string(), 1)];
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        tokenizers::Tokenizer::new(model)
            .save(dir.path().join("tokenizer.json"), false)
            .unwrap();
        let wire: crate::openai::ChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "fixture", "messages": [{"role": "user", "content": "weather"}],
                "tools": [{"type": "function", "function": {"name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}}}]
            }))
            .unwrap();
        let req: crate::ir::ChatRequest = wire.into();
        let bias = |steering: bool| {
            let tokenizer = crate::tokenizer::ChatTokenizer::from_model_dir(
                dir.path(),
                0,
                false,
                "qwen4_exp",
                "",
                None,
                false,
                false,
            )
            .unwrap();
            let mut state = crate::api::test_app_state::app_state(tokenizer);
            state.tool_call_start_token_id = Some(1);
            state.behavior.tool_call_steering = steering;
            let state = std::sync::Arc::new(state);
            let Ok(setup) = super::super::build_sampling(&state, &req, false, true, false, 0)
            else {
                panic!("sampling setup");
            };
            setup.logit_bias
        };
        assert_eq!(bias(true), [(1, 3.0)]);
        assert!(bias(false).is_empty());
    /// tools x response_format x tool_choice -> grammar. `none` names
    /// tool_choice="none"; `required` covers "required", a named function and
    /// a parser-required format.
    #[test]
    fn grammar_selection_matrix() {
        use GrammarChoice::*;
        // (tools, response_format, none, required, escape hatch) -> choice
        let cases = [
            // No tools.
            ((false, false, false, false, false), None),
            ((false, true, false, false, false), ResponseFormat),
            // Tools, no response_format: the tool grammar, as ever.
            ((true, false, false, false, false), ToolCall),
            ((true, false, true, false, false), ToolCall),
            ((true, false, false, true, false), ToolCall),
            // Both: auto/unset -> the union; none -> schema; required -> tools.
            ((true, true, false, false, false), ToolCallOrResponseFormat),
            ((true, true, true, false, false), ResponseFormat),
            ((true, true, false, true, false), ToolCall),
            // The escape hatch keeps auto mode unconstrained; it never
            // applies to required (`tool_grammar_escape_applies`).
            ((true, false, false, false, true), EscapeHatch),
            ((true, true, false, false, true), EscapeHatch),
            ((true, true, true, false, true), ResponseFormat),
        ];
        for ((tools, rf, none, required, hatch), want) in cases {
            assert_eq!(
                grammar_choice(tools, rf, none, required, hatch),
                want,
                "tools={tools} response_format={rf} none={none} required={required} hatch={hatch}"
            );
        }
    }

    /// The matrix's `required` input, from the wire: auto/unset and "none" are
    /// not required; "required" and a named function are.
    #[test]
    fn auto_and_none_are_not_required_but_required_and_named_are() {
        let named = ToolChoice::Specific {
            function: ToolChoiceFunction {
                name: "get_weather".to_string(),
            },
        };
        let mode = |m: &str| ToolChoice::Mode(m.to_string());
        let required = |tc: Option<&ToolChoice>| {
            tool_choice_required_for_parser(true, tc, Some("poolside_v1"))
        };

        assert!(!required(None));
        assert!(!required(Some(&mode("auto"))));
        assert!(!required(Some(&mode("none"))));
        assert!(required(Some(&mode("required"))));
        assert!(required(Some(&named)));
    }
}
