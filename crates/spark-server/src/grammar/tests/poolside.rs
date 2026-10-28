// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use xgrammar::{CompiledGrammar, GrammarMatcher};

fn grammar_accepts(compiled: &CompiledGrammar, input: &str) -> bool {
    let mut matcher =
        GrammarMatcher::new(compiled, None, true, -1).expect("GrammarMatcher::new failed");
    matcher.accept_string(input, false) && matcher.is_terminated()
}

#[test]
fn poolside_grammar_accepts_native_call() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");
    let call = "Working on it.\n<tool_call>get_weather<arg_key>location</arg_key>\
                <arg_value>Boston</arg_value></tool_call>";

    assert!(grammar_accepts(&compiled, call));
}

#[test]
fn poolside_auto_grammar_keeps_bare_opener_in_prose() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");
    let prose = "The opening tag is `<tool_call>` and this is ordinary prose.";

    assert!(grammar_accepts(&compiled, prose));
}

#[test]
fn poolside_grammar_rejects_malformed_argument_close() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");
    let malformed = "<tool_call>get_weather<arg_key>location</arg_key\
                     <arg_value>Boston</arg_value></tool_call>";

    assert!(!grammar_accepts(&compiled, malformed));
}

#[test]
fn poolside_grammar_rejects_unknown_parameter_key() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");
    let malformed = "<tool_call>get_weather<arg_key>city</arg_key>\
                     <arg_value>Boston</arg_value></tool_call>";

    assert!(!grammar_accepts(&compiled, malformed));
}

#[test]
fn poolside_required_grammar_rejects_unknown_tool() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), false, "</arg_value>")
        .expect("compile must succeed");
    let unknown = "<tool_call>lookup<arg_key>location</arg_key>\
                   <arg_value>Boston</arg_value></tool_call>";

    assert!(!grammar_accepts(&compiled, unknown));
}

#[test]
fn poolside_auto_grammar_keeps_unknown_xml_as_prose() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");
    let prose = "The delimiter <tool_call>lookup<arg_key>location</arg_key>\
                 <arg_value>Boston</arg_value></tool_call> is ordinary text.";

    assert!(grammar_accepts(&compiled, prose));
}

#[test]
fn poolside_grammar_accepts_markup_inside_argument_value() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");
    let call = "<tool_call>get_weather<arg_key>location</arg_key>\
                <arg_value><div>Boston</div></arg_value></tool_call>";

    assert!(grammar_accepts(&compiled, call));
}

#[test]
fn poolside_grammar_rejects_missing_required_argument() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");

    assert!(!grammar_accepts(
        &compiled,
        "<tool_call>get_weather</tool_call>"
    ));
}

#[test]
fn poolside_grammar_rejects_empty_tool_list() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let result = engine.compile_poolside_v1_tool_grammar(&[], true, "</arg_value>");

    assert!(matches!(result, Err(GrammarError::NoTools)));
}

#[test]
fn poolside_grammar_accepts_complete_zero_argument_call() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "get_status".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {}
            })),
        },
    }];
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&tools, true, "</arg_value>")
        .expect("compile must succeed");

    assert!(grammar_accepts(
        &compiled,
        "<tool_call>get_status</tool_call>"
    ));
}

#[test]
fn poolside_parser_reports_grammar_support() {
    assert!(crate::tool_parser::ToolCallFormat::PoolsideV1.has_grammar());
}

#[test]
fn poolside_auto_grammar_compiles_with_shared_tool_name_prefixes() {
    let mut short = test_tool_defs().remove(0);
    short.function.name = "foo".to_string();
    let mut long = short.clone();
    long.function.name = "foobar".to_string();
    let tools = vec![short, long];

    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&tools, true, "</arg_value>")
        .expect("shared-prefix Poolside names must compile");

    for name in ["foo", "foobar"] {
        let call = format!(
            "<tool_call>{name}<arg_key>location</arg_key><arg_value>Boston</arg_value></tool_call>"
        );
        assert!(grammar_accepts(&compiled, &call), "call={call:?}");
    }
}

#[test]
fn poolside_auto_grammar_disambiguates_mixed_zero_arg_prefix_names() {
    let zero = ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "foo".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {}
            })),
        },
    };
    let mut parameterized = zero.clone();
    parameterized.function.name = "foobar".to_string();
    parameterized.function.parameters = Some(serde_json::json!({
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
    }));

    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&[zero, parameterized], true, "</arg_value>")
        .expect("mixed prefix names must compile");

    assert!(grammar_accepts(&compiled, "<tool_call>foo</tool_call>"));
    assert!(grammar_accepts(
        &compiled,
        "<tool_call>foobar<arg_key>location</arg_key>\
         <arg_value>Boston</arg_value></tool_call>"
    ));
}
