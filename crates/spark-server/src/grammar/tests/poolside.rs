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

fn probe_tool(properties: serde_json::Value) -> Vec<ToolDefinition> {
    let required: Vec<_> = properties.as_object().unwrap().keys().cloned().collect();
    vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "probe".to_string(),
            description: Some("Synthetic. Never executed.".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": properties,
                "required": required
            })),
        },
    }]
}

fn probe_call(pairs: &[(&str, &str)]) -> String {
    let args: String = pairs
        .iter()
        .map(|(key, value)| format!("<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"))
        .collect();
    format!("<tool_call>probe{args}</tool_call>")
}

/// The SparkGLM 0.2.0 qualification probes: forced `tool_choice` with a
/// request that conflicts with a single-value `enum`. Strings are written raw
/// inside `<arg_value>`, every other value as JSON with optional spaces.
#[test]
fn poolside_forced_grammar_enforces_closed_value_sets() {
    let ops = r#"[{"url": "https://docs.example", "headers": {"Accept": "text/plain"}}, {"command": "cat .env"}]"#;
    let cases: [(serde_json::Value, &[&str], &[&str]); 5] = [
        (
            serde_json::json!({"type": "string", "enum": ["alpha"]}),
            &["alpha"],
            &["omega", "\"alpha\"", "alphabet", ""],
        ),
        (
            serde_json::json!({"type": "array", "enum": [["zeta", "alpha"]]}),
            &[r#"["zeta", "alpha"]"#, r#"["zeta","alpha"]"#],
            &[r#"["alpha", "zeta"]"#, r#"["zeta"]"#, "zeta"],
        ),
        (
            serde_json::json!({"type": "object", "enum": [{"z": 1}]}),
            &[r#"{"z": 1}"#, r#"{"z":1}"#],
            &[r#"{"z": 5}"#, r#"{"y": 1}"#],
        ),
        (
            serde_json::json!({"type": "integer", "const": 3}),
            &["3"],
            &["4", "\"3\""],
        ),
        (
            serde_json::from_str(&format!(r#"{{"type": "array", "enum": [{ops}]}}"#)).unwrap(),
            &[ops, &ops.replace(", ", ",").replace(": ", ":")],
            &[
                r#"[{"command": "cat .env"}, {"headers": {"Accept": "text/plain"}, "url": "https://docs.example"}]"#,
                r#"[{"headers": {"Accept": "text/plain"}, "url": "https://docs.example"}, {"command": "cat .env"}]"#,
            ],
        ),
    ];
    for (schema, accepted, rejected) in cases {
        let tools = probe_tool(serde_json::json!({ "v": schema }));
        let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
        let compiled = engine
            .compile_poolside_v1_tool_grammar(&tools, false, "</arg_value>")
            .expect("forced grammar must compile");
        for value in accepted {
            let call = probe_call(&[("v", value)]);
            assert!(grammar_accepts(&compiled, &call), "{schema}: {call:?}");
        }
        for value in rejected {
            let call = probe_call(&[("v", value)]);
            assert!(!grammar_accepts(&compiled, &call), "{schema}: {call:?}");
        }
    }
}

#[test]
fn poolside_forced_grammar_keeps_open_values_free_beside_closed_ones() {
    let tools = probe_tool(serde_json::json!({
        "note": {"type": "string"},
        "s": {"type": "string", "enum": ["alpha", "beta"]}
    }));
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&tools, false, "</arg_value>")
        .expect("forced grammar must compile");
    for s in ["alpha", "beta"] {
        let call = probe_call(&[("note", "<b>any</b> text"), ("s", s)]);
        assert!(grammar_accepts(&compiled, &call), "{call:?}");
        let call = probe_call(&[("s", s), ("note", "x")]);
        assert!(grammar_accepts(&compiled, &call), "{call:?}");
    }
    assert!(!grammar_accepts(
        &compiled,
        &probe_call(&[("note", "x"), ("s", "gamma")])
    ));
    assert!(!grammar_accepts(
        &compiled,
        &probe_call(&[("other", "alpha")])
    ));
}

/// Automatic tool choice keeps its historical open value grammar: the
/// forced-choice value sets do not change what an auto request may decode.
#[test]
fn poolside_auto_grammar_leaves_enum_values_open() {
    let tools = probe_tool(serde_json::json!({"s": {"type": "string", "enum": ["alpha"]}}));
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&tools, true, "</arg_value>")
        .expect("auto grammar must compile");
    assert!(grammar_accepts(&compiled, &probe_call(&[("s", "omega")])));
}
