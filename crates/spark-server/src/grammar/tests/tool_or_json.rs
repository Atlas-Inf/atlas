// SPDX-License-Identifier: AGPL-3.0-only

//! `compile_tool_or_json_grammar`: tool calls OR the response_format JSON,
//! the branch picked by the first output token (ToolEvalBench TC-65..69).

use super::*;
use crate::tool_parser::{HermesParser, PoolsideV1Parser, ToolCallParser};
use xgrammar::{CompiledGrammar, GrammarMatcher};

const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,
    "properties":{"city":{"type":"string"},
                  "conditions":{"type":"string","enum":["sunny","rain"]},
                  "temp_c":{"type":"integer"}},
    "required":["city","conditions","temp_c"]}"#;
const ANSWER: &str = r#"{"city": "Boston", "conditions": "sunny", "temp_c": 21}"#;
const CALL: &str = "<tool_call>get_weather<arg_key>location</arg_key>\
                    <arg_value>Boston</arg_value></tool_call>";

fn engine() -> GrammarEngine {
    GrammarEngine::new(&test_vocab(), &[130]).unwrap()
}

fn union(parser: &dyn ToolCallParser, schema: Option<&str>) -> CompiledGrammar {
    engine()
        .compile_tool_or_json_grammar(parser, &test_tool_defs(), schema)
        .expect("the parser has a tool grammar")
        .expect("compile must succeed")
}

fn accepts(compiled: &CompiledGrammar, input: &str) -> bool {
    let mut matcher = GrammarMatcher::new(compiled, None, true, -1).unwrap();
    matcher.accept_string(input, false) && matcher.is_terminated()
}

#[test]
fn the_tool_branch_accepts_calls_from_the_first_token() {
    let g = union(&PoolsideV1Parser, Some(SCHEMA));

    assert!(accepts(&g, CALL));
    assert!(accepts(&g, &format!("{CALL}{CALL}")), "parallel calls");
}

#[test]
fn the_json_branch_accepts_the_bare_schema_answer() {
    let g = union(&PoolsideV1Parser, Some(SCHEMA));

    assert!(accepts(&g, ANSWER));
    assert!(accepts(
        &g,
        "{\n  \"city\": \"Boston\",\n  \"conditions\": \"rain\",\n  \"temp_c\": -3\n}"
    ));
}

/// The ToolEvalBench failure: a correct object in a ```json fence.
#[test]
fn fenced_json_and_prose_are_refused() {
    let g = union(&PoolsideV1Parser, Some(SCHEMA));

    for bad in [
        format!("```json\n{ANSWER}\n```"),
        format!("Here is the result: {ANSWER}"),
        format!("{ANSWER}\nDone."),
        "The weather in Boston is sunny.".to_string(),
        // Prose before a call: the union's tool branch starts at token 1.
        format!("Let me check.\n{CALL}"),
    ] {
        assert!(!accepts(&g, &bad), "{bad:?}");
    }
}

#[test]
fn schema_violations_are_refused() {
    let g = union(&PoolsideV1Parser, Some(SCHEMA));

    for bad in [
        r#"{"city": "Boston", "conditions": "snow", "temp_c": 21}"#,
        r#"{"city": "Boston", "temp_c": 21}"#,
        r#"{"city": "Boston", "conditions": "sunny", "temp_c": "21"}"#,
        r#"{"city": "Boston", "conditions": "sunny", "temp_c": 21, "extra": 1}"#,
    ] {
        assert!(!accepts(&g, bad), "{bad}");
    }
}

#[test]
fn json_object_unions_any_json_with_the_tool_calls() {
    let g = union(&PoolsideV1Parser, None);

    assert!(accepts(&g, r#"{"anything": [1, 2]}"#));
    assert!(accepts(&g, CALL));
    assert!(!accepts(&g, "```json\n{\"anything\": 1}\n```"));
}

/// The union is built from the parser's own tags, for any format.
#[test]
fn the_union_follows_the_parser_format() {
    let g = union(&HermesParser, Some(SCHEMA));
    let hermes = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
                  {\"location\": \"Boston\"}}\n</tool_call>";

    assert!(accepts(&g, hermes));
    assert!(accepts(&g, ANSWER));
    assert!(!accepts(&g, CALL), "a poolside call is not a hermes call");
}

/// The JSON branch is scoped to its own compile: a later tool-only grammar
/// on the same engine still allows free text around calls.
#[test]
fn a_later_tool_grammar_is_not_unioned() {
    let mut engine = engine();
    engine
        .compile_tool_or_json_grammar(&PoolsideV1Parser, &test_tool_defs(), Some(SCHEMA))
        .unwrap()
        .unwrap();
    let tools_only = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .unwrap();

    assert!(accepts(&tools_only, &format!("Let me check.\n{CALL}")));
}

#[test]
fn an_oversized_or_invalid_schema_is_an_error() {
    let huge = format!(
        r#"{{"type":"object","description":"{}"}}"#,
        "x".repeat(super::super::compile_misc::MAX_SCHEMA_TEXT_BYTES)
    );
    for schema in [huge.as_str(), r#"{"type": 7}"#] {
        let result = engine().compile_tool_or_json_grammar(
            &PoolsideV1Parser,
            &test_tool_defs(),
            Some(schema),
        );
        assert!(
            matches!(result, Some(Err(_))),
            "{}",
            &schema[..20.min(schema.len())]
        );
    }
}
