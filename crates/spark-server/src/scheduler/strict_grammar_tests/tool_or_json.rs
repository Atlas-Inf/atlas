// SPDX-License-Identifier: AGPL-3.0-only

//! The tool-call-or-JSON union (`GrammarSpec::ToolCallOrJson`) through the
//! scheduler: a strict grammar whose JSON branch fails loud on a fence, whose
//! tool branch takes a call, and whose native GLM opener ends reasoning as
//! output on the emit path and in the strict verify walk alike.

use super::*;
use crate::scheduler::strict_spec::{row_masks, row_masks_with_history};
use crate::scheduler::test_support::{GRAMMAR_TOOL_CALL as TOOL_CALL, GRAMMAR_VOCAB, THINK_END};
use crate::tool_parser::{FunctionDefinition, PoolsideV1Parser, ToolDefinition};

const CALL_BODY: &str = "get_weather<arg_key>location</arg_key><arg_value>Boston</arg_value>";

fn union_spec(schema: Option<&str>) -> GrammarSpec {
    GrammarSpec::ToolCallOrJson {
        tools: vec![ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: None,
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                })),
            },
        }],
        parser: std::sync::Arc::new(PoolsideV1Parser),
        schema: schema.map(str::to_string),
    }
}

pub(super) fn union(opens_in_thinking: bool) -> GrammarState {
    let mut sink = ResponseSink::Blocking(None);
    compile_grammar_state(
        &mut engine(),
        &Some(union_spec(Some(SCHEMA))),
        &[EOS],
        opens_in_thinking,
        Some(TOOL_CALL),
        None,
        &mut sink,
    )
    .unwrap()
    .expect("poolside_v1 has a tool grammar")
}

fn emit_all(a: &mut crate::scheduler::types::ActiveSeq, sched: &SchedCtx, toks: &[u32]) {
    for &t in toks {
        emit_token(a, t, None, sched);
    }
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

#[test]
fn the_union_is_a_strict_grammar_that_knows_the_native_opener() {
    let thinking_off = union(false);
    assert!(thinking_off.is_strict());
    assert!(thinking_off.masks_first_token());
    assert!(!union(true).masks_first_token(), "reasoning first");
    assert!(thinking_off.opens_output(TOOL_CALL));
    // Only a grammar with a tool branch takes the opener as output.
    let mut sink = ResponseSink::Blocking(None);
    let json_only = compile_grammar_state(
        &mut engine(),
        &Some(GrammarSpec::JsonSchema {
            schema: SCHEMA.into(),
        }),
        &[EOS],
        false,
        Some(TOOL_CALL),
        None,
        &mut sink,
    )
    .unwrap()
    .unwrap();
    assert!(!json_only.opens_output(TOOL_CALL));
}

#[test]
fn the_json_answer_streams_and_a_fence_fails_loud() {
    let sched = SchedCtx::for_test();
    let (mut a, _rx) = test_seq(vec![], 40, None, 8);
    a.finished = false;
    a.grammar_state = Some(union(false));
    emit_all(&mut a, &sched, &bytes(r#"{"bridge":"Tower"}"#));
    assert!(!a.finished && a.engine_error.is_none());

    let (mut a, _rx) = test_seq(vec![], 40, None, 8);
    a.finished = false;
    a.grammar_state = Some(union(false));
    emit_token(&mut a, b'`' as u32, None, &sched);
    assert!(a.finished);
    assert!(
        a.engine_error
            .as_deref()
            .is_some_and(|e| e.contains("response_format"))
    );
}

#[test]
fn a_native_opener_ends_reasoning_into_the_tool_branch() {
    let mut sched = SchedCtx::for_test();
    sched.limits.glm_tool_boundary = Some(TOOL_CALL);
    let (mut a, _rx) = test_seq(vec![], 200, None, 8);
    a.finished = false;
    a.inside_thinking = true;
    a.tools_present = true;
    a.tool_call_start_token = Some(TOOL_CALL);
    a.grammar_state = Some(union(true));
    emit_all(&mut a, &sched, &bytes("hmm"));
    emit_token(&mut a, TOOL_CALL, None, &sched);
    assert!(!a.inside_thinking, "the opener ended reasoning");
    emit_all(&mut a, &sched, &bytes(CALL_BODY));
    emit_all(&mut a, &sched, &bytes("</tool_call>"));
    assert!(
        !a.finished && a.engine_error.is_none(),
        "{:?}",
        a.engine_error
    );
    assert!(
        a.grammar_state.as_mut().unwrap().stop_legal(&[EOS]),
        "the call is complete"
    );
}

/// The strict verify walk (`strict_spec`) agrees with the emit path: rows
/// after an opener drafted inside `<think>` are masked by the tool branch.
#[test]
fn the_verify_walk_masks_rows_after_a_drafted_opener() {
    let mut gs = union(true);
    let mut drafts = bytes("ok");
    drafts.push(TOOL_CALL);
    drafts.extend(bytes("gZ"));
    let v = row_masks(
        &mut gs,
        true,
        (None, Some(THINK_END)),
        &drafts,
        GRAMMAR_VOCAB,
    )
    .unwrap();
    // `<tool_call>g` begins `get_weather`; `Z` cannot follow.
    assert_eq!(v.drafts, drafts[..4]);
    assert_eq!(gs.num_history_steps(), 0, "the walk rolls back");

    // Without the opener every row is still reasoning: nothing is trimmed.
    let mut json_only = strict().strict_output(true);
    let v = row_masks(
        &mut json_only,
        true,
        (None, Some(THINK_END)),
        &drafts,
        GRAMMAR_VOCAB,
    )
    .unwrap();
    assert_eq!(v.drafts, drafts);
}

#[test]
fn the_verify_walk_keeps_a_quoted_native_opener_in_reasoning() {
    let mut gs = union(true);
    let history = [b'`' as u32];
    let mut drafts = vec![TOOL_CALL, b'`' as u32, TOOL_CALL];
    drafts.extend(bytes(CALL_BODY));
    let v = row_masks_with_history(
        &mut gs,
        true,
        (None, Some(THINK_END)),
        &drafts,
        GRAMMAR_VOCAB,
        &history,
    )
    .unwrap();
    assert_eq!(&v.drafts[..3], &drafts[..3]);
    assert!(
        v.drafts.len() > 3,
        "the real opener should enter the tool branch"
    );
    assert_eq!(gs.num_history_steps(), 0, "the walk rolls back");
}
