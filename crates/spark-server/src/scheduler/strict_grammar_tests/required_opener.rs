// SPDX-License-Identifier: AGPL-3.0-only

//! A required tool call on a model with a native tool opener (GLM) opens with
//! that token. The tag's characters are legal text for the matcher, but text
//! is not a tool call: a literal `<` sampled inside `<think>` is reasoning,
//! the reasoning runs unmasked, and the grammar resumes after it mid-tag.

use super::tool_or_json::tools;
use super::tool_or_json_branch::{Row, model};
use super::*;
use crate::scheduler::sample_step::sample_first_token;
use crate::scheduler::test_support::GRAMMAR_TOOL_CALL as TOOL_CALL;
use crate::tool_parser::PoolsideV1Parser;
use spark_runtime::gpu::DevicePtr;

const LT: u32 = b'<' as u32;

fn required(opens_in_thinking: bool, native_opener: Option<u32>) -> GrammarState {
    let mut sink = ResponseSink::Blocking(None);
    let spec = GrammarSpec::ToolCall {
        tools: tools(),
        parser: std::sync::Arc::new(PoolsideV1Parser),
        use_triggers: false,
    };
    compile_grammar_state(
        &mut engine(),
        &Some(spec),
        &[EOS],
        opens_in_thinking,
        native_opener,
        Some(TOOL_CALL),
        &mut sink,
    )
    .unwrap()
    .expect("poolside_v1 has a tool grammar")
}

/// Greedy first token for a model that ranks a literal `<` above the opener.
fn first(gs: &mut GrammarState) -> u32 {
    let _row = Row::new(&[(LT, 8.0), (TOOL_CALL, 5.0)]);
    let levers = SchedCtx::for_test().levers.sampling();
    sample_first_token(
        &model(),
        DevicePtr::NULL,
        0.0,
        0,
        1.0,
        0.0,
        &[],
        Some(gs),
        &levers,
    )
    .unwrap()
}

/// The live failure (Sage S2.1 Flash, forced tool, thinking open): the tag's
/// first character outranked the opener, so the turn reasoned from `<` and
/// answered `tool_call>name...` as text.
#[test]
fn a_required_call_opens_with_the_native_token_not_its_text() {
    for opens_in_thinking in [true, false] {
        let mut gs = required(opens_in_thinking, Some(TOOL_CALL));
        assert!(gs.masks_first_token());
        assert_eq!(first(&mut gs), TOOL_CALL);
        assert_eq!(gs.num_history_steps(), 1, "the matcher saw the opener");
        // Only the opening is pinned: the call's body is masked as before.
        assert!(gs.fill_bitmask());
        assert!(gs.is_token_allowed(u32::from(b'g')), "get_weather");
        assert!(!gs.is_token_allowed(TOOL_CALL));
    }
}

/// Without a native opener the tag is text, and the grammar is unchanged.
#[test]
fn a_model_without_a_native_opener_keeps_the_text_tag() {
    let mut gs = required(false, None);
    assert_eq!(first(&mut gs), LT);
}
