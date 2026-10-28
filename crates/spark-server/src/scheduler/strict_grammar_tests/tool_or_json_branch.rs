// SPDX-License-Identifier: AGPL-3.0-only

//! The union's branch token (`GrammarState::choose_branch`): the assistant
//! turn after a tool result, tools + response_format set. The model's mass
//! picks the branch on host logits — a masked fence counts as answering —
//! and the token is never picked in a verify row.

use super::tool_or_json::union;
use super::*;
use crate::scheduler::cancel_test_model::{SCRIPT, Script, TestModel};
use crate::scheduler::decode_logits_step::process_decode_logits;
use crate::scheduler::sample_step::sample_first_token;
use crate::scheduler::strict_spec::row_masks;
use crate::scheduler::test_support::{
    GRAMMAR_TOOL_CALL as TOOL_CALL, GRAMMAR_VOCAB, THINK_END, THINK_START,
};
use spark_runtime::gpu::DevicePtr;

const FENCE: u32 = b'`' as u32;
const BRACE: u32 = b'{' as u32;

/// One scripted logits row for this thread, cleared on drop.
struct Row;

impl Row {
    fn new(pairs: &[(u32, f32)]) -> Self {
        let mut row = vec![0.0f32; GRAMMAR_VOCAB];
        for &(t, v) in pairs {
            row[t as usize] = v;
        }
        SCRIPT.with(|s| {
            *s.borrow_mut() = Some(Script {
                rows: vec![row],
                cursor: 0,
            })
        });
        Row
    }
}

impl Drop for Row {
    fn drop(&mut self) {
        SCRIPT.with(|s| *s.borrow_mut() = None);
    }
}

fn model() -> TestModel {
    TestModel {
        tokens: vec![],
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    }
}

/// The first token of a thinking-off turn (greedy), and the grammar after it.
fn first(pairs: &[(u32, f32)]) -> (u32, GrammarState) {
    let _row = Row::new(pairs);
    let mut gs = union(false);
    assert!(gs.branch_pending());
    let levers = SchedCtx::for_test().levers.sampling();
    let tok = sample_first_token(
        &model(),
        DevicePtr::NULL,
        0.0,
        0,
        1.0,
        0.0,
        &[],
        Some(&mut gs),
        &levers,
    )
    .unwrap();
    (tok, gs)
}

#[test]
fn a_plain_json_top_token_answers_in_json() {
    let (tok, gs) = first(&[(BRACE, 8.0), (TOOL_CALL, 5.0)]);
    assert_eq!(tok, BRACE);
    assert!(!gs.branch_pending());
}

/// The live failure: GLM means to answer with a ```json fence. Masked token
/// by token, the tool opener (its runner-up) beat `{` and every answer
/// became another tool call.
#[test]
fn a_fenced_answer_intent_answers_in_json_not_a_tool_call() {
    let (tok, _) = first(&[(FENCE, 10.0), (TOOL_CALL, 4.0), (BRACE, 3.0)]);
    assert_eq!(tok, BRACE);
}

#[test]
fn a_model_that_means_to_call_still_calls() {
    let (tok, gs) = first(&[(TOOL_CALL, 10.0), (FENCE, 2.0), (BRACE, 1.0)]);
    assert_eq!(tok, TOOL_CALL);
    assert!(!gs.branch_pending());
}

/// Serial decode right after `</think>`: the same choice on the decode path.
#[test]
fn serial_decode_after_think_answers_in_json() {
    let _row = Row::new(&[(FENCE, 10.0), (TOOL_CALL, 4.0), (BRACE, 3.0)]);
    let (mut a, _rx) = test_seq(vec![THINK_END], 20, None, 8);
    a.finished = false;
    a.think_ended = true;
    a.tools_present = true;
    a.tool_call_end_token = None;
    a.grammar_state = Some(union(true));
    assert!(a.grammar_branch_pending());
    let mut rows = vec![a];
    let none = None;
    process_decode_logits(
        &model(),
        &mut rows,
        DevicePtr::NULL,
        std::time::Instant::now(),
        none,
        none,
        none,
        none,
        false,
        &SchedCtx::for_test(),
    );
    let a = &rows[0];
    assert!(a.engine_error.is_none(), "{:?}", a.engine_error);
    assert_eq!(a.output_tokens.last(), Some(&BRACE));
    assert!(!a.grammar_branch_pending());
}

/// The branch token never runs in a verify row: the batch decodes it
/// serially, then speculates again.
#[test]
fn the_branch_token_is_never_speculated() {
    let (mut a, _rx) = test_seq(vec![1, 2, 3], 20, None, 8);
    a.post_think_emitted = 100;
    a.grammar_state = Some(union(true));
    let eligible = |a: &crate::scheduler::types::ActiveSeq| {
        mtp_gate::seq_spec_eligible(a, true, 0, true, true)
    };
    assert!(!eligible(&a), "pending after </think>");
    a.inside_thinking = true;
    assert!(eligible(&a), "reasoning still speculates");
    a.inside_thinking = false;
    assert!(a.grammar_state.as_mut().unwrap().accept_token(BRACE));
    assert!(eligible(&a), "decided: the JSON speculates");
}

/// A drafted `</think>` ends the verify before the branch row.
#[test]
fn the_verify_walk_stops_at_a_drafted_think_close() {
    let tags = (Some(THINK_START), Some(THINK_END));
    let mut gs = union(true);
    let drafts = [b'o' as u32, THINK_END, BRACE];
    let v = row_masks(&mut gs, true, tags, &drafts, GRAMMAR_VOCAB).unwrap();
    assert_eq!(v.drafts, [b'o' as u32], "the bonus row picks at </think>");
    assert!(!v.dead);

    // As the only draft: a dead `<think>` its (reasoning) row refuses.
    let v = row_masks(&mut gs, true, tags, &drafts[1..], GRAMMAR_VOCAB).unwrap();
    assert_eq!(v.drafts, [THINK_START]);
    assert!(v.dead);
    let row0 = &v.masks[..GRAMMAR_VOCAB.div_ceil(32)];
    let allowed = |t: u32| row0[t as usize / 32] >> (t % 32) & 1 == 1;
    assert!(!allowed(THINK_START) && allowed(THINK_END) && allowed(FENCE));
    assert_eq!(gs.num_history_steps(), 0);

    // A JSON-only grammar has no branch row: `</think>` is drafted as ever.
    let mut json_only = strict().strict_output(true);
    let v = row_masks(&mut json_only, true, tags, &drafts, GRAMMAR_VOCAB).unwrap();
    assert_eq!(v.drafts, drafts);
}
