// SPDX-License-Identifier: AGPL-3.0-only

//! #55: the spinning detector must never hard-mask `<tool_call>`.
//!
//! Quinn's tool-eval-bench TC-07 under context pressure: the harness
//! fills the context with short pure-text ASSISTANT turns, so under the
//! old scan `recent_short >= 5` fired on the FIRST real turn and the
//! spinning arm hard-masked the tool-call opener the model had just
//! announced. The model re-announced until `finish_reason: length`.
//! With the mask off, the sweep scores 100% — the signal was pure noise
//! in these scenarios.
//!
//! Short prose history must not alter tool availability or tool-call bias.

use super::loop_detect::check_loops;
use crate::ir::{ContentPart, Message, Role};

fn text_turn(t: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![ContentPart::Text(t.to_string())],
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

fn user_turn(t: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentPart::Text(t.to_string())],
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

#[test]
fn filler_turns_never_trip_spinning() {
    // 8 injected filler turns: short pure-text assistant messages the
    // harness put in the history. None was engine-generated.
    let filler: Vec<Message> = (0..8)
        .map(|i| text_turn(&format!("(ref #2a-2864-000{i}) [id:2a-2864-000{i}]")))
        .collect();
    let mut messages = filler;
    messages.push(user_turn("Find the Q3 budget report and email the total."));

    let out = check_loops(&messages, true);
    assert!(
        !out.suppress_tool_call,
        "filler alone must not suppress tool calls"
    );
    assert_eq!(
        out.tool_call_repeat_count, 0,
        "filler alone must not seed the bias decay"
    );
}

#[test]
fn varied_short_prose_never_discourages_the_next_tool() {
    let messages: Vec<Message> = (0..8)
        .map(|i| text_turn(&format!("Working on step {i}.")))
        .collect();
    let out = check_loops(&messages, true);
    assert!(!out.suppress_tool_call);
    assert_eq!(out.tool_call_repeat_count, 0);
}

#[test]
fn suppress_verdict_keeps_its_hard_mask() {
    // The Suppress verdict (signature-based, genuinely repeated calls)
    // keeps its P1-5 hard mask: three identical succeeding calls. Only
    // the spinning arm was softened.
    use crate::ir::message::ToolCall;
    let call = || Message {
        role: Role::Assistant,
        content: vec![],
        tool_calls: vec![ToolCall {
            id: "c".into(),
            name: "search".into(),
            arguments: serde_json::json!({"query": "quarterly budget spreadsheet totals Q3 2026 revenue expenses office supplies travel"}),
        }],
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    };
    let messages = vec![user_turn("x"), call(), call(), call()];
    let out = check_loops(&messages, true);
    assert!(
        out.suppress_tool_call,
        "the Suppress verdict must keep its hard mask (repeated succeeding calls)"
    );
}
