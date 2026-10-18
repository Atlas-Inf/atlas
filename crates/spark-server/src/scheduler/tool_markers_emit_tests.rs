// SPDX-License-Identifier: AGPL-3.0-only
//! The envelope-stuck guard on a real Nemotron 3.5 Super VL token stream: a
//! `write` whose `content` is 1500 tokens (with `</div>` closes inside) must
//! stream to `</tool_call>` with the served tokenizer's markers, and is what
//! the Qwen-hardcoded markers used to kill at 1024 tokens.

use super::fixtures::{NEMOTRON, QWEN};
use super::{TEST_LOCK, ToolMarkers, set};
use crate::scheduler::emit_step::update_tool_param_state;
use crate::scheduler::test_support::test_seq;

const NEMO_TOOL_CALL: u32 = 14;
const NEMO_TOOL_CALL_END: u32 = 15;

/// `<tool_call>` `<function=write>\n` `<parameter=path>\n/tmp/a.html\n</parameter>\n`
/// `<parameter=content>\n` + body + `</parameter>\n</function>` `</tool_call>`,
/// in Nemotron ids (tokenizer.json, 2026-10-18).
fn nemotron_write(body_tokens: usize) -> Vec<u32> {
    let mut t = vec![NEMO_TOOL_CALL];
    t.extend([1060, 5165, 1061, 16539, 1561]); // <function=write>\n
    t.extend([1060, 31960, 1061, 7368, 1561]); // <parameter=path>\n
    t.extend([1047, 20307, 22139, 7120, 1010]); // /tmp/a.html\n
    t.extend([1885, 31960, 1561]); // </parameter>\n
    t.extend([1060, 31960, 1061, 5431, 1561]); // <parameter=content>\n
    for i in 0..body_tokens {
        // Mostly prose ids, with a `</div>\n` (</ div >\n) every 50 tokens.
        if i % 50 == 49 {
            t.extend([1885, 1950, 1561]);
        } else {
            t.push(41199 + (i as u32 % 7));
        }
    }
    t.extend([1885, 31960, 1561]); // </parameter>\n
    t.extend([1885, 5165, 1062]); // </function>
    t.push(NEMO_TOOL_CALL_END);
    t
}

/// Drive the guard the way `emit_token` does (state update, then push).
fn run(markers: Option<ToolMarkers>, tokens: &[u32]) -> (bool, Option<&'static str>, bool) {
    set(markers);
    let (mut a, _rx) = test_seq(Vec::new(), 100_000, None, 0);
    a.tool_call_start_token = Some(NEMO_TOOL_CALL);
    a.tool_call_end_token = Some(NEMO_TOOL_CALL_END);
    for &tok in tokens {
        update_tool_param_state(&mut a, tok, true);
        a.output_tokens.push(tok);
        if a.finished {
            break;
        }
    }
    set(None);
    (a.finished, a.guard_stop, a.inside_tool_body)
}

#[test]
fn nemotron_markers_let_a_large_write_finish() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (finished, guard, inside) = run(Some(NEMOTRON), &nemotron_write(1500));
    assert!(!finished, "guard fired: {guard:?}");
    assert_eq!(guard, None);
    assert!(!inside, "</tool_call> should close the body");
}

#[test]
fn qwen_ids_on_a_nemotron_stream_reproduce_the_kill() {
    // The pre-fix behaviour: Qwen ids never match a Nemotron `<parameter=`,
    // so the file content counts as envelope and trips the 1024 cap.
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (finished, guard, _) = run(Some(QWEN), &nemotron_write(1500));
    assert!(finished);
    assert_eq!(guard, Some("tool_envelope_stuck"));
}

#[test]
fn unresolved_markers_leave_the_cap_off() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (finished, guard, _) = run(None, &nemotron_write(1500));
    assert!(!finished && guard.is_none());
}

#[test]
fn a_never_closed_envelope_is_still_cut() {
    // The runaway the cap exists for: `<tool_call>` then 1100 envelope tokens
    // that never open a parameter value.
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut t = vec![NEMO_TOOL_CALL];
    t.extend(std::iter::repeat_n(41199u32, 1100));
    let (finished, guard, _) = run(Some(NEMOTRON), &t);
    assert!(finished);
    assert_eq!(guard, Some("tool_envelope_stuck"));
}
