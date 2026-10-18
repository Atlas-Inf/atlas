// SPDX-License-Identifier: AGPL-3.0-only
//! The envelope-stuck guard on real tool-call token streams: a value of 1500
//! tokens (with `</div>` closes inside) must stream to `</tool_call>` with the
//! served tokenizer's marker classes, and is what the five hard-coded Qwen ids
//! used to kill at 1024 tokens — on Nemotron for every key, on Qwen for keys
//! whose `=` merges into the key (`=new` `_str`).

use super::fixtures::{nemotron, qwen};
use super::{ToolMarkers, set};
use crate::scheduler::emit_step::update_tool_param_state;
use crate::scheduler::test_support::test_seq;

const NEMO_TOOL_CALL: u32 = 14;
const NEMO_TOOL_CALL_END: u32 = 15;
const QWEN_TOOL_CALL: u32 = 151_657;
const QWEN_TOOL_CALL_END: u32 = 151_658;

/// Nemotron ids (tokenizer.json, 2026-10-18): `<tool_call>`
/// `<function=write>\n<parameter=path>\n/tmp/a.html\n</parameter>\n`
/// `<parameter=content>\n` + body + `</parameter>\n</function>` `</tool_call>`.
fn nemotron_write(body_tokens: usize) -> Vec<u32> {
    let mut t = vec![NEMO_TOOL_CALL];
    t.extend([1060, 5165, 1061, 16539, 1561]); // <function=write>\n
    t.extend([1060, 31960, 1061, 7368, 1561]); // <parameter=path>\n
    t.extend([1047, 20307, 22139, 7120, 1010]); // /tmp/a.html\n
    t.extend([1885, 31960, 1561]); // </parameter>\n
    t.extend([1060, 31960, 1061, 5431, 1561]); // <parameter=content>\n
    body(&mut t, body_tokens, [1885, 1950, 1561]); // </div>\n every 50
    t.extend([1885, 31960, 1561]); // </parameter>\n
    t.extend([1885, 5165, 1062]); // </function>
    t.push(NEMO_TOOL_CALL_END);
    t
}

/// Qwen ids: `<parameter=new_str>` tokenizes as `<` `parameter` `=new` `_str`
/// `>` (tokenizer.json, 2026-10-18); `=new` is a stand-in id from the fixture.
fn qwen_new_str(body_tokens: usize) -> Vec<u32> {
    let mut t = vec![QWEN_TOOL_CALL, 198];
    t.extend([27, 1688, 28, 8_888, 29, 198]); // <function=replace>\n (`=` standalone)
    t.extend([27, 15704, 900_002, 4_444, 29, 198]); // <parameter=new_str>\n
    body(&mut t, body_tokens, [510, 611, 29]); // </div> every 50
    t.extend([198, 510, 15704, 29, 198]); // \n</parameter>\n
    t.extend([510, 1688, 29, 198]); // </function>\n
    t.push(QWEN_TOOL_CALL_END);
    t
}

fn body(t: &mut Vec<u32>, n: usize, close_tag: [u32; 3]) {
    for i in 0..n {
        if i % 50 == 49 {
            t.extend(close_tag);
        } else {
            t.push(41_199 + (i as u32 % 7));
        }
    }
}

/// Drive the guard the way `emit_token` does (state update, then push).
fn run(
    markers: Option<ToolMarkers>,
    open: u32,
    close: u32,
    tokens: &[u32],
) -> (bool, Option<&'static str>, bool) {
    set(markers);
    let (mut a, _rx) = test_seq(Vec::new(), 100_000, None, 0);
    a.tool_call_start_token = Some(open);
    a.tool_call_end_token = Some(close);
    // The fixture builds a finished sequence; this one is mid-generation.
    a.finished = false;
    a.guard_stop = None;
    for &tok in tokens {
        update_tool_param_state(&mut a, tok, true);
        a.output_tokens.push(tok);
        if a.finished {
            break;
        }
    }
    (a.finished, a.guard_stop, a.inside_tool_body)
}

fn nemo(markers: Option<ToolMarkers>, tokens: &[u32]) -> (bool, Option<&'static str>, bool) {
    run(markers, NEMO_TOOL_CALL, NEMO_TOOL_CALL_END, tokens)
}

#[test]
fn nemotron_classes_let_a_large_write_finish() {
    let (finished, guard, inside) = nemo(Some(nemotron()), &nemotron_write(1500));
    assert!(!finished, "guard fired: {guard:?}");
    assert!(!inside, "</tool_call> should close the body");
}

#[test]
fn legacy_qwen_ids_on_a_nemotron_stream_reproduce_the_kill() {
    // The old behaviour: Qwen ids never match a Nemotron `<parameter=`, so
    // the file content counts as envelope and trips the 1024 cap.
    let (finished, guard, _) = nemo(Some(ToolMarkers::legacy_qwen()), &nemotron_write(1500));
    assert!(finished);
    assert_eq!(guard, Some("tool_envelope_stuck"));
}

#[test]
fn qwen_classes_recognise_a_merged_eq_key() {
    let t = qwen_new_str(1500);
    let (finished, guard, inside) = run(Some(qwen()), QWEN_TOOL_CALL, QWEN_TOOL_CALL_END, &t);
    assert!(!finished, "guard fired: {guard:?}");
    assert!(!inside);
    // ...which the five legacy ids did not: `=new` is not the `=` id.
    let (finished, guard, _) = run(
        Some(ToolMarkers::legacy_qwen()),
        QWEN_TOOL_CALL,
        QWEN_TOOL_CALL_END,
        &t,
    );
    assert!(finished);
    assert_eq!(guard, Some("tool_envelope_stuck"));
}

#[test]
fn unresolved_markers_leave_the_cap_off() {
    let (finished, guard, _) = nemo(None, &nemotron_write(1500));
    assert!(!finished && guard.is_none());
}

#[test]
fn a_never_closed_envelope_is_still_cut() {
    // The runaway the cap exists for: `<tool_call>` then 1100 envelope tokens
    // that never open a parameter value.
    let mut t = vec![NEMO_TOOL_CALL];
    t.extend(std::iter::repeat_n(41_199u32, 1100));
    let (finished, guard, _) = nemo(Some(nemotron()), &t);
    assert!(finished);
    assert_eq!(guard, Some("tool_envelope_stuck"));
}
