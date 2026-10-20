// SPDX-License-Identifier: AGPL-3.0-only

//! Strict grammars on the serial decode path: the first token after prefill,
//! a refusal in `process_decode_logits`, and the tool-turn content guards.

use super::*;
use crate::scheduler::cancel_test_model::TestModel;
use crate::scheduler::decode_logits_content::handle_content_token;
use crate::scheduler::decode_logits_step::process_decode_logits;
use crate::scheduler::sample_step::sample_first_token;
use crate::scheduler::types::{ActiveSeq, tool_request_for};
use spark_runtime::gpu::DevicePtr;

/// The test model's f32 5.0 at id 101 reads, as BF16, as a maximum at id 203.
const RAW_TOP: u32 = 203;
/// JSON may open with these: suppressed too, every candidate is grammar-illegal.
const JSON_STARTS: [u32; 5] = [
    b'{' as u32,
    b' ' as u32,
    b'\n' as u32,
    b'\t' as u32,
    b'\r' as u32,
];

fn model() -> TestModel {
    TestModel {
        tokens: vec![],
        host_logits: true,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    }
}

/// A suppress list forces the host sampler (the test model has no device argmax).
fn first(gs: Option<&mut GrammarState>, suppress: &[u32]) -> anyhow::Result<u32> {
    let levers = SchedCtx::for_test().levers.sampling();
    sample_first_token(
        &model(),
        DevicePtr::NULL,
        0.0,
        0,
        1.0,
        0.0,
        suppress,
        gs,
        &levers,
    )
}

#[test]
fn the_first_token_is_masked_and_consumed_by_a_strict_grammar() {
    assert_eq!(first(None, &[2047]).unwrap(), RAW_TOP, "unmasked baseline");
    let mut gs = strict();
    let tok = first(Some(&mut gs), &[2047]).unwrap();
    assert_eq!(tok, b'{' as u32);
    assert_eq!(gs.num_history_steps(), 1);
}

#[test]
fn a_first_token_inside_think_is_neither_masked_nor_consumed() {
    let mut gs = compile(GrammarSpec::JsonObject, true).unwrap();
    assert_eq!(first(Some(&mut gs), &[2047]).unwrap(), RAW_TOP);
    assert_eq!(
        gs.num_history_steps(),
        0,
        "the grammar starts after </think>"
    );
}

#[test]
fn a_first_token_refusal_fails_a_strict_grammar_only() {
    let suppress = [&JSON_STARTS[..], &[2047]].concat();
    let err = first(Some(&mut strict()), &suppress).unwrap_err();
    assert!(format!("{err:#}").contains("response_format"), "{err:#}");
    // A tool grammar keeps going; the emit-path disengage handles it later.
    assert!(first(Some(&mut lenient()), &suppress).is_ok());
}

#[test]
fn a_strict_refusal_in_serial_decode_ends_the_response_with_an_error() {
    // ATLAS_FORCE_TEMP_ZERO takes the raw argmax past every mask: the model's
    // top token (`e`) cannot open JSON, so the strict grammar refuses it.
    let mut levers = crate::scheduler::levers::SchedLevers::defaults();
    levers.force_temp_zero = true;
    let mut sched = SchedCtx::for_test();
    sched.levers = std::sync::Arc::new(levers);
    let (mut a, _rx) = test_seq(vec![], 20, None, 8);
    a.finished = false;
    a.think_ended = true;
    a.tool_call_end_token = None;
    a.grammar_state = Some(strict());
    let mut rows = vec![a];
    process_decode_logits(
        &model(),
        &mut rows,
        DevicePtr::NULL,
        std::time::Instant::now(),
        None,
        None,
        None,
        None,
        false,
        &sched,
    );
    let a = &rows[0];
    assert!(a.finished);
    assert!(
        a.engine_error
            .as_deref()
            .is_some_and(|e| e.contains("response_format"))
    );
    assert!(
        a.output_tokens.is_empty(),
        "the refused token is not emitted"
    );
}

/// Feed `len` content tokens through the serial content guards; returns the
/// position at which a guard cut or rolled back the response, if any.
fn first_cut(a: &mut ActiveSeq, len: usize, token_at: impl Fn(usize) -> u32) -> Option<usize> {
    let (model, sched) = (model(), SchedCtx::for_test());
    // Armed here (GLM's MODEL.toml disarms it) to cover the strictest case.
    sched.levers.set_loop_watchdog(true);
    a.finished = false;
    a.think_ended = true;
    a.remaining = len + 1;
    a.min_tokens = 0;
    (0..len).find(|&i| {
        let cut = handle_content_token(a, &model, &sched, false) || a.finished;
        a.output_tokens.push(token_at(i));
        cut
    })
}

#[test]
fn tool_turn_guards_never_cut_a_strict_json_answer() {
    assert!(!tool_request_for(Some(&strict()), false));
    assert!(tool_request_for(Some(&lenient()), false));
    assert!(tool_request_for(None, true));
    assert!(!tool_request_for(None, false));
    // 12,288 tokens of period-8 JSON-shaped repetition: past the 3,072-token
    // inter-tool prose budget and a content loop by the watchdog's measure.
    let looping = |i: usize| (b'a' as u32) + (i % 8) as u32;
    let (mut a, _rx) = test_seq(vec![], 0, None, 8);
    a.grammar_state = Some(strict());
    a.tool_request = tool_request_for(a.grammar_state.as_ref(), false);
    assert_eq!(first_cut(&mut a, 12_288, looping), None);
    // A tool turn with the same output is still cut.
    let (mut a, _rx) = test_seq(vec![], 0, None, 8);
    a.grammar_state = Some(lenient());
    a.tool_request = true;
    assert!(first_cut(&mut a, 12_288, looping).is_some());
}

#[test]
fn a_plain_request_is_never_cut_for_length() {
    // No tools, no response_format: no prose budget, no post-think cap. Only a
    // real repetition loop trips the content watchdog, never length alone.
    let varied = |i: usize| ((i * 7919) % 2000) as u32 + 1;
    let (mut a, _rx) = test_seq(vec![], 0, None, 8);
    assert!(!a.tool_request && a.grammar_state.is_none());
    assert_eq!(first_cut(&mut a, 12_288, varied), None);
    let (mut a, _rx) = test_seq(vec![], 0, None, 8);
    let looping = |i: usize| (b'a' as u32) + (i % 8) as u32;
    let cut = first_cut(&mut a, 12_288, looping).expect("a real loop is still caught");
    assert!(cut < 512, "caught within a few watchdog strides, at {cut}");
}
