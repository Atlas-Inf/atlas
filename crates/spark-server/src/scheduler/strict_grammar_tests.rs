// SPDX-License-Identifier: AGPL-3.0-only

//! Strict (`response_format`) grammars: armed strict or refused loudly, never
//! speculated, deferred past an opening `<think>`, and fail-loud on a refusal
//! where tool grammars disengage.

use crate::api::GrammarSpec;
use crate::grammar::{GrammarEngine, GrammarState};
use crate::scheduler::emit_step::{compile_grammar_state, emit_token};
use crate::scheduler::{
    mtp_gate, sched_ctx::SchedCtx, test_support::test_seq, types::ResponseSink,
};

#[path = "strict_grammar_tests/decode.rs"]
mod decode;

const EOS: u32 = crate::scheduler::test_support::GRAMMAR_EOS;
const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,
    "properties":{"bridge":{"type":"string"}},"required":["bridge"]}"#;

fn engine() -> Option<GrammarEngine> {
    Some(crate::scheduler::test_support::grammar_engine())
}

fn compile(spec: GrammarSpec, opens_in_thinking: bool) -> Option<GrammarState> {
    let mut sink = ResponseSink::Blocking(None);
    compile_grammar_state(
        &mut engine(),
        &Some(spec),
        &[EOS],
        opens_in_thinking,
        &mut sink,
    )
    .unwrap()
}

fn strict() -> GrammarState {
    compile(
        GrammarSpec::JsonSchema {
            schema: SCHEMA.into(),
        },
        false,
    )
    .unwrap()
}

/// A plain (non-strict) JSON matcher: the disengage contract tool grammars keep.
fn lenient() -> GrammarState {
    let mut engine = engine().unwrap();
    let compiled = engine.compile_json_grammar().unwrap();
    GrammarState::new(&compiled, engine.vocab_size()).unwrap()
}

#[test]
fn response_format_grammars_are_strict_and_defer_past_an_open_think() {
    for spec in [
        GrammarSpec::JsonObject,
        GrammarSpec::JsonSchema {
            schema: SCHEMA.into(),
        },
    ] {
        let thinking_off = compile(spec.clone(), false).unwrap();
        assert!(thinking_off.is_strict());
        assert!(thinking_off.masks_first_token());
        // The prompt opens `<think>`: the first token is reasoning.
        assert!(!compile(spec, true).unwrap().masks_first_token());
    }
    let tool_like = lenient();
    assert!(!tool_like.is_strict());
    assert!(
        tool_like.masks_first_token(),
        "non-strict grammars keep masking token 1"
    );
}

#[test]
fn an_unarmable_response_format_is_reported_not_decoded_unconstrained() {
    let (tx, mut rx) = tokio::sync::oneshot::channel();
    let mut sink = ResponseSink::Blocking(Some(tx));
    let err = compile_grammar_state(
        &mut None,
        &Some(GrammarSpec::JsonObject),
        &[EOS],
        false,
        &mut sink,
    )
    .err()
    .expect("an unarmable response_format is an error");
    assert!(format!("{err:#}").contains("response_format"));
    let sent = rx.try_recv().expect("client is told");
    let sent = sent.err().expect("an error, not a completion");
    assert!(
        sent.is::<crate::api::InvalidRequestError>(),
        "a blocking client gets HTTP 400, which SDKs do not retry"
    );
    // No grammar requested: nothing to arm, nothing reported.
    let mut sink = ResponseSink::Blocking(None);
    assert!(
        compile_grammar_state(&mut None, &None, &[EOS], false, &mut sink)
            .unwrap()
            .is_none()
    );
}

#[test]
fn a_strict_refusal_ends_the_response_with_an_error() {
    let sched = SchedCtx::for_test();
    let (mut a, _rx) = test_seq(vec![], 20, None, 8);
    a.finished = false;
    a.grammar_state = Some(strict());
    emit_token(&mut a, b'{' as u32, None, &sched);
    assert!(!a.finished);
    emit_token(&mut a, b'Z' as u32, None, &sched);
    assert!(a.finished);
    assert!(
        a.engine_error
            .as_deref()
            .is_some_and(|e| e.contains("response_format"))
    );
    assert_eq!(
        a.output_tokens,
        vec![b'{' as u32],
        "the refused token is not emitted"
    );
    assert!(
        a.grammar_state.is_some(),
        "a strict grammar never disengages"
    );
}

#[test]
fn a_lenient_refusal_still_disengages_and_continues() {
    let sched = SchedCtx::for_test();
    let (mut a, _rx) = test_seq(vec![], 20, None, 8);
    a.finished = false;
    a.grammar_state = Some(lenient());
    emit_token(&mut a, b'{' as u32, None, &sched);
    emit_token(&mut a, b'Z' as u32, None, &sched);
    assert!(!a.finished && a.engine_error.is_none());
    assert!(a.grammar_state.is_none());
    assert_eq!(a.output_tokens, vec![b'{' as u32, b'Z' as u32]);
}

#[test]
fn a_strict_grammar_never_dispatches_speculation() {
    // Every regime that would otherwise speculate: before/after `</think>`,
    // DFlash raw-argmax or masked MTP verify, with or without spec-in-think.
    for inside_thinking in [false, true] {
        for spec_think in [false, true] {
            for raw in [false, true] {
                let (mut a, _rx) = test_seq(vec![1, 2, 3], 20, None, 8);
                a.inside_thinking = inside_thinking;
                a.post_think_emitted = 100;
                let eligible = |a: &crate::scheduler::types::ActiveSeq| {
                    mtp_gate::seq_spec_eligible(a, spec_think, 0, raw, false)
                };
                let baseline = eligible(&a);
                a.grammar_state = Some(lenient());
                assert_eq!(
                    eligible(&a),
                    baseline,
                    "tool grammars keep today's dispatch"
                );
                a.grammar_state = Some(strict());
                assert!(
                    !eligible(&a),
                    "think={inside_thinking} spec_think={spec_think} raw={raw}"
                );
                // ATLAS_GLM_STRICT_SPEC: the masked verify serves it like any
                // other sequence, so it no longer holds the batch serial.
                assert_eq!(
                    mtp_gate::seq_spec_eligible(&a, spec_think, 0, raw, true),
                    baseline
                );
            }
        }
    }
}

#[test]
fn a_schema_too_large_to_compile_is_a_400_not_a_crash() {
    let (tx, mut rx) = tokio::sync::oneshot::channel();
    let mut sink = ResponseSink::Blocking(Some(tx));
    let schema = crate::grammar::tests::schema_limits::too_many_rules_schema();
    let err = compile_grammar_state(
        &mut engine(),
        &Some(GrammarSpec::JsonSchema { schema }),
        &[EOS],
        false,
        &mut sink,
    )
    .err()
    .expect("refused");
    assert!(format!("{err:#}").contains("grammar too large"), "{err:#}");
    let sent = rx
        .try_recv()
        .expect("client is told")
        .err()
        .expect("an error");
    assert!(sent.is::<crate::api::InvalidRequestError>());
}
