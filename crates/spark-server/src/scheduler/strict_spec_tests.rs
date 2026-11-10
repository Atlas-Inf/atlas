// SPDX-License-Identifier: AGPL-3.0-only

//! Row masks of a strict speculative verify, after vLLM's regression cases
//! for #14702 / #44297: one mask per row from the matcher state after the
//! drafts before it, the bonus row masked, trimming at the first rejected
//! draft or `-1` pad, an exact rollback, and the `</think>` boundary.

use super::{StrictVerify, enabled, prepare, rides_batch, row_masks, width_word};
use crate::grammar::GrammarState;
use crate::scheduler::test_support::{
    GRAMMAR_VOCAB as VOCAB, THINK_END, THINK_START, grammar_engine, test_seq,
};

const WORDS: usize = VOCAB.div_ceil(32);
const TAGS: (Option<u32>, Option<u32>) = (Some(THINK_START), Some(THINK_END));
const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,
    "properties":{"bridge":{"type":"string"}},"required":["bridge"]}"#;

fn strict() -> GrammarState {
    let mut engine = grammar_engine();
    let compiled = engine.compile_json_schema(SCHEMA).unwrap();
    GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .strict_output(false)
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

/// The mask a fresh matcher reports after `prefix`, tags cleared: what each
/// row must equal.
fn mask_after(prefix: &[u32]) -> Vec<u32> {
    let mut gs = strict();
    for &t in prefix {
        assert!(gs.accept_token(t), "fixture prefix {prefix:?}");
    }
    assert!(gs.fill_bitmask());
    let mut m: Vec<u32> = gs.bitmask_data().iter().map(|&w| w as u32).collect();
    for t in [THINK_START, THINK_END] {
        m[t as usize / 32] &= !(1 << (t % 32));
    }
    m
}

fn rows(v: &StrictVerify) -> Vec<&[u32]> {
    assert_eq!(v.masks.len(), (v.drafts.len() + 1) * WORDS);
    v.masks.chunks(WORDS).collect()
}

fn allowed(row: &[u32], t: u32) -> bool {
    row[t as usize / 32] >> (t % 32) & 1 == 1
}

#[test]
fn every_row_and_the_bonus_row_carry_the_state_after_the_drafts_before_them() {
    let drafts = bytes(r#"{"bridge":"#);
    let mut gs = strict();
    let v = row_masks(&mut gs, false, TAGS, &drafts, VOCAB).unwrap();
    assert_eq!(v.drafts, drafts, "every draft is legal: none trimmed");
    for (r, row) in rows(&v).into_iter().enumerate() {
        assert_eq!(row, mask_after(&drafts[..r]), "row {r}");
    }
    // The walk is rolled back: the live matcher still starts the object.
    assert_eq!(gs.num_history_steps(), 0);
    assert!(gs.accept_token(b'{' as u32));
}

#[test]
fn a_rejected_draft_trims_the_window_and_rolls_back_exactly() {
    let mut gs = strict();
    assert!(gs.accept_token(b'{' as u32));
    let before = gs.num_history_steps();
    let drafts = bytes(r#""bZ"x"#);
    let v = row_masks(&mut gs, false, TAGS, &drafts, VOCAB).unwrap();
    // `"b` is a legal key prefix of "bridge"; `Z` is not.
    assert_eq!(v.drafts, bytes(r#""b"#));
    let r = rows(&v);
    assert!(
        !allowed(r[2], b'Z' as u32),
        "the bonus row refuses the rejected draft"
    );
    assert_eq!(gs.num_history_steps(), before);
    assert!(
        gs.accept_token(b'"' as u32),
        "matcher back at its pre-walk state"
    );
}

#[test]
fn a_pad_draft_is_no_draft_never_an_unconstrained_slot() {
    let mut gs = strict();
    let v = row_masks(
        &mut gs,
        false,
        TAGS,
        &[b'{' as u32, u32::MAX, b'"' as u32],
        VOCAB,
    )
    .unwrap();
    assert_eq!(v.drafts, [b'{' as u32]);
    assert_eq!(
        rows(&v)[1],
        mask_after(&[b'{' as u32]),
        "bonus row stays masked"
    );
    // A pad first: the verify keeps width 2 with a draft row 0 refuses, so
    // the step emits only its masked bonus.
    let v = row_masks(&mut gs, false, TAGS, &[u32::MAX, b'{' as u32], VOCAB).unwrap();
    let r = rows(&v);
    assert_eq!(v.drafts.len(), 1);
    assert!(!allowed(r[0], v.drafts[0]));
    assert_eq!(r[0], mask_after(&[]));
    assert_eq!(r[1], r[0]);
    assert_eq!(gs.num_history_steps(), 0);
}

#[test]
fn think_end_inside_the_window_switches_masks_at_the_next_row() {
    // vLLM #44297: rows inside reasoning are unmasked, the matcher never
    // consumes reasoning, and the row after `</think>` starts the grammar.
    let mut drafts = bytes("ok");
    drafts.push(THINK_END);
    drafts.extend(bytes(r#"{""#));
    let mut gs = strict();
    let v = row_masks(&mut gs, true, TAGS, &drafts, VOCAB).unwrap();
    assert_eq!(v.drafts, drafts);
    let r = rows(&v);
    for (i, row) in r[..3].iter().enumerate() {
        assert!(
            row.iter().all(|&w| w == u32::MAX),
            "reasoning row {i} unmasked"
        );
    }
    assert_eq!(r[3], mask_after(&[]), "start state after </think>");
    assert_eq!(r[4], mask_after(&[b'{' as u32]));
    assert_eq!(r[5], mask_after(&bytes(r#"{""#)));
    assert_eq!(gs.num_history_steps(), 0);
}

#[test]
fn thinking_off_rows_refuse_the_think_tags_as_the_serial_path_does() {
    let mut gs = strict();
    let v = row_masks(&mut gs, false, TAGS, &[THINK_START, b'{' as u32], VOCAB).unwrap();
    assert!(!allowed(rows(&v)[0], THINK_START));
    assert_eq!(
        v.drafts.len(),
        1,
        "the tag is trimmed, a dead draft keeps width 2"
    );
    assert!(!allowed(rows(&v)[0], v.drafts[0]));
}

#[test]
fn only_strict_sequences_mask_their_verify() {
    // A batch mixes strict and other sequences: theirs keep today's verify,
    // with no flag on the width word and nothing staged.
    let drafts = bytes("{\"");
    let (mut plain, _rx) = test_seq(vec![1], 20, None, 8);
    assert_eq!(prepare(&mut plain, &drafts, VOCAB).unwrap(), None);
    assert_eq!(width_word(3, &None), 3);
    let mut engine = grammar_engine();
    let compiled = engine.compile_json_grammar().unwrap();
    let (mut tool, _rx) = test_seq(vec![1], 20, None, 8);
    tool.grammar_state = Some(GrammarState::new(&compiled, VOCAB).unwrap());
    assert_eq!(prepare(&mut tool, &drafts, VOCAB).unwrap(), None);
    let (mut a, _rx) = test_seq(vec![1], 20, None, 8);
    a.grammar_state = Some(strict());
    let v = prepare(&mut a, &drafts, VOCAB).unwrap();
    assert_eq!(v.as_ref().map(|v| v.drafts.clone()), Some(drafts));
    assert_eq!(width_word(3, &v), 3 | spark_model::model::MASKED_VERIFY);
}

#[test]
fn a_stop_token_draft_holds_the_state_and_a_finished_matcher_constrains_nothing() {
    use crate::scheduler::test_support::GRAMMAR_EOS as EOS;
    let answer = bytes(r#"{"bridge":"x"}"#);
    let mut gs = strict().with_stop_tokens(&[EOS]);
    let mut drafts = answer.clone();
    drafts.extend([EOS, b'a' as u32]);
    let v = row_masks(&mut gs, false, TAGS, &drafts, VOCAB).unwrap();
    // The answer is complete, so the stop token is legal; nothing follows it.
    assert_eq!(v.drafts.len(), answer.len() + 1);
    let r = rows(&v);
    assert!(allowed(r[answer.len()], EOS));
    assert_eq!(
        r[answer.len() + 1],
        r[answer.len()],
        "a stop token does not advance"
    );
    assert_eq!(gs.num_history_steps(), 0);
    // A matcher that consumed its stop token constrains nothing: rows allow
    // everything but the think tags, and drafts are not trimmed.
    let mut done = strict();
    for &t in &answer {
        assert!(done.accept_token(t));
    }
    assert!(done.accept_token(EOS) && done.is_terminated());
    let v = row_masks(&mut done, false, TAGS, &bytes("ab"), VOCAB).unwrap();
    assert_eq!(v.drafts, bytes("ab"));
    assert!(
        rows(&v)
            .iter()
            .all(|r| allowed(r, b'a' as u32) && !allowed(r, THINK_END))
    );
}

#[test]
fn enabled_arms_by_verify_capability() {
    use crate::scheduler::cancel_test_model::{TestModel, VerifyScript};
    use crate::scheduler::levers::SchedLevers;
    use std::sync::Arc;

    let model = |verify: Option<Arc<VerifyScript>>| TestModel {
        tokens: vec![],
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify,
    };
    // A scripted verify is argmax-only unless it sets `full_logits`.
    let argmax_only = model(Some(Arc::new(VerifyScript::default())));
    let full_logits = model(None);

    let mut levers = SchedLevers::defaults();
    // The GLM arm masks rows on device: argmax-only head, DFlash lane only.
    levers.glm_strict_spec = true;
    assert!(enabled(&levers, &argmax_only, true));
    assert!(!enabled(&levers, &argmax_only, false));
    assert!(!enabled(&levers, &full_logits, true));
    // The full-logits arm needs no device masks: any speculation lane.
    levers.glm_strict_spec = false;
    levers.strict_spec = true;
    assert!(enabled(&levers, &full_logits, true));
    assert!(enabled(&levers, &full_logits, false));
    assert!(!enabled(&levers, &argmax_only, true));
    // Only its MTP lane shares the batched verify.
    assert!(rides_batch(&levers, &full_logits, false));
    assert!(!rides_batch(&levers, &full_logits, true));
    assert!(!rides_batch(&levers, &argmax_only, false));
    levers.strict_spec = false;
    assert!(!enabled(&levers, &full_logits, true));
    assert!(!rides_batch(&levers, &full_logits, false));
    levers.glm_strict_spec = true;
    assert!(!rides_batch(&levers, &argmax_only, true));
}

mod step {
    use super::*;
    use crate::scheduler::cancel_test_model::{TestModel, VerifyScript};
    use crate::scheduler::sched_ctx::SchedCtx;
    use crate::scheduler::verify_dflash_step::step_verify_dflash;
    use spark_model::model::MASKED_VERIFY;
    use spark_model::traits::Model;
    use std::sync::Arc;

    const LAST: u32 = b'\n' as u32;

    fn run(drafts: &[u32], picks: &str) -> (crate::scheduler::types::ActiveSeq, Vec<String>) {
        let script = Arc::new(VerifyScript {
            picks: bytes(picks),
            ..Default::default()
        });
        let model = TestModel {
            tokens: vec![],
            host_logits: false,
            cancel_after_sampling: None,
            cancel_after_row_commit: None,
            verify: Some(script.clone()),
        };
        let (mut a, _rx) = test_seq(vec![], 50, None, 8);
        a.finished = false;
        a.think_ended = true;
        a.last_token = LAST;
        a.tool_call_end_token = None;
        a.grammar_state = Some(strict());
        let sched = SchedCtx::for_test();
        let ctx = sched.verify_logits_ctx(Some(THINK_END), Some(THINK_START), None, None);
        step_verify_dflash(
            &model,
            &mut a,
            &sched,
            drafts,
            &[],
            drafts.len(),
            &ctx,
            true,
        );
        let log = script.log.lock().unwrap().clone();
        (a, log)
    }

    #[test]
    fn a_strict_verify_trims_masks_and_keeps_the_wire_order() {
        // `{"b` is legal, `Z` is not: three drafts verify in four rows.
        let (a, log) = run(&bytes(r#"{"bZ"#), r#"{"br"#);
        let rows = 4;
        assert_eq!(
            log[..6],
            [
                format!("upload {rows} rows {} words", rows * WORDS),
                "seq_cmd 0xfffffff5".to_string(),
                format!("cmd {:#x}", rows as u32 | MASKED_VERIFY),
                format!("tokens {:?}", [LAST, b'{' as u32, b'"' as u32, b'b' as u32]),
                format!("send {rows}"),
                format!("verify {:?}", [LAST, b'{' as u32, b'"' as u32, b'b' as u32]),
            ],
            "masks are uploaded before the command and sent after the tokens"
        );
        assert_eq!(log[6], "cmd 0x3", "three drafts accepted");
        assert!(a.engine_error.is_none() && !a.finished);
        assert_eq!(
            a.output_tokens,
            bytes(r#"{"br"#),
            "accepted drafts, then the bonus"
        );
        // The matcher advanced exactly by what was emitted.
        let mut gs = a.grammar_state.unwrap();
        assert!(gs.accept_token(b'i' as u32) && !gs.accept_token(b'Z' as u32));
    }

    #[test]
    fn a_dead_draft_verify_emits_its_bonus_and_feeds_no_drafter_stats() {
        let before = format!("{:?}", test_seq(vec![], 1, None, 1).0.spec_adapt.survival);
        let (a, log) = run(&bytes("Z"), "{{");
        assert_eq!(log[2], format!("cmd {:#x}", 2 | MASKED_VERIFY), "width 2");
        assert_eq!(a.output_tokens, bytes("{"));
        assert_eq!(format!("{:?}", a.spec_adapt.survival), before);
    }

    /// `ATLAS_STRICT_SPEC` arm: a verify with full logits masks every row in
    /// the host pick pipeline, so the step sends no device masks, sets no
    /// `MASKED_VERIFY` width flag and does not trim drafts — an illegal draft
    /// rides the verify and is rejected at the accept prefix instead.
    #[test]
    fn a_full_logits_strict_verify_skips_the_device_mask_walk() {
        use crate::scheduler::cancel_test_model::SCRIPT;

        // `{"bridge":"` is a legal prefix (the raw literal ends at the ':'
        // — its closing `"` is the delimiter); the trailing SOH (token 1) is
        // not legal inside the JSON string, so the pipeline must reject it.
        let mut drafts = bytes(r#"{"bridge":"#);
        drafts.push(b'"' as u32);
        drafts.push(1);
        let rows = drafts.len() + 1;
        // Scripted logits: row i argmaxes draft i, except the row predicting
        // the illegal draft (index drafts.len()-1), where it loses to 'x'
        // (the mask drops token 1 outright). The last row is the bonus slot.
        let mut script_rows: Vec<Vec<f32>> = (0..rows)
            .map(|i| {
                let mut row = vec![0.0f32; VOCAB];
                let t = match i.cmp(&(drafts.len() - 1)) {
                    std::cmp::Ordering::Less => drafts[i] as usize,
                    std::cmp::Ordering::Equal => b'x' as usize,
                    std::cmp::Ordering::Greater => b'y' as usize,
                };
                row[t] = 10.0;
                if i == drafts.len() - 1 {
                    row[1] = 50.0; // illegal decoy: masked to -inf
                }
                row
            })
            .collect();
        SCRIPT.with(|s| {
            *s.borrow_mut() = Some(crate::scheduler::cancel_test_model::Script {
                rows: std::mem::take(&mut script_rows),
                cursor: 0,
            })
        });
        let mut picks = drafts.clone();
        picks.extend([b'x' as u32, b'y' as u32]);
        let script = Arc::new(VerifyScript {
            picks,
            full_logits: true,
            ..Default::default()
        });
        let model = TestModel {
            tokens: vec![],
            host_logits: false,
            cancel_after_sampling: None,
            cancel_after_row_commit: None,
            verify: Some(script.clone()),
        };
        assert!(!model.verify_logits_argmax_only());
        let (mut a, _rx) = test_seq(vec![], 50, None, 8);
        a.finished = false;
        a.think_ended = true;
        a.last_token = LAST;
        a.tool_call_end_token = None;
        // Neutral penalties: the repeated '"' in the JSON prefix would
        // otherwise be penalised and flip a scripted argmax, exactly as it
        // would for serial decode.
        a.repetition_penalty = 1.0;
        a.frequency_penalty = 0.0;
        a.presence_penalty = 0.0;
        a.dry_multiplier = 0.0;
        a.grammar_state = Some(strict());
        let sched = SchedCtx::for_test();
        let ctx = sched.verify_logits_ctx(Some(THINK_END), Some(THINK_START), None, None);
        step_verify_dflash(
            &model,
            &mut a,
            &sched,
            &drafts,
            &[],
            drafts.len(),
            &ctx,
            true,
        );
        SCRIPT.with(|s| *s.borrow_mut() = None);
        let log = script.log.lock().unwrap().clone();
        assert!(
            !log.iter()
                .any(|e| e.starts_with("upload") || e.starts_with("send")),
            "full-logits strict verify must not stage device masks: {log:?}"
        );
        assert!(
            log.contains(&format!("cmd {:#x}", rows as u32)),
            "width word carries no MASKED_VERIFY flag: {log:?}"
        );
        assert!(
            log.iter()
                .any(|e| e.starts_with("tokens [") && e.ends_with(", 1]")),
            "the illegal draft rode the verify untrimmed: {log:?}"
        );
        assert!(a.engine_error.is_none(), "{:?}", a.engine_error);
        assert_eq!(
            a.output_tokens,
            bytes(r#"{"bridge":"x"#),
            "legal drafts accepted, the masked bonus corrects the illegal one"
        );
        // The matcher advanced exactly by what was emitted.
        let mut gs = a.grammar_state.unwrap();
        assert!(gs.accept_token(b'y' as u32));
    }
}
