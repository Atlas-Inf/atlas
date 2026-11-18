// SPDX-License-Identifier: AGPL-3.0-only

//! Strict structured-output owners in the owner batch
//! (`ATLAS_GLM_STRICT_OWNER_BATCH`): who is admitted, the row masks the one
//! traversal carries, and a dead draft's verdict.

use super::{owner_rides, verify_owner_batch_with};
use crate::grammar::GrammarState;
use crate::scheduler::cancel_test_model::{SCRIPT, TestModel, VerifyScript};
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::{
    GRAMMAR_VOCAB as VOCAB, RespRx, THINK_END, THINK_START, grammar_engine, test_seq,
};
use crate::scheduler::types::ActiveSeq;
use std::sync::Arc;

const WORDS: usize = VOCAB.div_ceil(32);
/// The strict owner's last emitted token.
const LAST: u32 = b'\n' as u32;
/// The free owner's one draft, and the bonus its verify picks after it.
const FREE: [u32; 2] = [41, 77];
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

fn allowed(row: &[u32], t: u32) -> bool {
    row[t as usize / 32] >> (t % 32) & 1 == 1
}

/// The mask a fresh matcher reports after `prefix`, think tags cleared (the
/// grammar allows neither where these tests look): what a strict owner's row
/// must equal outside reasoning.
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

/// `active` = a free owner holding one draft, then a strict owner holding
/// `drafts`. The receivers must outlive the step.
fn owners(drafts: &[u32]) -> (Vec<ActiveSeq>, Vec<RespRx>) {
    let (mut free, rx_free) = test_seq(vec![], 50, None, 8);
    free.pending_drafts = vec![FREE[0]];
    let (mut json, rx_json) = test_seq(vec![], 50, None, 8);
    json.think_ended = true;
    json.last_token = LAST;
    json.tool_call_end_token = None;
    json.grammar_state = Some(strict());
    json.pending_drafts = drafts.to_vec();
    let mut active = vec![free, json];
    for a in &mut active {
        a.finished = false;
        a.min_tokens = 0;
    }
    (active, vec![rx_free, rx_json])
}

/// One owner-batch partition of `active` with the switch at `strict_ok`,
/// against a model whose verify picks `picks` (owner-major). Returns the
/// sequences left to the serial path, the wire log and the row masks the
/// traversal carried.
fn step(
    active: &mut [ActiveSeq],
    picks: Vec<u32>,
    strict_ok: bool,
) -> (Vec<usize>, Vec<String>, Vec<u32>) {
    SCRIPT.with(|s| *s.borrow_mut() = None);
    let script = Arc::new(VerifyScript {
        picks,
        owner_batch: true,
        ..Default::default()
    });
    let model = TestModel {
        tokens: vec![],
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: Some(script.clone()),
    };
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(Some(THINK_END), Some(THINK_START), None, None);
    let mut serial: Vec<usize> = (0..active.len()).collect();
    verify_owner_batch_with(
        &model,
        active,
        &sched,
        &mut serial,
        7,
        7,
        false,
        &ctx,
        true,
        strict_ok,
    );
    let log = script.log.lock().unwrap().clone();
    let masks = script.masks.lock().unwrap().clone();
    (serial, log, masks)
}

#[test]
fn a_strict_owner_joins_the_owner_batch_only_with_the_switch_on() {
    // `{` is a legal draft, `Z` is not: the strict owner can verify one.
    let json = bytes("{Z");
    let picks = vec![FREE[0], FREE[1], b'{' as u32, b'"' as u32];

    // Switch off: it is left to the serial verify, its drafts untouched.
    let (mut active, _rx) = owners(&json);
    assert!(owner_rides(&active[0], false) && !owner_rides(&active[1], false));
    let (serial, log, _) = step(&mut active, picks.clone(), false);
    assert!(serial.contains(&1), "strict owner stays serial: {serial:?}");
    assert_eq!(active[1].pending_drafts, json);
    assert!(active[1].output_tokens.is_empty());
    assert!(!log.iter().any(|e| e.starts_with("masks")), "{log:?}");

    // Switch on: one traversal carries the free owner and the strict one.
    let (mut active, _rx) = owners(&json);
    assert!(owner_rides(&active[0], true) && owner_rides(&active[1], true));
    let (serial, log, masks) = step(&mut active, picks, true);
    assert!(serial.is_empty(), "both owners ride: {serial:?}");
    assert_eq!(log[0], format!("masks {} words", 2 * 2 * WORDS));
    assert_eq!(
        log[1],
        format!("owners 2x2 {:?}", [0, FREE[0], LAST, b'{' as u32]),
        "the strict owner carries the drafts its walk kept"
    );
    assert_eq!(active[0].output_tokens, FREE);
    assert_eq!(active[1].output_tokens, bytes("{\""));
    // The matcher advanced by exactly what was emitted.
    let gs = active[1].grammar_state.as_mut().unwrap();
    assert!(gs.accept_token(b'b' as u32) && !gs.accept_token(b'Z' as u32));

    // Rows are owner-major in the batch's owner order: the free owner's
    // allow everything, the strict owner's are its walk's.
    let (free_rows, json_rows) = masks.split_at(2 * WORDS);
    assert!(free_rows.iter().all(|&w| w == u32::MAX));
    assert_eq!(&json_rows[..WORDS], mask_after(&[]));
    assert_eq!(&json_rows[WORDS..], mask_after(&bytes("{")));
}

#[test]
fn a_tool_grammar_never_rides_the_owner_batch() {
    // Its picks come from the host pipeline; no switch admits it.
    let mut engine = grammar_engine();
    let compiled = engine.compile_json_grammar().unwrap();
    let (mut tool, _rx) = test_seq(vec![], 1, None, 8);
    tool.grammar_state = Some(GrammarState::new(&compiled, VOCAB).unwrap());
    assert!(!owner_rides(&tool, false) && !owner_rides(&tool, true));
}

#[test]
fn a_dead_draft_in_the_batch_emits_its_bonus_and_feeds_no_drafter_stats() {
    let fresh = format!("{:?}", test_seq(vec![], 1, None, 1).0.spec_adapt.survival);
    // `Z` cannot open the object: the strict owner verifies a dead draft.
    let (mut active, _rx) = owners(&bytes("Z"));
    let picks = vec![FREE[0], FREE[1], b'{' as u32, b'{' as u32];
    let (serial, log, masks) = step(&mut active, picks, true);
    assert!(serial.is_empty(), "{serial:?}");
    let (row0, row1) = masks[2 * WORDS..].split_at(WORDS);
    assert_eq!(row0, mask_after(&[]));
    assert_eq!(row1, row0, "the bonus row repeats the dead draft's");
    let dead = (0..VOCAB as u32).find(|&t| !allowed(row0, t)).unwrap();
    assert_eq!(
        log[1],
        format!("owners 2x2 {:?}", [0, FREE[0], LAST, dead]),
        "the draft sent is one its own row refuses"
    );
    assert_eq!(active[1].output_tokens, bytes("{"));
    assert_eq!(format!("{:?}", active[1].spec_adapt.survival), fresh);
    assert_ne!(
        format!("{:?}", active[0].spec_adapt.survival),
        fresh,
        "the free owner's verify still counts"
    );
}

#[test]
fn a_dead_think_end_draft_is_carried_as_walked_and_not_walked_again() {
    // Inside reasoning, with the output branch still to choose, a `</think>`
    // draft gives way to a dead `<think>` its row refuses. Walked a second
    // time, that `<think>` would be an ordinary draft on an open row.
    let fresh = format!("{:?}", test_seq(vec![], 1, None, 1).0.spec_adapt.survival);
    let (mut active, _rx) = owners(&[THINK_END]);
    let json = &mut active[1];
    json.grammar_state = Some(strict().with_answer_start(vec![-1i32; WORDS].into_boxed_slice()));
    json.inside_thinking = true;
    json.think_ended = false;
    json.think_start_token = Some(THINK_START);
    json.think_end_token = Some(THINK_END);
    let picks = vec![FREE[0], FREE[1], b'o' as u32, b'o' as u32];
    let (serial, log, masks) = step(&mut active, picks, true);
    assert!(serial.is_empty(), "{serial:?}");
    assert_eq!(
        log[1],
        format!("owners 2x2 {:?}", [0, FREE[0], LAST, THINK_START])
    );
    let (row0, row1) = masks[2 * WORDS..].split_at(WORDS);
    assert!(!allowed(row0, THINK_START), "row 0 refuses the dead draft");
    assert!(allowed(row0, b'o' as u32) && allowed(row0, THINK_END));
    assert_eq!(row1, row0);
    assert_eq!(format!("{:?}", active[1].spec_adapt.survival), fresh);
}

#[test]
fn a_strict_refusal_in_the_batch_ends_that_owner_alone() {
    // The scripted head ignores the masks and picks `Z` as the strict
    // owner's bonus, which its grammar refuses at emission. That request
    // ends with an error, as it would alone; the free owner is untouched.
    let (mut active, _rx) = owners(&bytes("{"));
    let picks = vec![FREE[0], FREE[1], b'{' as u32, b'Z' as u32];
    let (serial, _, _) = step(&mut active, picks, true);
    assert!(serial.is_empty(), "{serial:?}");
    let refused = active[1].engine_error.as_deref();
    assert!(
        refused.is_some_and(crate::scheduler::emit_step::is_strict_refusal),
        "{refused:?}"
    );
    assert!(active[1].finished);
    assert!(active[0].engine_error.is_none() && !active[0].finished);
    assert_eq!(active[0].output_tokens, FREE);
}
