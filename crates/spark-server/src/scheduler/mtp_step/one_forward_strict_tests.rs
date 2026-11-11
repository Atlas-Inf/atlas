// SPDX-License-Identifier: AGPL-3.0-only

//! Strict (`response_format`) sequences on the batched verify under
//! `ATLAS_STRICT_SPEC` (`strict_spec::rides_batch`): the routing, and whole
//! steps against scripted logits rows, where the host pipeline masks every
//! member's rows at that member's own matcher state.

use super::*;
use crate::grammar::GrammarState;
use crate::scheduler::test_support::{
    GRAMMAR_VOCAB as VOCAB, THINK_END, THINK_START, grammar_engine,
};

const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,
    "properties":{"bridge":{"type":"string"}},"required":["bridge"]}"#;
/// A raw top the grammar never allows (SOH).
const ILLEGAL: u32 = 1;

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

/// [`seqs`] past `<think>`, penalty-free, each holding a fresh strict matcher.
fn strict_seqs(spec: Vec<(u32, Vec<u32>)>) -> (Vec<ActiveSeq>, Vec<RespRx>) {
    let (mut active, rx) = seqs(spec);
    for a in &mut active {
        a.think_ended = true;
        a.tool_call_end_token = None;
        a.repetition_penalty = 1.0;
        a.frequency_penalty = 0.0;
        a.presence_penalty = 0.0;
        a.dry_multiplier = 0.0;
        a.grammar_state = Some(strict());
    }
    (active, rx)
}

/// A logits row whose legal top is `top`; with `illegal_above`, [`ILLEGAL`]
/// outscores it, so the raw argmax is a token the grammar refuses.
fn row(top: u32, illegal_above: bool) -> Vec<f32> {
    let mut r = vec![0.0f32; VOCAB];
    r[top as usize] = 40.0;
    if illegal_above {
        r[ILLEGAL as usize] = 50.0;
    }
    r
}

fn rows(tops: &str) -> Vec<Vec<f32>> {
    bytes(tops).into_iter().map(|t| row(t, false)).collect()
}

impl Stub {
    /// The raw argmax of each scripted logits row, where a test scripted the
    /// verify's rows (`bootstrap_rows`); `picks` otherwise.
    pub(super) fn scripted_picks(&self, picks: Vec<u32>) -> Vec<u32> {
        if self.bootstrap_rows.len() < picks.len() {
            return picks;
        }
        self.bootstrap_rows[..picks.len()]
            .iter()
            .map(|r| {
                let top = r.iter().copied().fold(f32::MIN, f32::max);
                r.iter().position(|&v| v == top).unwrap() as u32
            })
            .collect()
    }
}

/// A step of `nd` drafts with `ATLAS_STRICT_SPEC` set, and the dynamic depth
/// where `deep`.
fn strict_step(model: &Stub, active: &mut [ActiveSeq], nd: usize, deep: bool) {
    let sched = strict_sched(deep);
    let ctx = sched.verify_logits_ctx(Some(THINK_END), Some(THINK_START), None, None);
    step_mtp(model, active, &sched, nd, &ctx, false);
}

fn strict_sched(deep: bool) -> SchedCtx {
    let mut levers = SchedLevers::defaults();
    levers.strict_spec = true;
    levers.mtp_dynamic_depth = deep;
    SchedCtx::new(
        crate::scheduler::vocab_masks::VocabMasks::default(),
        std::sync::Arc::new(levers),
        std::sync::Arc::new(crate::scheduler::snapshot::SnapshotCell::default()),
        crate::scheduler::limits::SchedLimits::NONE,
        crate::scheduler::helpers::WatchdogParams::default(),
    )
}

/// With the lever, strict sequences ride the batched verify and a draftless
/// one joins as a decode row; the DFlash batch does not take them.
#[test]
fn the_lever_routes_strict_sequences_into_the_batched_verify() {
    let model = Stub::new(true);
    let sched = strict_sched(false);
    for dflash in [false, true] {
        let (mut active, _rx) = strict_seqs(vec![
            (10, vec![11, 12, 13]),
            (20, vec![21, 22, 23]),
            (30, vec![31, 32, 33]),
            (40, vec![]),
        ]);
        active[1].grammar_state = None;
        active[2].grammar_state = None;
        let boot: &[usize] = if dflash { &[] } else { &[3] };
        let p = super::super::plan_verify(
            &model,
            &sched,
            &mut active,
            &[0, 1, 2],
            boot,
            3,
            None,
            dflash,
        );
        if dflash {
            assert_eq!(p.serial, vec![0], "the DFlash batch keeps strict out");
            assert_eq!(p.batch, vec![1, 2]);
        } else {
            assert!(p.serial.is_empty());
            let mut batch = p.batch.clone();
            batch.sort_unstable();
            assert_eq!(batch, vec![0, 1, 2, 3]);
            assert_eq!(p.decode_rows, vec![3], "a draftless strict sequence");
        }
    }
}

/// Two strict sequences share ONE forward. Each member's rows are masked at
/// its own matcher state: the first accepts its legal drafts, the second
/// (sampled) holds an illegal draft whose row's raw top is illegal too, and
/// emits the grammar's pick there instead.
#[test]
fn strict_sequences_share_one_forward_and_stay_grammar_legal() {
    let mut model = Stub::new(true);
    model.bootstrap_rows = rows(r#"{"br"#);
    model.bootstrap_rows.extend([
        row(b'{' as u32, false),
        row(b'"' as u32, true),
        row(b'b' as u32, false),
    ]);
    let second = vec![b'{' as u32, ILLEGAL, b'"' as u32];
    let (mut active, _rx) = strict_seqs(vec![(10, bytes(r#"{"b"#)), (20, second.clone())]);
    active[1].temperature = 0.8;
    strict_step(&model, &mut active, 3, false);

    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.other, Vec::<&str>::new(), "no per-sequence verify");
    // D-Cut prunes the second member's last draft, as for any batch.
    let mut tokens = vec![10];
    tokens.extend(bytes(r#"{"b"#));
    tokens.push(20);
    tokens.extend(&second[..2]);
    assert_eq!(seen.batched, vec![(vec![4, 3], tokens)]);
    for a in &active {
        assert!(a.engine_error.is_none(), "{:?}", a.engine_error);
        assert!(!a.finished);
    }
    assert_eq!(active[0].output_tokens[1..], bytes(r#"{"br"#));
    assert_eq!(
        active[1].output_tokens[1..],
        bytes(r#"{""#),
        "the legal draft, then the masked pick in place of the illegal one"
    );
    assert_eq!(seen.commits, vec![(4, 4), (2, 3)]);
    // Each matcher advanced exactly by what its sequence emitted.
    let gs = active[0].grammar_state.as_mut().unwrap();
    assert!(gs.accept_token(b'i' as u32) && !gs.accept_token(b'Z' as u32));
    let gs = active[1].grammar_state.as_mut().unwrap();
    assert!(!gs.accept_token(b'Z' as u32) && gs.accept_token(b'b' as u32));
}

/// A member's window crossing `</think>`: its rows inside reasoning are
/// unmasked and the first answer row is masked at the grammar's start.
#[test]
fn a_batched_strict_window_crossing_think_end_masks_the_answer_rows() {
    let mut model = Stub::new(true);
    model.bootstrap_rows = vec![
        row(b'a' as u32, false),
        row(THINK_END, false),
        row(b'{' as u32, true),
        row(b'"' as u32, false),
    ];
    model.bootstrap_rows.extend(rows(r#"{"b"#));
    let crossing = vec![b'a' as u32, THINK_END, ILLEGAL];
    let (mut active, _rx) = strict_seqs(vec![(10, crossing), (20, bytes(r#"{"b"#))]);
    active[0].think_ended = false;
    active[0].inside_thinking = true;
    active[0].enable_thinking = true;
    active[0].think_end_token = Some(THINK_END);
    active[0].think_start_token = Some(THINK_START);
    strict_step(&model, &mut active, 3, false);

    assert_eq!(model.seen.lock().unwrap().other, Vec::<&str>::new());
    for a in &active {
        assert!(a.engine_error.is_none(), "{:?}", a.engine_error);
    }
    assert_eq!(
        active[0].output_tokens[1..],
        [b'a' as u32, THINK_END, b'{' as u32]
    );
    assert!(!active[0].inside_thinking);
    let gs = active[0].grammar_state.as_mut().unwrap();
    assert!(!gs.accept_token(b'Z' as u32) && gs.accept_token(b'"' as u32));
    // D-Cut left the second member two drafts: both accepted, then its bonus.
    assert_eq!(active[1].output_tokens[1..], bytes(r#"{"b"#));
}

/// A lone strict request whose ceiling the controller raised verifies its
/// deep window in the batched forward, as a grammarless one does
/// (`a_lone_deep_window_rides_the_batched_verify`); the per-sequence verify
/// would cut it to 3 drafts.
#[test]
fn a_lone_strict_deep_window_rides_the_batched_verify() {
    // Seven drafts along the key; the raised ceiling of 5 verifies 6 rows.
    let key = bytes(r#"{"bridge"#);
    let mut model = Stub::new(true);
    model.bootstrap_rows = key[..6].iter().map(|&t| row(t, false)).collect();
    let (mut active, _rx) = strict_seqs(vec![(10, key[..7].to_vec())]);
    promote(&mut active[0], 1);
    strict_step(&model, &mut active, 7, true);

    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.other, Vec::<&str>::new());
    let mut tokens = vec![10];
    tokens.extend(&key[..5]);
    assert_eq!(seen.batched, vec![(vec![6], tokens)]);
    assert!(active[0].engine_error.is_none());
    assert_eq!(active[0].output_tokens[1..], key[..6]);
    let gs = active[0].grammar_state.as_mut().unwrap();
    assert!(gs.accept_token(key[6]));
}
