// SPDX-License-Identifier: AGPL-3.0-only

//! Copy drafts on the MTP path (`copy_drafts::mtp`,
//! `ATLAS_QWEN_COPY_DRAFTS`) through `step_mtp` on the exact-lane stub, at
//! the copy knobs' defaults (an 8-token match, 7 drafts): a copy stands in
//! for the head's propose, verifies and rolls back like the head's drafts,
//! and settles as a copy round.

use super::{BONUS, DRAFT, REFUSED, Stub, deep_step_with, promote};
use crate::scheduler::dflash_conf_width::COPY_CONF;
use crate::scheduler::levers::SchedLevers;
use crate::scheduler::test_support::{RespRx, test_seq};
use crate::scheduler::types::ActiveSeq;

/// A deep step (7-draft ceiling) with copies on or off.
fn step(model: &Stub, active: &mut [ActiveSeq], copies: bool) {
    let mut levers = SchedLevers::defaults();
    levers.qwen_copy_drafts = copies;
    deep_step_with(model, active, levers);
}

/// A live, greedy, penalty-free request on pool slot `slot` over `prompt`
/// that has emitted `reply` (its last token pending) and holds the head's
/// `drafts`, with its response receiver.
fn request(slot: usize, prompt: &[u32], reply: &[u32], drafts: &[u32]) -> (ActiveSeq, RespRx) {
    let (mut a, rx) = test_seq(reply.to_vec(), 50, None, 0);
    a.finished = false;
    a.min_tokens = 0;
    a.lz_penalty = 0.0;
    a.seq.slot_idx = slot;
    a.seq.tokens = prompt
        .iter()
        .chain(&reply[..reply.len() - 1])
        .copied()
        .collect();
    a.seq.seq_len = a.seq.tokens.len();
    a.pending_drafts = drafts.to_vec();
    (a, rx)
}

/// A prompt the reply below quotes: it re-emits `99..=106`, the target then
/// picks `BONUS` (member 0's bonus row), and the prompt goes on with
/// `1, 2, third, 4, ..`.
fn prompt(third: u32) -> Vec<u32> {
    vec![
        50, 99, 100, 101, 102, 103, 104, 105, 106, BONUS, 1, 2, third, 4, 5, 6, 7, 8, 9, 60,
    ]
}

/// A lone request quoting its prompt: after its verify the copy stands in
/// for the head's propose, then verifies past the head's own depth (5) to the
/// step's ceiling (7) in ONE forward, settles as a copy round and trims no
/// drafter row. The next propose is the head's again, at its own depth.
#[test]
fn a_lone_copy_replaces_the_heads_propose_and_verifies_to_the_ceiling() {
    let model = Stub::new(true);
    let (a, _rx) = request(0, &prompt(3), &[99, 100, 101, 102], &[103, 104, 105, 106]);
    let mut active = vec![a];
    promote(&mut active[0], 1);
    step(&model, &mut active, true);
    {
        let seen = model.seen.lock().unwrap();
        assert_eq!(seen.batched, vec![(vec![5], vec![102, 103, 104, 105, 106])]);
        assert_eq!(seen.proposes, 0, "the copy stood in for the head");
        assert_eq!(seen.trims, 1, "the head's round trims its rows");
    }
    assert_eq!(active[0].pending_drafts, [1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(active[0].draft_conf(), [COPY_CONF; 7]);

    step(&model, &mut active, true);
    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.batched[1], (vec![8], vec![BONUS, 1, 2, 3, 4, 5, 6, 7]));
    assert_eq!(
        active[0].output_tokens,
        [
            99, 100, 101, 102, 103, 104, 105, 106, BONUS, 1, 2, 3, 4, 5, 6, 7, BONUS
        ]
    );
    assert_eq!(seen.commits, vec![(5, 5), (8, 8)]);
    assert_eq!(seen.trims, 1, "a copy round trims no drafter row");
    assert_eq!(
        active[0].spec_adapt.copy.tally(),
        ([1, 0], [7, 0], [7, 0], false)
    );
    // Nothing to copy after it: the head drafts again, at its own depth.
    assert_eq!(seen.proposes, 1);
    assert_eq!(active[0].pending_drafts, vec![DRAFT; 5]);
}

/// A copy the target parts from at its third draft keeps two: the verify
/// rewinds the rejected tail and commits the kept prefix as for the head's
/// drafts, emits the target's own pick there, and books a missed round.
#[test]
fn a_partly_kept_copy_rolls_back_like_the_heads_drafts() {
    let model = Stub::new(true);
    let (a, _rx) = request(
        0,
        &prompt(REFUSED),
        &[99, 100, 101, 102],
        &[103, 104, 105, 106],
    );
    let mut active = vec![a];
    promote(&mut active[0], 1);
    step(&model, &mut active, true);
    assert_eq!(active[0].pending_drafts, [1, 2, REFUSED, 4, 5, 6, 7]);
    let before = active[0].seq.tokens.len();
    step(&model, &mut active, true);
    let a = &active[0];
    assert!(
        a.output_tokens.ends_with(&[BONUS, 1, 2, BONUS]),
        "{:?}",
        a.output_tokens
    );
    assert_eq!(a.last_token, BONUS);
    // Eight rows joined the sequence; the five past the kept prefix left it.
    assert_eq!(
        (a.seq.tokens.len(), a.seq.seq_len),
        (before + 3, before + 3)
    );
    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.commits[1], (3, 8));
    assert_eq!(seen.trims, 1, "a copy round trims no drafter row");
    assert_eq!(a.spec_adapt.copy.tally(), ([1, 0], [7, 0], [2, 0], true));
}

/// In a batch only the member whose context holds a copy skips the head; the
/// other proposes as before. Beside another member the copy then verifies at
/// the head's depth (3): only a lone request verifies past it.
#[test]
fn in_a_batch_a_copy_replaces_only_its_own_propose_and_keeps_the_ladder() {
    let model = Stub::new(true);
    let reply = [99, 100, 101, 102, 103, 104, 105];
    let (a, _rx0) = request(0, &prompt(3), &reply, &[106]);
    let (b, _rx1) = request(1, &[200, 201, 202, 203], &[300, 301], &[302]);
    let mut active = vec![a, b];
    step(&model, &mut active, true);
    {
        let seen = model.seen.lock().unwrap();
        assert_eq!(seen.batched, vec![(vec![2, 2], vec![105, 106, 301, 302])]);
        assert_eq!(seen.proposes, 1, "the head proposed for the other member");
    }
    assert_eq!(active[0].pending_drafts, [1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(active[1].pending_drafts, vec![DRAFT; 3]);

    step(&model, &mut active, true);
    let seen = model.seen.lock().unwrap();
    let (ks, tokens) = &seen.batched[1];
    assert_eq!(ks.iter().max(), Some(&4), "the head's depth of 3: {ks:?}");
    assert!(tokens.starts_with(&[BONUS, 1, 2, 3]), "{tokens:?}");
}

/// A decode row bootstraps: the head proposes for it even where its context
/// holds a copy (its first propose builds the drafter's prompt context).
#[test]
fn a_decode_row_always_proposes_from_the_head() {
    let model = Stub::new(true);
    let (a, _rx0) = request(0, &[10, 11, 12], &[20, 21], &[22]);
    // Member 1's row picks `BONUS + 1`, which its prompt goes on from.
    let quoted = [
        50,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        BONUS + 1,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        60,
    ];
    let (b, _rx1) = request(1, &quoted, &[99, 100, 101, 102, 103, 104, 105], &[]);
    let mut active = vec![a, b];
    step(&model, &mut active, true);
    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.batched, vec![(vec![2, 1], vec![21, 22, 105])]);
    assert_eq!(active[1].last_token, BONUS + 1);
    assert_eq!(active[1].pending_drafts, vec![DRAFT; 3]);
    assert_eq!(seen.proposes, 1, "one batched propose for both");
}

/// With the lever off the same request is the head's alone: it proposes
/// after every verify, verifies at its own depth and books no copy round.
#[test]
fn with_copies_off_the_head_drafts_every_round() {
    let model = Stub::new(true);
    let (a, _rx) = request(0, &prompt(3), &[99, 100, 101, 102], &[103, 104, 105, 106]);
    let mut active = vec![a];
    promote(&mut active[0], 1);
    step(&model, &mut active, false);
    step(&model, &mut active, false);
    let seen = model.seen.lock().unwrap();
    assert_eq!(
        seen.batched[1],
        (vec![6], vec![BONUS, DRAFT, DRAFT, DRAFT, DRAFT, DRAFT])
    );
    assert_eq!((seen.proposes, seen.trims), (2, 2));
    assert_eq!(
        active[0].spec_adapt.copy.tally(),
        ([0, 0], [0, 0], [0, 0], false)
    );
}

/// The lever is inert where verify rows are not serial decode's.
#[test]
fn copies_need_an_exact_verify() {
    let model = Stub::new(false);
    let reply = [99, 100, 101, 102, 103, 104, 105];
    let (a, _rx0) = request(0, &prompt(3), &reply, &[106]);
    let (b, _rx1) = request(1, &[200, 201, 202, 203], &[300, 301], &[302]);
    let mut active = vec![a, b];
    step(&model, &mut active, true);
    assert_eq!(active[0].pending_drafts, vec![DRAFT; 3]);
    assert_eq!(
        model.seen.lock().unwrap().proposes,
        1,
        "one batched propose"
    );
}
