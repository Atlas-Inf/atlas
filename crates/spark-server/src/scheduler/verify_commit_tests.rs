// SPDX-License-Identifier: AGPL-3.0-only

//! Every verify path commits the accepted rows BEFORE it emits, with the
//! rejected rows already rolled back.
//!
//! An emit can finish the sequence (stop token, budget, guard) and return.
//! `finish_sequence` then caches `seq.tokens` together with the sequence's
//! live recurrent state, and the worker rank has already committed the step.
//! A path that returned before its commit left the head's state at the
//! pre-verify position while `seq.tokens` and the worker were past it. With
//! KDA records every accept folds rows in, full accepts included.
//!
//! The model also reads `seq.tokens.len()` at the commit as the accepted
//! length (`finish_leaf::leaf_save`), so each case checks the length the
//! stub saw there: the pre-verify length plus the committed rows.

use super::lifecycle_tests::{COMMITS, StubModel, VERIFY_PICKS};
use super::sched_ctx::SchedCtx;
use super::test_support::{EOS, test_seq};
use super::types::ActiveSeq;
use super::verify_dflash_batch_step::apply_dflash_accept;
use super::verify_dflash_step::verify_dflash_tail;
use super::verify_k4_verdict::{K4Hidden, k4_apply_verdict};

const PRE: usize = 40;

/// A live sequence as `decode_verify_dflash` leaves it: `seq_len` and
/// `seq.tokens` advanced by the whole `k`-row verify block.
fn verified(k: usize) -> ActiveSeq {
    let (mut a, rx) = test_seq(vec![10], 20, None, PRE + k);
    // A dropped receiver reads as a client disconnect: keep it alive.
    std::mem::forget(rx);
    a.finished = false;
    a.min_tokens = 0;
    a.seq.tokens = (0..(PRE + k) as u32).collect();
    a
}

fn commits() -> Vec<(usize, usize, usize)> {
    COMMITS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

/// Run the per-sequence tail over `drafts` against the target's `argmax`.
fn tail(a: &mut ActiveSeq, drafts: &[u32], argmax: &[u32]) -> Option<usize> {
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    let tokens: Vec<u32> = std::iter::once(a.last_token)
        .chain(drafts.iter().copied())
        .collect();
    verify_dflash_tail(
        &StubModel::default(),
        a,
        &sched,
        drafts,
        &[],
        drafts.len(),
        &ctx,
        true,
        &tokens,
        argmax.to_vec(),
        false,
        0.0,
        true,
        true,
    )
}

#[test]
fn tail_commits_when_the_bonus_token_finishes_the_sequence() {
    // Two of four drafts accepted; the bonus (row 2) is the stop token.
    let mut a = verified(5);
    assert_eq!(tail(&mut a, &[1, 2, 3, 4], &[1, 2, EOS[0], 9, 9]), None);
    assert!(a.finished, "the stop token must finish the sequence");
    assert_eq!(a.seq.tokens.len(), PRE + 3);
    assert_eq!(a.seq.seq_len, PRE + 3);
    assert_eq!(commits(), vec![(PRE + 3, 3, 5)]);
}

#[test]
fn tail_commits_when_an_accepted_draft_finishes_the_sequence() {
    // All four drafts accepted; draft 1 is the stop token, so the emit loop
    // returns with two accepted rows still un-emitted. The state still has to
    // hold every accepted row, exactly as `seq.tokens` and the worker do.
    let mut a = verified(5);
    let drafts = [1, EOS[0], 3, 4];
    assert_eq!(tail(&mut a, &drafts, &[1, EOS[0], 3, 4, 7]), None);
    assert!(a.finished);
    assert_eq!(a.output_tokens, vec![10, 1, EOS[0]]);
    assert_eq!(commits(), vec![(PRE + 5, 5, 5)]);
}

#[test]
fn tail_commits_once_when_the_sequence_continues() {
    let mut a = verified(5);
    assert_eq!(tail(&mut a, &[1, 2, 3, 4], &[1, 2, 8, 9, 9]), Some(4));
    assert!(!a.finished);
    assert_eq!(a.last_token, 8);
    assert_eq!(commits(), vec![(PRE + 3, 3, 5)]);
}

#[test]
fn batched_accept_commits_when_an_emit_finishes_the_sequence() {
    let sched = SchedCtx::for_test();
    let accept = |drafts: &[u32], argmax: &[u32]| {
        let mut a = verified(4);
        apply_dflash_accept(
            &StubModel::default(),
            &mut a,
            &sched,
            drafts,
            argmax,
            3,
            true,
        );
        (a.finished, commits())
    };
    // The bonus is the stop token.
    assert_eq!(
        accept(&[1, 2, 3], &[1, EOS[0], 9, 9]),
        (true, vec![(PRE + 2, 2, 4)])
    );
    // An accepted draft is the stop token.
    assert_eq!(
        accept(&[EOS[0], 2, 3], &[EOS[0], 2, 3, 7]),
        (true, vec![(PRE + 4, 4, 4)])
    );
    // No finish: still exactly one commit.
    assert_eq!(
        accept(&[1, 2, 3], &[1, 5, 9, 9]),
        (false, vec![(PRE + 2, 2, 4)])
    );
}

/// A live sequence before its verify: `PRE` tokens, `last_token` the anchor.
fn unverified() -> ActiveSeq {
    let (mut a, rx) = test_seq(vec![10], 20, None, PRE);
    std::mem::forget(rx);
    a.finished = false;
    a.min_tokens = 0;
    a.seq.tokens = (0..PRE as u32).collect();
    a
}

#[test]
fn k4_verdict_commits_before_an_emit_can_finish_the_sequence() {
    let sched = SchedCtx::for_test();
    let verdict = |drafts: &[u32], v: &[u32], na: usize| {
        let mut a = verified(drafts.len() + 1);
        let hidden = K4Hidden::DeferPropose;
        let model = StubModel::default();
        k4_apply_verdict(&model, &mut a, &sched, drafts, v, vec![], 3, na, hidden, 0);
        (a.finished, a.seq.tokens.len(), commits())
    };
    // Full accept, an accepted draft is the stop token: the path that used
    // to return from the emit loop without a commit.
    assert_eq!(
        verdict(&[1, EOS[0], 3], &[1, EOS[0], 3, 7], 3),
        (true, PRE + 4, vec![(PRE + 4, 4, 4)])
    );
    // Full accept, the bonus is the stop token.
    assert_eq!(
        verdict(&[1, 2, 3], &[1, 2, 3, EOS[0]], 3),
        (true, PRE + 4, vec![(PRE + 4, 4, 4)])
    );
    // Full accept that continues, at K=2 too: still exactly one commit.
    assert_eq!(
        verdict(&[1], &[1, 7], 1),
        (false, PRE + 2, vec![(PRE + 2, 2, 2)])
    );
    // Partial accept: the rollback comes first, then the commit.
    assert_eq!(
        verdict(&[1, 2, 3], &[1, EOS[0], 9, 9], 1),
        (true, PRE + 2, vec![(PRE + 2, 2, 4)])
    );
    assert_eq!(
        verdict(&[1, 2, 3], &[5, 9, 9, 9], 0),
        (false, PRE + 1, vec![(PRE + 1, 1, 4)])
    );
}

/// Run a whole K=2 or K=3 step against the scripted `picks`.
fn k_step(drafts: &[u32], picks: &[u32]) -> (bool, usize, Vec<(usize, usize, usize)>) {
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    let (model, mut a) = (StubModel::default(), unverified());
    VERIFY_PICKS.with(|p| *p.borrow_mut() = picks.to_vec());
    if drafts.len() == 1 {
        super::verify_k2_step::step_verify_k2(&model, &mut a, &sched, drafts, 1, &ctx, false);
    } else {
        super::verify_k3_step::step_verify_k3(&model, &mut a, &sched, drafts, 2, &ctx, false);
    }
    assert_eq!(a.engine_error, None);
    (a.finished, a.seq.tokens.len(), commits())
}

#[test]
fn k2_and_k3_steps_commit_before_an_emit_can_finish_the_sequence() {
    // Full accepts that finish on an accepted draft or on the bonus.
    assert_eq!(
        k_step(&[EOS[0]], &[EOS[0], 7]),
        (true, PRE + 2, vec![(PRE + 2, 2, 2)])
    );
    assert_eq!(
        k_step(&[1], &[1, EOS[0]]),
        (true, PRE + 2, vec![(PRE + 2, 2, 2)])
    );
    assert_eq!(
        k_step(&[1, EOS[0]], &[1, EOS[0], 7]),
        (true, PRE + 3, vec![(PRE + 3, 3, 3)])
    );
    assert_eq!(
        k_step(&[1, 2], &[1, 2, EOS[0]]),
        (true, PRE + 3, vec![(PRE + 3, 3, 3)])
    );
    // Rejects and partial accepts: rolled back, then committed.
    assert_eq!(
        k_step(&[1], &[EOS[0], 9]),
        (true, PRE + 1, vec![(PRE + 1, 1, 2)])
    );
    assert_eq!(
        k_step(&[1, 2], &[1, EOS[0], 9]),
        (true, PRE + 2, vec![(PRE + 2, 2, 3)])
    );
    assert_eq!(
        k_step(&[1, 2], &[EOS[0], 9, 9]),
        (true, PRE + 1, vec![(PRE + 1, 1, 3)])
    );
}

#[test]
fn ngram_verify_commits_the_rolled_back_prefix_before_it_emits() {
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    let step = |drafts: &[u32], picks: &[u32]| {
        let (model, mut a) = (StubModel::default(), unverified());
        let mut proposer = crate::ngram::NgramProposer::new(3);
        VERIFY_PICKS.with(|p| *p.borrow_mut() = picks.to_vec());
        a.pending_drafts = drafts.to_vec();
        let active = std::slice::from_mut(&mut a);
        super::spec_step::step_ngram(&model, active, &sched, &mut proposer, false, &ctx);
        assert_eq!(a.engine_error, None);
        (a.finished, a.seq.tokens.len(), commits())
    };
    // Full accept that finishes on a draft, at K=4 and K=2.
    assert_eq!(
        step(&[1, EOS[0], 3], &[1, EOS[0], 3, 7]),
        (true, PRE + 4, vec![(PRE + 4, 4, 4)])
    );
    assert_eq!(
        step(&[EOS[0]], &[EOS[0], 7]),
        (true, PRE + 2, vec![(PRE + 2, 2, 2)])
    );
    // Partial accept: one of two drafts, the correction is the stop token.
    assert_eq!(
        step(&[1, 2], &[1, EOS[0], 9]),
        (true, PRE + 2, vec![(PRE + 2, 2, 3)])
    );
}
