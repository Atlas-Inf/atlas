// SPDX-License-Identifier: AGPL-3.0-only

//! The shared single/batched K-row verdict must account at its output boundary.

use super::super::*;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::verify_k4_verdict::{K4Hidden, k4_apply_verdict};

fn verdict(
    remaining: usize,
    drafts: &[u32],
    picks: &[u32],
    accepted: usize,
    configure: impl FnOnce(&mut ActiveSeq),
) -> (ActiveSeq, RespRx) {
    let (mut a, rx) = test_seq(vec![], remaining, None, 32);
    a.finished = false;
    a.min_tokens = 0;
    configure(&mut a);
    let before = a.seq.seq_len;
    // The model forward has committed the anchor plus all draft input rows.
    a.seq.tokens = vec![7; before];
    a.seq.tokens.push(a.last_token);
    a.seq.tokens.extend_from_slice(drafts);
    a.seq.seq_len += drafts.len() + 1;
    k4_apply_verdict(
        &StubModel::default(),
        &mut a,
        &SchedCtx::for_test(),
        drafts,
        picks,
        vec![],
        drafts.len(),
        accepted,
        K4Hidden::DeferPropose,
        0,
    );
    assert!(a.engine_error.is_none(), "{:?}", a.engine_error);
    a.mtp_acct.record_verify_emitted(a.seq.seq_len - before);
    (a, rx)
}

#[test]
fn prediction_usage_k4_cap_and_bonus_preserve_raw_acceptance() {
    for (budget, expected) in [(1, 1), (2, 2), (4, 3)] {
        let (a, rx) = verdict(budget, &[41, 42, 43], &[41, 42, 43, 44], 3, |_| {});
        assert_eq!(a.mtp_acct.accepted_total(), 3);
        let response = finish_and_recv(a, rx);
        assert_eq!(response.output_tokens.len(), budget);
        assert_eq!(response.accepted_prediction_tokens, expected);
    }
}

#[test]
fn prediction_usage_k4_eos_retains_only_emitted_drafts() {
    let eos = EOS[0];
    let (a, rx) = verdict(8, &[41, eos, 43], &[41, eos, 43, 44], 3, |_| {});
    assert_eq!(a.mtp_acct.accepted_total(), 3);
    let response = finish_and_recv(a, rx);
    assert_eq!(response.output_tokens, vec![41, eos]);
    assert_eq!(response.accepted_prediction_tokens, 2);
}

#[test]
fn prediction_usage_k4_partial_and_rejected_bonus_are_not_drafts() {
    for (accepted, budget, expected_output) in [(0, 1, vec![51]), (1, 2, vec![41, 52])] {
        let picks = if accepted == 0 {
            [51, 52, 53, 54]
        } else {
            [41, 52, 53, 54]
        };
        let (a, rx) = verdict(budget, &[41, 42, 43], &picks, accepted, |_| {});
        assert_eq!(a.mtp_acct.accepted_total(), accepted as u64);
        let response = finish_and_recv(a, rx);
        assert_eq!(response.output_tokens, expected_output);
        assert_eq!(response.accepted_prediction_tokens, accepted);
    }
}

#[test]
fn prediction_usage_k4_suppressed_draft_does_not_credit_bonus() {
    let (a, rx) = verdict(1, &[601], &[601, 42], 1, |a| {
        a.think_end_token = Some(601);
        a.inside_thinking = false;
    });
    assert_eq!(a.mtp_acct.accepted_total(), 1);
    let response = finish_and_recv(a, rx);
    assert_eq!(response.output_tokens, vec![42]);
    assert_eq!(response.accepted_prediction_tokens, 0);
}
