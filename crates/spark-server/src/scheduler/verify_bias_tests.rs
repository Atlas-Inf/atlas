// SPDX-License-Identifier: AGPL-3.0-only

//! Verify-position bias contract: `PositionKind::Verify` must apply exactly
//! the base bias FinalDecode would at the same sequence state — the
//! `<tool_call>` opener nudge included — and the fast-greedy gates must not
//! bypass it.

use super::sample_step::{PositionKind, advance_in_tool_flag, penalty_params_for};
use super::test_support::test_seq;

#[test]
fn verify_params_carry_sequence_base_bias() {
    let (mut a, _rx) = test_seq(vec![], 16, None, 32);
    a.logit_bias = vec![(151644, 3.0)];
    let decode = penalty_params_for(
        &a,
        PositionKind::FinalDecode,
        0.0,
        None,
        a.logit_bias.clone(),
    );
    let verify = penalty_params_for(&a, PositionKind::Verify, 0.0, None, a.logit_bias.clone());
    assert_eq!(decode.logit_bias, verify.logit_bias);
    assert_eq!(verify.logit_bias, vec![(151644, 3.0)]);
}

#[test]
fn verify_bias_strips_opener_inside_tool_body() {
    let (mut a, _rx) = test_seq(vec![], 16, None, 32);
    a.logit_bias = vec![(151644, 3.0)];
    a.tool_call_start_token = Some(151644);
    a.inside_tool_body = true;
    let verify = penalty_params_for(&a, PositionKind::Verify, 0.0, None, a.logit_bias.clone());
    assert!(verify.logit_bias.iter().all(|&(id, _)| id != 151644));
}

#[test]
fn in_tool_flag_advances_and_rolls_back() {
    // Opener sets, closer clears, anything else preserves, thinking freezes.
    assert!(advance_in_tool_flag(false, false, 7, Some(7), Some(8)));
    assert!(!advance_in_tool_flag(false, true, 8, Some(7), Some(8)));
    assert!(advance_in_tool_flag(false, true, 9, Some(7), Some(8)));
    assert!(!advance_in_tool_flag(false, false, 9, Some(7), Some(8)));
    assert!(!advance_in_tool_flag(true, false, 7, Some(7), Some(8)));
}

#[test]
fn fast_greedy_gate_blocks_on_bias() {
    use super::fast_greedy::{PenaltyGate, classify_penalties};
    let (mut a, _rx) = test_seq(vec![], 16, None, 32);
    a.logit_bias = vec![(151644, 3.0)];
    let p = penalty_params_for(&a, PositionKind::Verify, 0.0, None, a.logit_bias.clone());
    assert_eq!(classify_penalties(&p), PenaltyGate::Blocked);
}

#[test]
fn fast_gate_declines_when_closer_lands_mid_block() {
    // The confirmed loss (parallel_multiple_78): block starts inside a tool
    // body, `</tool_call>` lands mid-block, and the rows after it must see
    // the +3 opener bias again — a once-per-block gate predicts Neutral and
    // returns raw argmax (turn-ending EOS) where the slow path picks the
    // biased `<tool_call>`. The per-row gate must decline the fast path.
    use super::fast_greedy::{PenaltyGate, fast_gate_over_rows};
    const OPENER: u32 = 14;
    const CLOSER: u32 = 15;
    let (mut a, _rx) = test_seq(vec![], 16, None, 32);
    a.logit_bias = vec![(OPENER, 3.0)];
    a.tool_call_start_token = Some(OPENER);
    a.tool_call_end_token = Some(CLOSER);
    a.inside_tool_body = true;
    // Rows [content, closer, post-close, post-close]: row 2 sees
    // in_tool=false → bias applies → Blocked.
    let gate = fast_gate_over_rows(&mut a, &[100, CLOSER, 11, 1010]);
    assert_eq!(gate, PenaltyGate::Blocked);
    assert!(a.inside_tool_body, "flag restored after the gate scan");
}

#[test]
fn fast_gate_still_fires_without_mid_block_transition() {
    // Same in_tool block, no closer among the rows: the bias stays stripped
    // for the whole block — the fast path remains valid (the speedup must
    // not be lost).
    use super::fast_greedy::{PenaltyGate, fast_gate_over_rows};
    const OPENER: u32 = 14;
    const CLOSER: u32 = 15;
    let (mut a, _rx) = test_seq(vec![], 16, None, 32);
    a.logit_bias = vec![(OPENER, 3.0)];
    a.tool_call_start_token = Some(OPENER);
    a.tool_call_end_token = Some(CLOSER);
    a.inside_tool_body = true;
    assert_eq!(
        fast_gate_over_rows(&mut a, &[100, 1010, 200, 300]),
        PenaltyGate::Neutral
    );
    assert!(a.inside_tool_body);
    // Outside the body the +3 bias is live on every row → Blocked from row 0.
    a.inside_tool_body = false;
    assert_eq!(
        fast_gate_over_rows(&mut a, &[100, 1010, 200, 300]),
        PenaltyGate::Blocked
    );
    assert!(!a.inside_tool_body);
    // An OPENER mid-block transitions in_tool true → bias stripped later,
    // but row 0 (outside) already classifies Blocked.
    assert_eq!(
        fast_gate_over_rows(&mut a, &[OPENER, 100, 1010]),
        PenaltyGate::Blocked
    );
}

#[test]
fn slow_and_fast_agree_at_post_closer_tie() {
    // Mechanism test: at the row after the closer, the slow path applies
    // the +3 opener bias — the pick at a raw tie between EOS and
    // `<tool_call>` must be `<tool_call>` in BOTH paths. The fast path can
    // only promise that when it declines (Blocked), letting the slow path
    // run the biased pick.
    use super::fast_greedy::{PenaltyGate, fast_gate_over_rows};
    const OPENER: u32 = 14;
    const CLOSER: u32 = 15;
    const EOS: u32 = 11;
    let (mut a, _rx) = test_seq(vec![], 16, None, 32);
    a.logit_bias = vec![(OPENER, 3.0)];
    a.tool_call_start_token = Some(OPENER);
    a.tool_call_end_token = Some(CLOSER);
    a.inside_tool_body = true;
    // Block: [.., closer, EOS-raw-argmax, ..] — the row after the
    // closer has raw argmax EOS but slow-path pick <tool_call> (+3.0).
    assert_eq!(
        fast_gate_over_rows(&mut a, &[100, CLOSER, EOS, 1010]),
        PenaltyGate::Blocked,
        "fast path must decline so the slow path emits the biased pick"
    );
    // And the slow path side: with the flag advanced past the closer, the
    // params carry the opener bias again.
    a.inside_tool_body = advance_in_tool_flag(
        false,
        a.inside_tool_body,
        CLOSER,
        a.tool_call_start_token,
        a.tool_call_end_token,
    );
    let p = penalty_params_for(&a, PositionKind::Verify, 0.0, None, a.logit_bias.clone());
    assert!(
        p.logit_bias.contains(&(OPENER, 3.0)),
        "post-closer row must re-carry the +3 opener bias"
    );
}
