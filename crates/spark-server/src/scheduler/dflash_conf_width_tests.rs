// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// A pick the drafter is sure of, and one it is not.
const SURE: f32 = -0.001;
const UNSURE: f32 = -2.5;

/// The rule without its periodic full-width step.
const NO_PROBE: Params = Params {
    tau: DEFAULT_PARAMS.tau,
    probe: 0,
};

fn choose(policy: &mut Policy, owners: &[&[f32]], max: usize, params: Params) -> Option<usize> {
    policy.choose(owners.iter().copied(), max, params)
}

fn lone(conf: &[f32]) -> usize {
    choose(&mut Policy::default(), &[conf], conf.len(), NO_PROBE).unwrap()
}

#[test]
fn bins_cover_the_confidence_range_in_order() {
    assert_eq!(bin(f32::NEG_INFINITY), 0);
    assert_eq!(bin(-1.5), 1);
    assert_eq!(bin(-0.02), COPY_BIN - 2);
    assert_eq!(bin(0.0), COPY_BIN - 1);
    assert!(PRIOR[..COPY_BIN].windows(2).all(|w| w[0] < w[1]));
    // Copied drafts bin apart from every drafter confidence, by source.
    assert_eq!(bin(COPY_CONF), COPY_BIN);
    assert_eq!(bin(COPY_REPLY_CONF), COPY_REPLY_BIN);
    assert_eq!(COPY_REPLY_BIN, BINS - 1);
    assert!(is_copy(&[COPY_CONF]) && is_copy(&[COPY_REPLY_CONF]));
    assert!(!is_copy(&[SURE]) && !is_copy(&[]) && !is_copy(&[f32::NAN]));
}

#[test]
fn the_copy_priors_verify_a_prompt_copy_whole_and_a_reply_copy_narrow() {
    // Per-draft rates that keep 74% and 23% of five drafts.
    let kept = |p: f32| (1..=5).map(|j| p.powi(j)).sum::<f32>() / 5.0;
    assert!((kept(PRIOR[COPY_BIN]) - 0.74).abs() < 0.01);
    assert!((kept(PRIOR[COPY_REPLY_BIN]) - 0.23).abs() < 0.01);
    assert_eq!(lone(&[COPY_CONF; 7]), 7);
    assert_eq!(lone(&[COPY_REPLY_CONF; 7]), 2);
}

#[test]
fn copied_drafts_are_measured_and_calibrate_only_their_own_bin() {
    let mut policy = Policy::default();
    // The drafter's bins hold evidence, which copy rounds must not move.
    for _ in 0..30 {
        policy.calibration.record(&[SURE; 7], 7, 4);
        policy.calibration.record(&[-0.04; 7], 7, 2);
    }
    let drafter =
        |policy: &Policy| [SURE, -0.04, UNSURE].map(|c| policy.calibration.survival(&[c; 7]));
    let before = drafter(&policy);
    let reply_before = policy.calibration.survival(&[COPY_REPLY_CONF; 7]);
    let copies = [COPY_CONF; 7];
    for _ in 0..50 {
        policy.calibration.record(&copies, 7, 0);
    }
    // Rejected copies lower the prompt copies' survival, not the drafter's
    // nor the reply copies'.
    assert_eq!(drafter(&policy), before);
    assert_eq!(
        policy.calibration.survival(&[COPY_REPLY_CONF; 7]),
        reply_before
    );
    assert!(policy.calibration.survival(&copies)[0] < PRIOR[COPY_BIN] / 2.0);
    assert_eq!(choose(&mut policy, &[&copies[..]], 7, NO_PROBE), Some(2));
    // Drafter rounds leave the copies' evidence where it is.
    let copies_after = policy.calibration.survival(&copies);
    for _ in 0..200 {
        policy.calibration.record(&[SURE; 7], 7, 7);
    }
    assert_eq!(policy.calibration.survival(&copies), copies_after);
}

#[test]
fn copy_rounds_do_not_move_the_drafters_probe() {
    let params = Params {
        tau: DEFAULT_PARAMS.tau,
        probe: 4,
    };
    let mut policy = Policy::default();
    let unsure: &[f32] = &[UNSURE; 7];
    let copies: &[f32] = &[COPY_REPLY_CONF; 7];
    let mut widths = Vec::new();
    for _ in 0..4 {
        widths.push(choose(&mut policy, &[unsure], 7, params).unwrap());
        // A copy round between every drafter round: probed on its own count.
        choose(&mut policy, &[copies], 7, params).unwrap();
    }
    assert_eq!(widths, [2, 2, 2, 7]);
    assert_eq!((policy.steps, policy.copy_steps), (4, 4));
    // A batch with any drafter owner is the drafter's step.
    choose(&mut policy, &[copies, unsure], 7, params).unwrap();
    assert_eq!((policy.steps, policy.copy_steps), (5, 4));
}

#[test]
fn a_sure_block_verifies_wide_and_an_unsure_one_narrow() {
    assert_eq!(lone(&[SURE; 7]), 7);
    // Two drafts is the floor for a lone owner: its 2-row verify costs more
    // than its 3-row one, so one draft never pays.
    assert_eq!(lone(&[UNSURE; 7]), 2);
}

#[test]
fn the_draft_is_cut_at_the_first_position_below_the_threshold() {
    let cal = Calibration::default();
    for cut in 2..7 {
        let mut conf = [SURE; 7];
        conf[cut] = UNSURE;
        let survival = cal.survival(&conf);
        assert!(survival[cut - 1] > DEFAULT_PARAMS.tau && survival[cut] < DEFAULT_PARAMS.tau);
        assert_eq!(lone(&conf), cut, "cut at position {cut}");
    }
}

#[test]
fn survival_is_the_running_product() {
    // Sure picks after an unsure one cannot recover the block: every later
    // draft is reached only through the unsure one.
    let cal = Calibration::default();
    let survival = cal.survival(&[SURE, SURE, UNSURE, SURE, SURE, SURE, SURE]);
    assert!(survival.windows(2).all(|w| w[1] <= w[0]));
    assert!(survival[6] < survival[2] && survival[2] < 0.2);
    // Middling picks fall below the threshold together, none of them alone.
    let middling = [-0.04; 7];
    assert!(cal.survival(&middling)[0] > 2.0 * DEFAULT_PARAMS.tau);
    assert!((2..=4).contains(&lone(&middling)));
}

#[test]
fn a_higher_threshold_cuts_earlier() {
    let conf = [-0.04; 7];
    let at = |tau| {
        let params = Params { tau, probe: 0 };
        choose(&mut Policy::default(), &[&conf], 7, params).unwrap()
    };
    assert!(at(0.6) < at(0.25) && at(0.25) < at(0.05));
    assert_eq!(at(0.0), 7);
}

#[test]
fn every_probe_period_verifies_the_full_width() {
    let mut policy = Policy::default();
    let params = Params {
        tau: DEFAULT_PARAMS.tau,
        probe: 4,
    };
    let widths: Vec<usize> = (0..12)
        .map(|_| choose(&mut policy, &[&[UNSURE; 7]], 7, params).unwrap())
        .collect();
    assert_eq!(widths, [2, 2, 2, 7, 2, 2, 2, 7, 2, 2, 2, 7]);
    // The probe respects the caller's cap.
    let mut policy = Policy::default();
    let probe_every_step = Params {
        tau: DEFAULT_PARAMS.tau,
        probe: 1,
    };
    assert_eq!(
        choose(&mut policy, &[&[UNSURE; 7]], 5, probe_every_step),
        Some(5)
    );
}

#[test]
fn owners_share_one_width() {
    let (sure, unsure): (&[f32], &[f32]) = (&[SURE; 7], &[UNSURE; 7]);
    let mut policy = Policy::default();
    let mut width = |owners: &[&[f32]]| choose(&mut policy, owners, 7, NO_PROBE).unwrap();
    assert_eq!(width(&[sure, sure]), 7);
    // One sure owner keeps the batch wider than the unsure owner alone
    // would be, and narrower than it would be alone.
    let mixed = width(&[sure, unsure]);
    assert!(mixed >= width(&[unsure, unsure]) && mixed <= 7);
    // Every extra row costs more with more owners: four unsure owners are
    // no wider than one.
    assert!(width(&[unsure; 4]) <= width(&[unsure]));
    // 8 owners x 4 rows is the widest batch the 32-row budget admits.
    assert_eq!(width(&[sure; 8]), 3);
}

#[test]
fn unmeasured_drafts_leave_the_callers_policy() {
    let mut policy = Policy::default();
    let sure = [SURE; 7];
    // No confidences (a drafter that does not report them), fewer than the
    // drafts held, or a non-finite one: not a number to cut on.
    assert_eq!(choose(&mut policy, &[&[]], 7, NO_PROBE), None);
    assert_eq!(choose(&mut policy, &[&sure[..3]], 7, NO_PROBE), None);
    assert_eq!(choose(&mut policy, &[&sure, &[]], 7, NO_PROBE), None);
    assert_eq!(choose(&mut policy, &[], 7, NO_PROBE), None);
    assert_eq!(choose(&mut policy, &[&sure], 0, NO_PROBE), None);
    // A grammar-truncated owner holds fewer drafts than were proposed: the
    // confidences' prefix describes them.
    assert_eq!(choose(&mut policy, &[&sure], 3, NO_PROBE), Some(3));
}

/// Below a request's `min_tokens` floor the selector bans end tokens: a row
/// whose every scored candidate is banned reports NaN, and a banned pick's
/// runner-up is reported as sure. Neither may cut or widen on a number the
/// target's acceptance does not back.
#[test]
fn min_tokens_bans_fall_back_and_recalibrate() {
    let mut policy = Policy::default();
    let mut conf = [SURE; 7];
    conf[1] = f32::NAN;
    assert_eq!(choose(&mut policy, &[&conf], 7, NO_PROBE), None);
    // The NaN sits past the drafts this owner holds: measured.
    assert_eq!(choose(&mut policy, &[&conf], 1, NO_PROBE), Some(1));

    // Forced continuation: "sure" runner-up picks that the target rejects at
    // the second draft. The table learns that within a few dozen verifies...
    let forced = [SURE; 7];
    assert_eq!(choose(&mut policy, &[&forced], 7, NO_PROBE), Some(7));
    for _ in 0..40 {
        policy.calibration.record(&forced, 7, 1);
    }
    assert_eq!(choose(&mut policy, &[&forced], 7, NO_PROBE), Some(2));
    // ...and unlearns it once sure picks are accepted again.
    for _ in 0..200 {
        policy.calibration.record(&forced, 7, 7);
    }
    assert_eq!(choose(&mut policy, &[&forced], 7, NO_PROBE), Some(7));
}

#[test]
fn record_observes_only_reached_positions() {
    let mut cal = Calibration::default();
    let conf = [SURE; 7];
    let top = bin(SURE);
    // Three drafts verified, the first accepted: position 0 accepted,
    // position 1 rejected, position 2 never reached, 3.. never verified.
    cal.record(&conf, 3, 1);
    assert_eq!((cal.verified[0][top], cal.accepted[0][top]), (1.0, 1.0));
    assert_eq!((cal.verified[1][top], cal.accepted[1][top]), (1.0, 0.0));
    assert!(cal.verified[2..].iter().all(|row| row[top] == 0.0));
    // A full accept observes every verified position and nothing past it.
    let mut cal = Calibration::default();
    cal.record(&conf, 3, 3);
    assert!((0..3).all(|d| cal.accepted[d][top] == 1.0));
    assert_eq!(cal.verified[3][top], 0.0);
}

#[test]
fn a_cut_position_cannot_stay_mislearned() {
    // Deep drafts were rejected for a while at a confidence the rule now
    // cuts on: the cell is no longer verified, so nothing would correct it.
    let mut cal = Calibration::default();
    let conf = [-0.04; 7];
    let (depth, b) = (4, bin(-0.04));
    for _ in 0..60 {
        cal.record(&conf, 7, 4);
    }
    let learned = cal.rate(depth, b);
    assert!(learned < 0.2);
    // Verifies that stop short of that position still decay its cell back
    // towards the bin's all-depth rate, which the verified positions keep
    // current.
    for _ in 0..1500 {
        cal.record(&conf, 3, 3);
    }
    assert!(cal.rate(depth, b) > 0.75);
    assert!(cal.rate(depth, b) <= cal.rate(0, b));
}

#[test]
fn rows_are_priced_at_todays_step_cost() {
    // One owner: 5.68 ms a row from 3 to 8 rows; its 2-row verify is slower
    // than its 3-row one, so a lone owner never verifies one draft.
    assert!((row_ms() - 5.68).abs() < 1e-3);
    assert!(((step_ms(1, 8) - step_ms(1, 3)) / 5.0 - row_ms()).abs() < 1e-3);
    assert!(step_ms(1, 2) > step_ms(1, 3));
    // Four owners: 3.2 ms an owner-row, linear (not the concave 2026-09-28
    // table, which priced deep rows too cheaply).
    assert!((step_ms(4, 5) - step_ms(4, 4) - 4.0 * 3.2).abs() < 1e-3);
    assert!((step_ms(4, 8) - step_ms(4, 4) - 16.0 * 3.2).abs() < 1e-3);
    // More owners cost more per step and less per owner-row, and never past
    // the 32-row verify budget.
    for n in 1..8 {
        assert!(step_ms(n + 1, 3) > step_ms(n, 3));
        assert!(step_ms(n + 1, 4) - step_ms(n + 1, 3) >= step_ms(n, 4) - step_ms(n, 3));
    }
    assert!(step_ms(8, 4).is_finite() && step_ms(8, 5).is_infinite());
}

/// Four prose streams whose drafter is unsure past its first drafts: the
/// shared width stays within a 3-row pin (two drafts), where the stale
/// table's cheap deep rows verified wider.
#[test]
fn four_unsure_prose_owners_stay_within_a_three_row_pin() {
    let prose: &[f32] = &[-0.05, -0.3, -0.6, -0.9, -1.2, -1.6, -2.0];
    let width = choose(&mut Policy::default(), &[prose; 4], 7, NO_PROBE).unwrap();
    assert!(width <= 2, "four unsure owners verified {width} drafts");
    // Alone, the same drafts pay for no more rows than together.
    assert!(lone(prose) >= width);
}

#[test]
fn confidences_travel_with_the_drafts() {
    let (mut a, _rx) = crate::scheduler::test_support::test_seq(vec![1], 8, None, 4);
    a.pending_drafts = vec![5, 6, 7];
    a.pending_draft_conf = vec![-0.1, -0.2, -0.3];
    assert_eq!(a.draft_conf(), [-0.1, -0.2, -0.3]);
    // Stale (not draft for draft): not measured.
    a.pending_draft_conf.truncate(2);
    assert!(a.draft_conf().is_empty());
    a.pending_draft_conf.push(-0.3);
    // Taking the drafts takes their confidences and leaves neither pending.
    assert_eq!(a.take_drafts(), (vec![5, 6, 7], vec![-0.1, -0.2, -0.3]));
    assert!(a.pending_drafts.is_empty() && a.pending_draft_conf.is_empty());
    // A drafter that reports no confidences leaves none, even over stale ones.
    a.pending_draft_conf = vec![-0.5];
    a.set_proposed_drafts(vec![8, 9]);
    assert_eq!(a.pending_drafts, [8, 9]);
    assert!(a.pending_draft_conf.is_empty());
    assert_eq!(a.take_drafts(), (vec![8, 9], Vec::new()));
}
