// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn unscored_logprob_admission_keeps_dflash_requests_on_single_sequence_path() {
    let model = Stub::new(false);
    let sched = SchedCtx::for_test();
    for dflash in [false, true] {
        let (mut active, _rx) = seqs(vec![
            (10, vec![11, 12, 13]),
            (20, vec![21, 22, 23]),
            (30, vec![31, 32, 33]),
        ]);
        active[0].top_logprobs = Some(5);
        let plan = super::super::plan_verify(
            &model,
            &sched,
            &mut active,
            &[0, 1, 2],
            &[],
            3,
            None,
            dflash,
        );
        if dflash {
            assert_eq!(
                plan.serial,
                vec![0],
                "the single-sequence DFlash path reports scores"
            );
            assert_eq!(plan.batch, vec![1, 2], "plain requests still batch");
        } else {
            assert!(plan.serial.is_empty());
            assert_eq!(plan.batch, vec![0, 1, 2], "native MTP batch reports scores");
        }
    }
}

#[test]
fn batched_logprob_read_failure_aborts_every_advanced_member() {
    let mut model = Stub::new(false);
    model.fail_logits = true;
    let (mut active, _rx) = seqs(vec![(10, vec![11, 12, 13]), (20, vec![21, 22, 23])]);
    active[0].top_logprobs = Some(2);
    step(&model, &mut active);
    assert!(
        active
            .iter()
            .all(|a| a.finished && a.engine_error.is_some()),
        "one failed score read invalidates every already-forwarded member"
    );
    assert_eq!(active[0].output_tokens, vec![10]);
    assert_eq!(active[1].output_tokens, vec![20]);
    assert!(model.seen.lock().unwrap().commits.is_empty());
}
