// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::scheduler::cancel_test_model::{SCRIPT, Script, TestModel, VerifyScript};
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::test_seq;

#[test]
fn processed_scores_match_serial_with_history_and_batched_row_offset() {
    for row_base in [0, 1] {
        for temperature in [0.0, 0.7] {
            let mut row = vec![0.0f32; 2048];
            row[300] = 8.0;
            row[1500] = 7.0;
            let mut rows = vec![vec![20.0; 2048]; row_base];
            rows.extend([row.clone(), row.clone()]);
            SCRIPT.with(|s| *s.borrow_mut() = Some(Script { rows, cursor: 0 }));
            let model = TestModel {
                tokens: vec![],
                host_logits: false,
                cancel_after_sampling: None,
                cancel_after_row_commit: None,
                verify: None,
            };
            let (mut a, _rx) = test_seq(vec![], 1, None, 8);
            let (mut serial, _serial_rx) = test_seq(vec![], 1, None, 8);
            for a in [&mut a, &mut serial] {
                a.finished = false;
                a.min_tokens = 0;
                a.temperature = temperature;
                a.top_k = 1;
                a.top_logprobs = Some(2);
                a.logit_bias = vec![(1500, 1.5)];
                a.presence_penalty = 2.0;
            }
            let sched = SchedCtx::for_test();
            let ctx = sched.verify_logits_ctx(None, None, None, None);
            let picked = requested(&model, 2, &mut a, &ctx, row_base, None)
                .unwrap()
                .unwrap();
            assert!(a.output_tokens.is_empty(), "shadow history restored");
            let rejected = requested(&model, 2, &mut a, &ctx, row_base, Some(&[1]))
                .unwrap()
                .unwrap();
            assert_eq!(rejected.tokens, vec![1500]);
            assert_eq!(
                rejected.logprobs.len(),
                1,
                "do not process rows past first rejected draft"
            );
            assert!(a.output_tokens.is_empty());

            let bytes: Vec<u8> = row
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            for i in 0..2 {
                let (token, lp) = crate::scheduler::decode_logits_seq::process_seq_logits(
                    &model,
                    &mut serial,
                    &bytes,
                    0,
                    2048,
                    2,
                    false,
                    &ctx,
                    false,
                );
                let lp = lp.unwrap();
                assert_eq!(picked.tokens[i], token);
                assert_eq!(picked.logprobs[i].token_id, lp.token_id);
                assert_eq!(picked.logprobs[i].logprob, lp.logprob);
                assert_eq!(picked.logprobs[i].top, lp.top);
                assert!(lp.logprob < -0.1, "scores precede top-k renormalization");
                serial.output_tokens.push(token);
            }
            assert_eq!(
                picked.tokens,
                vec![1500, 300],
                "second row sees accepted history"
            );
            SCRIPT.with(|s| *s.borrow_mut() = None);
        }
    }
}

#[test]
fn unrequested_scoring_performs_no_model_read_or_pick() {
    SCRIPT.with(|s| *s.borrow_mut() = None);
    let model = TestModel {
        tokens: vec![],
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    };
    let (mut a, _rx) = test_seq(vec![], 1, None, 8);
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    assert!(
        requested(&model, 2, &mut a, &ctx, 0, None)
            .unwrap()
            .is_none()
    );
}

use crate::scheduler::verify_dflash_step::step_verify_dflash;
use std::sync::Arc;

fn model(script: Arc<VerifyScript>) -> TestModel {
    TestModel {
        tokens: vec![],
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: Some(script),
    }
}

#[test]
fn dense_logprobs_report_the_biased_selection_distribution() {
    for temperature in [0.0, 0.7] {
        let mut row = vec![0.0f32; 2048];
        row[300] = 12.0;
        row[1500] = -8.0;
        SCRIPT.with(|s| {
            *s.borrow_mut() = Some(Script {
                rows: vec![row.clone(); 2],
                cursor: 0,
            })
        });
        let script = Arc::new(VerifyScript {
            picks: vec![300; 2],
            ..Default::default()
        });
        let model = model(script.clone());
        let (mut a, _rx) = test_seq(vec![], 1, None, 8);
        a.finished = false;
        a.min_tokens = 0;
        a.top_logprobs = Some(2);
        a.temperature = temperature;
        a.top_k = 1;
        a.logit_bias = vec![(1500, 100.0)];
        let sched = SchedCtx::for_test();
        let ctx = sched.verify_logits_ctx(None, None, None, None);
        step_verify_dflash(&model, &mut a, &sched, &[1], &[], 1, &ctx, true);
        assert!(a.engine_error.is_none(), "{:?}", a.engine_error);
        assert_eq!(a.output_tokens, vec![1500]);
        row[1500] += 100.0;
        let expected = spark_model::traits::logprob_of(&row, 1500, 2);
        assert_eq!(a.logprobs_data[0].logprob, expected.0);
        assert_eq!(a.logprobs_data[0].top, expected.1);
        SCRIPT.with(|s| *s.borrow_mut() = None);
    }
}

#[test]
fn scored_dense_thinking_boundary_commits_once_and_stops_at_rejection() {
    for boundary_row in [0, 1] {
        let mut row = vec![0.0f32; 2048];
        row[1500] = 12.0;
        SCRIPT.with(|s| {
            *s.borrow_mut() = Some(Script {
                rows: vec![row; 2],
                cursor: 0,
            })
        });
        let script = Arc::new(VerifyScript {
            picks: vec![1500; 2],
            ..Default::default()
        });
        let model = model(script.clone());
        let (mut a, _rx) = test_seq(vec![], boundary_row + 1, None, 8);
        a.finished = false;
        a.min_tokens = 0;
        a.top_logprobs = Some(2);
        a.inside_thinking = true;
        a.think_end_token = Some(99);
        a.force_end_thinking = true;
        a.thinking_budget = Some(128);
        a.thinking_tokens = if boundary_row == 0 { 384 } else { 128 };
        if boundary_row == 1 {
            a.sentence_defer_count = crate::scheduler::confidence::MAX_SENTENCE_DEFER_TOKENS - 1;
        }
        let sched = SchedCtx::for_test();
        let ctx = sched.verify_logits_ctx(Some(99), None, None, None);
        step_verify_dflash(&model, &mut a, &sched, &[1500], &[], 1, &ctx, true);
        assert!(a.engine_error.is_none(), "{:?}", a.engine_error);
        assert_eq!(
            a.output_tokens,
            if boundary_row == 0 {
                vec![99]
            } else {
                vec![1500, 99]
            }
        );
        assert_eq!(a.logprobs_data.len(), boundary_row + 1);
        assert_eq!(a.logprobs_data[boundary_row].token_id, 99);
        assert_eq!(a.logprobs_data[boundary_row].logprob, 0.0);
        assert!(!a.inside_thinking && a.think_ended && a.think_force_closed);
        assert_eq!(
            a.remaining, 0,
            "only the emitted boundary consumes the budget"
        );
        assert!(
            script
                .log
                .lock()
                .unwrap()
                .iter()
                .any(|x| x == &format!("cmd {boundary_row:#x}"))
        );
        SCRIPT.with(|s| *s.borrow_mut() = None);
    }
}
