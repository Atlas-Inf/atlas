// SPDX-License-Identifier: AGPL-3.0-only

use crate::scheduler::cancel_test_model::{SCRIPT, Script, TestModel, VerifyScript};
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::test_seq;
use crate::scheduler::verify_dflash_step::{step_verify_dflash, step_verify_glm_long_with};
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
fn split_logprobs_fetch_full_rows_before_the_worker_verdict() {
    for temperature in [0.0, 0.7] {
        let mut row = vec![0.0f32; 2048];
        // The chosen token is on the remote shard; a local-only normalizer
        // would disagree with the whole-row reference.
        row[1500] = 12.0;
        row[300] = 11.0;
        SCRIPT.with(|s| {
            *s.borrow_mut() = Some(Script {
                rows: vec![row.clone(); 2],
                cursor: 0,
            })
        });
        let script = Arc::new(VerifyScript {
            picks: vec![1500; 2],
            ..Default::default()
        });
        let model = model(script.clone());
        let (mut a, _rx) = test_seq(vec![], 1, None, 8);
        a.finished = false;
        a.min_tokens = 0;
        a.top_logprobs = Some(2);
        a.temperature = temperature;
        a.top_k = 1;
        let sched = SchedCtx::for_test();
        let ctx = sched.verify_logits_ctx(None, None, None, None);
        step_verify_dflash(&model, &mut a, &sched, &[1], &[], 1, &ctx, true);
        assert!(a.engine_error.is_none());
        assert_eq!(a.output_tokens, vec![1500]);
        let lp = &a.logprobs_data[0];
        let expected = spark_model::traits::logprob_of(&row, 1500, 2);
        assert_eq!(lp.token_id, 1500);
        assert_eq!(lp.logprob, expected.0);
        assert_eq!(lp.top, expected.1);
        let log = script.log.lock().unwrap();
        let full = log
            .iter()
            .position(|e| e == "full 2")
            .expect("whole vocabulary gathered");
        let verdict = log
            .iter()
            .position(|e| e == "cmd 0x0")
            .expect("verdict sent");
        assert!(full < verdict, "{log:?}");
        SCRIPT.with(|s| *s.borrow_mut() = None);
    }
}

#[test]
fn split_logprobs_report_the_biased_selection_distribution() {
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
fn unsupported_sampled_split_head_fails_before_any_worker_command() {
    SCRIPT.with(|s| *s.borrow_mut() = None);
    let script = Arc::new(VerifyScript {
        picks: vec![10; 2],
        ..Default::default()
    });
    let model = model(script.clone());
    let (mut a, _rx) = test_seq(vec![], 1, None, 8);
    a.finished = false;
    a.temperature = 0.7;
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    step_verify_dflash(&model, &mut a, &sched, &[1], &[], 1, &ctx, true);
    assert!(a.finished && a.engine_error.is_some());
    assert!(a.output_tokens.is_empty());
    assert!(script.log.lock().unwrap().is_empty());
}

#[test]
fn unsupported_owner_group_is_rejected_before_traversal() {
    SCRIPT.with(|s| *s.borrow_mut() = None);
    let model = model(Arc::new(VerifyScript::default()));
    let (mut a, _rx) = test_seq(vec![], 1, None, 8);
    a.finished = false;
    a.temperature = 0.7;
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    let mut traversed = false;
    step_verify_glm_long_with(
        &model,
        &mut [&mut a],
        &sched,
        1,
        &ctx,
        true,
        &[],
        &mut |_, _, _| {
            traversed = true;
            Ok(vec![])
        },
    );
    assert!(!traversed);
    assert!(a.finished && a.engine_error.is_some());
}

#[test]
fn unsupported_split_head_keeps_greedy_without_logprobs() {
    SCRIPT.with(|s| *s.borrow_mut() = None);
    let script = Arc::new(VerifyScript {
        picks: vec![10; 2],
        ..Default::default()
    });
    let model = model(script.clone());
    let (mut a, _rx) = test_seq(vec![], 1, None, 8);
    a.finished = false;
    a.min_tokens = 0;
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    step_verify_dflash(&model, &mut a, &sched, &[1], &[], 1, &ctx, true);
    assert!(a.engine_error.is_none());
    assert_eq!(a.output_tokens, vec![10]);
    assert!(
        !script
            .log
            .lock()
            .unwrap()
            .iter()
            .any(|e| e.starts_with("full"))
    );
}

/// The same injected errors exercise real single/owner entry points. EP
/// cases run in a child test process because terminate() deliberately uses
/// _exit and must not be replaced by a test-only production branch.
fn failed_exchange(operation: &str, is_ep: bool, owners: bool) {
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
        is_ep,
        fail_candidates: operation == "candidates",
        fail_full_rows: operation == "logprobs",
        ..Default::default()
    });
    let model = model(script.clone());
    let (mut a, _rx) = test_seq(vec![], 1, None, 8);
    let (mut b, _rx_b) = test_seq(vec![], 1, None, 8);
    for seq in [&mut a, &mut b] {
        seq.finished = false;
        seq.min_tokens = 0;
        seq.temperature = 0.7;
        seq.top_k = 1;
        seq.top_logprobs = (operation == "logprobs").then_some(2);
        seq.pending_drafts = vec![1];
    }
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    if owners {
        step_verify_glm_long_with(
            &model,
            &mut [&mut a, &mut b],
            &sched,
            1,
            &ctx,
            true,
            &[],
            &mut |rows, tokens, seqs| {
                for (owner, seq) in seqs.iter_mut().enumerate() {
                    seq.seq_len += rows;
                    seq.tokens
                        .extend_from_slice(&tokens[owner * rows..(owner + 1) * rows]);
                }
                Ok(vec![1500; 4])
            },
        );
        assert!(b.finished && b.engine_error.is_some());
        assert!(!script.log.lock().unwrap().iter().any(|e| e == "owner 1"));
    } else {
        step_verify_dflash(&model, &mut a, &sched, &[1], &[], 1, &ctx, true);
    }
    assert!(!is_ep, "an uncertain EP exchange must never return");
    assert!(a.finished && a.engine_error.is_some());
    assert!(a.output_tokens.is_empty());
    assert!(!script.log.lock().unwrap().iter().any(|e| e == "cmd 0x0"));
    SCRIPT.with(|s| *s.borrow_mut() = None);
}

#[test]
fn non_ep_split_exchange_failure_stops_request_and_owner_group() {
    for operation in ["candidates", "logprobs"] {
        for owners in [false, true] {
            failed_exchange(operation, false, owners);
        }
    }
}

#[test]
fn ep_split_exchange_failure_exits_rank_without_next_owner_or_verdict() {
    const CHILD: &str = "ATLAS_TEST_SPLIT_EXCHANGE_EXIT";
    if let Ok(case) = std::env::var(CHILD) {
        let (operation, owners) = case.split_once(':').unwrap();
        failed_exchange(operation, true, owners == "owners");
        panic!("fatal EP exchange failure returned");
    }
    let full = concat!(
        module_path!(),
        "::ep_split_exchange_failure_exits_rank_without_next_owner_or_verdict"
    );
    let (_, name) = full.split_once("::").unwrap();
    for operation in ["candidates", "logprobs"] {
        for scope in ["single", "owners"] {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture"])
                .env(CHILD, format!("{operation}:{scope}"))
                .status()
                .unwrap();
            assert_eq!(status.code(), Some(74), "{operation}:{scope}: {status}");
        }
    }
}

#[test]
fn scored_split_thinking_boundary_commits_once_and_stops_at_rejection() {
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
