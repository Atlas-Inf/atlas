// SPDX-License-Identifier: AGPL-3.0-only

//! `verify_pick_batch` (members' host pipelines on the rayon pool) picks and
//! leaves state exactly as `verify_pick_all_with_pipeline` per member does.

use super::verify_pick_batch;
use crate::scheduler::cancel_test_model::{SCRIPT, Script, TestModel};
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::{THINK_END, THINK_START, test_seq};
use crate::scheduler::types::ActiveSeq;
use crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline;
use spark_runtime::sampler::argmax_first_wins_f32;
use std::sync::Arc;

const V: usize = 2048;

fn sched() -> SchedCtx {
    let masks = crate::scheduler::vocab_masks::VocabMasks {
        boundary: Some((0..V as u32).map(|t| t == 46).collect()),
        mid_word: Some((0..V as u32).map(|t| (65..=90).contains(&t)).collect()),
        ..Default::default()
    };
    let watchdog = crate::scheduler::helpers::WatchdogParams {
        confidence_early_stop: true,
        confidence_run_length: 2,
        ..Default::default()
    };
    SchedCtx::new(
        masks,
        Arc::new(crate::scheduler::levers::SchedLevers::defaults()),
        Arc::default(),
        crate::scheduler::limits::SchedLimits::NONE,
        watchdog,
    )
}

/// Member `m`: thinking members around F2's 400-token gate (with and
/// without a presence penalty) and plain-chat members the GPU arms serve.
fn member(m: usize) -> ActiveSeq {
    let (mut a, rx) = test_seq(vec![300, 301 + m as u32], 400, None, 40);
    std::mem::forget(rx);
    a.finished = false;
    a.min_tokens = 0;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a.inside_thinking = m % 4 != 3;
    a.enable_thinking = true;
    a.thinking_tokens = [398, 420, 10][m % 3];
    a.presence_penalty = if m.is_multiple_of(2) { 0.5 } else { 0.0 };
    a
}

fn rows(seed: u64, n: usize) -> Vec<Vec<f32>> {
    let mut x = seed;
    let mut next = move || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) as u32
    };
    (0..n)
        .map(|_| {
            let mut r: Vec<f32> = (0..V).map(|_| (next() % 8) as f32).collect();
            let win = (100 + next() % 600) as usize;
            match next() % 8 {
                0 => r[THINK_END as usize] = 12.0,
                1 => {
                    r[win] = 12.0;
                    r[win + 1] = 12.0;
                }
                2..=4 => r[win] = 40.0,
                _ => r[win] = 12.0,
            }
            r
        })
        .collect()
}

#[test]
fn batch_picks_equal_per_member_picks() {
    let model = TestModel {
        tokens: Vec::new(),
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    };
    for seed in 1..=12u64 {
        let ks = [4usize, 3, 4, 1, 2, 4, 4, 3];
        let mut off = vec![0usize];
        for k in ks {
            off.push(off.last().unwrap() + k);
        }
        let stream = rows(seed, off[ks.len()]);
        let gpu: Vec<u32> = stream.iter().map(|r| argmax_first_wins_f32(r)).collect();
        SCRIPT.with(|s| {
            *s.borrow_mut() = Some(Script {
                rows: stream,
                cursor: 0,
            })
        });
        let s = sched();
        let ctx = s.verify_logits_ctx(Some(THINK_END), Some(THINK_START), None, None);

        let mut want: Vec<ActiveSeq> = (0..ks.len()).map(member).collect();
        let want_picks: Vec<Vec<u32>> = want
            .iter_mut()
            .enumerate()
            .map(|(i, a)| {
                verify_pick_all_with_pipeline(&model, &gpu[off[i]..off[i + 1]], a, &ctx, off[i])
            })
            .collect();

        let mut got: Vec<ActiveSeq> = (0..ks.len()).map(member).collect();
        let mut refs: Vec<&mut ActiveSeq> = got.iter_mut().collect();
        let got_picks = verify_pick_batch(&model, &gpu, &off, &mut refs, &ctx);

        assert_eq!(got_picks, want_picks, "seed {seed}");
        for (g, w) in got.iter().zip(&want) {
            let state = |a: &ActiveSeq| {
                (
                    a.inside_thinking,
                    a.thinking_tokens,
                    a.force_end_thinking,
                    a.consecutive_confident,
                    a.sentence_defer_count,
                    a.output_tokens.clone(),
                )
            };
            assert_eq!(state(g), state(w), "seed {seed}");
        }
        SCRIPT.with(|s| *s.borrow_mut() = None);
    }
}
