// SPDX-License-Identifier: AGPL-3.0-only

//! A sampled request speculating on the split head commits exactly serial
//! decode's tokens under a seed (the verify rows' bits equal the decode's, as
//! `ATLAS_GLM_CANONICAL_VERIFY` makes them): from the candidates where they
//! decide each row, from the whole rows where they do not (penalties, a pure
//! temperature draw). The harness is `loop_steer_tests.rs`'s: one scripted
//! logits stream, decoded serially (`process_decode_logits`) and as verify
//! spans (`split_verify_picks` + the accept prefix).

use super::split_verify_picks;
use crate::scheduler::cancel_test_model::{SCRIPT, Script, TestModel, VerifyScript};
use crate::scheduler::decode_logits_step::process_decode_logits;
use crate::scheduler::emit_step::emit_token;
use crate::scheduler::levers::SchedLevers;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::test_seq;
use crate::scheduler::types::ActiveSeq;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::sampler::argmax_first_wins_f32;
use std::sync::Arc;

const V: usize = 2048;
const N: usize = 60;

fn sched() -> SchedCtx {
    SchedCtx::new(
        Default::default(),
        Arc::new(SchedLevers::defaults()),
        Arc::default(),
        crate::scheduler::limits::SchedLimits::NONE,
        Default::default(),
    )
}

fn model(verify: &Arc<VerifyScript>) -> TestModel {
    TestModel {
        tokens: Vec::new(),
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: Some(verify.clone()),
    }
}

/// BF16-exact logits: a few strong tokens over a flat tail, so drafts of the
/// serial picks are often right and the draws still vary.
fn rows(seed: u64) -> Vec<Vec<f32>> {
    let mut x = seed;
    let mut next = move || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) as u32
    };
    (0..N + 8)
        .map(|_| {
            let mut r: Vec<f32> = (0..V).map(|_| (next() % 64) as f32 / 16.0).collect();
            for _ in 0..6 {
                r[(next() as usize) % V] = 10.0 + (next() % 8) as f32 / 4.0;
            }
            r
        })
        .collect()
}

fn at(cursor: usize) {
    SCRIPT.with(|s| s.borrow_mut().as_mut().unwrap().cursor = cursor);
}

/// (temperature, top_k, top_p, min_p, presence).
type Params = (f32, u32, f32, f32, f32);

/// A sampled chat sequence.
fn seq(p: Params, seed: u64) -> ActiveSeq {
    let (mut a, rx) = test_seq(vec![300, 301], 1000, None, 40);
    std::mem::forget(rx);
    a.finished = false;
    a.min_tokens = 0;
    (a.temperature, a.top_k, a.top_p, a.min_p, a.presence_penalty) = p;
    a.seed = Some(seed);
    a
}

fn serial(m: &TestModel, p: Params, seed: u64) -> Vec<u32> {
    let s = sched();
    let mut batch = vec![seq(p, seed)];
    for t in 0..N {
        at(t);
        let none = None;
        process_decode_logits(
            m,
            &mut batch,
            DevicePtr::NULL,
            std::time::Instant::now(),
            none,
            none,
            none,
            none,
            false,
            &s,
        );
    }
    batch.pop().unwrap().output_tokens
}

fn speculative(m: &TestModel, p: Params, seed: u64, want: &[u32]) -> Vec<u32> {
    let s = sched();
    let ctx = s.verify_logits_ctx(None, None, None, None);
    let stream = SCRIPT.with(|s| s.borrow().as_ref().unwrap().rows.clone());
    let mut a = seq(p, seed);
    let mut pos = 0usize;
    let mut step = 0usize;
    while pos < N {
        let k = [5usize, 3, 8][step % 3].min(N - pos + 1);
        step += 1;
        at(pos);
        let gpu: Vec<u32> = (0..k)
            .map(|i| argmax_first_wins_f32(&stream[pos + i]))
            .collect();
        // The serial picks, every third one wrong.
        let drafts: Vec<u32> = (0..k - 1)
            .map(|i| {
                let d = want.get(2 + pos + i).copied().unwrap_or(0);
                if (pos + i) % 3 == 2 { d ^ 1 } else { d }
            })
            .collect();
        let picks = split_verify_picks(m, &gpu, &drafts, &mut a, &ctx).unwrap();
        let accepted = drafts
            .iter()
            .zip(&picks)
            .take_while(|(d, v)| d == v)
            .count();
        assert!(
            picks.len() == accepted + 1,
            "the span ends at its first rejection"
        );
        for &d in &drafts[..accepted] {
            emit_token(&mut a, d, None, &s);
        }
        emit_token(&mut a, picks[accepted], None, &s);
        pos += accepted + 1;
    }
    a.output_tokens.truncate(2 + N);
    a.output_tokens
}

#[test]
fn split_head_speculation_samples_what_serial_decode_samples() {
    // (params, whole rows fetched)
    let cases: [(Params, bool); 5] = [
        ((0.7, 8, 0.95, 0.0, 0.0), false),
        ((1.0, 0, 1.0, 0.3, 0.0), false),
        ((1.0, 0, 0.5, 0.0, 0.0), false),
        ((1.0, 0, 1.0, 0.0, 0.0), true),
        ((0.7, 8, 0.95, 0.0, 0.5), true),
    ];
    for (c, &(p, whole)) in cases.iter().enumerate() {
        for seed in 0..4u64 {
            SCRIPT.with(|s| {
                *s.borrow_mut() = Some(Script {
                    rows: rows(seed * 31 + c as u64),
                    cursor: 0,
                })
            });
            let verify = Arc::new(VerifyScript::default());
            let m = model(&verify);
            let want = serial(&m, p, seed);
            let got = speculative(&m, p, seed, &want);
            assert_eq!(got, want, "case {c} seed {seed}");
            let log = verify.log.lock().unwrap();
            assert_eq!(
                log.iter().any(|e| e.starts_with("full")),
                whole,
                "case {c} seed {seed}: {log:?}"
            );
            SCRIPT.with(|s| *s.borrow_mut() = None);
        }
    }
}
