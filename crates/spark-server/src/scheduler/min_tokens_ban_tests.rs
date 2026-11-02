// SPDX-License-Identifier: AGPL-3.0-only

//! The qwen4_exp min_tokens end-token ban (`min_tokens_ban`): speculation
//! commits exactly serial decode's tokens under it (single-sequence spans of
//! every width and the batched verify's picks), nothing banned is picked
//! below the floor, a request without min_tokens is unaffected, and, on a
//! forced-length run past its natural end, how many verify steps the ban
//! saves.
//!
//! The harness is `think_commit/exact_tests.rs`'s: one deterministic logits
//! stream, a row per position, decoded serially (`process_decode_logits`)
//! and as verify spans (`verify_pick_all_with_pipeline` + the K-verdict:
//! drafts accepted while they equal the picks, then the first pick that
//! differs). Every pick, emitted or discarded, consumes a position.

use super::{banned_ids, first_token_suppress};
use crate::scheduler::cancel_test_model::{SCRIPT, Script, TestModel};
use crate::scheduler::decode_logits_step::process_decode_logits;
use crate::scheduler::emit_step::emit_token;
use crate::scheduler::levers::SchedLevers;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::test_seq;
use crate::scheduler::types::ActiveSeq;
use crate::scheduler::verify_pipeline_helper::{verify_pick_all_with_pipeline, verify_pick_batch};
use spark_model::traits::EosBan;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::sampler::argmax_first_wins_f32;
use std::sync::Arc;

const V: usize = 2048;
/// The model end tokens (`<|im_end|>`, `<|endoftext|>` stand-ins).
const END: u32 = 900;
const ALT: u32 = 901;
const MODEL_END: [u32; 2] = [END, ALT];
/// Tokens already out before the stream (`test_seq`).
const PRIOR: usize = 2;
/// A generation budget no stream here reaches.
const BUDGET: usize = 1000;

#[derive(Clone, Copy, Debug)]
struct Case {
    /// `ignore_eos`: the request has no end tokens.
    ignore_eos: bool,
    min_tokens: usize,
    /// The target ban (`ATLAS_QWEN4EXP_EOS_BAN`) armed, else the plain ban.
    target: bool,
    presence: f32,
    /// `SchedLevers::fast_greedy_chat` (the verify GPU-argmax arm).
    chat_fast: bool,
    seed: u64,
}

fn sched(c: Case) -> SchedCtx {
    let mut levers = SchedLevers::defaults();
    levers.fast_greedy_chat = c.chat_fast;
    SchedCtx::new(
        Default::default(),
        Arc::new(levers),
        Arc::default(),
        crate::scheduler::limits::SchedLimits::NONE,
        Default::default(),
    )
}

fn model() -> TestModel {
    TestModel {
        tokens: Vec::new(),
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    }
}

/// A chat sequence armed as admission arms it.
fn chat_seq(c: Case, remaining: usize) -> ActiveSeq {
    let (mut a, rx) = test_seq(vec![300, 301], remaining, None, 40);
    std::mem::forget(rx);
    a.finished = false;
    a.min_tokens = c.min_tokens;
    a.eos_tokens = if c.ignore_eos { Vec::new() } else { vec![END] };
    a.presence_penalty = c.presence;
    a.seq.eos_ban = if c.target {
        EosBan::targeted(0, c.min_tokens, &MODEL_END, &a.eos_tokens)
    } else {
        EosBan::new(0, c.min_tokens, &a.eos_tokens)
    };
    a
}

/// Small integer logits (exact in BF16), one designated winner a row; the
/// end tokens win a third of the rows, alone or tied with a higher id.
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
            r[win] = 11.0;
            match next() % 16 {
                0..=3 => r[END as usize] = 12.0,
                4 => r[ALT as usize] = 12.0,
                5 => {
                    // A tie the lower end token wins unbanned.
                    r[END as usize] = 12.0;
                    r[950] = 12.0;
                }
                6 => {
                    r[win] = 12.0;
                    r[win + 1 + (next() % 300) as usize] = 12.0;
                }
                7..=9 => r[win] = 40.0,
                _ => r[win] = 12.0,
            }
            r
        })
        .collect()
}

fn script(stream: Vec<Vec<f32>>) {
    SCRIPT.with(|s| {
        *s.borrow_mut() = Some(Script {
            rows: stream,
            cursor: 0,
        })
    });
}

fn at(cursor: usize) {
    SCRIPT.with(|s| s.borrow_mut().as_mut().unwrap().cursor = cursor);
}

fn stream() -> Vec<Vec<f32>> {
    SCRIPT.with(|s| s.borrow().as_ref().unwrap().rows.clone())
}

/// Serial decode: the sequence and its pick at every position.
fn serial(c: Case, n: usize) -> (ActiveSeq, Vec<u32>) {
    let s = sched(c);
    let mut batch = vec![chat_seq(c, BUDGET)];
    let mut picks = Vec::new();
    for t in 0..n {
        at(t);
        let none = None;
        process_decode_logits(
            &model(),
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
        picks.push(batch[0].last_token);
        if batch[0].finished {
            break;
        }
    }
    (batch.pop().unwrap(), picks)
}

/// Verify spans of `ks` rows (cycled) over `n` positions, `budget` tokens;
/// `draft(p)` drafts position `p`. Returns the sequence and its steps.
fn speculative(
    c: Case,
    n: usize,
    budget: usize,
    ks: &[usize],
    draft: impl Fn(usize) -> u32,
) -> (ActiveSeq, usize) {
    let s = sched(c);
    let ctx = s.verify_logits_ctx(None, None, None, None);
    let stream = stream();
    let mut a = chat_seq(c, budget);
    let (mut p, mut step) = (0usize, 0usize);
    while p < n && !a.finished {
        let k = ks[step % ks.len()].min(n - p);
        step += 1;
        at(p);
        let gpu: Vec<u32> = (0..k)
            .map(|i| argmax_first_wins_f32(&stream[p + i]))
            .collect();
        let drafts: Vec<u32> = (0..k - 1).map(|i| draft(p + i)).collect();
        let picks = verify_pick_all_with_pipeline(&model(), &gpu, &mut a, &ctx, 0);
        let accepted = drafts
            .iter()
            .zip(&picks)
            .take_while(|(d, v)| d == v)
            .count();
        for &d in &drafts[..accepted] {
            emit_token(&mut a, d, None, &s);
            if a.finished {
                return (a, step);
            }
        }
        emit_token(&mut a, picks[accepted], None, &s);
        p += accepted + 1;
    }
    (a, step)
}

/// Serial vs every span shape; returns serial decode's sequence and picks.
fn check(c: Case) -> (ActiveSeq, Vec<u32>) {
    const N: usize = 120;
    script(rows(c.seed, N));
    let (want, reference) = serial(c, N);
    // Serial decode's own picks, every fifth one wrong.
    let draft = |p: usize| {
        let d = reference.get(p).copied().unwrap_or(0);
        if p % 5 == 4 { d ^ 1 } else { d }
    };
    for ks in [&[2usize][..], &[4], &[8], &[5, 2, 3, 1], &[3, 6]] {
        let (got, _) = speculative(c, reference.len(), BUDGET, ks, draft);
        assert_eq!(got.output_tokens, want.output_tokens, "{c:?} K={ks:?}");
        assert_eq!(got.finished, want.finished, "{c:?} K={ks:?}");
    }
    SCRIPT.with(|s| *s.borrow_mut() = None);
    (want, reference)
}

#[test]
fn verify_spans_commit_exactly_what_serial_decode_commits_under_the_ban() {
    let mut banned_wins = 0;
    for seed in 1..=16 {
        for (ignore_eos, min_tokens) in [(true, 30), (false, 30), (true, 75), (false, 75)] {
            for (presence, chat_fast) in [(0.0, true), (0.0, false), (0.5, true)] {
                let c = Case {
                    ignore_eos,
                    min_tokens,
                    target: true,
                    presence,
                    chat_fast,
                    seed,
                };
                let (a, picks) = check(c);
                // Below the floor no end token is picked, so none is discarded:
                // every pick there is output.
                let below = (min_tokens - PRIOR).min(picks.len());
                assert!(
                    picks[..below].iter().all(|t| !MODEL_END.contains(t)),
                    "{c:?}"
                );
                assert_eq!(&a.output_tokens[PRIOR..PRIOR + below], &picks[..below]);
                // The stream does put an end token on top below the floor.
                let s = rows(seed, below);
                banned_wins += s
                    .iter()
                    .filter(|r| MODEL_END.contains(&argmax_first_wins_f32(r)))
                    .count();
            }
        }
    }
    assert!(banned_wins > 100, "{banned_wins}");
}

#[test]
fn without_the_target_ban_speculation_still_matches_serial_decode() {
    // The plain ban (GLM, or the switch off): end tokens below the floor are
    // picked and discarded (or emitted under ignore_eos), as before.
    for seed in 1..=8 {
        for ignore_eos in [true, false] {
            let c = Case {
                ignore_eos,
                min_tokens: 30,
                target: false,
                presence: 0.0,
                chat_fast: true,
                seed,
            };
            let (_, picks) = check(c);
            assert!(picks.contains(&END), "seed {seed}");
        }
    }
}

#[test]
fn without_min_tokens_the_target_ban_changes_nothing() {
    for seed in 1..=8 {
        for ignore_eos in [true, false] {
            let run = |target| {
                check(Case {
                    ignore_eos,
                    min_tokens: 0,
                    target,
                    presence: 0.0,
                    chat_fast: true,
                    seed,
                })
            };
            let ((on, on_picks), (off, off_picks)) = (run(true), run(false));
            assert_eq!(on.output_tokens, off.output_tokens, "seed {seed}");
            assert_eq!(on_picks, off_picks, "seed {seed}");
            assert_eq!(on.finished, off.finished);
        }
    }
    let c = chat_seq(
        Case {
            ignore_eos: true,
            min_tokens: 0,
            target: true,
            presence: 0.0,
            chat_fast: true,
            seed: 0,
        },
        8,
    );
    assert_eq!((c.seq.eos_ban.floor, c.seq.eos_ban.target), (0, false));
    assert_eq!(banned_ids(&c), None);
    // No target ban is installed in this process: the first token keeps the
    // end tokens it always suppressed.
    assert_eq!(first_token_suppress(&[END], 30), vec![END]);
    assert!(first_token_suppress(&[], 30).is_empty());
}

#[test]
fn the_batched_verify_picks_what_each_member_picks_under_the_ban() {
    // Members at different counts, so spans sit below, across and above
    // their floors (straddling spans take the host pipeline).
    for seed in 1..=12u64 {
        let ks = [4usize, 3, 8, 1, 2, 4, 5, 3];
        let mut off = vec![0usize];
        for k in ks {
            off.push(off.last().unwrap() + k);
        }
        let stream = rows(seed, off[ks.len()]);
        let gpu: Vec<u32> = stream.iter().map(|r| argmax_first_wins_f32(r)).collect();
        script(stream);
        let member = |m: usize| {
            chat_seq(
                Case {
                    ignore_eos: m.is_multiple_of(2),
                    min_tokens: [3, 4, 6, 9, 30, 2][m % 6],
                    target: m != 7,
                    presence: if m.is_multiple_of(3) { 0.5 } else { 0.0 },
                    chat_fast: true,
                    seed,
                },
                64,
            )
        };
        let s = sched(Case {
            ignore_eos: true,
            min_tokens: 0,
            target: true,
            presence: 0.0,
            chat_fast: true,
            seed,
        });
        let ctx = s.verify_logits_ctx(None, None, None, None);
        let mut want: Vec<ActiveSeq> = (0..ks.len()).map(member).collect();
        let want_picks: Vec<Vec<u32>> = want
            .iter_mut()
            .enumerate()
            .map(|(i, a)| {
                verify_pick_all_with_pipeline(&model(), &gpu[off[i]..off[i + 1]], a, &ctx, off[i])
            })
            .collect();
        let mut got: Vec<ActiveSeq> = (0..ks.len()).map(member).collect();
        let mut refs: Vec<&mut ActiveSeq> = got.iter_mut().collect();
        let got_picks = verify_pick_batch(&model(), &gpu, &off, &mut refs, &ctx);
        assert_eq!(got_picks, want_picks, "seed {seed}");
        // A member wholly below its floor never picks an end token.
        for (i, picks) in want_picks.iter().enumerate() {
            let a = member(i);
            if a.seq.eos_ban.target && PRIOR + ks[i] <= a.min_tokens {
                assert!(picks.iter().all(|t| !MODEL_END.contains(t)), "member {i}");
            }
        }
        SCRIPT.with(|s| *s.borrow_mut() = None);
    }
}

#[test]
fn a_forced_run_past_its_natural_end_commits_one_token_a_step_without_the_ban() {
    // The structured shape: the answer ends after 20 tokens, then the model
    // wants its end token at every position, a newline second. The draft
    // head holds the first 800 ids only (`--mtp-vocab` 100k against
    // <|im_end|> 248046), so it drafts the newline and never the end token.
    // K=4, 60 forced tokens (min_tokens + ignore_eos).
    const NL: u32 = 10;
    const N: usize = 200;
    let stream: Vec<Vec<f32>> = (0..N)
        .map(|p| {
            let mut r = vec![0.0f32; V];
            if p < 20 {
                r[100 + p] = 9.0;
            } else {
                r[END as usize] = 9.0;
                r[NL as usize] = 8.0;
            }
            r
        })
        .collect();
    let draft = |rows: &[Vec<f32>], p: usize| argmax_first_wins_f32(&rows[p][..800]);
    let forced = 60;
    let mut steps = Vec::new();
    for (ignore_eos, target) in [(true, false), (true, true), (false, false), (false, true)] {
        script(stream.clone());
        let c = Case {
            ignore_eos,
            min_tokens: PRIOR + forced,
            target,
            presence: 0.0,
            chat_fast: true,
            seed: 0,
        };
        let s = stream.clone();
        let (a, n) = speculative(c, N, forced, &[4], |p| draft(&s, p));
        SCRIPT.with(|s| *s.borrow_mut() = None);
        steps.push((a.output_tokens.len() - PRIOR, n));
    }
    // ignore_eos: 5 steps of 4 for the answer, then one end token a step.
    assert_eq!(steps[0], (forced, 5 + 40));
    // The ban: the newline is the pick, and the drafts match it.
    assert_eq!(steps[1], (forced, 15));
    // min_tokens alone: each end token is discarded but still draws on the
    // budget, so the run ends at max_tokens with the answer and the one end
    // token the exhausted budget stops on.
    assert_eq!(steps[2], (20 + 1, 5 + 40));
    assert_eq!(steps[3], (forced, 15));
}
