// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::scheduler::cancel_test_model::{SCRIPT, Script, TestModel, VerifyScript};
use crate::scheduler::levers::SchedLevers;
use crate::scheduler::test_support::test_seq;
use spark_model::traits::Model;
use std::sync::Arc;

fn run(masked: bool, inside: bool, raw: &[u32]) -> (Vec<u32>, Vec<String>, Vec<u32>) {
    run_armed(masked, inside, raw, false)
}

fn run_armed(
    masked: bool,
    inside: bool,
    raw: &[u32],
    armed: bool,
) -> (Vec<u32>, Vec<String>, Vec<u32>) {
    run_mode(masked, inside, raw, armed, true, None)
}

fn run_mode(
    masked: bool,
    inside: bool,
    raw: &[u32],
    armed: bool,
    split: bool,
    reject: Option<usize>,
) -> (Vec<u32>, Vec<String>, Vec<u32>) {
    let rows: Vec<Vec<f32>> = raw
        .iter()
        .enumerate()
        .map(|(i, &tok)| {
            let mut row = vec![0.0; 2048];
            row[41 + i] = 8.0;
            row[tok as usize] = 12.0;
            row
        })
        .collect();
    SCRIPT.with(|s| {
        *s.borrow_mut() = Some(Script {
            rows: rows.clone(),
            cursor: 0,
        })
    });
    let script = Arc::new(VerifyScript {
        picks: raw.to_vec(),
        ..Default::default()
    });
    let model = TestModel {
        tokens: vec![],
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: split.then(|| script.clone()),
    };
    let mut levers = SchedLevers::defaults();
    levers.dflash_masked_verify = masked;
    let sched = crate::scheduler::sched_ctx::SchedCtx::new(
        Default::default(),
        Arc::new(levers),
        Arc::default(),
        crate::scheduler::limits::SchedLimits {
            code_fence_token: Some(97),
            ..crate::scheduler::limits::SchedLimits::NONE
        },
        Default::default(),
    );
    let mut ctx = sched.verify_logits_ctx(Some(99), None, None, None);
    ctx.boundary_mask = Some(vec![true; 2048].into());
    let make = || {
        let (mut a, rx) = test_seq(vec![], 20, None, 32);
        std::mem::forget(rx);
        a.finished = false;
        a.min_tokens = 0;
        a.temperature = 0.0;
        a.think_end_token = Some(99);
        a.think_start_token = Some(98);
        a.inside_thinking = inside;
        a.think_ended = !inside;
        a.force_end_thinking = armed;
        a
    };
    let mut serial = make();
    let mut want = vec![];
    for row in rows {
        let bytes: Vec<u8> = row
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let (tok, _) = crate::scheduler::decode_logits_seq::process_seq_logits(
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
        want.push(tok);
        emit_token(&mut serial, tok, None, &sched);
    }
    let mut a = make();
    // Use serial tokens as drafts: a correct verifier accepts them, then emits
    // the same bonus. No logprobs are requested, matching the failed gate.
    let mut drafts = want[..want.len() - 1].to_vec();
    if let Some(i) = reject {
        drafts[i] = 123;
    }
    let mut tokens = vec![a.last_token];
    tokens.extend_from_slice(&drafts);
    let before = a.seq.seq_len;
    let picks = if split {
        model.decode_verify_dflash(&tokens, &mut a.seq, 0).unwrap()
    } else {
        a.seq.tokens.extend_from_slice(&tokens);
        a.seq.seq_len += tokens.len();
        raw.to_vec()
    };
    verify_dflash_tail(
        &model,
        &mut a,
        &sched,
        &drafts,
        &[],
        drafts.len(),
        &ctx,
        true,
        &tokens,
        picks,
        false,
        0.0,
        true,
        true,
    );
    assert!(a.engine_error.is_none(), "{:?}", a.engine_error);
    if masked && split && reject.is_none() {
        assert_eq!(a.inside_thinking, serial.inside_thinking);
        assert_eq!(a.think_ended, serial.think_ended);
        assert_eq!(a.in_code_fence, serial.in_code_fence);
        assert_eq!(a.force_end_thinking, serial.force_end_thinking);
        assert_eq!(a.sentence_defer_count, serial.sentence_defer_count);
        assert_eq!(a.remaining, serial.remaining);
    }
    let wire = script.log.lock().unwrap().clone();
    SCRIPT.with(|s| *s.borrow_mut() = None);
    if let Some(i) = reject {
        serial.output_tokens.truncate(i + 1);
        assert_eq!(a.seq.seq_len, before + i + 1, "rejected suffix rolled back");
        assert_eq!(
            a.mtp_acct.accepted_output_total(),
            i as u64,
            "only emitted drafts count"
        );
    }
    (a.output_tokens, wire, serial.output_tokens)
}

#[test]
fn masked_path_matches_serial_after_close_and_across_in_span_close() {
    for (inside, raw) in [(false, vec![99, 98, 44]), (true, vec![99, 99, 98])] {
        let (got, wire, want) = run(true, inside, &raw);
        assert_eq!(got, want, "inside={inside}; wire={wire:?}");
        assert!(wire.contains(&"full 3".to_string()), "{wire:?}");
        let full = wire.iter().position(|s| s == "full 3").unwrap();
        let verdict = wire.iter().position(|s| s == "cmd 0x2").unwrap();
        assert!(
            full < verdict,
            "rows gathered before accepted-prefix verdict"
        );
    }
}

#[test]
fn raw_path_reproduces_post_close_mismatch_without_logprobs() {
    let (got, wire, want) = run(false, false, &[99, 98, 44]);
    assert_ne!(got, want);
    assert!(!wire.iter().any(|s| s.starts_with("full ")));
}

#[test]
fn masked_path_armed_fence_must_match_serial() {
    let (got, wire, want) = run_armed(true, true, &[97, 44, 45], true);
    assert_eq!(
        got, want,
        "armed fence must retain per-row pipeline decisions; wire={wire:?}"
    );
}

#[test]
fn masked_pipeline_rejection_and_empty_drafts_keep_bonus_and_rollback() {
    for split in [true] {
        for reject in [0, 1] {
            let (got, _, want) = run_mode(true, true, &[97, 44, 45], true, split, Some(reject));
            assert_eq!(got, want, "split={split} reject={reject}");
        }
        let (got, _, want) = run_mode(true, false, &[99], false, split, None);
        assert_eq!(got, want, "empty draft still returns processed bonus");
    }
}
