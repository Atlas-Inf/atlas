// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::scheduler::test_support::test_request;

// The forensic C=128 ladder shape (2026-08-15, Qwen3.8-27B/GB10):
// pool 102k tokens, per-request demand ~1226 tokens (prompt ~202 +
// max_tokens 1024), 128 requests ⇒ 157k demand. block_size 16.
const BS: usize = 16;
const POOL_BLOCKS: usize = 102_000 / BS; // 6375
const PROMPT: usize = 202;
const MAX_TOK: usize = 1024;
const MAX_SEQ_LEN: usize = 8192;

fn req_blocks() -> usize {
    blocks_for_tokens(PROMPT + MAX_TOK, BS) // 77
}

#[test]
fn block_math_matches_legacy_formula() {
    // Same `tokens / block_size + 1` shape admission always used.
    assert_eq!(blocks_for_tokens(0, 16), 1);
    assert_eq!(blocks_for_tokens(15, 16), 1);
    assert_eq!(blocks_for_tokens(16, 16), 2);
    assert_eq!(blocks_for_tokens(1226, 16), 77);
}

#[test]
fn watermark_caps_the_decode_reservation() {
    let d = SeqDemand {
        current_tokens: 200,
        budget_tokens: 4096,
    };
    // Watermark below max_tokens: reserve prompt + watermark.
    assert_eq!(
        seq_commitment_blocks(&d, 512, MAX_SEQ_LEN, BS),
        blocks_for_tokens(200 + 512, BS)
    );
    // Watermark above max_tokens: the request's own budget bounds it.
    assert_eq!(
        seq_commitment_blocks(&d, usize::MAX, MAX_SEQ_LEN, BS),
        blocks_for_tokens(200 + 4096, BS)
    );
    // Watermark 0 = legacy prompt-only reservation.
    assert_eq!(
        seq_commitment_blocks(&d, 0, MAX_SEQ_LEN, BS),
        blocks_for_tokens(200, BS)
    );
    // The served context ceiling clamps the depth: no sequence can grow
    // past max_seq_len, so nothing more is ever reserved.
    let long = SeqDemand {
        current_tokens: 8000,
        budget_tokens: 4096,
    };
    assert_eq!(
        seq_commitment_blocks(&long, usize::MAX, MAX_SEQ_LEN, BS),
        blocks_for_tokens(MAX_SEQ_LEN, BS)
    );
}

#[test]
fn fits_everything_admission_unchanged() {
    // C<=64 rung: 64 × 77 = 4928 blocks ≤ 6375 — ALL admitted, exactly
    // as the pre-gate code admitted them. Pins the no-regression claim.
    let reqs = vec![(PROMPT, MAX_TOK); 64];
    let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, MAX_SEQ_LEN, MAX_SEQ_LEN, BS);
    assert_eq!(n, 64);
    assert!(!forced);
}

#[test]
fn c128_overflow_queues_instead_of_admit_then_shoot() {
    // The measured failure: 128 requests whose true demand (157k tokens)
    // exceeds the 102k pool. The gate admits what fits and QUEUES the
    // rest — no admit-then-preempt thrash.
    let reqs = vec![(PROMPT, MAX_TOK); 128];
    let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, MAX_SEQ_LEN, MAX_SEQ_LEN, BS);
    assert_eq!(n, POOL_BLOCKS / req_blocks()); // 82: every admitted seq fits fully
    assert!(n < 128);
    assert!(!forced);
    // The admitted set can never exhaust the pool.
    assert!(n * req_blocks() <= POOL_BLOCKS);
}

#[test]
fn in_flight_commitments_reduce_capacity() {
    // 40 active sequences mid-decode still owe their remaining budget.
    let demands: Vec<SeqDemand> = (0..40)
        .map(|_| SeqDemand {
            current_tokens: 600,
            budget_tokens: 700,
        })
        .collect();
    let committed = committed_blocks(&demands, MAX_SEQ_LEN, MAX_SEQ_LEN, BS);
    assert_eq!(committed, 40 * blocks_for_tokens(1300, BS));
    let reqs = vec![(PROMPT, MAX_TOK); 128];
    let (n, _) = admit_count(POOL_BLOCKS, committed, &reqs, MAX_SEQ_LEN, MAX_SEQ_LEN, BS);
    assert_eq!(n, (POOL_BLOCKS - committed) / req_blocks());
}

#[test]
fn admission_stops_at_first_misfit_no_head_of_line_bypass() {
    // A huge request mid-queue blocks later small ones from jumping it.
    let reqs = vec![
        (PROMPT, MAX_TOK),
        (100_000, MAX_TOK), // cannot fit
        (PROMPT, MAX_TOK),  // must NOT bypass
    ];
    let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, usize::MAX, 0, BS);
    assert_eq!(n, 1);
    assert!(!forced);
}

#[test]
fn liveness_oversized_lone_request_still_admits() {
    // Nothing in flight + a request bigger than the whole pool: admit it
    // (runtime back-pressure applies), never queue it forever.
    let reqs = vec![(200_000, MAX_TOK)];
    let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, usize::MAX, 0, BS);
    assert_eq!(n, 1);
    assert!(forced);
    // But with ANYTHING in flight it waits its turn.
    let (n, forced) = admit_count(POOL_BLOCKS, 10, &reqs, usize::MAX, 0, BS);
    assert_eq!(n, 0);
    assert!(!forced);
}

#[test]
fn watermark_zero_reserves_prompt_only_legacy() {
    // Escape hatch pinned: watermark 0 ⇒ the C=128 set is admitted in
    // full (128 × blocks(202) = 128 × 13 = 1664 ≤ 6375) — byte-for-byte
    // the legacy prompt-only admission decision.
    let reqs = vec![(PROMPT, MAX_TOK); 128];
    let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, 0, MAX_SEQ_LEN, BS);
    assert_eq!(n, 128);
    assert!(!forced);
}

#[test]
fn ep_oversized_blocking_request_gets_error_before_allocation() {
    let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
    let request = test_request!(Blocking, response_tx,);
    assert!(reject_oversized(vec![request], 16, 262144, 262144, 16).is_empty());
    let error = response_rx
        .try_recv()
        .unwrap()
        .err()
        .expect("capacity error");
    assert!(error.to_string().contains("pool has 16 usable blocks"));
}

#[test]
fn ep_oversized_stream_gets_error_before_allocation() {
    let (token_tx, mut token_rx) = tokio::sync::mpsc::channel(4);
    let request = test_request!(Streaming, token_tx,
        cancel_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),);
    assert!(reject_oversized(vec![request], 16, 262144, 262144, 16).is_empty());
    match token_rx.try_recv().unwrap() {
        StreamEvent::Error(message) => {
            assert!(message.contains("pool has 16 usable blocks"))
        }
        _ => panic!("expected explicit capacity error"),
    }
    assert!(token_rx.try_recv().is_err());
}

#[test]
fn ep_zero_capacity_rejects_even_a_single_request() {
    let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
    let request = test_request!(Blocking, response_tx,);
    assert!(reject_oversized(vec![request], 0, 262144, 262144, 16).is_empty());
    assert!(response_rx.try_recv().unwrap().is_err());
}
