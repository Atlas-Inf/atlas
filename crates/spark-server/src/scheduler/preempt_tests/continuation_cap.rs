// SPDX-License-Identifier: AGPL-3.0-only

//! BUG 1 (production, 2026-11-25): a prompt longer than one chunk died at
//! its CONTINUATION — the continue loop sized the continuation chunk to the
//! whole remaining tail (16983 for a 25176-token prompt at the 8193-token
//! default budget) instead of min(remaining, budget), blowing the buffer
//! arena: "Prefill chunk (16983 tokens) exceeds buffer arena capacity
//! (8193)". Production dodged it with `--max-prefill-tokens 32768`, so any
//! agent session crossing 32k prompt tokens died mid-prefill.

use super::PreemptStubModel;
use super::queue_first_chunk::{Tick, chunks, request};

/// Tick drives the real phases with this budget for both
/// `max_prefill_tokens` and `max_batch_tokens` (see `Tick::admit`).
const BUDGET: usize = 16;

/// A prompt longer than one chunk cycles ceil(N/B) chunks, each at (or
/// under) the budget, through the REAL start + continue phases, and the
/// prefill completes: chunk 0 at admission, then budget-sized continuations
/// until the exact tail.
#[test]
fn long_prompt_cycles_ceil_n_over_b_chunks_under_the_budget() {
    let model = PreemptStubModel::default(); // single-rank; standard continue path
    let mut t = Tick::new(&model, false, false);
    let n = 40;
    let (req, _rx) = request(n);
    t.admit(vec![req]);
    for _ in 0..4 {
        t.step_spec(false, false);
    }
    let calls = chunks(&model);
    assert_eq!(
        calls,
        vec![(0, BUDGET), (BUDGET, BUDGET), (2 * BUDGET, n - 2 * BUDGET)],
        "ceil({n}/{BUDGET}) = {} chunks",
        n.div_ceil(BUDGET)
    );
    assert!(calls.iter().all(|&(_, len)| len > 0 && len <= BUDGET));
    assert_eq!(calls.iter().map(|&(_, len)| len).sum::<usize>(), n);
    assert!(t.queue().is_empty(), "the prefill completes");
    assert_eq!(t.active.len(), 1, "the sequence decodes");
}

/// An MLA model that cannot chunk, with a prompt past the buffer arena:
/// the request fails cleanly — the client gets the error, nothing wedges
/// in the prefill queue, the sequence frees, and the serve takes the next
/// request. (On the buggy main the continuation asked for all 24 remaining
/// tokens at once and the arena guard killed the request mid-loop instead.)
#[test]
fn unchunkable_mla_prompt_past_the_arena_fails_cleanly() {
    let model = PreemptStubModel {
        mla_unchunkable: true,
        ..Default::default()
    };
    let mut t = Tick::new(&model, false, false);
    let (bad, mut rx_bad) = request(40);
    t.admit(vec![bad]);
    assert_eq!(
        chunks(&model),
        vec![(0, BUDGET)],
        "chunk 0 honors the budget"
    );
    t.step_spec(false, false);
    assert!(
        rx_bad.try_recv().unwrap().is_err(),
        "the client gets the error (invalid_request/400)"
    );
    assert!(t.queue().is_empty(), "nothing wedged in the prefill queue");
    assert!(t.active.is_empty());
    // The serve survives: the next request prefills and runs.
    let (good, mut rx_good) = request(8);
    t.admit(vec![good]);
    t.step_spec(false, false);
    assert!(t.queue().is_empty());
    assert_eq!(t.active.len(), 1, "the follow-up request is served");
    assert!(
        rx_good.try_recv().is_err(),
        "still running — no error frame was sent for the follow-up"
    );
}
