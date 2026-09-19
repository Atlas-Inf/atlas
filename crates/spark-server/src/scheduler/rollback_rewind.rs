// SPDX-License-Identifier: AGPL-3.0-only

//! Token-buffer rewind for Phase-C rollback (`rollback.rs`): the pure
//! `Vec`/scalar core (`rewind_buffers`) and the `ActiveSeq` glue that
//! applies it (`apply_rollback`). Child module of `rollback` (it needs the
//! scheduler's private `ActiveSeq`); split out for the ≤500 LoC cap.

use super::*;

/// Token-buffer rewind applied to a generated-token buffer and the
/// paged-attention sequence buffers. Pure over plain `Vec`s / scalars so
/// it is unit-testable without an [`ActiveSeq`] (which carries channels,
/// `Instant`s and a `SequenceState`). This is the load-bearing KV-rewind
/// step — see the module doc on why lowering `seq_len` *is* the
/// attention rewind.
///
/// Returns the new `seq_len`.
pub fn rewind_buffers(
    output_tokens: &mut Vec<u32>,
    seq_tokens: &mut Vec<u32>,
    seq_len: usize,
    keep_len: usize,
    unfed_tail: usize,
) -> usize {
    let dropped = output_tokens.len().saturating_sub(keep_len);
    if dropped == 0 {
        return seq_len;
    }
    output_tokens.truncate(keep_len);
    // `seq_tokens` grows only when the model decodes a token, so it holds
    // the prompt plus every generated token the model has FED — that is
    // every generated token except the `unfed_tail` newest ones the caller
    // pushed to `output_tokens` but has not decoded yet (0 at a watchdog
    // that runs before the step's push, 1 at one that runs after it). The
    // boundary token is re-fed as `last_token` on the next step, from the
    // SSM/aux snapshot taken before it was fed, so it must leave
    // `seq_tokens` too: pop every fed token from the boundary onwards. A
    // fixed `dropped` decoded the boundary token twice at the before-push
    // site ("QSA: decode at pos N+1 but N tokens ingested"); a fixed
    // `dropped + 1` discarded a real token at the after-push site
    // ("pos N but N+1 ingested").
    let fed_outputs = (output_tokens.len() + dropped).saturating_sub(unfed_tail);
    let pops = (fed_outputs + 1).saturating_sub(keep_len);
    let mut new_seq_len = seq_len;
    for _ in 0..pops {
        if seq_tokens.pop().is_some() {
            new_seq_len = new_seq_len.saturating_sub(1);
        }
    }
    new_seq_len
}

/// Apply the truncation + KV/position rewind + watchdog-state reset to a
/// live [`ActiveSeq`]. Delegates the buffer rewind to [`rewind_buffers`].
pub(super) fn apply_rollback(
    a: &mut ActiveSeq,
    keep_len: usize,
    dropped: usize,
    unfed_tail: usize,
) {
    // 1+2. Truncate the generated-token buffer and rewind the
    //       attention-KV cursor (`seq.tokens` + `seq_len`).
    a.seq.seq_len = rewind_buffers(
        &mut a.output_tokens,
        &mut a.seq.tokens,
        a.seq.seq_len,
        keep_len,
        unfed_tail,
    );

    // 3. Restore the generation budget that the dropped tokens consumed.
    //    Every dropped token decremented `remaining` by one (since the §C-1
    //    hard-limit fix, thinking tokens draw down the budget too, not just
    //    content) — and the watchdogs that call this only ever fire
    //    post-`</think>`, dropping content tokens, so `dropped` is an exact
    //    count of budget to refund.
    a.remaining = a.remaining.saturating_add(dropped);
    a.content_tokens = a.content_tokens.saturating_sub(dropped as u32);

    // 4. Re-point the decode cursor at the boundary token.
    if let Some(&last) = a.output_tokens.last() {
        a.last_token = last;
    }

    // 5. Rewind the grammar FSM by the same token count so the
    //    constrained-decoding matcher stays in sync with the truncated
    //    token stream. Every dropped token is a post-`</think>` content
    //    token (the watchdogs that call this fire after thinking has
    //    closed) and was therefore fed to `grammar_state.accept_token`,
    //    so `rollback(dropped)` is exact. Reuses the existing
    //    spec-decode grammar-rewind path (`GrammarState::rollback`).
    if let Some(ref mut gs) = a.grammar_state {
        gs.rollback(dropped);
    }

    // 6. Reset the watchdog accumulators so the just-cleared window does
    //    not immediately re-trigger before fresh tokens arrive.
    a.prose_tokens_since_last_tool = 0;
    a.consecutive_confident = 0;
}
