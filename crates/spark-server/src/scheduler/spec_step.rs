// SPDX-License-Identifier: AGPL-3.0-only

//! Self-speculative + NGram speculative decoding step + grammar helpers.

use super::*;

/// Self-speculative step: draft via layer-skipping, verify with full model.
/// Combines bootstrap + verify in one step (no pipeline).
///
/// `verify_ctx` is plumbed into the verify-time argmax replacement so
/// each verify position runs through the full 8-stage pre-sample
/// pipeline instead of falling through unmasked. See
/// `verify_pipeline_helper` for the rationale.
pub fn step_self_spec(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let a = &mut active[0];

    // 1. Full-model decode to get token_0
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
        tracing::error!("EP broadcast self-spec token: {e:#}");
        a.engine_error = Some(format!("{e:#}"));
        a.finished = true;
        return;
    }
    let logits = match model.decode(a.last_token, &mut a.seq, 0) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("self-spec decode error: {e:#}");
            a.engine_error = Some(format!("{e:#}"));
            a.finished = true;
            return;
        }
    };
    let token_0 = match model.argmax_on_device(logits, 0) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("self-spec argmax error: {e:#}");
            a.engine_error = Some(format!("{e:#}"));
            a.finished = true;
            return;
        }
    };

    // 2. Draft phase: layer-skipping for cheap predictions
    let seq_len_before_draft = a.seq.seq_len;
    let tokens_before_draft = a.seq.tokens.len();

    let mut draft_tokens = Vec::with_capacity(num_drafts);
    let mut draft_token = token_0;
    for _ in 0..num_drafts {
        let logits = match model.decode_draft(draft_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("self-spec draft error: {e:#}");
                break;
            }
        };
        draft_token = match model.argmax_on_device(logits, 0) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("self-spec draft argmax error: {e:#}");
                break;
            }
        };
        draft_tokens.push(draft_token);
    }

    // 3. Rewind to pre-draft state (SSM unchanged since we skipped SSM layers)
    a.seq.seq_len = seq_len_before_draft;
    a.seq.tokens.truncate(tokens_before_draft);

    if draft_tokens.is_empty() {
        // No drafts: emit token_0 and continue
        emit_token(a, token_0, None, sched);
        if !a.finished {
            a.last_token = token_0;
        }
        return;
    }

    // 4. Checkpoint SSM states before verification
    if let Err(e) = model.checkpoint_ssm_states(&mut a.seq) {
        tracing::error!("self-spec checkpoint: {e:#}");
        a.engine_error = Some(format!("{e:#}"));
        a.finished = true;
        return;
    }
    let seq_len_before_verify = a.seq.seq_len;

    // 5. Verify: run full model on [token_0, d1, ..., dK]
    let mut verify_tokens = vec![token_0];
    verify_tokens.extend_from_slice(&draft_tokens);

    let verified_argmax = match model.decode_verify(&verify_tokens, &mut a.seq, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("self-spec verify error: {e:#}");
            a.engine_error = Some(format!("{e:#}"));
            a.finished = true;
            return;
        }
    };

    // Phase C-2 (2026-05-24): replay the pre-sample
    // logits-processor pipeline per verify position. `decode_verify`
    // wrote `[verify_tokens.len(), vocab]` BF16 into `logits_buffer`;
    // the helper copies it D2H and applies the same 8-stage pipeline
    // used in the non-MTP path. Falls back to the raw argmax on D2H
    // failure (see helper).
    let verified = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
        model,
        &verified_argmax,
        a,
        verify_ctx,
        0,
    );

    // 6. Compare draft vs verified, count acceptances
    let n_drafts = draft_tokens.len();
    let mut num_accepted = 0;

    emit_token(a, token_0, None, sched);
    if a.finished {
        return;
    }

    for i in 0..n_drafts {
        if draft_tokens[i] == verified[i] {
            emit_token(a, draft_tokens[i], None, sched);
            if a.finished {
                return;
            }
            num_accepted += 1;
        } else {
            emit_token(a, verified[i], None, sched);
            if a.finished {
                return;
            }
            a.last_token = verified[i];
            break;
        }
    }

    if num_accepted == n_drafts && n_drafts > 0 {
        emit_token(a, verified[n_drafts], None, sched);
        if !a.finished {
            a.last_token = verified[n_drafts];
        }
    } else if num_accepted < n_drafts {
        // Already set a.last_token above in the break
    } else {
        a.last_token = token_0;
    }

    // 7. Rollback extra verify tokens
    // tokens_added = token_0 (always kept) + accepted drafts
    let tokens_added = 1 + num_accepted;
    let expected_seq_len = seq_len_before_verify + tokens_added;

    if a.seq.seq_len > expected_seq_len {
        let extra = a.seq.seq_len - expected_seq_len;
        for _ in 0..extra {
            a.seq.seq_len -= 1;
            a.seq.tokens.pop();
        }
        // +1 because token_0 is always accepted in the verify batch
        if let Err(e) = model.rollback_ssm_states(&mut a.seq, num_accepted + 1) {
            tracing::error!("self-spec rollback: {e:#}");
        }
    }
}

/// N-gram speculative step: CPU proposer + CUDA-graphed K=2 verify.
///
/// Two-phase pipeline (same as MTP but with N-gram proposer instead):
/// 1. Bootstrap: regular decode → argmax → N-gram propose → pending_drafts
/// 2. Verify: decode_verify_graphed(K=2) → accept/reject → SSM rollback
pub fn step_ngram(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    proposer: &mut NgramProposer,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let a = &mut active[0];

    if !a.pending_drafts.is_empty() {
        // ── Phase B: Verify pending draft ──
        let drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        a.pending_draft_conf.clear();
        step_ngram_verify(model, a, sched, &drafts, proposer, verify_ctx);
    } else {
        // ── Phase A: Bootstrap decode + N-gram propose ──
        if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
            tracing::error!("EP broadcast ngram bootstrap: {e:#}");
            a.engine_error = Some(format!("{e:#}"));
            a.finished = true;
            return;
        }
        let logits = match model.decode(a.last_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("ngram bootstrap decode error: {e:#}");
                a.engine_error = Some(format!("{e:#}"));
                a.finished = true;
                return;
            }
        };
        let tok = match model.argmax_on_device(logits, 0) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("ngram bootstrap argmax error: {e:#}");
                a.engine_error = Some(format!("{e:#}"));
                a.finished = true;
                return;
            }
        };

        // Observe the token for future predictions
        proposer.observe(&a.seq.tokens, tok);

        emit_token(a, tok, None, sched);
        if a.finished {
            return;
        }
        a.last_token = tok;

        // N-gram propose (CPU-only, zero GPU cost): a chain of up to
        // `num_drafts` tokens — prompt-lookup continuation extended through
        // the learned n-gram table (llama.cpp-style). Chain is capped at 3:
        // the graphed verifies top out at K=4.
        // `a.last_token` is sampled-but-not-pushed (verify_b comment): the
        // searchable context ends at the emitted token, not tokens.last().
        let mut ngram_ctx = a.seq.tokens.clone();
        ngram_ctx.push(a.last_token);
        let chain = proposer.propose_chain(&ngram_ctx);
        if !chain.is_empty() {
            a.pending_drafts = chain;

            // Checkpoint SSM for potential rollback during verify
            if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
                tracing::error!("ngram start_checkpoint_async: {e:#}");
            }
        }
        // If no proposal: next iteration will be another bootstrap (regular decode)
    }
}

/// Verify an N-gram draft chain via the CUDA-graphed K=2/3/4 verify paths.
pub fn step_ngram_verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    proposer: &mut NgramProposer,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let t_sync = Instant::now();
    if let Err(e) = model.sync_secondary() {
        tracing::error!("ngram sync_secondary: {e:#}");
        a.engine_error = Some(format!("{e:#}"));
        a.finished = true;
        return;
    }
    let sync_us = t_sync.elapsed().as_micros();

    // Verify width k = drafts + 1; the graphed verifies top out at K=4.
    let nd = drafts.len().min(3);
    let k = nd + 1;
    let ep_cmd = match k {
        2 => 0xFFFFFFF2u32,
        3 => 0xFFFFFFF3u32,
        _ => 0xFFFFFFF4u32,
    };
    let mut tokens = Vec::with_capacity(k);
    tokens.push(a.last_token);
    tokens.extend_from_slice(&drafts[..nd]);

    // EP: broadcast verify command + tokens
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, ep_cmd) {
        tracing::error!("EP broadcast ngram verify cmd: {e:#}");
        a.engine_error = Some(format!("{e:#}"));
        a.finished = true;
        return;
    }
    for &t in &tokens {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast ngram verify token: {e:#}");
            a.engine_error = Some(format!("{e:#}"));
            a.finished = true;
            return;
        }
    }

    let t_verify = Instant::now();
    let verified_raw: Vec<u32> = match k {
        2 => model
            .decode_verify_graphed(&[tokens[0], tokens[1]], &mut a.seq, 0)
            .map(|r| r.to_vec()),
        3 => model
            .decode_verify_graphed_k3(&[tokens[0], tokens[1], tokens[2]], &mut a.seq, 0)
            .map(|r| r.to_vec()),
        _ => model
            .decode_verify_graphed_k4(&[tokens[0], tokens[1], tokens[2], tokens[3]], &mut a.seq, 0)
            .map(|r| r.to_vec()),
    }
    .unwrap_or_else(|e| {
        tracing::error!("ngram decode_verify_graphed (k={k}): {e:#}");
        a.engine_error = Some(format!("{e:#}"));
        a.finished = true;
        Vec::new()
    });
    if a.finished {
        return;
    }
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();

    // Pipeline picks per verify row (penalties/masks honored).
    let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
        model,
        &verified_raw,
        a,
        verify_ctx,
        0,
    );
    let v: Vec<u32> = (0..k)
        .map(|i| processed.get(i).copied().unwrap_or(verified_raw[i]))
        .collect();

    // Accept-prefix: draft[i] must equal the verified pick at row i.
    let mut na = 0usize;
    while na < nd && drafts[na] == v[na] {
        na += 1;
    }

    // EP: broadcast accept count to worker
    if let Err(e) = model.ep_broadcast_cmd(na as u32) {
        tracing::error!("EP broadcast ngram verify result: {e:#}");
        a.engine_error = Some(format!("{e:#}"));
        a.finished = true;
        return;
    }

    if na < nd {
        // Rewind the rejected tail: seq_len and tokens roll back
        // rows, then commit_accepted_prefix rewinds the aux (QSA indexer /
        // PLE carry) by the same count and re-checkpoints.
        a.seq.seq_len -= nd - na;
        for _ in 0..(nd - na) {
            a.seq.tokens.pop();
        }
    }
    if let Err(e) = model.commit_accepted_prefix(&mut a.seq, na + 1, k) {
        tracing::error!("ngram commit_accepted_prefix (k={k} na={na}): {e:#}");
        a.engine_error = Some(format!("{e:#}"));
        a.finished = true;
        return;
    }
    if na < nd {
        // Keep a fresh SSM checkpoint for the next verify — the commit above
        // already re-checkpointed after the rewind, matching the K-paths.
    } else {
        // Full accept: commit was a no-op; checkpoint for the next verify.
        if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
            tracing::error!("ngram accept checkpoint: {e:#}");
        }
    }

    // Observe accepted context into the dynamic table, then emit.
    for j in 0..na {
        let n = a.seq.tokens.len();
        if n > 0 {
            let last = a.seq.tokens[n - 1];
            proposer.observe(&a.seq.tokens[..n - 1], last);
        }
        emit_token(a, drafts[j], None, sched);
        if a.finished {
            return;
        }
    }
    proposer.observe(&a.seq.tokens, v[na]);
    emit_token(a, v[na], None, sched);
    if a.finished {
        return;
    }
    a.last_token = v[na];

    if na == nd {
        proposer.accepts += na as u64;
    } else {
        proposer.rejects += 1;
    }

    // Propose next chain — same off-by-one: last_token is emitted but not
    // yet pushed to seq.tokens.
    let mut ngram_ctx = a.seq.tokens.clone();
    ngram_ctx.push(a.last_token);
    let chain = proposer.propose_chain(&ngram_ctx);
    if !chain.is_empty() {
        a.pending_drafts = chain;
    }

    tracing::debug!(
        "NGRAM detail: drafts={:?} v={:?} na={} seq_len={}",
        &drafts[..nd],
        v,
        na,
        a.seq.seq_len
    );
    if a.seq.seq_len.is_multiple_of(50) {
        tracing::info!(
            "NGRAM K{k} {}: sync={sync_us}us verify={verify_us}us cache={} seq_len={} na={na}/{nd}",
            if na == nd { "ACCEPT" } else { "REJECT" },
            proposer.len(),
            a.seq.seq_len,
        );
    }
}

/// Fill the XGrammar bitmask for the current matcher position and clone it
/// into an owned `Vec<i32>` the caller can pass into MTP draft sampling.
///
/// Returns `None` when grammar is inactive, the sequence is currently inside
/// a `<think>` span (matcher is paused), the grammar has already terminated,
/// or `fill_bitmask` reported no constraint. In all those cases MTP should
/// fall back to its unconstrained GPU-argmax path.
///
/// The owned copy is small (~ceil(vocab/32)*4 bytes, ~32KB for 100k vocab)
/// and is necessary because the matcher is borrowed mutably by the scheduler
/// between `fill_bitmask` and the subsequent `accept_token` calls inside
/// `emit_token`, while the MTP propose call borrows the model immutably —
/// cloning sidesteps the lifetime overlap.
pub fn mtp_grammar_mask_for(a: &mut ActiveSeq) -> Option<Vec<i32>> {
    if a.inside_thinking {
        return None;
    }
    let gs = a.grammar_state.as_mut()?;
    if gs.is_terminated() {
        return None;
    }
    if !gs.fill_bitmask() {
        return None;
    }
    Some(gs.bitmask_data().to_vec())
}

/// BUG#4 clamp, complete fix (2026-07-09): when a grammar is active, propose
/// only ONE draft. `run_mtp_propose_multi` masks every draft position with
/// the SAME position-0 bitmask snapshot (`mtp_head` warns "mask held fixed
/// across draft positions"), so draft\[1..\] is drafted against a stale mask —
/// grammar-illegal continuations get proposed, truncated at the boundary
/// (`truncate_drafts_at_grammar_boundary`), and acceptance collapses. The
/// original BUG#4 fix (2026-06-02) applied this clamp only in the Phase-A
/// bootstrap (`mtp_step.rs`); the five verify-path re-propose sites
/// (`verify_k2_step`, `verify_k3_step`) kept passing raw `num_drafts`, which
/// is why the warning spammed on every step after the first during grammar-
/// constrained tool calls (live opencode 42.5k session, 2026-07-09). SSOT
/// for all six propose sites — semantics identical to the bootstrap clamp
/// (`grammar_state.is_some()`). No-op when grammar is inactive: full K kept.
pub fn effective_drafts_under_grammar(a: &ActiveSeq, num_drafts: usize) -> usize {
    if a.grammar_state.is_some() {
        1
    } else {
        num_drafts
    }
}

/// Truncate a draft list at the first token the grammar would
/// reject *if it were the next emitted token at that draft position*.
///
/// Required for K=3+ MTP paths where `run_mtp_propose_multi` uses a
/// SINGLE bitmask snapshot (taken at the start of propose) for all N
/// drafts. The mask correctly constrains `drafts[0]` but does not
/// reflect the post-`drafts[0]` grammar state — so `drafts[1]` may
/// cross a structural boundary (e.g. `drafts[0] = </function>`
/// closing a tool body, then `drafts[1] = <parameter=` which is
/// invalid in the outer free-text grammar state).
///
/// Without this guard, the spec verifier accepts the cross-boundary
/// span (the model's actual sample matches whatever the in-tool
/// distribution happened to produce), `emit_token` advances the
/// grammar past `</function>`, and the next `accept_token` for
/// `drafts[1]` returns false silently — the token is already in
/// `output_tokens`, but the grammar is desync'd from the output
/// stream. Subsequent bitmasks are wrong.
///
/// Reference: arXiv:2512.15834 ("Speculative Tool Calls"). The
/// canonical fix is to re-run the grammar mask from a fresh outer
/// state for each draft; we approximate cheaply by simulating
/// `accept_token` per draft and truncating at the first rejection,
/// rolling the state back when done. The verifier then accepts at
/// most the validated prefix.
///
/// Returns the number of drafts that pass grammar validation.
/// Mutates `gs` transiently but restores it via `rollback`. K=2
/// (num_drafts=1) callers can skip this — a single draft uses its
/// own up-to-date mask.
pub fn truncate_drafts_at_grammar_boundary(gs: &mut GrammarState, drafts: &[u32]) -> usize {
    if drafts.len() < 2 || gs.is_terminated() {
        return drafts.len();
    }
    // BUG#3 (2026-06-02): roll back ACTUAL matcher advances (history delta), not
    // the `accepted` tally. `accept_token` returns true for stop/EOS tokens and
    // in the terminated state WITHOUT advancing the matcher; counting rollback
    // from `accepted` over-rewinds when such a token is in the draft span
    // (corrupt state / rollback panic). `accepted` still drives truncation.
    let steps_before = gs.num_history_steps();
    let mut accepted = 0usize;
    for &tok in drafts {
        if !gs.accept_token(tok) {
            break;
        }
        accepted += 1;
    }
    let advanced = gs.num_history_steps().saturating_sub(steps_before);
    if advanced > 0 {
        gs.rollback(advanced);
    }
    if accepted < drafts.len() {
        tracing::warn!(
            kept = accepted,
            dropped = drafts.len() - accepted,
            "spec-decode boundary: truncated drafts crossing grammar transition"
        );
    }
    accepted
}
