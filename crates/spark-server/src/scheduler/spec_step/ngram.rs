// SPDX-License-Identifier: AGPL-3.0-only

//! The n-gram speculative lane: CPU proposer + CUDA-graphed K=2/3/4 verify.
//!
//! Split out of `spec_step.rs` for the file-size cap. The seam is a real one —
//! this is the whole n-gram lane (bootstrap, propose, verify, accept,
//! re-propose), and nothing here is shared with the self-speculative step.
//!
//! ★ The lane is confined to `seq_len + 4 <= verify_context_limit()` — the
//! QSA inert bound, 2051 on Flash-Next — because the verify paths refuse an
//! ACTIVE selection past it. The gate is at the dispatch site in `mod.rs`,
//! not here; this module assumes it was satisfied.

use super::*;

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
    adaptive_sampling: bool,
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
        // Sample through the canonical per-seq pipeline — the old
        // argmax_on_device here collapsed every bootstrap token to greedy for
        // requests carrying temperature/penalties/grammar (same class of bug
        // as the DFlash process-wide raw-argmax gate).
        let vocab_size = model.vocab_size();
        let logits_fp32 = model.decode_logits_fp32();
        let elem = if logits_fp32 { 4 } else { 2 };
        let mut buf = sched.scratch.host_bytes.borrow_mut().split_off(0);
        buf.resize(vocab_size * elem, 0);
        if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
            tracing::error!("ngram copy_logits_to_host: {e:#}");
            a.engine_error = Some(format!("{e:#}"));
            a.finished = true;
            *sched.scratch.host_bytes.borrow_mut() = buf;
            return;
        }
        let (tok, lp) = crate::scheduler::decode_logits_seq::process_seq_logits(
            model,
            a,
            &buf,
            0,
            vocab_size,
            elem,
            logits_fp32,
            verify_ctx,
            adaptive_sampling,
        );
        *sched.scratch.host_bytes.borrow_mut() = buf;

        // Observe the token for future predictions
        proposer.observe(&a.seq.tokens, tok);

        emit_token(a, tok, lp, sched);
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

    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        crate::scheduler::logprobs::extract_verify_logprobs(model, &v, top_logprobs, 0)
    } else {
        Vec::new()
    };

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
        // Rewind the rejected tail: seq_len and tokens roll back nd-na
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

    // Learn each accepted draft with its true preceding context — the
    // verify already pushed [anchor, d0..na] into seq.tokens, so accepted
    // draft j sits at index N-na+j.
    let n = a.seq.tokens.len();
    for j in 0..na {
        let idx = n - na + j;
        if idx > 0 {
            proposer.observe(&a.seq.tokens[..idx], a.seq.tokens[idx]);
        }
        emit_token(a, drafts[j], verify_lps.get(j).cloned(), sched);
        if a.finished {
            return;
        }
    }
    proposer.observe(&a.seq.tokens, v[na]);
    emit_token(a, v[na], verify_lps.get(na).cloned(), sched);
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
