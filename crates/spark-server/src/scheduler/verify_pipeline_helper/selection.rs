// SPDX-License-Identifier: AGPL-3.0-only

//! One verify driver; explicit copy-failure policy, unchanged per-row pipeline.
use super::selection_io::CopyFailurePolicy;
use super::{ActiveSeq, LogitsContext, Model, fast_masked, verify_pick_with_pipeline};

pub(super) fn select(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    row_base: usize,
    policy: CopyFailurePolicy,
) -> anyhow::Result<Vec<u32>> {
    anyhow::ensure!(
        policy == CopyFailurePolicy::LegacyFallback || a.grammar_state.is_none(),
        "checked verify selection requires a grammarless request"
    );
    use crate::scheduler::mtp_timing::Phase;
    let k = argmax_ids.len();
    if k == 0 {
        return Ok(Vec::new());
    }

    // ── CHAT FAST PATH (2026-07-08): masked-greedy == raw-argmax guard ──
    // See `fast_masked` module docs: for a grammarless request with no
    // forced/stateful stage armed and argmax-preserving penalties, the
    // pipeline provably cannot change any pick, so the raw argmax IS the
    // masked pick and the [K, vocab] D2H is skipped entirely. Any
    // ineligible position falls through to the slow path for the call.
    let masked_verify = fast_masked::masked_verify_required(
        ctx.sampling.dflash_masked_verify,
        model.is_lightning_dspark_product(),
    );
    if let Some(picks) =
        fast_masked::try_chat_fast_path(model, argmax_ids, a, ctx, row_base, policy, masked_verify)?
    {
        return Ok(picks);
    }

    // ── FAST PATH (#3, 2026-06-02): on-GPU greedy pick under grammar ──
    //
    // Culprit #3 (regression hunt): the slow path below D2H-copies the full
    // [K, vocab] logits, CPU-dequants 248k BF16→F32 per position, and runs the
    // 8-stage pipeline + argmax — ~1-3 ms/token of host/PCIe serialization on
    // the dominant MTP verify path, the structural reason vLLM (GPU sampling)
    // out-decodes Atlas on tool/grammar workloads.
    //
    // But when decoding is GREEDY (temp=0 or ATLAS_FORCE_TEMP_ZERO), penalties
    // are neutral, and we're not inside <think>, the masked-greedy pick at each
    // verify position is EXACTLY the GPU argmax (`argmax_ids[i]`, already
    // computed by decode_verify_graphed*) WHENEVER that argmax is grammar-
    // allowed — because the global max that is also in the allowed set is, by
    // definition, the max over the allowed set. So we can emit it directly with
    // NO D2H/dequant/pipeline. This fires for the bulk of content tokens (the
    // permissive value ladder allows almost everything). We fall back to the
    // slow pipeline per-call only when some position's argmax is grammar-
    // DISALLOWED (structural/forced positions — rare) or the regime isn't
    // pure-greedy. The speculative matcher advance + history-delta rollback
    // (BUG#3) are preserved identically to the slow path, so on fallback the
    // matcher is restored to its exact pre-call state.
    //
    // Skipped in this fast path: the WS/AM/think/forced quality nudges. Those
    // are either no-ops in the content/greedy/neutral regime or acceptable
    // speed-for-quality trades (we hold a measured accuracy margin over vLLM).
    // Kill-switch: ATLAS_DISABLE_FAST_GREEDY=1.
    //
    // #237 (fix 4a): the all-penalties-neutral requirement is relaxed to the
    // SSOT `fast_greedy` gate — reduce-only penalties (rep>=1.0, presence/
    // frequency>=0, LZ/DRY off, no bias) provably cannot flip an argmax whose
    // token is NOT in the scoped penalty history and whose raw logit is > 0
    // (see `fast_greedy` module docs for the proof). The membership test uses
    // the SAME scoped history the slow path hands to
    // `apply_penalties_and_bias` (`penalty_history_scope`), which is also
    // deliberately STALE across positions ≥ 1 exactly like the slow path
    // (output_tokens does not grow until `emit_token`, after this helper).
    // P1-3 (2026-07-09): this temp==0 gate is load-bearing for verify-time
    // sampling — at temperature > 0 the fast GPU-argmax shortcut must NOT
    // fire, so every position routes through the slow pipeline below where
    // the temp>0 sampling branch (step 4a in `verify_pick_with_pipeline`)
    // draws from the processed logits. Confirmed and kept as-is.
    let fast_penalty_gate = if ctx.sampling.fast_greedy_grammar
        && a.grammar_state.is_some()
        && !a.inside_thinking
        && (a.temperature == 0.0 || ctx.sampling.force_temp_zero)
    {
        crate::scheduler::fast_greedy::classify_penalties(
            &crate::scheduler::sample_step::penalty_params_for(
                a,
                crate::scheduler::sample_step::PositionKind::Verify,
                0.0,
                None,
                Vec::new(),
            ),
        )
    } else {
        crate::scheduler::fast_greedy::PenaltyGate::Blocked
    };
    if fast_penalty_gate != crate::scheduler::fast_greedy::PenaltyGate::Blocked {
        let t_fast = std::time::Instant::now();
        let vocab = model.vocab_size();
        let logits_base = model.logits_buffer_ptr();
        // Scoped history for the ReduceOnly immunity test — cloned before the
        // `&mut a.grammar_state` borrow below.
        let scoped_history: Vec<u32> =
            if fast_penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly {
                crate::scheduler::sample_step::penalty_history_scope(
                    &a.output_tokens,
                    ctx.tool_call_end_token,
                )
                .to_vec()
            } else {
                Vec::new()
            };
        let before = a.grammar_state.as_ref().map(|gs| gs.num_history_steps());
        let mut fast: Vec<u32> = Vec::with_capacity(k);
        let mut all_allowed = true;
        // Scoped block so `gs`'s mutable borrow ends before the post-loop
        // rollback re-borrows `a.grammar_state`. let-else (not `.expect()`)
        // keeps clippy happy — `is_some()` is gated in the `if` condition above.
        {
            let Some(gs) = a.grammar_state.as_mut() else {
                unreachable!("grammar_state present (gated by is_some above)")
            };
            for (i, &tok) in argmax_ids.iter().enumerate() {
                // ReduceOnly regime: the argmax must be penalty-immune (not in
                // the scoped history + raw logit > 0) or we take the slow path.
                if fast_penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly
                    && !crate::scheduler::fast_greedy::argmax_immune(tok, &scoped_history, || {
                        crate::scheduler::fast_greedy::logit_is_positive(
                            model,
                            logits_base,
                            row_base + i,
                            vocab,
                            tok,
                        )
                    })
                {
                    all_allowed = false;
                    break;
                }
                let allowed = if gs.is_terminated() {
                    true // no further constraint past grammar completion
                } else {
                    gs.fill_bitmask();
                    gs.is_token_allowed(tok)
                };
                if !allowed {
                    all_allowed = false;
                    break;
                }
                fast.push(tok);
                // Speculatively advance so position i+1's bitmask reflects the
                // post-emit state (mirrors the slow path). Skip after the last.
                if i + 1 < k && !gs.is_terminated() {
                    let _ = gs.accept_token(tok);
                }
            }
        }
        // Roll back the speculative advances to the exact pre-call state
        // (history delta — stop/terminated tokens don't advance; BUG#3).
        if let (Some(b), Some(gs)) = (before, a.grammar_state.as_mut()) {
            let adv = gs.num_history_steps().saturating_sub(b);
            if adv > 0 {
                gs.rollback(adv);
            }
        }
        ctx.timing.record(Phase::FastGreedy, t_fast);
        if all_allowed && fast.len() == k {
            return Ok(fast); // no D2H, no CPU pipeline — all positions GPU-greedy + grammar-legal
        }
        // else: fall through to the slow path (matcher restored above).
    }

    // ── GRAMMARLESS fast-greedy (2026-07-30): the same #237 gate, minus the
    // bitmask ──
    //
    // The fast path above required `grammar_state.is_some()`, so plain chat —
    // the EASIEST regime (no mask to consult at all) — unconditionally paid
    // the slow tail below: per SEQUENCE per STEP, a blocking D2H of its
    // [K+1, vocab] logits rows (2 x 248,077 x BF16 = 992,308 B on the 27B at
    // the 16:1 ladder). The C=16 profile (PROGRESS_LOG 6.12) measured 2,541
    // such copies = 2.5 GB per 16x300-token burst, ~16 stream-drain waits per
    // step — the single largest slice of the host-bound decode wall.
    //
    // Eligibility mirrors the grammar arm exactly: greedy (temp==0 or forced),
    // not inside thinking, penalties classified by the SSOT `fast_greedy`
    // gate (Neutral, or ReduceOnly with the per-token immunity proof — same
    // scoped history, same `logit_is_positive` 2-byte probe). When every
    // position qualifies, the GPU argmax IS the masked-greedy pick and the
    // [K,vocab] D2H is skipped entirely.
    //
    // Same behavioral trade #237 shipped for grammar sequences: GPU-argmax
    // tie-breaking near equal logits can differ from the host FP32 scan, so
    // emitted tokens are NOT byte-invariant vs the slow path at near-ties.
    // Kill switch: ATLAS_NO_FAST_GREEDY_CHAT=1 restores the slow path.
    let chat_fast_gate = if !masked_verify
        && ctx.sampling.fast_greedy_chat
        && a.grammar_state.is_none()
        && !a.inside_thinking
        && (a.temperature == 0.0 || ctx.sampling.force_temp_zero)
    {
        crate::scheduler::fast_greedy::classify_penalties(
            &crate::scheduler::sample_step::penalty_params_for(
                a,
                crate::scheduler::sample_step::PositionKind::Verify,
                0.0,
                None,
                Vec::new(),
            ),
        )
    } else {
        crate::scheduler::fast_greedy::PenaltyGate::Blocked
    };
    if chat_fast_gate != crate::scheduler::fast_greedy::PenaltyGate::Blocked {
        let t_fast = std::time::Instant::now();
        let vocab = model.vocab_size();
        let logits_base = model.logits_buffer_ptr();
        let scoped_history: Vec<u32> =
            if chat_fast_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly {
                crate::scheduler::sample_step::penalty_history_scope(
                    &a.output_tokens,
                    ctx.tool_call_end_token,
                )
                .to_vec()
            } else {
                Vec::new()
            };
        let mut all_immune = true;
        for (i, &tok) in argmax_ids.iter().enumerate() {
            if chat_fast_gate != crate::scheduler::fast_greedy::PenaltyGate::Neutral
                && !policy.immune(tok, &scoped_history, || {
                    crate::scheduler::fast_greedy::logit_is_positive_checked(
                        model,
                        logits_base,
                        row_base + i,
                        vocab,
                        tok,
                    )
                })?
            {
                all_immune = false;
                break;
            }
        }
        ctx.timing.record(Phase::FastGreedy, t_fast);
        if all_immune {
            return Ok(argmax_ids.to_vec()); // no D2H, no CPU pipeline
        }
        // else: some position needs the penalty-aware pipeline — slow path.
    }

    let vocab = model.vocab_size();
    // BF16 always for verify path: `decode_verify_graphed_*` writes BF16
    // to `logits_buffer()`. The FP32-lm_head path (Gemma-4 dense) does
    // not go through verify (no MTP for dense Gemma).
    let elem_bytes = 2usize;
    let total = k * vocab * elem_bytes;
    let t_d2h = std::time::Instant::now();
    let mut buf = vec![0u8; total];
    if let Err(error) = model.copy_logits_to_host(
        model
            .logits_buffer_ptr()
            .offset(row_base * vocab * elem_bytes),
        &mut buf,
    ) {
        return match policy {
            CopyFailurePolicy::LegacyFallback => Ok(argmax_ids.to_vec()),
            CopyFailurePolicy::Propagate => Err(error),
        };
    }
    ctx.timing.record(Phase::D2h, t_d2h);

    let mut picks: Vec<u32> = Vec::with_capacity(k);
    // Snapshot the matcher's history depth BEFORE speculative advances so we
    // roll back exactly the ACTUAL advances afterward. BUG#3 (2026-06-02):
    // stop/EOS and terminated tokens return true from `accept_token` WITHOUT
    // advancing the matcher, so a count of `accept_token`→true calls would
    // over-rewind. `emit_token` (run after this helper) re-advances from the
    // restored, clean state.
    let grammar_steps_before = a.grammar_state.as_ref().map(|gs| gs.num_history_steps());

    for i in 0..k {
        let slice = &buf[i * vocab * elem_bytes..(i + 1) * vocab * elem_bytes];
        // P1-3 (2026-07-09): `i` threads the verify-position index down for
        // the per-position seed offset of the temp>0 sampling branch.
        let pick = verify_pick_with_pipeline(slice, false, vocab, a, ctx, i);
        picks.push(pick);

        // Speculatively advance the matcher with `pick[i]` so the next
        // position's bitmask reflects post-emit state. Skip on the last
        // position (no next position to mask) and when the seq has no
        // grammar (nothing to advance).
        if i + 1 < k
            && let Some(ref mut gs) = a.grammar_state
            && !a.inside_thinking
        {
            // Matcher advance can fail if `pick` is not in the current
            // bitmask. If our pipeline correctly applied the bitmask,
            // pick is the argmax over masked logits → MUST be in the
            // bitmask → advance MUST succeed. The defensive check
            // exists for forced-token fast-path returns where the
            // grammar may have terminated; those legitimately can't
            // advance further.
            if !gs.accept_token(pick) {
                tracing::debug!(
                    pick,
                    i,
                    "verify_pick: grammar speculative advance refused — pipeline picked a token outside the current bitmask. \
                     This indicates a stale bitmask in the pipeline or a forced-token fastpath that terminated grammar. \
                     Stopping speculation here; the real `accept_token` in emit_token will fail and end the response."
                );
                break;
            }
            // accept_token advanced the matcher as a side effect; the rollback
            // below counts the ACTUAL advances from matcher history (BUG#3).
        }
    }

    // Roll back exactly the ACTUAL speculative advances (history delta) so the
    // matcher returns to its pre-call state; `emit_token` then re-advances it
    // normally. BUG#3: counting from accept_token→true calls over-rewinds when
    // a stop/EOS/terminated token (which returns true WITHOUT advancing) lands
    // in the verified span.
    if let (Some(before), Some(gs)) = (grammar_steps_before, a.grammar_state.as_mut()) {
        let advanced = gs.num_history_steps().saturating_sub(before);
        if advanced > 0 {
            gs.rollback(advanced);
        }
    }

    Ok(picks)
}
