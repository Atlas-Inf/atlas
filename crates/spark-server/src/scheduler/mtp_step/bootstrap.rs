// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence Phase-A bootstrap of the MTP step (moved verbatim from
//! `mtp_step.rs` for the 500-line cap): the decode, sample, emit and propose
//! of one draftless sequence, or DFlash's fused propose that defers its
//! verify to Phase B.

use super::glm_repair::mark_engine_error;
use super::*;

/// Bootstrap one draftless sequence. Returns true when its drafts were
/// stashed for Phase B's batched verify (DFlash at n >= 2) instead.
#[allow(clippy::too_many_arguments)]
pub(super) fn bootstrap_one(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    ladder_nd: usize,
    n_active: usize,
    glm_repaired_narrow: bool,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) -> bool {
    // DFlash path: skip the standalone M=1 decode. The fused pass already
    // computes every position's logit in one weight sweep, so the "next
    // decoded token" is the bonus token at result[num_accepted] — the logit
    // at the position immediately after the accepted prefix (§8 vLLM
    // bonus-token pattern). Propose initial drafts using the DFlash hidden
    // already captured at row 0 by the previous step's fused pass (or
    // prefill), then route through step_verify_k3/k2 which handles the
    // fused forward, accept/reject, bonus-token emit, and re-propose for
    // the next step. This replaces the two-sweep sequence (M=1 decode here
    // + M=1+k fused in Phase B) with a single M=1+k fused sweep.
    if dflash_verify_raw_argmax
        && !sched.levers.dflash_seam_serial
        && crate::scheduler::adaptive_spec::spec_allowed(a, sched)
    {
        // A strict grammar here speculates at full width: its verify
        // masks every row and trims the drafts (`strict_spec`).
        let eff = if a.grammar_state.is_some() && !a.strict_grammar() {
            1
        } else {
            num_drafts
        };
        let _gmask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            eff,
            &mut a.seq,
            0,
            _gmask.as_deref(),
        ) {
            Ok(init) if !init.is_empty() => {
                // n>=2: do not verify here. Stash drafts so Phase B can
                // run one decode_verify_batched over every ready seq.
                // In-loop step_verify_dflash left only 1 seq with drafts
                // (shared propose scratch / graph) and never hit Phase B.
                a.set_proposed_drafts(init);
                if n_active >= 2 && !dspark_batch_verify_disabled() {
                    return true;
                }
                let (init, conf) = a.take_drafts();
                if dflash_verify_raw_argmax || glm_repaired_narrow {
                    step_verify_dflash(
                        model,
                        a,
                        sched,
                        &init,
                        &conf,
                        num_drafts,
                        verify_ctx,
                        dflash_verify_raw_argmax,
                    );
                } else if eff >= 3 && init.len() >= 3 {
                    step_verify_k4(
                        model,
                        a,
                        sched,
                        &init,
                        num_drafts,
                        verify_ctx,
                        dflash_verify_raw_argmax,
                    );
                } else if eff >= 2 && init.len() >= 2 {
                    step_verify_k3(
                        model,
                        a,
                        sched,
                        &init,
                        num_drafts,
                        verify_ctx,
                        dflash_verify_raw_argmax,
                    );
                } else {
                    step_verify_k2(
                        model,
                        a,
                        sched,
                        &init,
                        num_drafts,
                        verify_ctx,
                        dflash_verify_raw_argmax,
                    );
                }
                return false;
            }
            Ok(_) => {
                tracing::warn!(
                    "DFlash bootstrap propose returned empty slot={} seq_len={}",
                    a.seq.slot_idx,
                    a.seq.seq_len
                );
                // Lightning product fail-closed boundary: an empty
                // product proposal is an admission violation, not a
                // silent no-spec fallback. Generic DFlash/MTP keeps the
                // legacy fall-through below. The guard marker makes the
                // truncation client-visible ("length" family), never a
                // natural "stop".
                if crate::scheduler::helpers::handle_dspark_bootstrap_proposal_failure(
                    model,
                    a,
                    crate::scheduler::helpers::ProposalOutcome::Empty,
                ) {
                    return false;
                }
            }
            Err(e) => {
                tracing::error!("DFlash bootstrap propose: {e:#}");
                if crate::scheduler::helpers::handle_dspark_bootstrap_proposal_failure(
                    model,
                    a,
                    crate::scheduler::helpers::ProposalOutcome::Error,
                ) {
                    return false;
                }
            }
        }
        // Rare fallback: propose failed or returned empty (e.g. drafter not
        // yet primed). Fall through to the standalone decode below so the
        // sequence emits its next token rather than stalling. Never
        // reached for the Lightning product (handled above).
    }

    // Non-DFlash path (or DFlash-propose fallback): EP broadcast + standalone decode.
    // EP: broadcast token to worker before decode (worker runs decode in lockstep).
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
        tracing::error!("EP broadcast bootstrap token: {e:#}");
        mark_engine_error(a, format!("EP broadcast bootstrap token failed: {e:#}"));
        return false;
    }
    let logits = match model.decode(a.last_token, &mut a.seq, 0) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("bootstrap decode error: {e:#}");
            mark_engine_error(a, format!("bootstrap decode failed: {e:#}"));
            return false;
        }
    };
    // Picked exactly as plain decode picks this row (`fast_greedy::pick_row`):
    // the GPU argmax only where the pipeline provably keeps it, otherwise the
    // full pipeline — inside `<think>` and right after it included, where the
    // raw argmax used to be taken unmasked.
    let tok = match crate::scheduler::fast_greedy::pick_row(model, logits, a, verify_ctx, None) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("bootstrap sample error: {e:#}");
            mark_engine_error(a, format!("bootstrap sampling failed: {e:#}"));
            return false;
        }
    };

    // Extract logprobs from bootstrap decode logits (single position).
    let lp = if let Some(k) = a.top_logprobs {
        extract_single_logprobs(model, logits, tok, k)
    } else {
        None
    };

    emit_token(a, tok, lp, sched);
    if a.finished {
        return false;
    }
    a.last_token = tok;
    // Adaptive speculation: count serial tokens toward the re-probe window.
    crate::scheduler::adaptive_spec::tick_serial(a, sched);

    // Ctx-holes fix (ATLAS_DFLASH_SERIAL_APPEND=1), COMPLEMENT-GATED:
    // the serial ctx-append fires iff propose() will NOT run this
    // iteration, so append and propose decode-append can never both
    // cover one token — double-append impossible by construction
    // (that was the cuMemcpyDtoDAsync status-1 crash).
    // `spec_allowed` is evaluated exactly once (it mutates re-probe
    // state); its verdict is reused for the propose gate below.
    // Exception — re-probe RESUME: the token decoded on the un-suspend
    // iteration would otherwise fall in a hole (the stale
    // `skip_next_decode_append` set by the last suspended token makes
    // the propose below skip its decode-append). Append it here; the
    // skip flag this sets is consumed by that propose — one append,
    // no duplicate, seam covered.
    let was_suspended = crate::scheduler::adaptive_spec::is_suspended(a, sched);
    let will_propose = crate::scheduler::adaptive_spec::spec_allowed(a, sched);
    let reprobe_resume = was_suspended && will_propose;
    if sched.levers.dflash_unified_ctx {
        // Unified ctx commit: same complement-gate as the old serial
        // append — fire iff propose() will NOT run (or re-probe resume),
        // so commit and propose decode-append never both cover a token.
        if !will_propose || reprobe_resume {
            let base_pos = a.seq.seq_len.saturating_sub(1);
            if let Err(e) = model.commit_ctx(&mut a.seq, 1, base_pos) {
                tracing::error!("commit_ctx (mtp serial): {e:#}");
            }
        }
    } else if sched.levers.dflash_serial_append
        && (!will_propose || reprobe_resume)
        && let Err(e) = model.dflash_serial_ctx_append(&mut a.seq)
    {
        tracing::error!("dflash_serial_ctx_append: {e:#}");
    }

    if let Err(e) = model.save_hidden_for_mtp(0, 0) {
        tracing::error!("save_hidden_for_mtp: {e:#}");
        if spark_model::speculative::glm_repair_policy::enabled() {
            mark_engine_error(a, format!("save_hidden_for_mtp failed: {e:#}"));
        }
        return false;
    }
    let _mtp_grammar_mask = mtp_grammar_mask_for(a);
    // BUG#4 (2026-06-02): when a grammar is active, generate only ONE draft.
    // run_mtp_propose_multi (mtp_multi.rs) masks only draft[0] with the
    // position-0 bitmask and leaves draft[1..] UNMASKED, so multi-draft +
    // grammar desyncs — a draft[1] token can violate its true per-position
    // mask, get verified+accepted, then be refused by the matcher later
    // (→ truncation). A single draft uses its own up-to-date mask and is
    // sound; drafts.len()==1 routes verify to the K=2 path. Mask is a no-op
    // when grammar is inactive, so NVFP4/non-tool paths keep full K.
    // 2026-07-09: hoisted to the `effective_drafts_under_grammar` SSOT,
    // now also applied at the five verify-path re-propose sites that
    // previously bypassed this clamp (the "mask held fixed" warn spam).
    // Composed with the K-vs-batch ladder: the bootstrap propose is
    // sized for the current concurrency so the next verify is uniform
    // at the ladder width (no surplus drafts to truncate).
    let effective_num_drafts =
        crate::scheduler::spec_step::effective_drafts_under_grammar(a, ladder_nd);
    // Adaptive speculation: a suspended seq skips proposing entirely and
    // stays on this serial bootstrap path until the re-probe fires.
    // (`will_propose` is the single spec_allowed evaluation above.)
    if will_propose {
        match model.run_mtp_propose_multi(
            tok,
            a.seq.seq_len,
            effective_num_drafts,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(drafts) if !drafts.is_empty() => {
                tracing::debug!("MTP bootstrap: tok={tok} → drafts={drafts:?}");
                a.set_proposed_drafts(drafts);
            }
            Ok(_) => {
                tracing::warn!("MTP propose returned empty");
                if spark_model::speculative::glm_repair_policy::enabled() {
                    mark_engine_error(a, "MTP propose returned empty");
                }
            }
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
                if spark_model::speculative::glm_repair_policy::enabled() {
                    mark_engine_error(a, format!("MTP proposal failed: {e:#}"));
                }
            }
        }
    }

    // A terminal proposal failure must not start a checkpoint on the
    // partially-mutated proposer state before retirement frees it.
    if a.finished {
        return false;
    }

    if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
        tracing::error!("bootstrap start_checkpoint_async: {e:#}");
    }
    false
}
