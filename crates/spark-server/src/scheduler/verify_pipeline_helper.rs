// SPDX-License-Identifier: AGPL-3.0-only

//! Verify-time pre-sample LogitsProcessor pipeline (Phase C-2 wiring).
//!
//! The MTP / speculative-decode verify paths used to consume the raw
//! GPU `argmax_bf16` ID at every verify position, completely bypassing
//! the 8-stage [`crate::scheduler::logit_processors`] pipeline that the
//! non-MTP path runs on every sampled token. Result: tokens emitted
//! through verify (the dominant decode path when MTP is enabled —
//! every accepted/bonus token came from `decode_verify_graphed`) never
//! saw mid-word `</think>` defer, post-close think mask, tool-during-
//! think mask, forced think-end injection, pin-to-tool-call, forced-
//! token fast-path, or grammar bitmask. This is the root cause of
//! grammar desync, malformed tool calls, mid-word `</think>` cuts and
//! stray `<think>` re-entry observed on Qwen3.6-FP8 (opencode-session
//! transcripts, 2026-05-24).
//!
//! This module replays the same dequant + pipeline on a host-side copy
//! of the verify logits buffer (`[K, vocab]` BF16, written by
//! `decode_verify_graphed_*` into `model.logits_buffer_ptr()`), then
//! picks the resulting argmax. Cost: ~0.8 ms per verify position for a
//! ~256k vocab on host, mirroring the non-MTP `process_seq_logits` path
//! in `decode_logits_seq.rs`. The CUDA-graphed `argmax_bf16` saving of
//! ~0.5 ms/step is preserved for the **draft** path (drafts already go
//! through a separate grammar-bitmask path in MTP propose); only the
//! **verify-time** argmax is replaced.
//!
//! Per-position semantics: the pipeline is applied independently to
//! each verify position 0..K. For position 0 the `ActiveSeq` state is
//! exactly the post-`last_token` state, identical to the non-MTP
//! decode site. For positions ≥ 1, the driver SPECULATIVELY ADVANCES
//! the xgrammar matcher via `gs.accept_token(pick_{i-1})` between
//! positions, so each position's bitmask reflects the matcher state
//! that will actually exist at `emit_token` time on the accept path.
//! Speculative advances are rolled back via `gs.rollback(n)` once all
//! K positions have been picked; the real `emit_token` calls then
//! re-advance the matcher normally for the verified tokens that
//! actually get emitted.
//!
//! **DO NOT remove the speculative advance.** Prior versions emitted
//! position-1 argmax against position-0 bitmask, which desynced
//! xgrammar on the accept path and tripped the non-silent
//! `accept_token` kill switch (observed live on
//! opencode-realfix.jsonl 2026-05-24: every response ended with
//! `length` + `tok=198 output_len=30-60` because the bonus token was
//! masked at position 0's state — a `\n` legal at JSON-value-start
//! is not legal at JSON-comma-or-closebrace).
//!
//! Every other piece of state a position reads (`<think>` phase, the
//! mid-word and sentence-boundary lookbacks, the penalty history, the
//! pipeline's own counters) is advanced between positions by committing the
//! previous pick through the commit path's own rule, then restored
//! (`emit_step::SpanShadow`): position `i` sees what serial decode would.

mod argmax;
mod batch;
pub(in crate::scheduler) use batch::{applies as pick_batch_applies, verify_pick_batch};
mod fast_masked;
mod prepick;
mod scratch;
pub(in crate::scheduler) use scratch::HostRows;
mod selection;
mod selection_io;

use crate::scheduler::ActiveSeq;
use crate::scheduler::helpers::bf16_to_f32;
use crate::scheduler::logit_processors::LogitsContext;
use spark_model::traits::Model;

// `ATLAS_DISABLE_FAST_GREEDY` is now `SchedLevers::fast_greedy_grammar`,
// read off `LogitsContext::sampling` at the one site that gated on it.

// The DFlash verify statics are now `SchedLevers::dflash_*`.
// The `ATLAS_NO_MTP_VERIFY_SAMPLE` kill switch is now
// `SchedLevers::mtp_verify_sample`, carried on `LogitsContext`.

/// Per-position verify logits, dequantised + processed through the full
/// pre-sample pipeline. Returns the chosen token: either the forced
/// token from a [`crate::scheduler::logit_processors::forced_token::ForcedTokenFastPath`]
/// short-circuit, or the post-pipeline argmax.
///
/// `logits_bytes`: byte slice for ONE verify position; length
/// `vocab_size * 2` (BF16) or `vocab_size * 4` (FP32).
/// `is_fp32`: true when the model emits FP32 logits (Gemma-4 dense).
/// `a`: the active sequence; the pipeline mutates seq state in place
/// (F2 confidence arm, sentence_defer_count, etc.).
/// `ctx`: tokenizer special-token IDs used by the pipeline.
///
/// Mirrors the host-side path of `decode_logits_seq::process_seq_logits`
/// for byte-identical pipeline semantics.
pub fn verify_pick_with_pipeline(
    logits_bytes: &[u8],
    is_fp32: bool,
    vocab_size: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> u32 {
    use crate::scheduler::mtp_timing::Phase;
    // 1. Dequant per the same scheme as `process_seq_logits`, into a REUSED
    //    thread-local buffer rather than a fresh ~1 MB `Vec<f32>` per K position
    //    — see `scratch.rs`. Semantically inert: every `vocab_size` entry is
    //    overwritten before any read.
    let t_dequant = std::time::Instant::now();
    let mut f32_logits = scratch::DEQUANT_SCRATCH.with(|s| std::mem::take(&mut *s.borrow_mut()));
    dequant_into(logits_bytes, is_fp32, vocab_size, &mut f32_logits);
    ctx.timing.record(Phase::Dequant, t_dequant);
    // Hand the allocation back on EVERY exit below (forced-token short circuit,
    // temp>0 sample, argmax), or the next call allocates from scratch again and
    // the reuse is silently lost.
    let mut f32_logits = scratch::ScratchGuard(f32_logits);
    pick_dequantized(&mut f32_logits, vocab_size, a, ctx)
}

/// Step 1 of [`verify_pick_with_pipeline`]: one row's logits as F32, exactly
/// (BF16 -> F32 is a shift). `out` is cleared and refilled.
pub(super) fn dequant_into(
    logits_bytes: &[u8],
    is_fp32: bool,
    vocab_size: usize,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.reserve(vocab_size);
    if is_fp32 {
        out.extend((0..vocab_size).map(|j| {
            let off = j * 4;
            f32::from_le_bytes([
                logits_bytes[off],
                logits_bytes[off + 1],
                logits_bytes[off + 2],
                logits_bytes[off + 3],
            ])
        }));
    } else {
        out.extend((0..vocab_size).map(|j| {
            let lo = logits_bytes[j * 2];
            let hi = logits_bytes[j * 2 + 1];
            bf16_to_f32(lo, hi)
        }));
    }
}

/// Steps 2-4 of [`verify_pick_with_pipeline`] on an already-dequantized row,
/// which the pipeline then masks and penalises in place.
pub(super) fn pick_dequantized(
    f32_logits: &mut [f32],
    vocab_size: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> u32 {
    use crate::scheduler::mtp_timing::Phase;
    // 2. Build this position's penalty/bias params (Verify kind: greedy,
    //    seed-free, the request's bias — the builder still appends the A4
    //    floor and the rep/presence/freq/LZ/DRY gates from `a`). Cloned before
    //    the `&mut a` borrow in `process_position_logits`.
    //
    //    Without these penalties MTP-VERIFIED tokens were decided by a
    //    penalty-FREE argmax, so the MODEL.toml `repetition_penalty` /
    //    `dry_multiplier` never reached the dominant decode path and the
    //    model degenerated into repeated tool-call argument junk. The
    //    resulting emission is a penalty-aware ARGMAX (greedy) — an intended
    //    behavioral delta for speculative acceptance. Backward-compatible: a
    //    no-op when the penalties are neutral (rep==1.0, dry==0.0, etc.).
    let penalties = crate::scheduler::sample_step::verify_penalty_params(a);

    // 3. Unified per-position post-processing (SSOT shared with the non-MTP
    //    path): force-temp-zero bypass → pipeline (forced-token short
    //    circuit) → penalties+bias. A `Some(tok)` return is the forced /
    //    bypass token — emit directly, no argmax scan. R1: this does NOT
    //    advance the grammar matcher; the K-loop in
    //    `verify_pick_all_with_pipeline` owns `accept_token` / `rollback`.
    let t_proc = std::time::Instant::now();
    if let Some(tok) = crate::scheduler::logit_processors::process_position_logits(
        f32_logits,
        a,
        ctx,
        &penalties,
        crate::scheduler::sample_step::PositionKind::Verify,
    ) {
        ctx.timing.record(Phase::PipelineProc, t_proc);
        return tok;
    }
    ctx.timing.record(Phase::PipelineProc, t_proc);

    // 4a. P1-3 (2026-07-09): when the request asked for temperature > 0,
    //     SAMPLE from the processed logits instead of taking the argmax.
    //     The processors (grammar bitmask, think/tool sched, penalties+bias)
    //     already ran in place above, so masked tokens sit at -inf and the
    //     sampler's candidate filter excludes them — the sampled pick is
    //     grammar-mask-allowed by construction. This mirrors the non-MTP
    //     tail of `decode_logits_seq::process_seq_logits` exactly: neutral
    //     penalty params (penalties were applied in step 3, so the sampler's
    //     internal `apply_penalties_and_bias` is a no-op) + the sequence's
    //     temperature / top_k / top_p / top_n_sigma / min_p. min_p is the
    //     resolved request+MODEL.toml-floor value, subject to the P1-4
    //     ATLAS_NO_MTP_MINP kill-switch. The seed advances per emitted
    //     position (`output_tokens.len()`, which the span replay advances) —
    //     the offset FinalDecode uses for the same emitted position.
    //     Unreachable under ATLAS_FORCE_TEMP_ZERO (the bypass in
    //     `process_position_logits` returns Some(argmax) before this point);
    //     the guard is kept as documentation. Kill-switch:
    //     ATLAS_NO_MTP_VERIFY_SAMPLE=1 reverts to the pinned argmax below.
    if ctx.sampling.mtp_verify_sample && a.temperature > 0.0 && !ctx.sampling.force_temp_zero {
        let t_sample = std::time::Instant::now();
        let step_seed = a.seed.map(|s| s.wrapping_add(a.output_tokens.len() as u64));
        let sampler_shape = spark_runtime::sampler::SamplingParams {
            temperature: a.temperature,
            top_k: a.top_k,
            top_p: a.top_p,
            top_n_sigma: a.top_n_sigma,
            min_p: crate::scheduler::sample_step::effective_min_p(a.min_p, &ctx.sampling),
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            repetition_penalty_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            lz_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: penalties.dry_base,
            dry_allowed_length: penalties.dry_allowed_length,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: step_seed,
        };
        // SAFETY: `f32_logits` is a live Vec<f32> of `vocab_size` elements;
        // reinterpreting as bytes is the same cast the non-MTP sampler tail
        // uses (`decode_logits_seq.rs`).
        let f32_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
        let sampled =
            spark_runtime::sampler::sample_with_params_history(f32_bytes, &sampler_shape, &[]);
        // Recorded under the Argmax phase: it replaces the argmax pick and
        // keeps the mtp_timing phase set unchanged.
        ctx.timing.record(Phase::Argmax, t_sample);
        return sampled;
    }

    // 4. Argmax over the (now-masked-and-penalised) vector. Matches the
    //    sampler's argmax branch behaviour.
    let t_argmax = std::time::Instant::now();
    let best_id = argmax::argmax_first_wins(f32_logits);
    ctx.timing.record(Phase::Argmax, t_argmax);
    best_id
}

/// Convenience: copy the full `[K, vocab]` verify logits buffer to
/// host and apply [`verify_pick_with_pipeline`] to every position,
/// returning the K processed token IDs. Falls back to the raw argmax
/// IDs if the D2H copy fails (matches `verify_resample` and
/// `extract_verify_logprobs` failure semantics).
///
/// `argmax_ids` is the GPU-graphed argmax already returned by
/// `decode_verify_graphed*`; used as the fallback for the failure
/// path and as the array length source.
///
/// `row_base` (batched-MTP E12): first logits row of THIS sequence's
/// verify span within the shared `[R, vocab]` logits buffer. Single-
/// sequence verify paths pass 0 (rows 0..K — unchanged behaviour);
/// the batched K=4 verify passes `i*4` for sequence i so every read
/// (fast-path single-logit probes and the slow-path D2H) targets the
/// sequence's own rows.
pub fn verify_pick_all_with_pipeline(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    row_base: usize,
) -> Vec<u32> {
    selection::select(
        model,
        argmax_ids,
        a,
        ctx,
        row_base,
        selection_io::CopyFailurePolicy::LegacyFallback,
    )
    .unwrap_or_else(|_| argmax_ids.to_vec())
}

/// Checked, grammarless selection for the future paired driver (no serving caller yet).
///
/// Rejects an actual grammar before even an empty span. Unlike the legacy API,
/// ANY failed logits copy, including a fast-path probe, returns its error without
/// a later read or raw-ID success. Successful pipeline/fast-path semantics are shared.
pub fn verify_pick_all_with_pipeline_checked(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    row_base: usize,
) -> anyhow::Result<Vec<u32>> {
    selection::select(
        model,
        argmax_ids,
        a,
        ctx,
        row_base,
        selection_io::CopyFailurePolicy::Propagate,
    )
}
