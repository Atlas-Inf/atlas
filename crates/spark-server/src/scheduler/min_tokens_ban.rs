// SPDX-License-Identifier: AGPL-3.0-only

//! The qwen4_exp min_tokens end-token ban at the target pick
//! (`ATLAS_QWEN4EXP_EOS_BAN=1`, default off; `EosBan::target`).
//!
//! Below a request's `min_tokens` the target never picks an end token (the
//! model's, even under `ignore_eos`, and the request's own): vLLM's min_tokens
//! processor. Without the ban an end token picked there is discarded (or,
//! under `ignore_eos`, emitted) and fed back, and the MTP draft head (the
//! first `--mtp-vocab` ids, 100k) can never propose it, so every such pick
//! rejects the rest of its verify window: a forced-length request that has
//! finished its answer commits about one token a step.
//!
//! One rule for every pick, serial decode and every verify row alike: the
//! pick with `n` tokens out excludes the ban's ids while `n < min_tokens`,
//! the same count the emission's discard tests (`think_commit`), so with the
//! ban nothing is discarded below the floor. Two places apply it:
//!
//! * the host pipeline (`process_position_logits`, the SSOT of serial decode's
//!   host path and the verify slow path) masks the ids to -inf before any
//!   stage; a verify span's rows see their own counts there (`SpanShadow`);
//! * the GPU-argmax fast paths ([`fix_raw_picks`]): a raw argmax that is a
//!   banned id is replaced by the host argmax of that row with the ids at
//!   -inf, the slow path's own masked-row argmax. A raw argmax that is not a
//!   banned id is the masked row's argmax already (it is the row's maximum
//!   and the lowest index holding it), so those rows are untouched, and every
//!   fast-path proof built on "the raw argmax" holds for the masked row.
//!
//! The device heads, their graphs and the qwen4_exp vocabulary-split head
//! are untouched: that head assembles the full logits rows on both ranks
//! (`qwen4exp_lmhead_split`), so there is no per-rank partial argmax to ban.
//! Without the switch `EosBan::target` is false and nothing here acts.

use crate::scheduler::types::ActiveSeq;
use spark_model::traits::Model;
use spark_runtime::gpu::DevicePtr;

/// The ids the sequence's next pick may not take, when its ban covers it.
pub(in crate::scheduler) fn banned_ids(a: &ActiveSeq) -> Option<[u32; 4]> {
    let ban = a.seq.eos_ban;
    (ban.target && a.output_tokens.len() < a.min_tokens).then_some(ban.ids)
}

/// Mask the banned ids of the next pick in a host logits row (`f32`).
pub(in crate::scheduler) fn mask_row(logits: &mut [f32], a: &ActiveSeq) {
    if let Some(ids) = banned_ids(a) {
        mask_ids(logits, ids);
    }
}

fn mask_ids(logits: &mut [f32], ids: [u32; 4]) {
    for id in ids {
        if let Some(v) = logits.get_mut(id as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// The first token's suppress list (`sample_first_token`): the end tokens it
/// always suppresses, plus, under a target ban with `min_tokens`, the ban's
/// ids (the model end tokens, which `ignore_eos` leaves out of `eos_tokens`).
pub(in crate::scheduler) fn first_token_suppress(
    eos_tokens: &[u32],
    min_tokens: usize,
) -> Vec<u32> {
    let ban = spark_model::traits::EosBan::for_request(0, min_tokens, eos_tokens);
    let mut ids = eos_tokens.to_vec();
    if ban.target {
        ids.extend(ban.ids.iter().filter(|&&id| id != u32::MAX));
        ids.sort_unstable();
        ids.dedup();
    }
    ids
}

/// Replace every raw GPU argmax in `picks` (the span's rows, `vocab` apart
/// from `rows`) that is a banned id by its row's argmax with the ids
/// excluded. `Ok(false)` leaves `picks` alone and the caller must take the
/// host pipeline: the span straddles the floor and holds a banned id (only
/// the host pipeline counts the floor row by row; an in-span discard shifts
/// it), or a content-loop steer could touch it (`loop_steer`).
pub(in crate::scheduler) fn fix_raw_picks(
    model: &dyn Model,
    a: &ActiveSeq,
    picks: &mut [u32],
    rows: DevicePtr,
    fp32: bool,
) -> anyhow::Result<bool> {
    // A content-loop steer (`loop_steer`) is applied by the host pipeline,
    // row by row.
    if crate::scheduler::loop_steer::span_may_steer(a, picks) {
        return Ok(false);
    }
    let Some(ids) = banned_ids(a) else {
        return Ok(true);
    };
    if !picks.iter().any(|p| ids.contains(p)) {
        return Ok(true);
    }
    // Row r's count is at most `n + r`: every row is below the floor when
    // the last one is.
    if a.output_tokens.len() + picks.len() > a.min_tokens {
        return Ok(false);
    }
    let vocab = model.vocab_size();
    let elem = if fp32 { 4 } else { 2 };
    let mut bytes = vec![0u8; vocab * elem];
    let mut row = Vec::new();
    for (r, pick) in picks.iter_mut().enumerate() {
        if ids.contains(pick) {
            model.copy_logits_to_host(rows.offset(r * vocab * elem), &mut bytes)?;
            crate::scheduler::verify_pipeline_helper::dequant_into(&bytes, fp32, vocab, &mut row);
            mask_ids(&mut row, ids);
            *pick = spark_runtime::sampler::argmax_first_wins_f32(&row);
        }
    }
    Ok(true)
}

#[cfg(test)]
#[path = "min_tokens_ban_tests.rs"]
mod tests;
