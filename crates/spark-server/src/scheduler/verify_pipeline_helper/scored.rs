// SPDX-License-Identifier: AGPL-3.0-only

//! Select and score the same processed rows, before the span restores state.
//! As in serial decode, reported logprobs include masks, penalties and bias,
//! but precede temperature and top-k/top-p sampling transforms.

use super::{ActiveSeq, HostRows, LogitsContext, Model};
use crate::api::TokenLogprobs;

pub(in crate::scheduler) struct ScoredPicks {
    pub tokens: Vec<u32>,
    pub logprobs: Vec<TokenLogprobs>,
}

/// The opt-in reporting path. No logits I/O or selection for ordinary requests.
/// Dense batched requests read only this sequence's rows. Never rescore raw rows
/// or replay the processors after SpanShadow has restored the sequence.
pub(in crate::scheduler) fn requested(
    model: &dyn Model,
    rows: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    row_base: usize,
    drafts: Option<&[u32]>,
) -> anyhow::Result<Option<ScoredPicks>> {
    let Some(k) = a.top_logprobs else {
        return Ok(None);
    };
    let result = select(model, rows, a, ctx, row_base, drafts, k);
    if let Err(error) = &result {
        abort_exchange(model, a, "scored rows", error);
    }
    result.map(Some)
}

fn select(
    model: &dyn Model,
    rows: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    row_base: usize,
    drafts: Option<&[u32]>,
    k: u8,
) -> anyhow::Result<ScoredPicks> {
    let vocab = model.vocab_size();
    let mut buf = HostRows::take(&ctx.scratch.host_bytes, rows * vocab * 2);
    super::selection::copy_rows(model, row_base, ctx, &mut buf)?;
    let mut row = Vec::with_capacity(vocab);
    let mut logprobs = Vec::with_capacity(rows);
    let tokens = super::selection::span_picks(rows, a, ctx, drafts, |i, a| {
        super::dequant_into(
            &buf[i * vocab * 2..(i + 1) * vocab * 2],
            false,
            vocab,
            &mut row,
        );
        let token = super::pick_dequantized(&mut row, vocab, a, ctx);
        logprobs.push(crate::scheduler::logprobs::extract_logprobs_from_f32(
            &row, token, k as usize,
        ));
        Ok::<_, std::convert::Infallible>(token)
    })
    .unwrap_or_else(|never| match never {});
    Ok(ScoredPicks { tokens, logprobs })
}

/// A failed read after the forward leaves the worker protocol uncertain.
/// Abort the request; EP ranks follow the existing lifeline fatal-step policy.
pub(in crate::scheduler) fn abort_exchange(
    model: &dyn Model,
    a: &mut ActiveSeq,
    operation: &str,
    error: &anyhow::Error,
) {
    let reason = format!(
        "verify {operation} failed (slot={}, seq_len={}): {error:#}",
        a.seq.slot_idx, a.seq.seq_len
    );
    a.abort_on_engine_error(&reason);
    if model.is_ep() {
        tracing::error!("{reason}; EP verdict/exchange state uncertain; terminating rank");
        crate::ep_peer_lifeline::terminate();
    }
}

#[cfg(test)]
#[path = "scored_tests.rs"]
mod tests;
