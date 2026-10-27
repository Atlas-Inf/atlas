// SPDX-License-Identifier: AGPL-3.0-only

//! The first chunk of a prompt queued for it at admission
//! (`ATLAS_PREFILL_QUEUE_FIRST_CHUNK`, `phase_start_prefills::queue_first`),
//! run once the prompt is the head of the queue.
//!
//! It is the chunk `start_chunked_prefill` would have run inline, at the
//! length the admission computed, sent and run as every later chunk is
//! (`prefill_preempt::prefill_chunk_with_preemption` from position 0): the
//! same command words to the worker (its prompt length and every token, so
//! both ranks repeat one prefix-cache lookup) and the same KV preemption on
//! an agreed refusal, since by the time it runs it competes with decoding
//! sequences exactly as a later chunk does; a final refusal fails only its
//! request. Then the first-chunk normalization. Its setup reached the worker
//! at admission; this sends the chunk command right before the head runs
//! the chunk, so the worker runs the prompts' chunks in the head's order.
//! Never the two-phase prefill, which sends the worker nothing.

use spark_model::traits::Model;
use spark_runtime::gpu::DevicePtr;

use super::super::prefill_preempt::prefill_chunk_with_preemption;
use super::super::sample_first_token;
use super::super::types::{ActiveSeq, PrefillInProgress, ResponseSink};

/// Run `p`'s queued first chunk, if it has one; `completed` gets `(idx,
/// first token)` when it was the whole prompt, `(idx, None)` when it failed.
/// Returns false, having done nothing, for a prompt with no queued chunk.
/// `active` holds the preemption victims.
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    model: &dyn Model,
    p: &mut PrefillInProgress,
    idx: usize,
    active: &mut Vec<ActiveSeq>,
    prefill_stream: u64,
    prefill_event: u64,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    completed_indices: &mut Vec<(usize, Option<u32>)>,
) -> bool {
    let Some(chunk_len) = p.queued_first_chunk.take() else {
        return false;
    };
    let total = p.prompt_tokens.len();
    let is_last = chunk_len >= total;
    tracing::info!(
        "Queued prefill start: {total} prompt tokens, chunk_size={chunk_len}, queued {:.0} ms",
        p.request_start.elapsed().as_secs_f64() * 1e3,
    );
    let logits = (|| -> anyhow::Result<DevicePtr> {
        let logits = prefill_chunk_with_preemption(
            model,
            &p.prompt_tokens,
            &mut p.seq,
            p.disable_mtp,
            0,
            chunk_len,
            is_last,
            prefill_stream,
            active,
        )?;
        super::super::prefill_normalization::initial(model, &p.seq)?;
        Ok(logits)
    })();
    let logits = match logits {
        Ok(l) => l,
        Err(e) => {
            // The promotion reports it and frees the slot on every rank.
            tracing::error!("Queued prefill chunk failed: {e:#}");
            completed_indices.push((idx, None));
            return true;
        }
    };
    p.chunk_offset = chunk_len;
    if let Err(e) = model.record_event(prefill_event, prefill_stream) {
        tracing::warn!("Queued prefill record_event: {e:#}");
    }
    if let Err(e) = model.stream_wait_event(model.default_stream(), prefill_event) {
        tracing::warn!("Queued prefill stream_wait_event: {e:#}");
    }
    if is_last {
        match sample_first_token(
            model,
            logits,
            p.temperature,
            p.top_k,
            p.top_p,
            p.min_p,
            &crate::scheduler::min_tokens_ban::first_token_suppress(&p.eos_tokens, p.min_tokens),
            p.grammar_state.as_mut(),
            &sched.levers.sampling(),
        ) {
            Ok(first) => {
                tracing::info!("Prefill first token: {first}");
                completed_indices.push((idx, Some(first)));
            }
            Err(e) => {
                tracing::error!("Queued prefill sampling: {e:#}");
                completed_indices.push((idx, None));
            }
        }
    } else {
        tracing::info!("Prefill chunk {chunk_len}/{total} tokens");
    }
    true
}

/// A queued prompt whose streaming client is gone: nothing ran for it and
/// nobody reads its tokens, so it leaves the queue before its first chunk
/// (a buffered caller is caught by `ResponseSink::receiver_closed`). Prompts
/// already prefilling keep today's streaming contract (`cancel_flag`, read
/// at the emit and decode boundaries).
pub(super) fn abandoned(p: &PrefillInProgress) -> bool {
    p.queued_first_chunk.is_some()
        && matches!(&p.sink, ResponseSink::Streaming(tx) if tx.is_closed())
}
