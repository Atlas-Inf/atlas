// SPDX-License-Identifier: AGPL-3.0-only

//! Depth-aware KV admission: reserve DECODE room, not just the prompt.
//!
//! Admission used to size a request against its PROMPT only
//! (`blocks_needed = prompt_len / block_size + 1`) and never re-consult the
//! pool, so the pool was overcommitted by exactly each sequence's decode
//! depth. Measured on the C=128 ladder (Qwen3.8-27B/GB10): 128 sequences
//! admitted into a 102k-token pool against a 157k-token true demand, four
//! preemption waves, 171 decode-time preemptions. Preemption now RESUMES
//! victims (see `preempt`), but thrash is still pure overhead — the honest
//! fix is to not admit what cannot fit.
//!
//! Policy: a request reserves `prompt + min(max_tokens, WATERMARK)` tokens
//! of blocks (clamped to the served context ceiling), counted against the
//! TOTAL pool minus the same commitment of everything already in flight
//! (active, prefilling, spilled, requeued). Requests beyond capacity stay
//! in the pending queue — the existing queueing machinery — and admit as
//! earlier sequences finish. When everything genuinely fits, the gate
//! admits exactly what the old code admitted: behavior at C<=64 on the
//! measured ladder is unchanged (pinned in tests below).
//!
//! WATERMARK (PCND — explicit, documented default): default `max_seq_len`
//! (0/unlimited ⇒ no clamp), i.e. the reservation is the request's own
//! `max_tokens` — the honest conservative bound, since a request can never
//! generate past it. Operators who prefer overcommit (betting that real
//! generations stop early) set `ATLAS_KV_ADMIT_WATERMARK=<tokens>` lower;
//! `0` reserves prompt-only, which is the pre-gate legacy behavior, and any
//! override below the honest bound keeps a WARN so the C=128 failure mode
//! is at least attributable. The boot-time `KV OVERCOMMIT` warning in
//! `factory/build.rs` is unchanged and complementary (it sizes the pool;
//! this gates runtime admission).

use super::*;

/// Resolve the admission watermark once at scheduler start.
pub(super) fn resolve_admit_watermark(max_seq_len: usize) -> usize {
    let default = if max_seq_len > 0 {
        max_seq_len
    } else {
        usize::MAX
    };
    match std::env::var("ATLAS_KV_ADMIT_WATERMARK") {
        Err(_) => default,
        Ok(v) => match v.parse::<usize>() {
            Ok(w) => {
                if w < default {
                    tracing::warn!(
                        "ATLAS_KV_ADMIT_WATERMARK={w} < the honest bound ({default}): \
                         admission may OVERCOMMIT the KV pool; sequences past the \
                         watermark depth will hit decode-time preemption (resume, \
                         not kill — but pure overhead). 0 = legacy prompt-only \
                         reservation."
                    );
                }
                w
            }
            Err(_) => {
                tracing::warn!(
                    "ATLAS_KV_ADMIT_WATERMARK={v:?} is not an integer; using default {default}"
                );
                default
            }
        },
    }
}

/// SSOT block-count formula — the same shape admission has always used
/// (`tokens / block_size + 1`), kept so the gate never disagrees with the
/// legacy prompt sizing at watermark 0.
pub(super) fn blocks_for_tokens(tokens: usize, block_size: usize) -> usize {
    tokens / block_size.max(1) + 1
}

/// One in-flight sequence's KV demand, in tokens.
pub(super) struct SeqDemand {
    /// Tokens whose KV exists or must exist at resume (prompt + processed).
    pub current_tokens: usize,
    /// Generation still owed (`remaining` / `max_tokens`).
    pub budget_tokens: usize,
}

/// Blocks to reserve for one sequence: current + min(budget, watermark),
/// clamped to the served context ceiling (a sequence can never grow past
/// `max_seq_len`, so reserving beyond it would be dishonest the other way).
pub(super) fn seq_commitment_blocks(
    d: &SeqDemand,
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> usize {
    let mut depth = d
        .current_tokens
        .saturating_add(d.budget_tokens.min(watermark));
    if max_seq_len > 0 {
        depth = depth.min(max_seq_len.max(d.current_tokens));
    }
    blocks_for_tokens(depth, block_size)
}

/// Total reserved blocks for everything already in flight.
pub(super) fn committed_blocks(
    demands: &[SeqDemand],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> usize {
    demands
        .iter()
        .map(|d| seq_commitment_blocks(d, watermark, max_seq_len, block_size))
        .sum()
}

/// How many of `reqs` (`(prompt_len, max_tokens)`, in admission order) fit.
///
/// Returns `(admit_count, forced_oversize)`. Admission stops at the FIRST
/// request that does not fit (no head-of-line bypass: a small request must
/// not starve a big one that arrived first). LIVENESS: when nothing at all
/// is in flight and even the first request cannot fit, it is admitted
/// anyway (`forced_oversize = true`) — exactly today's behavior, where the
/// block allocator back-pressures at runtime — because queueing it forever
/// against an empty pool serves nobody.
pub(super) fn admit_count(
    total_blocks: usize,
    committed: usize,
    reqs: &[(usize, usize)],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> (usize, bool) {
    let mut used = committed;
    let mut n = 0usize;
    for &(prompt, max_tokens) in reqs {
        let need = seq_commitment_blocks(
            &SeqDemand {
                current_tokens: prompt,
                budget_tokens: max_tokens,
            },
            watermark,
            max_seq_len,
            block_size,
        );
        if used.saturating_add(need) <= total_blocks {
            used += need;
            n += 1;
        } else if n == 0 && committed == 0 {
            return (1, true);
        } else {
            break;
        }
    }
    (n, false)
}

fn reject_oversized(
    new_reqs: Vec<InferenceRequest>,
    total_blocks: usize,
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> Vec<InferenceRequest> {
    new_reqs
        .into_iter()
        .filter_map(|req| {
            let demand = SeqDemand {
                current_tokens: req.prompt_len(),
                budget_tokens: req.max_tokens(),
            };
            let need = seq_commitment_blocks(&demand, watermark, max_seq_len, block_size);
            if need <= total_blocks {
                return Some(req);
            }
            let message = format!(
                "Request needs {need} KV blocks including generation, \
                 but the pool has {total_blocks} usable blocks"
            );
            let mut sink = match req {
                InferenceRequest::Streaming { token_tx, .. } => ResponseSink::Streaming(token_tx),
                InferenceRequest::Blocking { response_tx, .. } => {
                    ResponseSink::Blocking(Some(response_tx))
                }
            };
            lifecycle::send_error_to_sink(&mut sink, &message);
            None
        })
        .collect()
}

/// Runtime gate: split this tick's drained requests into an admissible
/// prefix (returned) and an overflow tail (pushed back to the FRONT of the
/// pending queue, preserving arrival order ahead of newer requests).
#[allow(clippy::too_many_arguments)]
pub(super) fn gate_admissions(
    model: &dyn Model,
    pending: &std::sync::Arc<(Mutex<PendingQueue>, Condvar)>,
    new_reqs: Vec<InferenceRequest>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    swapped: &[SwappedSeq],
    preempted: &[PreemptedSeq],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> Vec<InferenceRequest> {
    if new_reqs.is_empty() {
        return new_reqs;
    }
    let total_blocks = model.num_total_blocks();
    if total_blocks == 0 {
        // Backend without a paged KV pool (or no occupancy info): nothing to
        // reserve against — admit as before.
        return new_reqs;
    }
    // Under EP a worker cannot back-pressure: its block allocation failure is
    // fatal to the pair. A request that can never fit the pool is refused
    // with an error instead of being forced in.
    let new_reqs = if model.is_ep() {
        reject_oversized(new_reqs, total_blocks, watermark, max_seq_len, block_size)
    } else {
        new_reqs
    };
    let mut demands: Vec<SeqDemand> =
        Vec::with_capacity(active.len() + prefilling.len() + swapped.len() + preempted.len());
    demands.extend(active.iter().map(|a| SeqDemand {
        // +1: the pending `last_token` decode input not yet in seq_len.
        current_tokens: a.seq.seq_len + 1,
        budget_tokens: a.remaining,
    }));
    demands.extend(prefilling.iter().map(|p| SeqDemand {
        current_tokens: p.prompt_tokens.len(),
        budget_tokens: p.max_tokens,
    }));
    demands.extend(swapped.iter().map(|s| SeqDemand {
        current_tokens: s.seq_len + 1,
        budget_tokens: s.remaining,
    }));
    demands.extend(preempted.iter().map(|p| SeqDemand {
        current_tokens: p.tokens.len() + 1,
        budget_tokens: p.a.remaining,
    }));
    let committed = committed_blocks(&demands, watermark, max_seq_len, block_size);
    let infos: Vec<(usize, usize)> = new_reqs
        .iter()
        .map(|r| (r.prompt_len(), r.max_tokens()))
        .collect();
    let (admit, forced) = admit_count(
        total_blocks,
        committed,
        &infos,
        watermark,
        max_seq_len,
        block_size,
    );
    if forced {
        tracing::warn!(
            "admitting a request whose reservation exceeds the whole KV pool \
             ({} prompt + min({}, watermark {}) tokens vs {} blocks); the block \
             allocator will back-pressure at runtime",
            infos[0].0,
            infos[0].1,
            watermark,
            total_blocks,
        );
    }
    // Overflow re-queues cycle through drain→gate every tick while parked;
    // log on CHANGE only, or a parked C=128 burst emits thousands of
    // identical lines per minute.
    static LAST_QUEUED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if admit >= new_reqs.len() {
        LAST_QUEUED.store(0, std::sync::atomic::Ordering::Relaxed);
        return new_reqs;
    }
    let mut admitted = new_reqs;
    let overflow = admitted.split_off(admit);
    if LAST_QUEUED.swap(overflow.len(), std::sync::atomic::Ordering::Relaxed) != overflow.len() {
        tracing::info!(
            "KV admission: {} of {} request(s) fit ({} blocks committed of {}); \
             {} queued until capacity frees",
            admitted.len(),
            admitted.len() + overflow.len(),
            committed,
            total_blocks,
            overflow.len(),
        );
    }
    let mut g = pending.0.lock();
    for (i, req) in overflow.into_iter().enumerate() {
        g.requests.insert(i, req);
    }
    admitted
}

#[cfg(test)]
#[path = "admission_tests.rs"]
mod tests;
