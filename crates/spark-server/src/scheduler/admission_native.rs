// SPDX-License-Identifier: AGPL-3.0-only

//! Native GLM shared-prefix accounting. Credit only physical ownership, and
//! acquire one proven warm follower before another admission decision.
use super::*;
use std::collections::HashSet;

pub(super) fn resident_commitment<'a>(
    demands: &[SeqDemand],
    residents: impl Iterator<Item = &'a SequenceState>,
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> usize {
    let mut physical = HashSet::new();
    let mut residents = residents;
    let mut future = 0usize;
    for demand in demands {
        let logical = seq_commitment_blocks(demand, watermark, max_seq_len, block_size);
        match residents.next() {
            Some(seq) => {
                // Count allocated speculative tails even when they extend
                // beyond the promised depth. Shared sealed IDs count once.
                physical.extend(seq.block_table.iter().copied());
                future = future.saturating_add(logical.saturating_sub(seq.block_table.len()));
            }
            None => future = future.saturating_add(logical),
        }
    }
    future.saturating_add(physical.len())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn admit_follower(
    model: &dyn Model,
    req: &InferenceRequest,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    committed: usize,
    total: usize,
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> bool {
    // Native TP2/TP4 chunk-0 runs inline. Packed prefill is the only EP path
    // that could defer it; never credit that path. One FIRST request only:
    // no intervening prefill may evict the snapshot inspected below.
    if !model.native_mtp_enabled()
        || !model.is_ep()
        || block_size != 16
        || model.packed_prefill_capacity().1 != 0
        || req.disable_mtp()
        || req.has_image_pixels()
        || req.prompt_logprobs().is_some()
    {
        return false;
    }
    let Some(owner) = active
        .first()
        .map(|a| &a.seq)
        .or_else(|| prefilling.first().map(|p| &p.seq))
    else {
        return false;
    };
    let tokens = req.prompt_tokens_arc();
    let adapter = model.adapter_id_for(req.adapter_slot());
    let eligible = |s: &SequenceState| {
        !s.disable_mtp
            && s.prefix_lookup_applied
            && s.prefix_lookup_skip
            && s.vision_cache_key.is_empty()
            && s.disk_block_ids.is_empty()
            && s.block_table.len() >= s.cached_prefix_blocks
            && s.adapter_id == adapter
    };
    if model.tokens_contain_vision_pad(&tokens)
        || active
            .iter()
            .any(|a| a.finished || a.disable_mtp || !eligible(&a.seq))
        || prefilling
            .iter()
            .any(|p| p.disable_mtp || !eligible(&p.seq))
    {
        return false;
    }
    if owner.cached_prefix_blocks == 0 || owner.marconi_skip_to == 0 {
        return false;
    }
    let candidate = SeqDemand {
        current_tokens: tokens.len(),
        budget_tokens: req.max_tokens(),
    };
    let need = seq_commitment_blocks(&candidate, watermark, max_seq_len, block_size);
    if committed.saturating_add(need.saturating_sub(owner.cached_prefix_blocks)) > total {
        return false;
    }
    // Private rows lag target by one row. Reserve every active future row,
    // plus two blocks per owner for an open/COW tail and <=8 speculative rows.
    // The candidate can share only sealed predictor rows below its snapshot.
    let mut private = need.saturating_sub(owner.marconi_skip_to.saturating_sub(1) / block_size) + 2;
    for a in active {
        let demand = SeqDemand {
            current_tokens: a.seq.seq_len + 1,
            budget_tokens: a.remaining,
        };
        let depth = seq_commitment_blocks(&demand, watermark, max_seq_len, block_size);
        private = private
            .saturating_add(depth.saturating_sub(a.seq.seq_len.saturating_sub(1) / block_size) + 2);
    }
    // A warm prefill already owns predictor rows through its restored
    // boundary even while the scheduler walks cached chunks below it.
    for p in prefilling {
        let demand = SeqDemand {
            current_tokens: p.prompt_tokens.len(),
            budget_tokens: p.max_tokens,
        };
        let depth = seq_commitment_blocks(&demand, watermark, max_seq_len, block_size);
        let ready = p.seq.seq_len.max(p.seq.marconi_skip_to).saturating_sub(1) / block_size;
        private = private.saturating_add(depth.saturating_sub(ready) + 2);
    }
    // A transport error cannot be retried as a cache miss: the rank command
    // stream may be incomplete. Fail-stop as for other fatal protocol errors.
    let Some(shared) = model
        .native_prefix_admission_blocks(owner, &tokens, req.session_hash(), private)
        .expect("native prefix admission rank probe failed")
    else {
        return false;
    };
    if shared > owner.cached_prefix_blocks
        || committed.saturating_add(need.saturating_sub(shared)) > total
    {
        return false;
    }
    tracing::info!(
        shared_blocks = shared,
        private_reserved = private,
        active = active.len(),
        prefilling = prefilling.len(),
        "native shared-prefix admission: one warm follower acquires its prefix this tick"
    );
    true
}
