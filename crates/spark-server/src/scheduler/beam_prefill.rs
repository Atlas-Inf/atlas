// SPDX-License-Identifier: AGPL-3.0-only

//! Shared beam-hypothesis resolution for the two prefill entry points
//! (`prefill_a_step`, `prefill_b_step`).

use anyhow::Result;
use spark_model::traits::{BeamReq, Model, SequenceState};

/// Resolve a beam request's winning hypothesis: the scheduler's pre-computed
/// result when present (this request was fused with others into ONE batched
/// search by the co-dispatch pre-pass), else a per-request `generate_beam_batch`.
pub(super) fn resolve_beam_hyp(
    model: &dyn Model,
    precomputed: Option<Vec<u32>>,
    seq: &SequenceState,
    prompt_tokens: &[u32],
    max_tokens: usize,
) -> Result<Vec<u32>> {
    if let Some(h) = precomputed {
        return Ok(h);
    }
    let beam_req = BeamReq {
        prompt_tokens: prompt_tokens.to_vec(),
        src_lang_id: seq.src_lang_id,
        tgt_lang_id: seq.tgt_lang_id,
        adapter_slot: seq.adapter_slot,
        num_beams: seq.num_beams as usize,
        max_new: max_tokens,
        length_penalty: seq.length_penalty,
        early_stopping: seq.early_stopping,
    };
    Ok(model
        .generate_beam_batch(std::slice::from_ref(&beam_req))?
        .pop()
        .unwrap_or_default())
}

/// Group beam requests by adapter — the model's single global LoRA gate forces
/// one adapter per fused batch — then pack each group into chunks whose summed
/// beam count does not exceed `cap`. `items` is `(req_idx, adapter_slot,
/// num_beams)`; returns chunks of `req_idx` (adapter order is deterministic).
pub(super) fn pack_beam_chunks(items: &[(usize, i32, usize)], cap: usize) -> Vec<Vec<usize>> {
    let mut groups: std::collections::BTreeMap<i32, Vec<(usize, usize)>> =
        std::collections::BTreeMap::new();
    for &(idx, adapter, nb) in items {
        groups.entry(adapter).or_default().push((idx, nb));
    }
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    for reqs in groups.into_values() {
        let (mut chunk, mut beams) = (Vec::new(), 0usize);
        for (idx, nb) in reqs {
            if !chunk.is_empty() && beams + nb > cap {
                chunks.push(std::mem::take(&mut chunk));
                beams = 0;
            }
            chunk.push(idx);
            beams += nb;
        }
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
    }
    chunks
}
