// SPDX-License-Identifier: AGPL-3.0-only

//! A prefill chunk carrying the active DFlash owners' verify rows (GLM, EP;
//! `Model::prefill_chunk_with_glm_owner_rows`): one target traversal instead
//! of the chunk followed by a separate owner-batched verify step, so every
//! weight is read once for both. The owners then run their ordinary tails and
//! batched re-propose, and sit out this tick's decode step.
//!
//! Opt-in (`ATLAS_GLM_FUSED_PREFILL_VERIFY=1`) until measured.

use spark_model::traits::Model;

use super::super::sample_first_token;
use super::super::types::{ActiveSeq, PrefillInProgress};
use super::super::verify_dflash_step::step_verify_glm_long_with;

/// The speculation settings of the step the owners would otherwise take.
pub(in crate::scheduler) struct SpecStep {
    pub num_drafts: usize,
    pub dflash_verify_raw_argmax: bool,
}

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_FUSED_PREFILL_VERIFY").as_deref() == Ok("1"))
}

/// Run `p`'s next chunk (at most `chunk_len` rows) carrying the active
/// owners' verify rows when the model can. Returns false, having done
/// nothing, when it cannot; otherwise the chunk ran or failed like an
/// ordinary chunk, and `rode` holds the owners' slots.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_fused_chunk(
    model: &dyn Model,
    p: &mut PrefillInProgress,
    idx: usize,
    active: &mut [ActiveSeq],
    chunk_len: usize,
    max_batch_tokens: usize,
    spec: &SpecStep,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    rode: &mut Vec<usize>,
) -> bool {
    if !enabled() || !spec.dflash_verify_raw_argmax {
        return false;
    }
    // `mtp_step`'s owner-batched width rule over the owners holding drafts.
    let eligible = |a: &ActiveSeq| a.grammar_state.is_none() && !a.finished;
    let lens: Vec<usize> = active
        .iter()
        .filter(|a| eligible(a))
        .map(|a| a.pending_drafts.len())
        .filter(|&len| len > 0)
        .collect();
    let Some(width) = lens
        .iter()
        .copied()
        .max_by_key(|&w| (lens.iter().filter(|&&len| len >= w).count() * (w + 1), w))
    else {
        return false;
    };
    let group: Vec<usize> = (0..active.len())
        .filter(|&i| eligible(&active[i]) && active[i].pending_drafts.len() >= width)
        .collect();
    let rows = width + 1;
    // The chunk makes room for the riding rows in the arena.
    let mut chunk_len = chunk_len.min(max_batch_tokens.saturating_sub(group.len() * rows));
    let is_last = p.chunk_offset + chunk_len >= p.prompt_tokens.len();
    if !is_last {
        chunk_len -= chunk_len % 64;
    }
    if !model.can_fuse_glm_prefill_verify(&p.prompt_tokens, &p.seq, chunk_len, group.len(), rows) {
        return false;
    }
    for &i in &group {
        active[i].pending_drafts.truncate(width);
        active[i].pending_draft_conf.truncate(width);
    }
    if let Err(e) = model.ep_broadcast_disable_mtp_for_seq(p.seq.slot_idx as u32, p.disable_mtp) {
        tracing::error!("EP broadcast fused chunk fence: {e:#}");
        completed_indices.push((idx, None));
        return true;
    }
    rode.extend(group.iter().map(|&i| active[i].seq.slot_idx));
    let mut batch: Vec<&mut ActiveSeq> = active
        .iter_mut()
        .enumerate()
        .filter(|(i, _)| group.contains(i))
        .map(|(_, a)| a)
        .collect();
    let stream = model.default_stream();
    let mut chunk_ran = false;
    let mut first = None;
    step_verify_glm_long_with(
        model,
        &mut batch,
        sched,
        spec.num_drafts,
        verify_ctx,
        spec.dflash_verify_raw_argmax,
        &mut |rows, tokens, seqs| {
            let (logits, ids) = model.prefill_chunk_with_glm_owner_rows(
                &p.prompt_tokens,
                &mut p.seq,
                p.chunk_offset,
                chunk_len,
                rows,
                tokens,
                seqs,
            )?;
            chunk_ran = true;
            p.chunk_offset += chunk_len;
            if let Err(e) = super::super::prefill_normalization::continuation(model, &p.seq, stream)
            {
                tracing::warn!("SSM state normalization failed: {e:#}");
            }
            // Before the owners' tails, which restore their rows over the
            // logits this sample reads.
            if is_last {
                first = Some(sample_first_token(
                    model,
                    logits,
                    p.temperature,
                    p.top_k,
                    p.top_p,
                    p.min_p,
                    &p.eos_tokens,
                    p.grammar_state.as_mut(),
                    &sched.levers.sampling(),
                ));
            }
            Ok(ids)
        },
    );
    tracing::info!(
        "Fused prefill chunk {}/{} tokens + {} owners x {rows} rows",
        p.chunk_offset,
        p.prompt_tokens.len(),
        group.len(),
    );
    if !chunk_ran {
        completed_indices.push((idx, None));
    } else if let Some(first) = first {
        match first {
            Ok(first) => completed_indices.push((idx, Some(first))),
            Err(e) => {
                tracing::error!("Fused prefill sampling: {e:#}");
                completed_indices.push((idx, None));
            }
        }
    }
    true
}
