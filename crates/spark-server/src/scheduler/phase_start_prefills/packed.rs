// SPDX-License-Identifier: AGPL-3.0-only

//! Packed multi-prompt prefill (`ATLAS_GLM_PACKED_PREFILL=1`, model side
//! `spark_model::model::glm_packed_prefill`): every new short prompt of a
//! tick in one target traversal, so the routed-expert weights a short pass is
//! bound by are read once for all of them instead of once per prompt.
//!
//! A prompt is packed when the model admits it (short, text only, no
//! restorable prefix-cache state) and the request needs nothing the packed
//! pass does not carry (no images, beams, prompt logprobs, adapter or
//! shared-prefix plant). With the switch on, such a prompt is packed whether
//! it arrives alone or with others, so its output does not depend on its
//! company; eligible prompts beyond one pass's capacity take another packed
//! pass in the same tick, never the ordinary path.
//!
//! After the pass each prompt is normalized, sampled and promoted exactly as
//! a prompt that finished its last chunk (`promote_completed_prefills`).

use spark_model::traits::{Model, SequenceState};

use super::super::phase_promote_prefills::promote_completed_prefills;
use super::super::sample_first_token;
use super::super::ttft_trace;
use super::super::types::{ActiveSeq, PrefillInProgress};
use crate::api::InferenceRequest;

/// Whether `req` may join a packed prefill whose passes hold `row_cap` rows.
pub(super) fn eligible(
    model: &dyn Model,
    req: &InferenceRequest,
    plants: bool,
    row_cap: usize,
) -> bool {
    let len = req.prompt_len();
    !plants
        && len >= 2
        && len <= row_cap
        && !req.has_image_pixels()
        && req.num_beams() <= 1
        && req.prompt_logprobs().is_none()
        && req.adapter_slot() < 0
        && model.packed_prefill_admits(
            req.prompt_tokens_arc().as_slice(),
            model.adapter_id_for(req.adapter_slot()),
        )
}

/// Split prompts of `lens` tokens, in order, into passes of at most `rows`
/// rows and `prompts` prompts each. Every prompt lands in some pass.
pub(super) fn plan(lens: &[usize], (rows, prompts): (usize, usize)) -> Vec<Vec<usize>> {
    let mut passes: Vec<Vec<usize>> = Vec::new();
    let mut used = 0usize;
    for (i, &len) in lens.iter().enumerate() {
        match passes.last_mut() {
            Some(pass) if pass.len() < prompts.max(1) && used + len <= rows => {
                pass.push(i);
                used += len;
            }
            _ => {
                passes.push(vec![i]);
                used = len;
            }
        }
    }
    passes
}

/// Run the packed passes over `ps` (prompts whose per-request setup ran,
/// nothing prefilled) and promote each prompt into `active`.
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    mut ps: Vec<PrefillInProgress>,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    active: &mut Vec<ActiveSeq>,
) {
    if ps.is_empty() {
        return;
    }
    let lens: Vec<usize> = ps.iter().map(|p| p.prompt_tokens.len()).collect();
    let passes = plan(&lens, model.packed_prefill_capacity());
    // Back to front, so the indices of the passes still to run stay valid.
    let mut groups: Vec<Vec<PrefillInProgress>> = Vec::with_capacity(passes.len());
    for pass in passes.iter().rev() {
        let start = pass[0];
        groups.push(ps.split_off(start));
    }
    groups.reverse();
    for mut group in groups {
        let rows: usize = group.iter().map(|p| p.prompt_tokens.len()).sum();
        let n = group.len();
        tracing::info!("Packed prefill: {n} prompt(s), {rows} rows");
        for p in &group {
            ttft_trace::packed(p.seq.slot_idx, n, rows);
            ttft_trace::mark(p.seq.slot_idx, ttft_trace::Mark::PrefillStart);
        }
        let result = {
            let (prompts, mut seqs): (Vec<&[u32]>, Vec<&mut SequenceState>) = group
                .iter_mut()
                .map(|p| (p.prompt_tokens.as_slice(), &mut p.seq))
                .unzip();
            model.prefill_packed(&prompts, &mut seqs)
        };
        for p in &group {
            ttft_trace::mark(p.seq.slot_idx, ttft_trace::Mark::PrefillSubmitted);
        }
        let completed: Vec<(usize, Option<u32>)> = match result {
            Ok(logits) => group
                .iter_mut()
                .zip(logits)
                .enumerate()
                .map(|(i, (p, logits))| {
                    p.chunk_offset = p.prompt_tokens.len();
                    if let Err(e) = super::super::prefill_normalization::initial(model, &p.seq) {
                        tracing::warn!("SSM state normalization failed: {e:#}");
                    }
                    let first = sample_first_token(
                        model,
                        logits,
                        p.temperature,
                        p.top_k,
                        p.top_p,
                        p.min_p,
                        &p.eos_tokens,
                        p.grammar_state.as_mut(),
                        &sched.levers.sampling(),
                    );
                    match first {
                        Ok(t) => {
                            tracing::info!("Packed prefill first token: {t}");
                            (i, Some(t))
                        }
                        Err(e) => {
                            tracing::error!("Packed prefill sampling: {e:#}");
                            (i, None)
                        }
                    }
                })
                .collect(),
            Err(e) => {
                tracing::error!("Packed prefill of {n} prompt(s) failed: {e:#}");
                (0..n).map(|i| (i, None)).collect()
            }
        };
        promote_completed_prefills(
            model,
            &mut group,
            completed,
            active,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            sched.limits.max_seq_len,
            sched.limits.glm_tool_boundary,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::plan;

    #[test]
    fn one_pass_while_everything_fits() {
        assert_eq!(plan(&[50, 54, 46], (8192, 16)), vec![vec![0, 1, 2]]);
        assert_eq!(
            plan(&[512; 8], (8192, 16)),
            vec![(0..8).collect::<Vec<_>>()]
        );
    }

    #[test]
    fn overflow_takes_another_pass_never_the_ordinary_path() {
        // Rows: 300 + 300 fit 700, the third starts a new pass.
        assert_eq!(plan(&[300, 300, 300], (700, 16)), vec![vec![0, 1], vec![2]]);
        // Prompts: at most two per pass.
        assert_eq!(
            plan(&[10, 10, 10, 10, 10], (8192, 2)),
            vec![vec![0, 1], vec![2, 3], vec![4]]
        );
        // Every prompt lands somewhere, in order.
        let lens = [100, 900, 50, 700, 300];
        let passes = plan(&lens, (1000, 3));
        let flat: Vec<usize> = passes.concat();
        assert_eq!(flat, (0..lens.len()).collect::<Vec<_>>());
        for pass in &passes {
            assert!(pass.iter().map(|&i| lens[i]).sum::<usize>() <= 1000 || pass.len() == 1);
            assert!(pass.len() <= 3);
        }
    }

    #[test]
    fn empty_plan_for_no_prompts() {
        assert!(plan(&[], (8192, 16)).is_empty());
    }
}
