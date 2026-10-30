// SPDX-License-Identifier: AGPL-3.0-only

//! Verify picks on an argmax-only vocabulary-split head that can still sample
//! (`Model::verify_split_sampling`: GLM over the TP2 pair or the TP4 chain).
//!
//! The head's logits buffer holds only this rank's part of each row, so the
//! host pipeline cannot read the rows. A sampled sequence whose pipeline only
//! masks or biases a few special tokens (the think and tool tags) picks from
//! the ranks' candidates (`spark_model::model::SplitRow`): its present tokens
//! at their logits, the absent ones NaN, through the same pipeline and the
//! same sampler as a whole row, which picks exactly what the whole row would
//! or answers that the candidates do not decide it. Only then, and for every
//! other sequence (penalties, grammar, a forced token, ...), are the rows
//! fetched whole and picked as on any other head. Rows are picked in order
//! and stop at the first that rejects its draft, so a sampled row's pick is
//! its sample and the draft is accepted exactly while the samples match it:
//! exact rejection sampling for a one-hot draft.

use super::{ActiveSeq, HostRows, LogitsContext, Model, pick_processed, verify_pick_with_pipeline};
use crate::scheduler::mtp_timing::Phase;

thread_local! {
    /// The vocabulary-wide row a candidates pick runs on, reused.
    static VIEW: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The split head's picks for verify rows `[0, argmax_ids.len())`, judged
/// against `drafts`: the span ends at the first pick that is not its draft.
pub(in crate::scheduler) fn split_verify_picks(
    model: &dyn Model,
    argmax_ids: &[u32],
    drafts: &[u32],
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> anyhow::Result<Vec<u32>> {
    let k = argmax_ids.len();
    let vocab = model.vocab_size();
    let t = std::time::Instant::now();
    let rows = if candidates_serve(a, ctx, k) {
        Some(model.verify_split_candidates(k, a.temperature, &specials(a, ctx))?)
    } else {
        None
    };
    ctx.timing.record(Phase::D2h, t);
    let mut whole: Option<HostRows> = None;
    let mut view = VIEW.with(|v| std::mem::take(&mut *v.borrow_mut()));
    let mut ids: Vec<u32> = Vec::new();
    let picks = super::selection::span_picks(k, a, ctx, Some(drafts), |i, a| {
        if let Some(row) = rows.as_ref().map(|r| &r[i]) {
            view.clear();
            view.resize(vocab, f32::NAN);
            ids.clear();
            for &(id, v) in &row.present {
                view[id as usize] = v;
                ids.push(id);
            }
            let absent = row.absent.map(|absent| (absent, ids.as_slice()));
            if let Some(pick) = pick_processed(&mut view, vocab, a, ctx, absent) {
                return Ok(pick);
            }
        }
        if whole.is_none() {
            let t = std::time::Instant::now();
            let mut buf = HostRows::take(&ctx.scratch.host_bytes, k * vocab * 2);
            model.verify_split_full_rows(k, &mut buf)?;
            ctx.timing.record(Phase::D2h, t);
            whole = Some(buf);
        }
        let buf = whole.as_ref().expect("fetched above");
        Ok::<_, anyhow::Error>(verify_pick_with_pipeline(
            &buf[i * vocab * 2..(i + 1) * vocab * 2],
            false,
            vocab,
            a,
            ctx,
        ))
    });
    VIEW.with(|v| *v.borrow_mut() = view);
    picks
}

/// Whether the candidates can serve this sequence's rows: a sampled row whose
/// pipeline acts on the special tokens alone (no grammar, neutral penalties
/// and bias, nothing forced, no loop steer, no F2 confidence sums).
fn candidates_serve(a: &ActiveSeq, ctx: &LogitsContext, k: usize) -> bool {
    use crate::scheduler::fast_greedy::{PenaltyGate, classify_penalties, tool_pin_armed};
    a.temperature > 0.0
        && ctx.sampling.mtp_verify_sample
        && !ctx.sampling.force_temp_zero
        && a.grammar_state.is_none()
        && !a.require_tool_call
        && !tool_pin_armed(a)
        && !(a.inside_thinking && a.force_end_thinking)
        && a.loop_steers >= a.loop_steer_max
        && !super::prepick::f2_may_run(a, ctx, k)
        && classify_penalties(&crate::scheduler::sample_step::verify_penalty_params(a))
            == PenaltyGate::Neutral
}

/// The tokens the pipeline masks or biases on such a row.
fn specials(a: &ActiveSeq, ctx: &LogitsContext) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::with_capacity(4);
    for id in [
        ctx.think_end_token,
        a.think_start_token,
        ctx.tool_call_start_token,
        ctx.glm_tool_boundary,
    ]
    .into_iter()
    .flatten()
    {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    ids
}

#[cfg(test)]
#[path = "split_rows_tests.rs"]
mod tests;
