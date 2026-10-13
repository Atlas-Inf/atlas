// SPDX-License-Identifier: AGPL-3.0-only

//! Speculative decoding for strict structured output (`ATLAS_GLM_STRICT_SPEC=1`).
//!
//! A port of vLLM's design: vllm-project/vllm#14702 ("Enable Speculative
//! Decoding with Structured Outputs") and #44297 ("Constrain bitmask and trim
//! grammar advance at the reasoning boundary"), tracked by RFC #48197.
//!
//! Before a verify the head walks the grammar matcher through the drafted
//! tokens. Row r's mask is the matcher state after `drafts[..r]`; the walk
//! stops at the first draft row r's mask refuses (or that is no token at all,
//! e.g. a `-1` pad), so the verify carries only drafts the grammar could
//! accept, and the last row's mask constrains the bonus/correction token.
//! The walk is rolled back afterwards: only `emit_token` advances the matcher,
//! with the tokens the verify actually emits. Rows before a `</think>` are
//! unconstrained and never fed to the matcher; the row after it starts from
//! the matcher's untouched start state (#44297).
//!
//! DFlash proposes a whole block at once, so its drafts cannot be masked per
//! position; they are proposed unmasked (`mtp_grammar_mask_for`: a draft-0
//! mask would cost the propose its CUDA graph and its rank split) and the
//! walk trims them.
//!
//! The GLM TP2 verify head picks with a raw argmax over a vocabulary split
//! across the ranks, which vLLM does not have: the masks travel to both ranks
//! (`spark_model::model::glm_verify_masks`) and each applies its slice before
//! its partial argmax, so every emitted token is the best one the grammar
//! allows at its own position. Rows of other sequences are untouched.

use anyhow::{Result, ensure};

use crate::grammar::GrammarState;

use super::ActiveSeq;
use super::levers::SchedLevers;

/// Strict sequences speculate only on the DFlash raw-argmax lane served by
/// the GLM vocab-split head, the one verify that applies the row masks.
pub(super) fn enabled(
    levers: &SchedLevers,
    model: &dyn spark_model::traits::Model,
    raw: bool,
) -> bool {
    levers.glm_strict_spec && raw && model.verify_logits_argmax_only()
}

/// A strict verify: the drafts it carries and one mask per verify row.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct StrictVerify {
    pub drafts: Vec<u32>,
    /// The only draft is a dead one row 0 refuses: the step emits its
    /// masked bonus alone and says nothing about the drafter.
    pub dead: bool,
    /// `(drafts.len() + 1) * ceil(vocab / 32)` words, row-major.
    pub masks: Vec<u32>,
}

/// Build the verify of `drafts` for a strict sequence, or `None` when `a`
/// has no strict grammar.
pub(super) fn prepare(
    a: &mut ActiveSeq,
    drafts: &[u32],
    vocab: usize,
) -> Result<Option<StrictVerify>> {
    if !a.strict_grammar() {
        return Ok(None);
    }
    let thinking = a.inside_thinking;
    let (think_start, think_end) = (a.think_start_token, a.think_end_token);
    let Some(gs) = a.grammar_state.as_mut() else {
        return Ok(None);
    };
    row_masks(gs, thinking, (think_start, think_end), drafts, vocab).map(Some)
}

/// The drafts (and their confidences) the verify carries.
pub(super) fn verified_drafts<'a>(
    strict: &'a Option<StrictVerify>,
    drafts: &'a [u32],
    conf: &'a [f32],
) -> (&'a [u32], &'a [f32]) {
    match strict {
        Some(s) => (&s.drafts, &conf[..conf.len().min(s.drafts.len())]),
        None => (drafts, conf),
    }
}

/// The generic-verify width word: `rows`, flagged when row masks follow.
pub(super) fn width_word(rows: usize, strict: &Option<StrictVerify>) -> u32 {
    let flag = strict
        .as_ref()
        .map_or(0, |_| spark_model::model::MASKED_VERIFY);
    rows as u32 | flag
}

/// Validate and upload the row masks, before any verify command is sent.
pub(super) fn upload(
    model: &dyn spark_model::traits::Model,
    strict: &Option<StrictVerify>,
) -> Result<()> {
    match strict {
        Some(s) => model.prepare_verify_row_masks(s.drafts.len() + 1, &s.masks),
        None => Ok(()),
    }
}

/// Broadcast the uploaded masks to every rank, after the verify tokens.
pub(super) fn send(
    model: &dyn spark_model::traits::Model,
    strict: &Option<StrictVerify>,
) -> Result<()> {
    match strict {
        Some(s) => model.send_verify_row_masks(s.drafts.len() + 1),
        None => Ok(()),
    }
}

/// The walk (module docs). Every verify carries at least one draft (its width
/// is at least 2): when the first is unusable it is replaced by a token row 0
/// refuses, so the step emits its masked bonus alone, or by token 0 when row 0
/// allows everything, which is then an ordinary draft.
pub(super) fn row_masks(
    gs: &mut GrammarState,
    inside_thinking: bool,
    (think_start, think_end): (Option<u32>, Option<u32>),
    drafts: &[u32],
    vocab: usize,
) -> Result<StrictVerify> {
    let words = vocab.div_ceil(32);
    ensure!(
        gs.bitmask_data().len() == words,
        "grammar bitmask has {} words, the model vocabulary {words}",
        gs.bitmask_data().len()
    );
    let steps_before = gs.num_history_steps();
    let mut thinking = inside_thinking;
    let mut kept: Vec<u32> = Vec::with_capacity(drafts.len());
    let mut masks: Vec<u32> = Vec::with_capacity((drafts.len() + 1) * words);
    let mut dead = false;
    let walk = loop {
        let r = kept.len();
        let row_at = masks.len();
        push_row_mask(gs, thinking, [think_start, think_end], words, &mut masks);
        let Some(&draft) = drafts.get(r) else {
            break Ok(());
        };
        let row = &masks[row_at..];
        let mut d = draft;
        if !allowed(row, d, vocab) {
            if r > 0 {
                break Ok(());
            }
            match (0..vocab as u32).find(|&t| !allowed(row, t, vocab)) {
                Some(dead_token) => {
                    kept.push(dead_token);
                    masks.extend_from_within(row_at..);
                    dead = true;
                    break Ok(());
                }
                None => d = 0,
            }
        }
        if thinking {
            thinking = think_end != Some(d);
        } else if !gs.accept_token(d) {
            break Err(anyhow::anyhow!(
                "grammar refused draft {d} that its own mask allows"
            ));
        }
        kept.push(d);
    };
    let advanced = gs.num_history_steps().saturating_sub(steps_before);
    if advanced > 0 {
        gs.rollback(advanced);
    }
    walk?;
    Ok(StrictVerify {
        drafts: kept,
        dead,
        masks,
    })
}

/// Row mask for the current walk state: everything inside `<think>`, else the
/// matcher's mask (everything once it constrains nothing) without the think
/// tags, as the serial path's post-close think mask does.
fn push_row_mask(
    gs: &mut GrammarState,
    thinking: bool,
    think_tags: [Option<u32>; 2],
    words: usize,
    out: &mut Vec<u32>,
) {
    let at = out.len();
    if !thinking && !gs.is_terminated() && gs.fill_bitmask() {
        out.extend(gs.bitmask_data().iter().map(|&w| w as u32));
    } else {
        out.resize(at + words, u32::MAX);
    }
    if !thinking {
        for t in think_tags.into_iter().flatten() {
            if let Some(w) = out[at..].get_mut(t as usize / 32) {
                *w &= !(1u32 << (t % 32));
            }
        }
    }
}

fn allowed(row: &[u32], t: u32, vocab: usize) -> bool {
    (t as usize) < vocab && row[t as usize / 32] >> (t % 32) & 1 == 1
}

#[cfg(test)]
#[path = "strict_spec_tests.rs"]
mod tests;
