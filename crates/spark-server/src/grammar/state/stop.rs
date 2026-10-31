// SPDX-License-Identifier: AGPL-3.0-only
//! EOS suppression for grammar-constrained output (extracted; identical content).

use super::GrammarState;

/// #192: whether the active grammar FORBIDS ending the turn at the current
/// matcher position — the single EOS-suppression predicate for both decode
/// paths (`decode_logits_step` and the MTP/emit path in `emit_step`).
///
/// Replaces the blanket `!is_terminated()` gate: a trigger-based
/// (tool_choice="auto") structural-tag matcher NEVER terminates — its
/// dispatch root loops forever and [`GrammarState::accept_token`] exempts
/// stop/EOS tokens from the matcher — so `!is_terminated()` suppressed EOS
/// for the WHOLE turn whenever no tool call completed (armed-but-unused
/// tools ran to `finish_reason="length"`; live probe #6, 2026-07-02). The
/// correct question is positional: may the response legally end HERE?
///
/// * `None` grammar (plain chat / disengaged)      → `false` (EOS free).
/// * terminated matcher                            → `false`.
/// * dispatch/preamble state, or between completed
///   calls (stop token grammar-legal)              → `false`.
/// * mid-structure (open tag body / JSON string)   → `true` (suppress; the
///   sampler's bitmask already excludes EOS here, so a sampled EOS is a
///   masked-path leak that must not end the turn mid-call).
///
/// Cost note: [`GrammarState::stop_legal`] fills a fresh bitmask — callers
/// must invoke this ONLY when the sampled token is an EOS token (rare),
/// never as a per-token predicate.
pub fn grammar_blocks_stop(gs: Option<&mut GrammarState>, eos_tokens: &[u32]) -> bool {
    match gs {
        None => false,
        Some(gs) => !gs.is_terminated() && !gs.stop_legal(eos_tokens),
    }
}
