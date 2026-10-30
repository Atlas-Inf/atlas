// SPDX-License-Identifier: AGPL-3.0-only
//! Quote- and code-aware dispatch for an automatic tool-call grammar.
//!
//! A model that spells its tool envelope with tokens that also occur in prose
//! (GLM-5.3, Qwen4_exp) writes `<tool_call>` inside a fence or a quote as an
//! example. The structural-tag matcher would take it as a trigger and enter a
//! call body, then refuse the next prose token and disengage. In the free
//! dispatch state a `<` inside a literal span reaches the matcher as a space;
//! a real tool body stays governed by its grammar alone.
//!
//! Ported from Enntity/atlas 19bfc90b, 6a1ccde4, baa93620 and 318569b8
//! (AGPL-3.0-only).

use super::GrammarState;
use crate::literal_xml::{LiteralXmlState, mask_literal_lt_bytes_with_state};

impl GrammarState {
    /// Enable quote/code-aware matching for an automatic tool-call grammar.
    /// The scheduler opts this in only for `ToolCall(use_triggers=true)` on a
    /// literal-XML model; required and response-format grammars are unchanged.
    #[must_use]
    pub fn with_auto_literal_dispatch(mut self, enabled: bool) -> Self {
        self.auto_literal_dispatch = enabled;
        if enabled {
            // Preserve the compiled matcher's control stops even when the
            // caller does not add model-specific EOS IDs with_stop_tokens.
            let mut stops = self.stop_tokens.to_vec();
            for token in self.matcher.stop_token_ids() {
                if token >= 0 && !stops.contains(&(token as u32)) {
                    stops.push(token as u32);
                }
            }
            self.stop_tokens = stops.into_boxed_slice();
            self.literal_lt_tokens = self
                .tokenizer_info
                .decoded_vocab()
                .iter()
                .enumerate()
                .filter_map(|(token_id, token)| token.contains(&b'<').then_some(token_id as u32))
                .collect::<Vec<_>>()
                .into_boxed_slice();
        } else {
            self.literal_lt_tokens = Box::new([]);
        }
        self
    }

    /// Re-admit tokens whose matcher view escapes a `<` inside a quote/code
    /// span. This is limited to the free dispatch state; markup and quotes in
    /// an actual tool body remain governed by the tool grammar.
    pub(super) fn restore_literal_candidates(&mut self) {
        for index in 0..self.literal_lt_tokens.len() {
            let token_id = self.literal_lt_tokens[index];
            let word = token_id as usize / 32;
            if self
                .bitmask_data
                .get(word)
                .is_some_and(|mask| (mask & (1i32 << (token_id % 32))) != 0)
            {
                continue;
            }
            let Some(token) = self
                .tokenizer_info
                .decoded_vocab()
                .get(token_id as usize)
                .cloned()
            else {
                continue;
            };
            let matcher_view = mask_literal_lt_bytes_with_state(&token, self.literal_xml_state);
            if matcher_view == token {
                continue;
            }
            // The original token was masked because its literal `<` looked
            // structural. Probe the transformed bytes against the same
            // dispatch state before restoring it; a token may close the
            // literal span and then contain an invalid real opener. The
            // probe is one matcher history step and is rolled back
            // immediately, so bitmask filling is state-preserving.
            if !self.matcher.accept_bytes(&matcher_view, false) {
                continue;
            }
            self.matcher.rollback(1);
            if let Some(mask) = self.bitmask_data.get_mut(word) {
                *mask |= 1i32 << (token_id % 32);
            }
        }
    }

    /// The literal-aware accept of a non-stop token, or `None` when this
    /// grammar takes tokens on the exact matcher path.
    pub(super) fn accept_literal_dispatch(&mut self, token_id: u32) -> Option<bool> {
        if !self.auto_literal_dispatch
            || self
                .tokenizer_info
                .special_token_ids()
                .contains(&(token_id as i32))
        {
            return None;
        }
        let token = self.tokenizer_info.decoded_vocab().get(token_id as usize)?;
        let in_dispatch = self.matcher.is_grammar_completed();
        let matcher_view = if in_dispatch {
            mask_literal_lt_bytes_with_state(token, self.literal_xml_state)
        } else {
            token.clone()
        };
        let mut actual_state = self.literal_xml_state;
        actual_state.feed_bytes(token);
        let accepted = self.matcher.accept_bytes(&matcher_view, false);
        if accepted {
            self.literal_xml_state = if in_dispatch && self.matcher.is_grammar_completed() {
                // Ordinary prose remains in dispatch, so preserve the
                // actual delimiter state for the next token.
                actual_state
            } else {
                // The matcher entered a tool body, or a body's close returned
                // it to dispatch: the classifier starts clean.
                LiteralXmlState::default()
            };
            self.literal_xml_history.push(self.literal_xml_state);
        }
        Some(accepted)
    }

    /// Rewind the lexical state with the matcher: `Some(n)` steps, or `None`
    /// for a reset to the start.
    pub(super) fn rollback_literal_state(&mut self, steps: Option<usize>) {
        if !self.auto_literal_dispatch {
            return;
        }
        let keep = steps.map_or(0, |n| self.literal_xml_history.len().saturating_sub(n));
        self.literal_xml_history.truncate(keep);
        self.literal_xml_state = self.literal_xml_history.last().copied().unwrap_or_default();
    }
}
