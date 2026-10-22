// SPDX-License-Identifier: AGPL-3.0-only

use super::GrammarState;
use crate::literal_xml::{LiteralXmlState, mask_literal_lt_bytes_with_state};

impl GrammarState {
    /// Enable quote/code-aware matching for an automatic tool-call grammar.
    /// The scheduler opts this in only for `ToolCall(use_triggers=true)`;
    /// required and response-format grammars remain unchanged.
    #[must_use]
    pub fn with_auto_literal_dispatch(mut self, enabled: bool) -> Self {
        self.auto_literal_dispatch = enabled;
        if enabled {
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

    /// Whether the native opener follows a quoted or fenced XML mention in
    /// the already emitted token history. This runs only at a candidate
    /// opener, so decoding the accumulated prefix is off the hot path.
    pub fn native_opener_is_literal(&self, output_tokens: &[u32]) -> bool {
        let mut state = crate::literal_xml::LiteralXmlState::default();
        for &token_id in output_tokens {
            let Some(bytes) = self.tokenizer_info.decoded_vocab().get(token_id as usize) else {
                return false;
            };
            state.feed_bytes(bytes);
        }
        state.finish_pending();
        state.is_literal()
    }

    /// Variant used by speculative verification, where the already emitted
    /// history and the drafts accepted so far live in separate slices. Avoid
    /// allocating a concatenated history on every verify row.
    pub fn native_opener_is_literal_after(
        &self,
        output_tokens: &[u32],
        suffix_tokens: &[u32],
    ) -> bool {
        let mut state = crate::literal_xml::LiteralXmlState::default();
        for &token_id in output_tokens.iter().chain(suffix_tokens) {
            let Some(bytes) = self.tokenizer_info.decoded_vocab().get(token_id as usize) else {
                return false;
            };
            state.feed_bytes(bytes);
        }
        state.finish_pending();
        state.is_literal()
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

    /// Accept a sampled token and advance the grammar state.
    ///
    /// Returns `true` if the token was accepted by the grammar.
    /// Returns `false` if the token violates the grammar (should not happen
    /// if the bitmask was applied correctly).
    ///
    /// Short-circuits with `true` once the matcher has reached its
    /// terminated (accepting) state — feeding tokens past the stop into
    /// xgrammar emits a `grammar_matcher.cc:493` warning ("matcher has
    /// terminated, but is trying to accept new token") for every trailing
    /// token in spec-decode draft runs (Discord 2026-05-08 universe06608).
    /// Returning `true` keeps the spec-decode boundary heuristic in
    /// `truncate_drafts_at_grammar_boundary` consistent — drafts past a
    /// completed grammar are not "rejected by grammar"; they are simply
    /// past the stop, which the EOS handler will terminate independently.
    pub fn accept_token(&mut self, token_id: u32) -> bool {
        // Any advance invalidates the per-position bitmask cache. Done
        // unconditionally (even on the early-return paths that don't move the
        // matcher) — over-invalidation only costs a refill, whereas a missed
        // invalidation would serve a stale mask and desync the NPDA.
        self.bitmask_valid = false;
        if self.matcher.is_terminated() {
            return true;
        }
        // Stop/EOS tokens (e.g. `<|im_end|>`, 248046) are control tokens that
        // terminate the turn — they are never part of the grammar's content
        // language. The model legitimately ends a turn with one (after a
        // text-only reply, or after a complete tool call) while the
        // structural-tag NPDA is still in a non-accepting state; feeding it to
        // `matcher.accept_token` then returns false ("refusal"), which callers
        // mis-read as a grammar desync and abort the response mid-stream
        // (`emit_step`: "Ending response to prevent cascading grammar-mask
        // corruption"). That truncates agentic turns and is the dominant cause
        // of the opencode webserver_ok gap vs vLLM (which parses tools
        // post-hoc and never constrains the stop token). Accept it
        // unconditionally and let the EOS handler terminate. This cannot
        // corrupt tool-call STRUCTURE — only non-stop tokens drive the matcher.
        if self.stop_tokens.contains(&token_id) {
            return true;
        }
        if self.auto_literal_dispatch
            && !self
                .tokenizer_info
                .special_token_ids()
                .contains(&(token_id as i32))
            && let Some(token) = self.tokenizer_info.decoded_vocab().get(token_id as usize)
        {
            let in_dispatch = self.matcher.is_grammar_completed();
            let matcher_view = if in_dispatch {
                mask_literal_lt_bytes_with_state(token, self.literal_xml_state)
            } else {
                token.clone()
            };
            let accepted = self.matcher.accept_bytes(&matcher_view, false);
            if accepted {
                if in_dispatch {
                    let mut actual_state = self.literal_xml_state;
                    actual_state.feed_bytes(token);
                    if self.matcher.is_grammar_completed() {
                        // Ordinary prose remains in dispatch, so preserve the
                        // actual delimiter state for the next token.
                        self.literal_xml_state = actual_state;
                    } else {
                        // The matcher left dispatch and entered a tool body.
                        self.literal_xml_state = LiteralXmlState::default();
                    }
                } else {
                    // The active body is governed solely by its grammar. A
                    // closing tag returns to dispatch with a clean classifier.
                    self.literal_xml_state = LiteralXmlState::default();
                }
                self.literal_xml_history.push(self.literal_xml_state);
            }
            return accepted;
        }
        self.matcher.accept_token(token_id as i32)
    }
}
