// SPDX-License-Identifier: AGPL-3.0-only
//! First-prefill grammar and tool-opener policy while reasoning is open.
use super::GrammarState;

impl GrammarState {
    /// Mark this as a `response_format` grammar (see the `strict` field).
    /// `opens_in_thinking`: the prompt leaves the model inside `<think>`.
    #[must_use]
    pub fn strict_output(mut self, opens_in_thinking: bool) -> Self {
        self.strict = true;
        self.opens_in_thinking = opens_in_thinking;
        self
    }

    /// A required tool grammar for a model with an explicit think-end starts
    /// after reasoning, just like its decode-step mask. This does not change
    /// the existing tool grammar's refusal policy.
    #[must_use]
    pub fn defer_first_token(mut self, tool_opener: Option<u32>) -> Self {
        self.opens_in_thinking = true;
        self.first_token_suppress = tool_opener;
        self
    }

    /// A required tool call on a model with a native tool opener starts with
    /// that token (see [`Self::pin_required_opener`]).
    #[must_use]
    pub fn require_native_opener(mut self, opener: Option<u32>) -> Self {
        self.required_opener = opener;
        self
    }

    /// At the grammar's start, leave only the required native opener legal.
    /// The matcher also takes the tag spelled out as text, but text is not a
    /// tool call: sampled inside `<think>`, a literal `<` is reasoning, the
    /// reasoning runs unmasked, and the grammar resumes after it mid-tag.
    pub(super) fn pin_required_opener(&mut self) {
        let Some(opener) = self.required_opener else {
            return;
        };
        if self.matcher.num_history_steps() != 0 || !self.is_token_allowed(opener) {
            return;
        }
        self.bitmask_data.fill(0);
        self.bitmask_data[(opener / 32) as usize] = 1i32 << (opener % 32);
    }

    /// Whether the first token sampled after prefill is grammar output.
    /// An explicit reasoning preamble starts before the grammar, as on every
    /// decode step. Native GLM tool grammars retain their opener mask; a strict
    /// union consumes a native opener through `opens_output` instead.
    pub fn masks_first_token(&self) -> bool {
        !self.opens_in_thinking
    }

    /// Match the decode-time tool-opener mask during required-tool reasoning.
    pub fn first_token_suppress(&self) -> Option<u32> {
        self.first_token_suppress
    }
}
