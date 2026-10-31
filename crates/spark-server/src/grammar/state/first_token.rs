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
