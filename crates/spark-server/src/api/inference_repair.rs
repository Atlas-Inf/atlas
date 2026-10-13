// SPDX-License-Identifier: AGPL-3.0-only

//! Request helpers used by model-specific admission fences.

use super::inference_types::InferenceRequest;

impl InferenceRequest {
    /// Whether this request carries a constrained-decoding grammar.
    pub fn has_grammar_spec(&self) -> bool {
        match self {
            InferenceRequest::Blocking { grammar_spec, .. }
            | InferenceRequest::Streaming { grammar_spec, .. } => grammar_spec.is_some(),
        }
    }

    /// Force native serial decode for a request that cannot enter a model's
    /// speculative collective.
    pub(crate) fn disable_mtp_for_fallback(&mut self) {
        match self {
            InferenceRequest::Blocking { disable_mtp, .. }
            | InferenceRequest::Streaming { disable_mtp, .. } => *disable_mtp = true,
        }
    }
}
