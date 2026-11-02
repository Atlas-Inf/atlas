// SPDX-License-Identifier: AGPL-3.0-only

//! Request admission for the ngram and self-speculative lanes.
use crate::scheduler::ActiveSeq;

pub(in crate::scheduler) fn self_spec(a: &ActiveSeq, slots_covered: bool) -> bool {
    // These lanes either omit scores or return raw rather than processed
    // logprobs. Requested scores continue through supported MTP or decode.
    slots_covered && a.grammar_state.is_none() && a.top_logprobs.is_none()
}

pub(in crate::scheduler) fn ngram(
    a: &ActiveSeq,
    slots_covered: bool,
    context_limit: Option<usize>,
) -> bool {
    self_spec(a, slots_covered)
        && a.temperature == 0.0
        && context_limit.is_none_or(|lim| a.seq.seq_len + 4 <= lim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::test_support::test_seq;

    #[test]
    fn unscored_logprob_admission_declines_ngram_and_self_spec() {
        let (mut a, _rx) = test_seq(vec![], 1, None, 40);
        a.temperature = 0.0;
        assert!(self_spec(&a, true));
        assert!(ngram(&a, true, Some(44)));
        assert!(!self_spec(&a, false));
        assert!(!ngram(&a, true, Some(43)));
        for requested in [Some(0), Some(5)] {
            a.top_logprobs = requested;
            assert!(
                !self_spec(&a, true),
                "logprob requests continue to supported MTP or serial decode"
            );
            assert!(!ngram(&a, true, Some(44)));
        }
        a.top_logprobs = None;
        assert!(self_spec(&a, true));
        assert!(ngram(&a, true, Some(44)));
        a.temperature = 0.7;
        assert!(!ngram(&a, true, Some(44)));
        assert!(self_spec(&a, true), "preserve previous self-spec admission");
    }
}
