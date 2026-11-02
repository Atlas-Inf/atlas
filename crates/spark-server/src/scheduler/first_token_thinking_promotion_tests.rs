// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn first_token_thinking_promotion_preserves_accounting_and_handles_close() {
    for (enabled, first, max_tokens, expected) in [
        (true, 20, 256, (false, true, true)),
        (true, 20, 1, (false, true, true)),
        (true, 30, 256, (true, false, false)),
        (false, 10, 256, (true, false, false)),
        (false, 30, 256, (false, true, false)),
        (false, 20, 256, (false, true, false)),
    ] {
        let (a, _rx) = super::super::test_support::test_seq(vec![], 256, None, 28);
        let now = Instant::now();
        let p = super::super::prefill_a_step_params::build_prefill_in_progress(
            std::sync::Arc::new(vec![7; 28]),
            0,
            a.seq,
            28,
            max_tokens,
            0,
            super::super::test_support::EOS.to_vec(),
            a.sink,
            None,
            now,
            0.0,
            0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            vec![],
            enabled,
            Some(128),
            None,
            64,
            false,
            false,
            false,
            false,
            None,
            None,
            None,
            None,
        );
        let spontaneous = !enabled && first == 10;
        let immediate = !spontaneous && max_tokens <= 1;
        let a = build_active_seq_from_prefill(
            p,
            first,
            spontaneous,
            false,
            0,
            immediate,
            now,
            Some(20),
            Some(10),
            None,
            None,
            0,
            None,
        );
        assert_eq!(
            (a.inside_thinking, a.think_ended, a.think_just_ended),
            expected,
            "enabled={enabled}, first={first}, max_tokens={max_tokens}"
        );
        assert_eq!(
            a.thinking_tokens, 0,
            "preserve existing first-token accounting"
        );
        assert_eq!(a.remaining, max_tokens - 1);
        assert_eq!(
            a.output_tokens,
            if spontaneous { vec![] } else { vec![first] }
        );
        assert_eq!(a.thinking_budget, Some(if spontaneous { 64 } else { 128 }));
        assert_eq!(a.finished, immediate);
        assert!(!a.force_end_thinking && !a.think_force_closed);
    }
}
