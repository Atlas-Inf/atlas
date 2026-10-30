// SPDX-License-Identifier: AGPL-3.0-only

use super::reconcile_prompt_thinking;

const START: u32 = 1;
const END: u32 = 2;

#[test]
fn an_open_think_in_the_prompt_tail_turns_reasoning_on() {
    // A template that always opens `<think>` (GLM-5.3) reasons whatever the
    // request resolved to; the server budget then applies, and a model with
    // none (`max_thinking_budget = 0`) gets none.
    let open = [9, 3, START];
    assert_eq!(
        reconcile_prompt_thinking(&open, Some(START), Some(END), false, None, 128),
        (true, Some(128))
    );
    assert_eq!(
        reconcile_prompt_thinking(&open, Some(START), Some(END), false, None, 0),
        (true, None)
    );
    // A request that already enabled thinking keeps its own budget.
    assert_eq!(
        reconcile_prompt_thinking(&open, Some(START), Some(END), true, Some(32), 0),
        (true, Some(32))
    );
}

#[test]
fn a_closed_empty_think_block_turns_reasoning_off() {
    for enabled in [false, true] {
        assert_eq!(
            reconcile_prompt_thinking(
                &[9, START, END],
                Some(START),
                Some(END),
                enabled,
                Some(32),
                128
            ),
            (false, None)
        );
    }
}

#[test]
fn unrelated_or_missing_prompt_markers_preserve_policy() {
    for enabled in [false, true] {
        for tokens in [&[9, 10][..], &[START, END, 3][..], &[END][..]] {
            assert_eq!(
                reconcile_prompt_thinking(tokens, Some(START), Some(END), enabled, Some(32), 128),
                (enabled, Some(32))
            );
        }
        assert_eq!(
            reconcile_prompt_thinking(&[START, END], None, Some(END), enabled, Some(32), 128),
            (enabled, Some(32))
        );
    }
    assert_eq!(
        reconcile_prompt_thinking(&[9, START], Some(START), None, false, None, 128),
        (true, Some(128))
    );
}
