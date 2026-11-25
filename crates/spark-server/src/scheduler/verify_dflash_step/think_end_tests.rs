// SPDX-License-Identifier: AGPL-3.0-only

//! The DFlash verify `</think>` rule, position by position.

use super::accept_with_forced_think_end as accept;
use crate::scheduler::confidence::MAX_SENTENCE_DEFER_TOKENS;

const END: u32 = 99;
/// Token 7 ends a sentence; every other id does not.
const MASK: [bool; 100] = {
    let mut m = [false; 100];
    m[7] = true;
    m
};

fn thinking_seq(armed: bool) -> crate::scheduler::ActiveSeq {
    let (mut a, _rx) = crate::scheduler::test_support::test_seq(vec![1, 2, 3], 10, None, 32);
    a.inside_thinking = true;
    a.force_end_thinking = armed;
    a.thinking_budget = Some(128);
    a.thinking_tokens = 130;
    a.output_tokens = vec![5];
    a
}

#[test]
fn unarmed_rows_keep_the_plain_accept_prefix() {
    let mut a = thinking_seq(false);
    let mut v = [1, 2, 7, 4, 5];
    assert_eq!(
        accept(&mut a, Some(END), Some(&MASK), &[1, 2, 7, 9], &mut v),
        3
    );
    assert_eq!(v, [1, 2, 7, 4, 5]);
}

#[test]
fn armed_row_closes_thinking_after_the_first_sentence_boundary() {
    let mut a = thinking_seq(true);
    let mut v = [1, 7, 3, 4, 5];
    // Drafts 1, 7, 3, 4 all match; the boundary is token 7 at position 1, so
    // position 2 becomes `</think>` and the step ends there.
    assert_eq!(
        accept(&mut a, Some(END), Some(&MASK), &[1, 7, 3, 4], &mut v),
        2
    );
    assert_eq!(v[2], END);
}

#[test]
fn a_boundary_just_before_the_step_injects_at_position_zero() {
    let mut a = thinking_seq(true);
    a.output_tokens = vec![7];
    let mut v = [1, 2, 3];
    assert_eq!(accept(&mut a, Some(END), Some(&MASK), &[1, 2], &mut v), 0);
    assert_eq!(v[0], END);
}

#[test]
fn deferral_ceiling_forces_the_close_without_a_boundary() {
    let mut a = thinking_seq(true);
    a.sentence_defer_count = MAX_SENTENCE_DEFER_TOKENS - 2;
    let mut v = [1, 2, 3, 4, 5];
    assert_eq!(
        accept(&mut a, Some(END), Some(&MASK), &[1, 2, 3, 4], &mut v),
        2
    );
    assert_eq!(v[2], END);
}

#[test]
fn deferring_positions_tick_the_counter() {
    let mut a = thinking_seq(true);
    let mut v = [1, 2, 3, 4];
    assert_eq!(
        accept(&mut a, Some(END), Some(&MASK), &[1, 2, 9], &mut v),
        2
    );
    assert_eq!(v, [1, 2, 3, 4]);
    assert_eq!(a.sentence_defer_count, 3);
}

#[test]
fn a_model_that_closes_thinking_itself_is_left_alone() {
    let mut a = thinking_seq(true);
    let mut v = [1, END, 7, 4];
    assert_eq!(
        accept(&mut a, Some(END), Some(&MASK), &[1, END, 7], &mut v),
        3
    );
    assert_eq!(v, [1, END, 7, 4]);
}

#[test]
fn a_code_fence_defers_the_boundary_but_not_the_ceiling() {
    let mut a = thinking_seq(true);
    a.in_code_fence = true;
    let mut v = [7, 2, 3];
    assert_eq!(accept(&mut a, Some(END), Some(&MASK), &[7, 2], &mut v), 2);
    a.thinking_tokens = 128 * 3;
    let mut v = [7, 2, 3];
    assert_eq!(accept(&mut a, Some(END), Some(&MASK), &[7, 2], &mut v), 0);
    assert_eq!(v[0], END);
}

#[test]
fn an_inkling_budget_close_is_end_message_and_the_step_ends_there() {
    // Inkling 2's think end is `<|end_message|>`: the forced accept writes it
    // where the budget's close lands, and the model opens its next block from
    // there. An `<|end_message|>` the model wrote itself is left alone.
    use crate::scheduler::test_support::inkling::{END, MESSAGE, THINKING};
    let mut a = thinking_seq(true);
    a.output_tokens = vec![MESSAGE, THINKING, 4];
    let mut v = [1, 7, 3, 4, 5];
    assert_eq!(
        accept(&mut a, Some(END), Some(&BOUNDARY), &[1, 7, 3, 4], &mut v),
        2
    );
    assert_eq!(v[2], END);
    let mut v = [1, END, MESSAGE, 4];
    assert_eq!(
        accept(
            &mut a,
            Some(END),
            Some(&BOUNDARY),
            &[1, END, MESSAGE],
            &mut v
        ),
        3
    );
    assert_eq!(v, [1, END, MESSAGE, 4]);
}

/// `MASK` widened to the test vocabulary the Inkling marker ids sit in:
/// token 7 ends a sentence.
const BOUNDARY: [bool; 2048] = {
    let mut m = [false; 2048];
    m[7] = true;
    m
};
