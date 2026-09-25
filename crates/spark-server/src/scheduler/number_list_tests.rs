// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the number-list watchdog exemption (2026-09-20
//! Flash-Next MinHeap incident). Split out of `helpers_tests.rs` to keep
//! it ≤500 LoC (CI file-size-cap). Logical child of `helpers` via
//! `#[path]`; `use super::*` resolves to helpers.rs items exactly as
//! before the split.

use super::thinking_loop_tests::{numeric_mask, varying_template_stream};
use super::*;

// ── Number-list exemption (2026-09-20 Flash-Next MinHeap incident) ────
// `values = [8, 3, 1, 6, 4, 2, 7, …]` normalizes to `N , N , N , …` —
// period-2 — and cut the response mid-list on both decode paths.
// Punctuation ids 200..=209 sit beside the numeric block 100..=199.

fn punct_mask() -> Vec<bool> {
    let mut m = vec![false; 1100];
    for (i, slot) in m.iter_mut().enumerate() {
        *slot = (200..=209).contains(&i);
    }
    m
}

/// `N ,` repeated — the normalized shape of `8, 3, 1, 6, 4, 2, 7,`.
fn number_list_stream(numbers: u32) -> Vec<u32> {
    let mut t: Vec<u32> = (900u32..990).collect(); // structural-noise prefix
    for k in 0..numbers {
        t.push(100 + k); // DISTINCT numeric token per element
        t.push(200); // `,`
    }
    t
}

#[test]
fn norm_exempts_distinct_number_list_when_punct_mask_present() {
    let t = number_list_stream(6);
    let mask = numeric_mask();
    let punct = punct_mask();
    assert!(
        !detect_content_token_loop_normalized_with(&t, &mask, Some(&punct), None),
        "6 distinct numbers separated by commas is data, not a loop"
    );
    assert!(
        detect_content_token_loop_normalized_with(&t, &mask, None, None),
        "punct == None keeps the legacy fail-open behaviour (fires)"
    );
}

#[test]
fn norm_still_fires_when_period_contains_a_word_token() {
    // The original Qwen3.6 degeneration shape `- B(46) = N\n- B(47) = M\n…`:
    // the period carries structural word tokens (ids 1..=11, not
    // punctuation), so the exemption must NOT apply.
    let t = varying_template_stream(5);
    let mask = numeric_mask();
    let punct = punct_mask();
    assert!(
        detect_content_token_loop_normalized_with(&t, &mask, Some(&punct), None),
        "word-bearing period is a real template loop, not a number list"
    );
}

#[test]
fn identical_numbers_still_caught_by_the_exact_detector() {
    // `1, 1, 1, 1, …` repeats literally — the exact path fires before the
    // normalized one is even consulted at the call sites.
    let mut t: Vec<u32> = (900u32..990).collect();
    for _ in 0..6 {
        t.extend([100, 200]); // identical `1 ,` pair every repeat
    }
    assert!(
        detect_content_token_loop_with(&t, None),
        "identical-number runaway is a byte-identical period-2 loop"
    );
}

#[test]
fn detect_token_loop_period_reports_the_matched_period() {
    // Normalized `N , N , N , N ,` (sentinel at 5): period-2, 4 repeats.
    let v: Vec<u32> = vec![
        NUMERIC_SENTINEL,
        200,
        NUMERIC_SENTINEL,
        200,
        NUMERIC_SENTINEL,
        200,
        NUMERIC_SENTINEL,
        200,
    ];
    assert_eq!(
        detect_token_loop_period(&v, CONTENT_LOOP_PERIOD_MIN, CONTENT_LOOP_PERIOD_MAX, 4, 0),
        Some(2)
    );
    // A stream with no anchored repeat reports no period.
    let flat: Vec<u32> = (900u32..990).collect();
    assert_eq!(
        detect_token_loop_period(
            &flat,
            CONTENT_LOOP_PERIOD_MIN,
            CONTENT_LOOP_PERIOD_MAX,
            4,
            0
        ),
        None
    );
}
