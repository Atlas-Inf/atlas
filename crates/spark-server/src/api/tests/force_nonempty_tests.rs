// SPDX-License-Identifier: AGPL-3.0-only

//! force_nonempty_content (vLLM deepseek_r1) — bin-side tests because the
//! free `extract_thinking` swap lives in `crate::api::inference_impl`.

use crate::reasoning_parser::ReasoningFormat;

#[test]
fn force_nonempty_promotes_reasoning_to_content() {
    let p = ReasoningFormat::DeepSeekR1.into_parser();
    // Unclosed think block: all output is reasoning, content is empty.
    let (reasoning, content) = p.extract_thinking("all reasoning, no answer", true);
    assert!(reasoning.is_some());
    assert!(content.is_empty());
    // The wrapper applies the vLLM swap: reasoning text becomes content.
    let (r2, c2) = crate::api::inference_impl::extract_thinking(
        "all reasoning, no answer",
        true,
        true,
        Some(&*p),
    );
    assert_eq!(c2, "all reasoning, no answer");
    assert!(r2.is_some());
}

#[test]
fn force_nonempty_off_keeps_empty_content() {
    let p = ReasoningFormat::DeepSeekR1.into_parser();
    let (_r, c) = crate::api::inference_impl::extract_thinking(
        "all reasoning, no answer",
        true,
        false,
        Some(&*p),
    );
    assert!(c.is_empty());
}
