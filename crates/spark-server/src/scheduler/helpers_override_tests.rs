// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for scheduler::helpers' override/env seams (request-level
//! repetition overrides, forced-token fast-path, EOS-block, Lightning
//! proposal-failure gating). Split out of `helpers_tests.rs` to keep both
//! ≤500 LoC (CI file-size-cap). Logical child of `helpers` via `#[path]`.

use super::*;

#[test]
fn override_loosens_content_loop_threshold() {
    // Three contiguous copies of a 22-token sentence — passes the
    // boot-default `CONTENT_LOOP_MIN_REPEATS=3` so the default
    // detector fires. With a stricter `min_count=4` override, the
    // detector must NOT fire on the same input. This proves the
    // override actually wins over the boot default.
    let sentence: Vec<u32> = (1000..1022).collect();
    let mut tokens: Vec<u32> = (0..100).collect(); // prior content
    tokens.extend(sentence.iter()); // r1
    tokens.extend(sentence.iter()); // r2
    tokens.extend(sentence.iter()); // r3

    // Default path: 3 repeats at period 22, MIN_REPEATS=3 ⇒ fires.
    assert!(
        detect_content_token_loop_with(&tokens, None),
        "default thresholds must still fire on 22-token × 3 repeat"
    );

    // Override path: min_count=4 ⇒ 3 repeats are insufficient.
    let strict = crate::api::inference_types::RepetitionDetectionParams {
        min_pattern_size: 2,
        max_pattern_size: 64,
        min_count: 4,
    };
    assert!(
        !detect_content_token_loop_with(&tokens, Some(strict)),
        "stricter min_count=4 override must suppress 3-repeat firing"
    );
}

#[test]
fn override_tightens_content_loop_threshold() {
    // Five contiguous copies of a 5-token block. Below the boot-default
    // CONTENT_LOOP_MIN_TOKENS the detector won't even consider firing,
    // so pad with prior content first. With period_min=5 .. period_max=5
    // + min_count=3 the override fires on (5 × 5 = 25) end-anchored
    // tokens — covered by the 5-repeat tail.
    let pat: Vec<u32> = vec![42, 43, 44, 45, 46];
    let mut tokens: Vec<u32> = (0u32..50).collect();
    for _ in 0..5 {
        tokens.extend(pat.iter());
    }
    let permissive = crate::api::inference_types::RepetitionDetectionParams {
        min_pattern_size: 5,
        max_pattern_size: 5,
        min_count: 3,
    };
    assert!(
        detect_content_token_loop_with(&tokens, Some(permissive)),
        "override (period=5, min_count=3) must catch 5×period-5 tail"
    );
}

#[test]
fn override_applies_to_thinking_loop() {
    // 4× period-10 fence loop — fires under boot default
    // THINK_LOOP_MIN_REPEATS=3.
    let pat: Vec<u32> = vec![7, 6, 5, 4, 3, 2, 1, 0, 9, 8];
    let mut tokens: Vec<u32> = (100u32..150).collect();
    for _ in 0..4 {
        tokens.extend(pat.iter());
    }
    assert!(
        detect_thinking_token_loop_with(&tokens, None, WatchdogParams::default()),
        "default thinking-loop thresholds must still fire on 4× period-10"
    );
    // Override demanding 6 repeats ⇒ 4 is insufficient ⇒ must not fire.
    let strict = crate::api::inference_types::RepetitionDetectionParams {
        min_pattern_size: 4,
        max_pattern_size: 20,
        min_count: 6,
    };
    assert!(
        !detect_thinking_token_loop_with(&tokens, Some(strict), WatchdogParams::default()),
        "stricter min_count=6 override must suppress 4-repeat firing"
    );
}

// ── Forced-token fast-path kill-switch parsing ──────────────────────────────

#[test]
fn forced_token_fastpath_default_enabled() {
    // Env unset → fast-path on (the default; output is bit-identical to
    // the sampled path so there is no reason to ship it off).
    assert!(parse_forced_token_fastpath(None));
}

#[test]
fn forced_token_fastpath_disabled_by_truthy() {
    // Explicit truthy values disable the fast-path (the kill-switch).
    assert!(!parse_forced_token_fastpath(Some("1")));
    assert!(!parse_forced_token_fastpath(Some("true")));
    assert!(!parse_forced_token_fastpath(Some("TRUE")));
    assert!(!parse_forced_token_fastpath(Some("  true  ")));
}

#[test]
fn forced_token_fastpath_enabled_by_falsy_or_junk() {
    // Anything that is not an explicit truthy value keeps it enabled —
    // `0`, `false`, empty, and junk all mean "do not disable".
    assert!(parse_forced_token_fastpath(Some("0")));
    assert!(parse_forced_token_fastpath(Some("false")));
    assert!(parse_forced_token_fastpath(Some("")));
    assert!(parse_forced_token_fastpath(Some("yes")));
}

/// Lock the batch4 mid-think EOS block in `decode_logits_step.rs` inert by
/// default: `honor_eos_inside_thinking` must stay FALSE (pre-p350
/// behaviour — a mid-`<think>` EOS is discarded, never honored as an
/// implicit close) and the think-loop watchdog must stay ENABLED unless a
/// MODEL.toml `[behavior]` table opts out. Qwen3.6-35B-A3B measurably
/// regresses when the close is honored (8/10 vs 10/10 agentic, #464), so a
/// default flip here is a production behavior change, not a tweak.
#[test]
fn watchdog_defaults_keep_mid_think_eos_block_inert() {
    let d = WatchdogParams::default();
    assert!(
        !d.honor_eos_inside_thinking,
        "honor_eos_inside_thinking must default OFF (opt-in per model)"
    );
    assert!(
        d.enable_think_loop_watchdog,
        "think-loop watchdog must default ON (opt-out per model)"
    );
}

#[test]
fn dspark_proposal_failure_fails_closed_only_for_lightning_product() {
    // The production decision shared by mtp_step bootstrap and
    // verify_dflash_step re-propose: generic DFlash/MTP keeps the legacy
    // log-and-fall-through; the Lightning product fails closed.
    assert!(crate::scheduler::helpers::dspark_proposal_failure_fails_closed(true));
    assert!(!crate::scheduler::helpers::dspark_proposal_failure_fails_closed(false));
}
