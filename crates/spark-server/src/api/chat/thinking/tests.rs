// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn policy() -> Policy {
    Policy {
        disable_thinking: false,
        model_default: false,
        thinking_in_tools: true,
        max_thinking_budget: 2048,
        effort_capped_at_ceiling: false,
        cap_at_max_tokens: true,
    }
}

#[test]
fn kill_switch_overrides_everything() {
    let (et, tb) = resolve(
        ThinkingDirective::On { budget: Some(512) },
        Policy {
            disable_thinking: true,
            ..policy()
        },
        4096,
        false,
    );
    assert!(!et);
    assert!(tb.is_none());
}

#[test]
fn unspecified_falls_to_model_default() {
    let (et, tb) = resolve(
        ThinkingDirective::Unspecified,
        Policy {
            model_default: true,
            ..policy()
        },
        4096,
        false,
    );
    assert!(et);
    // Defers to max_thinking_budget, capped at 90% of max_tokens.
    assert_eq!(tb, Some(2048));

    let (et, tb) = resolve(ThinkingDirective::Unspecified, policy(), 4096, false);
    assert!(!et);
    assert!(tb.is_none());
}

#[test]
fn explicit_budget_capped_at_90_pct_of_max_tokens() {
    let (et, tb) = resolve(
        ThinkingDirective::On { budget: Some(4096) },
        policy(),
        1000,
        false,
    );
    assert!(et);
    assert_eq!(tb, Some(900));
}

#[test]
fn budgetless_on_defers_to_model_cap() {
    let (et, tb) = resolve(
        ThinkingDirective::On { budget: None },
        policy(),
        4096,
        false,
    );
    assert!(et);
    assert_eq!(tb, Some(2048));
}

#[test]
fn effort_ladder_at_default_ceiling_matches_the_historical_absolutes() {
    // Regression pin for the symbolic-effort switch: with the built-in
    // default ceiling (256, `ModelBehavior::default()`), every effort
    // level resolves to EXACTLY the number the old wire-edge ladder
    // hardcoded. Fails if anyone re-anchors the ratios.
    use crate::ir::EffortLevel::*;
    let default_ceiling = Policy {
        max_thinking_budget: 256,
        cap_at_max_tokens: false,
        ..policy()
    };
    for (level, historical) in [
        (Minimal, 64),
        (Low, 128),
        (Medium, 256),
        (High, 512),
        (XHigh, 1024),
    ] {
        let (et, tb) = resolve(
            ThinkingDirective::OnEffort(level),
            default_ceiling,
            4096,
            false,
        );
        assert!(et);
        assert_eq!(tb, Some(historical), "level={level:?}");
    }
}

#[test]
fn effort_ladder_scales_with_the_operator_ceiling() {
    // The #328-family fix: `--max-thinking-budget` (folded into
    // `Policy::max_thinking_budget` by serve) now GOVERNS what a client's
    // qualitative effort means, instead of being silently outranked by a
    // hardcoded 256. Daniel's repro: flag 16256 + harness "medium".
    use crate::ir::EffortLevel::*;
    let operator = Policy {
        max_thinking_budget: 16256,
        cap_at_max_tokens: false,
        ..policy()
    };
    let (et, tb) = resolve(ThinkingDirective::OnEffort(Medium), operator, 32768, false);
    assert!(et);
    assert_eq!(tb, Some(16256));
    let (_, tb) = resolve(ThinkingDirective::OnEffort(Minimal), operator, 32768, false);
    assert_eq!(tb, Some(4064));
    // The 90%-of-max_tokens safety clamp still applies on top when the
    // model keeps `cap_thinking_at_max_tokens`.
    let capped = Policy {
        max_thinking_budget: 16256,
        ..policy()
    };
    let (_, tb) = resolve(ThinkingDirective::OnEffort(Medium), capped, 1000, false);
    assert_eq!(tb, Some(900));
}

#[test]
fn effort_cap_defaults_off_so_shipping_it_changes_nothing() {
    // The built-in default MUST stay `false`: with it, the parity test
    // above proves every non-opted-in model resolves byte-identically
    // to the pre-clamp code. Reads the same `ModelBehavior::default()`
    // the server boots with (SSOT constant in behavior_defaults.rs),
    // not a copy of the literal.
    assert!(!atlas_kernels::ModelBehavior::default().effort_capped_at_ceiling);
}

#[test]
fn effort_cap_clamps_the_ladder_at_the_ceiling() {
    // Opt-in shape for measured budget non-monotonicity (Qwen3.5-397B,
    // 2026-05-07 sweep: 256 thinking tokens scores WORSE than 128): a
    // client's boilerplate high/xhigh must not exceed the model's E.
    use crate::ir::EffortLevel::*;
    let p = Policy {
        max_thinking_budget: 128,
        effort_capped_at_ceiling: true,
        cap_at_max_tokens: false,
        ..policy()
    };
    for (level, expected) in [
        (Minimal, 32),
        (Low, 64),
        (Medium, 128),
        (High, 128),
        (XHigh, 128),
    ] {
        let (et, tb) = resolve(ThinkingDirective::OnEffort(level), p, 4096, false);
        assert!(et);
        assert_eq!(tb, Some(expected), "level={level:?}");
    }
}

#[test]
fn effort_cap_never_touches_an_explicit_client_budget() {
    // Precedence stays: explicit client token budget > effort mapping >
    // server flag > MODEL.toml > default. The clamp binds ONLY the
    // effort ladder — a client that states a number gets that number.
    let p = Policy {
        max_thinking_budget: 128,
        effort_capped_at_ceiling: true,
        cap_at_max_tokens: false,
        ..policy()
    };
    let (et, tb) = resolve(ThinkingDirective::On { budget: Some(4096) }, p, 8192, false);
    assert!(et);
    assert_eq!(tb, Some(4096));
}

#[test]
fn effort_is_explicit_for_thinking_in_tools_purposes() {
    // An effort-carrying request stated thinking intent, so the
    // `thinking_in_tools=false` tools-suppression must NOT apply —
    // parity with the old `On { budget }` encoding of the ladder.
    let p = Policy {
        thinking_in_tools: false,
        ..policy()
    };
    let (et, _) = resolve(
        ThinkingDirective::OnEffort(crate::ir::EffortLevel::Medium),
        p,
        4096,
        true,
    );
    assert!(et);
}

#[test]
fn cap_at_max_tokens_false_skips_the_90pct_clamp() {
    // vLLM single-budget: max_thinking_budget is the SOLE cap; a small
    // max_tokens no longer clamps thinking down to 90% of it.
    let no_cap = Policy {
        cap_at_max_tokens: false,
        ..policy()
    };
    // Explicit client budget passes through untouched (was 900 under cap).
    let (et, tb) = resolve(
        ThinkingDirective::On { budget: Some(4096) },
        no_cap,
        1000,
        false,
    );
    assert!(et);
    assert_eq!(tb, Some(4096));
    // Budgetless On → max_thinking_budget, not clamped to 0.9*max_tokens.
    let (_, tb) = resolve(ThinkingDirective::On { budget: None }, no_cap, 1000, false);
    assert_eq!(tb, Some(2048));
}

#[test]
fn no_server_ceiling_leaves_only_client_budgets() {
    // `max_thinking_budget = 0`: no server budget. Effort is the
    // template's alone, budgetless thinking is unbounded, and an explicit
    // client budget still binds (and is still clamped when the model
    // keeps the 90% cap).
    use crate::ir::EffortLevel::*;
    let none = Policy {
        max_thinking_budget: 0,
        cap_at_max_tokens: false,
        ..policy()
    };
    for directive in [
        ThinkingDirective::On { budget: None },
        ThinkingDirective::OnEffort(Low),
        ThinkingDirective::OnEffort(XHigh),
    ] {
        assert_eq!(resolve(directive, none, 4096, false), (true, None));
    }
    let model_default = Policy {
        model_default: true,
        ..none
    };
    assert_eq!(
        resolve(ThinkingDirective::Unspecified, model_default, 4096, false),
        (true, None)
    );
    assert_eq!(
        resolve(
            ThinkingDirective::On { budget: Some(64) },
            none,
            4096,
            false
        ),
        (true, Some(64))
    );
    let capped = Policy {
        cap_at_max_tokens: true,
        ..none
    };
    assert_eq!(
        resolve(ThinkingDirective::On { budget: None }, capped, 1000, false),
        (true, None)
    );
    assert_eq!(
        resolve(
            ThinkingDirective::On { budget: Some(4096) },
            capped,
            1000,
            false
        ),
        (true, Some(900))
    );
}

#[test]
fn tools_suppression_only_when_client_silent() {
    let no_tools_thinking = Policy {
        model_default: true,
        thinking_in_tools: false,
        ..policy()
    };
    // Silent client on a tool turn → suppressed.
    let (et, _) = resolve(
        ThinkingDirective::Unspecified,
        Policy {
            ..no_tools_thinking
        },
        4096,
        true,
    );
    assert!(!et);
    // Explicit enable survives the suppression.
    let (et, _) = resolve(
        ThinkingDirective::On { budget: None },
        Policy {
            model_default: true,
            thinking_in_tools: false,
            ..policy()
        },
        4096,
        true,
    );
    assert!(et);
    // Explicit disable is likewise respected (no double negation).
    let (et, _) = resolve(
        ThinkingDirective::Off,
        Policy {
            model_default: true,
            thinking_in_tools: false,
            ..policy()
        },
        4096,
        true,
    );
    assert!(!et);
}

#[test]
fn explicit_off_wins_over_model_default() {
    let (et, tb) = resolve(
        ThinkingDirective::Off,
        Policy {
            model_default: true,
            ..policy()
        },
        4096,
        false,
    );
    assert!(!et);
    assert!(tb.is_none());
}

#[test]
fn issue_517_tool_capped_ceiling_leaves_room_for_tool_call() {
    // Caller hoists min(max_tokens, tool_max_tokens) before resolve
    // (#517). 90% of the real ceiling (256) is 230, not 90% of the
    // raw client max_tokens (2048 -> 1843).
    let (et, tb) = resolve(ThinkingDirective::On { budget: None }, policy(), 256, true);
    assert!(et);
    assert_eq!(tb, Some(230));
    assert!(tb.unwrap() < 256);
}
