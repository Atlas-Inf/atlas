// SPDX-License-Identifier: AGPL-3.0-only
//
// Resolve `(enable_thinking, thinking_budget)` for a single request
// from the neutral thinking directive. Precedence (highest wins):
//   1. `--disable-thinking` CLI flag (forces OFF for every request)
//   2. The request directive (client channels resolved at the API edge,
//      or the server-level default directive when the client is silent)
//   3. MODEL.toml `[behavior].thinking_default`
//
// Lifted out of `chat::chat_completions_inner` (wave 4g); flipped from
// the OpenAI wire request to `ir::ThinkingDirective` (IR migration).

use std::sync::Arc;

use crate::AppState;
use crate::ir::ThinkingDirective;

pub(super) fn resolve_thinking(
    state: &Arc<AppState>,
    directive: ThinkingDirective,
    max_tokens: u32,
    tools_active: bool,
) -> (bool, Option<u32>) {
    resolve(
        directive,
        Policy {
            disable_thinking: state.disable_thinking,
            model_default: state.behavior.thinking_default,
            thinking_in_tools: state.behavior.thinking_in_tools,
            max_thinking_budget: state.behavior.max_thinking_budget,
            effort_capped_at_ceiling: state.behavior.effort_capped_at_ceiling,
            cap_at_max_tokens: state.behavior.cap_thinking_at_max_tokens,
        },
        max_tokens,
        tools_active,
    )
}

/// Generation allowance after the tools-active `--tool-max-tokens` shrink.
/// Single source of truth for thinking-budget resolution and sampling
/// `max_tokens` (#517). A tool turn must not receive a thinking budget
/// larger than the tokens it is actually allowed to emit.
pub(super) fn generation_max_tokens(
    max_tokens: usize,
    tools_active: bool,
    tool_max_tokens: usize,
) -> usize {
    if tools_active {
        max_tokens.min(tool_max_tokens)
    } else {
        max_tokens
    }
}

/// Server/model policy inputs, split from `AppState` so the resolution
/// core is a pure function.
#[derive(Clone, Copy)]
struct Policy {
    disable_thinking: bool,
    model_default: bool,
    thinking_in_tools: bool,
    max_thinking_budget: u32,
    /// MODEL.toml `[behavior].effort_capped_at_ceiling`: clamp the effort
    /// ladder at E (high/xhigh -> E instead of 2E/4E). Never touches an
    /// explicit client token budget. See `behavior_defaults.rs`.
    effort_capped_at_ceiling: bool,
    /// When false, `max_thinking_budget` is the SOLE thinking cap (vLLM
    /// single-budget); the 90%-of-max_tokens clamp is skipped.
    cap_at_max_tokens: bool,
}

fn resolve(
    directive: ThinkingDirective,
    policy: Policy,
    max_tokens: u32,
    tools_active: bool,
) -> (bool, Option<u32>) {
    if policy.disable_thinking {
        return (false, None);
    }
    let ceiling = atlas_kernels::thinking_ceiling(policy.max_thinking_budget);
    let (et, tb) = match directive {
        // No client/server directive → MODEL.toml decides. `None` budget
        // defers to the per-model `max_thinking_budget` below rather than
        // a conservative hardcoded default.
        ThinkingDirective::Unspecified => (policy.model_default, None),
        ThinkingDirective::Off => (false, None),
        ThinkingDirective::On { budget } => (true, budget),
        // A qualitative effort level scales against the model's effective
        // ceiling HERE — the one place that knows it — so MODEL.toml and
        // `--max-thinking-budget` govern what "medium" means. Resolved at
        // the wire edge it was a hardcoded absolute (medium = 256) that no
        // server knob could reach (#328 family).
        // Without a server ceiling an effort level is the template's alone.
        ThinkingDirective::OnEffort(level) => (
            true,
            ceiling.map(|e| effort_budget(level, e, policy.effort_capped_at_ceiling)),
        ),
    };
    // `thinking_in_tools=false` is the MODEL.toml DEFAULT for tool-
    // active turns: it suppresses thinking when the client is silent.
    // An explicit directive (enabled OR disabled — including the
    // server-level default directive) still wins.
    let et = if tools_active && !policy.thinking_in_tools && !directive.is_explicit() {
        false
    } else {
        et
    };
    let budget = if et {
        // No client budget and no server ceiling: thinking is unbounded
        // (the reference engines' behaviour), so there is nothing to clamp.
        let Some(b) = tb.or(ceiling) else {
            return (et, None);
        };
        if !policy.cap_at_max_tokens {
            // vLLM single-budget: `max_thinking_budget` (or an explicit
            // client budget) is the sole cap. Reasoning may use the full
            // generation budget; there is no second, max_tokens-derived cap.
            Some(b)
        } else {
            // 2026-05-23 sweep: dropped the 70% special case for
            // `tools_active && thinking_in_tools` (previously 7/10, now
            // 9/10 uniformly). With `thinking_in_tools=true` as the
            // project-wide default the 70% branch fired on every tool turn
            // and silently undermined the MODEL.toml `max_thinking_budget`
            // bump (opencode-style requests at max_tokens=2048 capped to
            // 1433 instead of 2048). 90% leaves headroom for content +
            // tool args without crippling reasoning chains that now run
            // naturally after the F1 reflection-penalty removal.
            let safety_cap_pct = 9;
            let max = ((max_tokens * safety_cap_pct) / 10).max(1);
            Some(b.min(max))
        }
    } else {
        None
    };
    (et, budget)
}

/// Token budget for a qualitative effort level, as a ratio of the model's
/// effective `max_thinking_budget` E: minimal=E/4, low=E/2, medium=E,
/// high=2E, xhigh=4E.
///
/// At the built-in default E=256 these are exactly the historical absolute
/// ladder (64/128/256/512/1024), so a server with no MODEL.toml value and
/// no `--max-thinking-budget` resolves byte-identically to the pre-symbolic
/// code — including high/xhigh EXCEEDING the ceiling, which the old ladder
/// also did (512 > 256). The `cap_at_max_tokens` clamp below still applies
/// on top, unchanged.
///
/// `capped_at_ceiling` (MODEL.toml `[behavior].effort_capped_at_ceiling`,
/// default false) clamps the ladder at E for models with MEASURED
/// degradation above their ceiling (budget non-monotonicity — Qwen3.5-397B,
/// 2026-05-07 sweep: 256 thinking tokens scores worse than 128). It binds
/// only this server-policy ladder; explicit client budgets never pass
/// through here.
fn effort_budget(
    level: crate::ir::EffortLevel,
    max_thinking_budget: u32,
    capped_at_ceiling: bool,
) -> u32 {
    use crate::ir::EffortLevel;
    let e = max_thinking_budget.max(1);
    let b = match level {
        EffortLevel::Minimal => (e / 4).max(1),
        EffortLevel::Low => (e / 2).max(1),
        EffortLevel::Medium => e,
        EffortLevel::High => e.saturating_mul(2),
        EffortLevel::XHigh => e.saturating_mul(4),
    };
    if capped_at_ceiling { b.min(e) } else { b }
}

#[cfg(test)]
mod tests;
