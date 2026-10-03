// SPDX-License-Identifier: AGPL-3.0-only
//
// `--reference-mode`: serve what the client asked for and nothing else.
//
// Benchmark rules that lock sampling (MLPerf Endpoints: "submitters must not
// modify the sampling parameters or thinking flags ... must not introduce
// additional sampling parameters"; the chat template "must not omit reasoning
// tokens from any previous turn") are broken by Atlas's chat-tuned defaults,
// each harmless on its own and every one silent:
//
//   * MODEL.toml `[behavior]` clamps: `temperature_max` (35B: 1.0 -> 0.7),
//     `min_p_floor`, the thinking budget (35B: 768 tokens) and its 90%-of-
//     max_tokens cap, the A4 `</think>` floor;
//   * the CLI sampling defaults `--default-min-p 0.08` /
//     `--default-top-n-sigma 1.0`, which reach every request that does not
//     name them (and a locked client may not name them);
//   * the `<tool_call>` logit-bias nudge and the `[sampling.*]` preset
//     penalties / DRY / LZ on tool turns;
//   * prompt text the client never sent: the parser tool system prompt
//     (rendered on top of the template's own tool block), the CWD hint, and a
//     jinja override that drops `preserve_thinking`;
//   * output guards that end a response early (content-loop, inter-tool
//     prose, in-think tool leak, and the `ATLAS_DISABLE_WATCHDOGS` family).
//
// This module owns the behaviour half; the request-path halves read
// `AppState::reference_mode`. Nothing here changes kernels or numerics.

/// Thinking budget that never binds: above any `max_tokens` a request can
/// carry, and small enough that the effort ladder's 4x cannot overflow u32.
pub(crate) const UNBOUNDED_THINKING_BUDGET: u32 = 1 << 20;

/// Neutralize every MODEL.toml `[behavior]` field that changes what the model
/// generates or what prompt it sees, relative to a reference server rendering
/// the checkpoint's own chat template with the client's sampling parameters.
pub(crate) fn neutralize_behavior(b: &mut atlas_kernels::ModelBehavior) {
    // Sampling clamps.
    b.temperature_max = 0.0;
    b.min_p_floor = 0.0;
    b.use_sampling_presets_for_core = false;
    b.use_sampling_preset_penalties_for_core = false;
    // Thinking: the client's max_tokens is the only cap.
    b.max_thinking_budget = UNBOUNDED_THINKING_BUDGET;
    b.cap_thinking_at_max_tokens = false;
    b.min_reasoning_floor_tokens = 0;
    b.honor_eos_inside_thinking = true;
    // Output guards.
    b.enable_loop_watchdog = false;
    b.enable_think_loop_watchdog = false;
    b.confidence_early_stop = false;
    b.rollback_resteer = false;
    b.max_inter_tool_prose = 0;
    b.max_post_think_content_tokens = UNBOUNDED_THINKING_BUDGET;
    b.tool_retry = false;
    // Prompt text: no grammar, no injected hints, no compacted tool schemas.
    b.disable_tool_grammar = true;
    b.disable_cwd_hint_injection = true;
    b.tscg = false;
}

/// The `[behavior]` fields `neutralize_behavior` changed, for the boot log.
pub(crate) fn describe_changes(
    before: &atlas_kernels::ModelBehavior,
    after: &atlas_kernels::ModelBehavior,
) -> Vec<String> {
    let mut out = Vec::new();
    macro_rules! diff {
        ($($f:ident),* $(,)?) => {$(
            if format!("{:?}", before.$f) != format!("{:?}", after.$f) {
                out.push(format!("{}={:?}->{:?}", stringify!($f), before.$f, after.$f));
            }
        )*};
    }
    diff!(
        temperature_max,
        min_p_floor,
        use_sampling_presets_for_core,
        use_sampling_preset_penalties_for_core,
        max_thinking_budget,
        cap_thinking_at_max_tokens,
        min_reasoning_floor_tokens,
        honor_eos_inside_thinking,
        enable_loop_watchdog,
        enable_think_loop_watchdog,
        confidence_early_stop,
        rollback_resteer,
        max_inter_tool_prose,
        max_post_think_content_tokens,
        tool_retry,
        disable_tool_grammar,
        disable_cwd_hint_injection,
        tscg,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 35B card's chat-tuned values, which an MLPerf run must not see.
    fn qwen36_35b_like() -> atlas_kernels::ModelBehavior {
        atlas_kernels::ModelBehavior {
            temperature_max: 0.7,
            min_p_floor: 0.05,
            max_thinking_budget: 768,
            max_inter_tool_prose: 3072,
            enable_loop_watchdog: true,
            ..atlas_kernels::ModelBehavior::default()
        }
    }

    #[test]
    fn every_clamp_and_guard_is_off() {
        let mut b = qwen36_35b_like();
        neutralize_behavior(&mut b);
        assert_eq!(b.temperature_max, 0.0, "temperature clamp");
        assert_eq!(b.min_p_floor, 0.0, "min-p floor");
        assert!(!b.use_sampling_presets_for_core && !b.use_sampling_preset_penalties_for_core);
        assert!(!b.cap_thinking_at_max_tokens && b.max_thinking_budget >= 1 << 20);
        assert_eq!(b.min_reasoning_floor_tokens, 0, "A4 </think> floor");
        assert!(b.honor_eos_inside_thinking);
        assert!(!b.enable_loop_watchdog && !b.enable_think_loop_watchdog);
        assert!(!b.confidence_early_stop && !b.rollback_resteer && !b.tool_retry);
        assert_eq!(b.max_inter_tool_prose, 0);
        assert!(b.disable_tool_grammar && b.disable_cwd_hint_injection && !b.tscg);
    }

    #[test]
    fn effort_ladder_cannot_overflow() {
        // `thinking::effort_budget` multiplies the ceiling by up to 4.
        assert!(UNBOUNDED_THINKING_BUDGET.checked_mul(4).is_some());
    }

    #[test]
    fn the_boot_log_names_what_changed() {
        let before = qwen36_35b_like();
        let mut after = before.clone();
        neutralize_behavior(&mut after);
        let changes = describe_changes(&before, &after);
        assert!(changes.iter().any(|c| c.starts_with("temperature_max=0.7")));
        assert!(changes.iter().any(|c| c.starts_with("max_thinking_budget=768")));
        assert!(describe_changes(&after, &after).is_empty());
    }
}
