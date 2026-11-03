// SPDX-License-Identifier: AGPL-3.0-only

//! `ModelBehavior`'s defaults, for a target whose MODEL.toml leaves a key out.

use crate::ModelBehavior;
use crate::behavior_defaults::{
    DEFAULT_EFFORT_CAPPED_AT_CEILING, DEFAULT_MAX_INTER_TOOL_PROSE, DEFAULT_MAX_THINKING_BUDGET,
};

impl Default for ModelBehavior {
    fn default() -> Self {
        Self {
            thinking_in_tools: true,
            max_thinking_budget: DEFAULT_MAX_THINKING_BUDGET,
            effort_capped_at_ceiling: DEFAULT_EFFORT_CAPPED_AT_CEILING,
            default_reasoning_effort: "",
            thinking_default: false,
            fp8_kv_calibration_tokens: 0,
            default_kv_dtype: "",
            default_num_drafts: 0,
            disable_tool_steering: false,
            template_owns_tool_definitions: false,
            disable_cwd_hint_injection: false,
            use_sampling_presets_for_core: false,
            use_sampling_preset_penalties_for_core: true,
            tool_call_parser: "",
            jinja_template: "",
            enable_loop_watchdog: false,
            enable_think_loop_watchdog: true,
            enable_stream_loop_guards: true,
            honor_eos_inside_thinking: false,
            min_reasoning_floor_tokens: 16,
            cap_thinking_at_max_tokens: true,
            min_p_floor: 0.0,
            temperature_max: 0.0,
            think_loop_min_repeats: 3,
            think_loop_scan_window: 160,
            confidence_early_stop: true,
            confidence_run_length: 30,
            fuzzy_repeat_tolerance_div: 12,
            max_inter_tool_prose: DEFAULT_MAX_INTER_TOOL_PROSE,
            max_post_think_content_tokens: 100_000,
            tscg: false,
            disable_tool_grammar: false,
            rollback_resteer: true,
            rom_head: "",
            tool_retry: true,
            preserve_thinking: None,
            verbatim_prompt: false,
            eos_ends_thinking: false,
            tool_call_steering: true,
            verbatim_reasoning_effort: false,
            ban_tool_call_without_tools: true,
        }
    }
}
