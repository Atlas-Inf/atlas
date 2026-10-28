// SPDX-License-Identifier: AGPL-3.0-only
//
// Sampling-preset selection, stop-token tokenisation, grammar-spec
// construction, and timeout / logprobs resolution.
//
// Lifted out of `chat::chat_completions_inner` (wave 4g).

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::AppState;
use crate::ir::ChatRequest;
use crate::tool_parser;

use super::super::compact::openai_error_response;
use super::super::inference_impl::tokenize_stop_sequences;
use super::super::inference_types::GrammarSpec;
use super::thinking;

pub(super) struct SamplingSetup {
    pub(super) temperature: f32,
    pub(super) top_k: u32,
    pub(super) top_p: f32,
    pub(super) top_n_sigma: f32,
    pub(super) min_p: f32,
    pub(super) repetition_penalty: f32,
    pub(super) presence_penalty: f32,
    pub(super) frequency_penalty: f32,
    pub(super) dry_multiplier: f32,
    pub(super) dry_base: f32,
    pub(super) dry_allowed_length: u32,
    pub(super) lz_penalty: f32,
    pub(super) logit_bias: Vec<(u32, f32)>,
    pub(super) max_tokens: usize,
    pub(super) stop_tokens: Vec<u32>,
    pub(super) tool_choice_required: bool,
    pub(super) grammar_spec: Option<GrammarSpec>,
    pub(super) timeout_at: Option<std::time::Instant>,
    pub(super) top_logprobs: Option<u8>,
}

fn tool_choice_required_for_parser(
    tools_active: bool,
    tool_choice: Option<&tool_parser::ToolChoice>,
    parser_name: Option<&str>,
) -> bool {
    if !tools_active {
        return false;
    }

    let explicit_required = tool_choice.is_some_and(|tc| {
        matches!(tc, tool_parser::ToolChoice::Mode(m) if m == "required")
            || matches!(tc, tool_parser::ToolChoice::Specific { .. })
    });
    let parser_required = matches!(parser_name, Some("minimax_xml"));

    explicit_required || parser_required
}

/// Whether the model-level tool-grammar escape hatch applies to THIS request.
///
/// `disable_tool_grammar` is a *model* property (MODEL.toml `[behavior]`, or
/// the `--disable-tool-grammar` override); `tool_choice_required` is a *request*
/// property. The escape hatch is scoped to the requests it was written for —
/// `serve_args.rs`: "skips XGrammar structural-tag enforcement on
/// `tool_choice="auto"` requests ... Matches vLLM's default behaviour in auto
/// mode (vLLM only grammar-constrains when tool_choice="required")".
///
/// Without the `!tool_choice_required` term the hatch also swallowed
/// `required`/specific requests, which is what `book/src/operations/tools.md`
/// documents the grammar as implementing: it "masks the no-tool-call path, so
/// the sampler can only produce a valid tool-call opening". Dropping the
/// grammar there did not leave `required` wholly unenforced — the scheduler's
/// legacy EOS-suppression backstop switches on precisely when the grammar is
/// absent (`prefill_a_step.rs`: `req_require_tool_call && grammar_state
/// .is_none() && tool_call_start_token.is_some()`) — but that path is weaker
/// (it needs a `tool_call_start_token` and only suppresses EOS) and it is not
/// the behaviour either doc promises.
///
/// The `minimax_xml` arm of `tool_choice_required_for_parser` is deliberately
/// covered by the same term: that parser's grammar is what stops the
/// `<invokeinvoke` / `<parameterparameter` degenerate-loop corruption class
/// (see `compile_minimax_xml_tool_grammar`), so the hatch must not remove it.
/// Whether a request that left its penalties unset inherits them from the
/// selected `[sampling.*]` preset. Tool-calling requests always do (the
/// `[sampling.tools]` row is agentic-turn tuned); core requests only when
/// the model opts in via `use_sampling_preset_penalties_for_core`.
fn preset_penalties_apply(tools_active: bool, core_penalties_opt_in: bool) -> bool {
    tools_active || core_penalties_opt_in
}

fn tool_grammar_escape_applies(disable_tool_grammar: bool, tool_choice_required: bool) -> bool {
    disable_tool_grammar && !tool_choice_required
}

/// The `<tool_call>` logit bias by how often the same call already repeated.
/// A tool-call-or-JSON union never takes the boost: the model's own mass
/// picks its branch (`grammar::branch`), and a boost after a call pulls the
/// next token toward yet another call instead of the answer.
fn tool_call_bias(repeat_count: usize, union: bool) -> f32 {
    match repeat_count {
        0 | 1 if union => 0.0,
        0 | 1 => 3.0,
        2 => 0.0,
        3 => -5.0,
        _ => -10.0,
    }
}

/// The grammar a request is decoded under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GrammarChoice {
    None,
    /// Tools armed, tool grammar off (`tool_grammar_escape_applies`).
    EscapeHatch,
    /// The response_format JSON alone.
    ResponseFormat,
    /// The tool-call grammar alone.
    ToolCall,
    /// Tool calls OR the response_format JSON (`GrammarSpec::ToolCallOrJson`).
    ToolCallOrResponseFormat,
}

/// response_format + tools coexistence. OpenAI's API allows both fields in
/// the same request; agentic pipelines routinely set both (the model emits a
/// tool call on turn N, then a schema-shaped final answer on turn N+1), and
/// OpenAI and vLLM enforce the schema on that answer while still allowing
/// tool calls:
///   * `tool_choice="none"` (or no tools) → the response_format alone
///     (without one, `"none"` keeps the triggered tool grammar, as ever)
///   * `"required"`/named (or a parser-required format) → the tool grammar
///     alone: the answer must be a call
///   * `"auto"`/unset → their union: the first output token picks a tool
///     call or the JSON. Enforcing only the tool grammar here let GLM wrap
///     a correct JSON answer in ```json fences, which strict clients reject.
///
/// The wire's `{"type":"text"}` was mapped to `None` at the edge, so
/// presence alone means a real constraint. The escape hatch keeps auto-mode
/// requests unconstrained, response_format or not.
fn grammar_choice(
    tools_active: bool,
    has_response_format: bool,
    tool_choice_none: bool,
    tool_choice_required: bool,
    escape_hatch: bool,
) -> GrammarChoice {
    if has_response_format && (!tools_active || tool_choice_none) {
        GrammarChoice::ResponseFormat
    } else if !tools_active {
        GrammarChoice::None
    } else if escape_hatch {
        GrammarChoice::EscapeHatch
    } else if has_response_format && !tool_choice_required {
        GrammarChoice::ToolCallOrResponseFormat
    } else {
        GrammarChoice::ToolCall
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
pub(super) fn build_sampling(
    state: &Arc<AppState>,
    req: &ChatRequest,
    enable_thinking: bool,
    tools_active: bool,
    suppress_tool_call: bool,
    tool_call_repeat_count: usize,
) -> Result<SamplingSetup, Response> {
    // Preset selection.
    let preset = if tools_active {
        &state.sampling_presets.tools
    } else if enable_thinking {
        &state.sampling_presets.thinking_text
    } else {
        &state.sampling_presets.non_thinking
    };
    // ATLAS_FORCE_TEMP_ZERO=1 — diagnostic override that forces fully greedy
    // deterministic decoding, ignoring client params AND MODEL.toml presets.
    // Used for layer-by-layer cosine comparison against vLLM (same env-var
    // contract on the vLLM side, VLLM_FORCE_TEMP_ZERO). At T=0 with identical
    // weights+tokens, two engines should produce bit-identical token streams;
    // any divergence localises a numerical bug.
    let force_temp_zero = std::env::var("ATLAS_FORCE_TEMP_ZERO")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // Core sampling defaults to generation_config.json unless MODEL.toml opts
    // into its category presets. Laguna's generation config ships 1.0/1.0 for
    // general evaluation, while its model card recommends 0.7/0.95 for
    // reliability; the model-owned opt-in preserves the historical behavior
    // for every other target. Explicit request values always win.
    let core_preset = state.behavior.use_sampling_presets_for_core;
    let temperature = if force_temp_zero {
        0.0
    } else {
        req.sampling.temperature.unwrap_or(if core_preset {
            preset.temperature
        } else {
            state.default_temperature
        })
    };
    let top_k = if force_temp_zero {
        0
    } else {
        req.sampling.top_k.unwrap_or(if core_preset {
            preset.top_k
        } else {
            state.default_top_k
        })
    };
    let top_p = if force_temp_zero {
        1.0
    } else {
        req.sampling.top_p.unwrap_or(if core_preset {
            preset.top_p
        } else {
            state.default_top_p
        })
    };
    // min_p / top_n_sigma precedence: explicit request > MODEL.toml preset >
    // CLI default. The preset arm was declared but never consulted, so
    // `--default-min-p 0.08` and `--default-top-n-sigma 1.0` reached every
    // request no matter what the model card said — and stacked on top of a
    // tight top_k they can gut the distribution. `Some(0.0)` is how a model
    // says "no filter"; `None` leaves the CLI default owning the field, which
    // is what every model without a `[sampling.*]` entry still gets.
    let top_n_sigma = if force_temp_zero {
        0.0
    } else {
        req.sampling
            .top_n_sigma
            .or(if core_preset {
                preset.top_n_sigma
            } else {
                None
            })
            .unwrap_or(state.default_top_n_sigma)
    };
    let min_p = if force_temp_zero {
        0.0
    } else {
        req.sampling
            .min_p
            .or(if core_preset { preset.min_p } else { None })
            .unwrap_or(state.default_min_p)
    };
    // Penalties: explicit request > preset, but a core (non-tool) request
    // inherits the preset penalties only when the model opts in
    // (`[behavior].use_sampling_preset_penalties_for_core`, default true).
    // A card's presence_penalty is tuned for short chat turns; over a
    // 20K-token single-file generation it suppresses every token already
    // emitted (see the flag's doc in `atlas_kernels::ModelBehavior`).
    // `--reference-mode`: an unset penalty is neutral, never the preset's.
    let preset_penalties = !state.reference_mode
        && preset_penalties_apply(
            tools_active,
            state.behavior.use_sampling_preset_penalties_for_core,
        );
    let repetition_penalty = if force_temp_zero {
        1.0
    } else {
        req.sampling
            .repetition_penalty
            .unwrap_or(if preset_penalties {
                preset.repetition_penalty
            } else {
                1.0
            })
    };
    let presence_penalty = if force_temp_zero {
        0.0
    } else {
        req.sampling
            .presence_penalty
            .unwrap_or(if preset_penalties {
                preset.presence_penalty
            } else {
                0.0
            })
    };
    let frequency_penalty = if force_temp_zero {
        0.0
    } else {
        req.sampling
            .frequency_penalty
            .unwrap_or(if preset_penalties {
                preset.frequency_penalty
            } else {
                0.0
            })
    };
    // Per-model server-side sampling SAFETY FLOOR/CEILING (MODEL.toml
    // [behavior]). Binds AFTER request/preset resolution so model stability
    // does NOT depend on the client volunteering safe params — the Claude-Code
    // loop fix (an unfloored min_p let the FP8/NVFP4 degenerate tail be sampled
    // into repetition loops; measured 2026-06-07: 0.05 → 4 watchdog fires
    // become 0). 0.0 = disabled (no-op). Skipped under force_temp_zero (that
    // diagnostic override deliberately drives greedy).
    let min_p = if !force_temp_zero && state.behavior.min_p_floor > 0.0 {
        min_p.max(state.behavior.min_p_floor)
    } else {
        min_p
    };
    let temperature = if !force_temp_zero && state.behavior.temperature_max > 0.0 {
        temperature.min(state.behavior.temperature_max)
    } else {
        temperature
    };
    // One line, once per process: what the sampler ACTUALLY runs with after
    // request > preset > CLI-default resolution. Sampling bugs are otherwise
    // invisible — the values live in three places and only their resolution
    // matters.
    {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                "sampling resolved (first request): temp={temperature:.3} top_p={top_p:.3} \
                 top_k={top_k} min_p={min_p:.4} top_n_sigma={top_n_sigma:.4} \
                 rep_pen={repetition_penalty:.3} (core_preset={core_preset})"
            );
        });
    }
    let dry_multiplier = if force_temp_zero || state.reference_mode {
        0.0
    } else {
        preset.dry_multiplier
    };
    let dry_base = preset.dry_base;
    let dry_allowed_length = preset.dry_allowed_length;
    let lz_penalty = if force_temp_zero || state.reference_mode {
        0.0
    } else {
        preset.lz_penalty
    };

    // OpenAI-style penalty range validation.
    if !(-2.0..=2.0).contains(&presence_penalty) {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("presence_penalty must be between -2.0 and 2.0, got {presence_penalty}"),
        ));
    }
    if !(-2.0..=2.0).contains(&frequency_penalty) {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("frequency_penalty must be between -2.0 and 2.0, got {frequency_penalty}"),
        ));
    }

    // Logit bias (already parsed to typed pairs at the API edge).
    let mut logit_bias: Vec<(u32, f32)> = if force_temp_zero {
        Vec::new()
    } else {
        req.logit_bias.clone()
    };

    // Tool-choice + parser-driven required mode.
    let tool_choice_required = tool_choice_required_for_parser(
        tools_active,
        req.tool_choice.as_ref(),
        state.tool_call_parser.as_ref().map(|p| p.name()),
    );

    let tool_choice_none = req
        .tool_choice
        .as_ref()
        .is_some_and(|tc| matches!(tc, tool_parser::ToolChoice::Mode(m) if m == "none"));
    let choice = grammar_choice(
        tools_active,
        req.response_format.is_some(),
        tool_choice_none,
        tool_choice_required,
        tool_grammar_escape_applies(state.behavior.disable_tool_grammar, tool_choice_required),
    );

    // Exponential `<tool_call>` bias decay. Skipped under ATLAS_FORCE_TEMP_ZERO
    // so the argmax is determined purely by raw logits (matches vLLM's path).
    if !force_temp_zero
        && !state.reference_mode
        && tools_active
        && !suppress_tool_call
        && let Some(tc_id) = state.tool_call_start_token_id
    {
        let union = choice == GrammarChoice::ToolCallOrResponseFormat;
        let bias = tool_call_bias(tool_call_repeat_count, union);
        if bias != 0.0 {
            logit_bias.push((tc_id, bias));
        }
    }

    // max_tokens cap when tools are active. Same ceiling as thinking
    // resolution (#517) — do not recompute min() here.
    let max_tokens =
        thinking::generation_max_tokens(req.max_tokens, tools_active, state.tool_max_tokens);
    if tools_active && max_tokens < req.max_tokens {
        tracing::info!(
            "Tool max_tokens cap: {} → {} (tool_max_tokens={})",
            req.max_tokens,
            max_tokens,
            state.tool_max_tokens
        );
    }

    // Stop tokens.
    //
    // #192: `</tool_call>` is deliberately NOT a stop token. It used to be
    // pushed here for every tools-active request, which (a) hard-stopped the
    // MTP/emit path at the FIRST closed tool call (the token hit the EOS
    // handler and was even dropped from the output), and (b) landed in the
    // grammar's stop-token exemption set, so the matcher never advanced
    // across the end-tag literal and desynced before a second call. vLLM
    // parity: generation continues past a closed call until natural EOS so
    // the model can emit parallel calls; the scheduler's tool watchdogs
    // (post-completion open cap, prose budget, loop detectors) bound run-on.
    let stop_tokens = tokenize_stop_sequences(&state.tokenizer, &req.stop);

    if choice == GrammarChoice::EscapeHatch {
        // Structure-snowballing escape hatch (arXiv:2604.06066): this
        // model tool-calls more reliably unconstrained. Tool calls are
        // still parsed from the output — just not grammar-enforced.
        // Scoped to auto-mode requests; `required`/specific keep the
        // grammar that enforces them (see `tool_grammar_escape_applies`).
        tracing::info!("MODEL.toml [behavior].disable_tool_grammar=true — tool-call grammar OFF");
    }
    let schema = || match req.response_format.as_ref() {
        Some(crate::ir::ResponseFormat::JsonSchema { schema, .. }) => Some(schema.to_string()),
        _ => None,
    };
    let tools = || {
        let mut tools = req.tools.clone();
        if let Some(tool_parser::ToolChoice::Specific { ref function }) = req.tool_choice {
            tools.retain(|t| t.function.name == function.name);
        }
        tools
    };
    let parser = state.tool_call_parser.as_ref().map(Arc::clone);
    // Grammar spec (XGrammar structural-tag enforcement).
    let grammar_spec: Option<GrammarSpec> = match choice {
        GrammarChoice::None | GrammarChoice::EscapeHatch => None,
        GrammarChoice::ResponseFormat => Some(match schema() {
            Some(schema) => GrammarSpec::JsonSchema { schema },
            None => GrammarSpec::JsonObject,
        }),
        GrammarChoice::ToolCall => parser.map(|parser| GrammarSpec::ToolCall {
            tools: tools(),
            parser,
            use_triggers: !tool_choice_required,
        }),
        GrammarChoice::ToolCallOrResponseFormat => {
            parser.map(|parser| GrammarSpec::ToolCallOrJson {
                tools: tools(),
                parser,
                schema: schema(),
            })
        }
    };

    // Timeout deadline (SSOT: AppState::request_deadline).
    let timeout_at = state.request_deadline(req.timeout_secs);

    // Pre-resolved from the wire's logprobs/top_logprobs pair at the edge.
    let top_logprobs = req.top_logprobs;

    Ok(SamplingSetup {
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        max_tokens,
        stop_tokens,
        tool_choice_required,
        grammar_spec,
        timeout_at,
        top_logprobs,
    })
}

// `resolve_top_logprobs` moved to the OpenAI edge (`openai/to_ir.rs`)
// — the envelope carries the already-resolved count.

#[cfg(test)]
#[path = "sampling_setup_tests.rs"]
mod sampling_setup_tests;
