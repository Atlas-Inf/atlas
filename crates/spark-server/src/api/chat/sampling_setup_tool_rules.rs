// SPDX-License-Identifier: AGPL-3.0-only
//
// Tool-choice and tool-call-bias rules of `sampling_setup`.

use crate::tool_parser;

pub(super) fn tool_choice_required_for_parser(
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
pub(super) fn preset_penalties_apply(tools_active: bool, core_penalties_opt_in: bool) -> bool {
    tools_active || core_penalties_opt_in
}

pub(super) fn tool_grammar_escape_applies(
    disable_tool_grammar: bool,
    tool_choice_required: bool,
) -> bool {
    disable_tool_grammar && !tool_choice_required
}

/// The `<tool_call>` logit bias by how often the same call already repeated.
/// A tool-call-or-JSON union never takes the boost: the model's own mass
/// picks its branch (`grammar::branch`), and a boost after a call pulls the
/// next token toward yet another call instead of the answer.
pub(super) fn tool_call_bias(repeat_count: usize, union: bool) -> f32 {
    match repeat_count {
        0 | 1 if union => 0.0,
        0 | 1 => 3.0,
        2 => 0.0,
        3 => -5.0,
        _ => -10.0,
    }
}
