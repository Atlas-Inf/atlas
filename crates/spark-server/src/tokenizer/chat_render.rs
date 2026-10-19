// SPDX-License-Identifier: AGPL-3.0-only

//! Core Jinja chat rendering shared by every `ChatTokenizer` apply path.
//!
//! Free function on purpose (SSOT): the golden-render tests in
//! `tokenizer/tests/` call [`render_chat`] against fixture templates, so the
//! exact context-construction code that produces production prompt bytes is
//! what the tests prove — not a re-implementation that can drift.

use anyhow::{Context, Result};

use super::chat_impl::preprocess_for_render;

/// Render-time flags for [`render_chat`]. Grouped so the apply-path
/// signatures stay readable as template knobs accumulate.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RenderFlags<'a> {
    pub enable_thinking: bool,
    pub disable_tool_steering: bool,
    /// Explicit client/server reasoning-effort string. `None` leaves the
    /// variable UNDEFINED while thinking is on, so the template's own default
    /// applies (Qwen3.8: `xhigh`; Mistral's override: `high`; GLM's: `max`),
    /// exactly as transformers/vLLM render an effort-silent request; `"none"`
    /// when thinking is off (Mistral's template would otherwise default to
    /// `high`). A model that wants another unset tier names it in MODEL.toml
    /// `[behavior].default_reasoning_effort`.
    pub reasoning_effort: Option<&'a str>,
    /// Leave an effort-silent thinking render's `reasoning_effort` UNDEFINED
    /// instead of the `"medium"` fallback, so the template's own default
    /// applies, as transformers/vLLM render it (MODEL.toml
    /// `[behavior].verbatim_prompt`).
    pub template_effort_default: bool,
    /// Tri-state `preserve_thinking` (Qwen3.6+ dense templates). `None` =
    /// the variable is left UNDEFINED in the Jinja context so the model
    /// template's own default applies (Qwen3.6 strips historical `<think>`
    /// blocks unless true; Qwen3.8 keeps them unless explicitly false).
    /// `Some(_)` pins it. Never pass `None` as Jinja `none` — Qwen3.8's
    /// `preserve_thinking is undefined` test distinguishes the two and
    /// `none` would silently flip its default from keep to strip.
    pub preserve_thinking: Option<bool>,
    /// Diagnostic "continue final message" mode: when true AND the last
    /// message is an assistant turn, render without a generation prompt and
    /// strip the trailing `<|im_end|>` so the assistant content becomes the
    /// final prefill token(s). The OpenAI-variant path pins this false.
    pub allow_continue_final: bool,
    /// Render thinking-off as a closed, empty think block where the template
    /// would otherwise open its own low-effort reasoning (GLM-5.3 without
    /// tools; the template's `glm_empty_think_off`). Set for `response_format`
    /// requests: thinking off must leave no reasoning, and the grammar must
    /// own the first generated token. OR'd with `ATLAS_GLM_EMPTY_THINK_OFF=1`.
    pub empty_think_off: bool,
}

/// Apply Atlas preprocessing and render the chat template to a string.
///
/// This is the single place production prompt bytes are produced for
/// Jinja-encoded models; both `apply_chat_template_jinja_with_effort` and
/// `apply_chat_template_openai_with_effort` delegate here.
pub(crate) fn render_chat(
    env: &minijinja::Environment<'static>,
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    flags: RenderFlags<'_>,
) -> Result<String> {
    let tmpl = env
        .get_template("chat")
        .context("Failed to get compiled template")?;

    // Atlas cross-cutting preprocessing (F76 arg-parse + autoclose-think
    // + think-control), applied to the model's OWN template so the
    // per-model jinja overrides that used to encode these are no longer
    // required. Inline `<|think_on|>`/`<|think_off|>` tokens, when
    // present, override the caller's `enable_thinking`.
    let (messages_for_render, enable_thinking) =
        preprocess_for_render(messages, flags.enable_thinking);
    let messages_val = minijinja::Value::from_serialize(&messages_for_render);
    let tools_val = tools.map(minijinja::Value::from_serialize);

    // Diagnostic "continue final message" mode (standard convention): see
    // [`RenderFlags::allow_continue_final`].
    let continue_final = flags.allow_continue_final
        && messages
            .last()
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            == Some("assistant");

    // Pass enable_thinking as-is to the template. The Qwen3.5 template uses it
    // to emit <think>\n (thinking) or <think>\n\n</think>\n\n (no thinking).
    // Mistral template uses reasoning_effort instead.
    // The api.rs layer controls enable_thinking based on thinking_in_tools MODEL.toml.
    // Mistral's template defaults `reasoning_effort` to "high" when
    // undefined, so we must explicitly pass "none" to disable thinking.
    //
    // Unset + thinking on ⇒ UNDEFINED: the template's own default tier
    // (see `RenderFlags::reasoning_effort`). The server-level default
    // (`--default-chat-template-kwargs`, MODEL.toml
    // `default_reasoning_effort`) is resolved in api/chat/prepare.rs.
    let reasoning_effort: minijinja::Value = if let Some(effort) = flags.reasoning_effort {
        effort.into()
    } else if enable_thinking && flags.template_effort_default {
        minijinja::Value::UNDEFINED
    } else if enable_thinking {
        minijinja::Value::UNDEFINED
    } else {
        "none".into()
    };
    // Tri-state → UNDEFINED (not `none`!) when unset; see RenderFlags doc.
    let preserve_thinking = flags
        .preserve_thinking
        .map(minijinja::Value::from)
        .unwrap_or(minijinja::Value::UNDEFINED);
    let ctx = minijinja::context! {
        messages => messages_val,
        tools => tools_val.unwrap_or(minijinja::Value::UNDEFINED),
        add_generation_prompt => !continue_final,
        enable_thinking => enable_thinking,
        reasoning_effort => reasoning_effort,
        preserve_thinking => preserve_thinking,
        disable_tool_steering => flags.disable_tool_steering,
        add_vision_id => false,
        glm_empty_think_off => flags.empty_think_off || glm_empty_think_off(),
    };

    let mut rendered = tmpl.render(ctx).map_err(|e| {
        tracing::error!("Jinja template error: {e:#}");
        anyhow::anyhow!("Failed to render Jinja chat template: {e}")
    })?;

    if continue_final {
        // Strip the trailing end-of-turn so the assistant content is the
        // last prefill token (qwen-style templates close with
        // `<|im_end|>\n`). Trim trailing whitespace first, then the marker.
        let trimmed = rendered.trim_end();
        let stripped = trimmed.strip_suffix("<|im_end|>").unwrap_or(trimmed);
        rendered = stripped.to_string();
        tracing::info!("continue_final_message: stripped trailing EOT for prefill A/B");
    }

    Ok(rendered)
}

/// `ATLAS_GLM_EMPTY_THINK_OFF=1`: GLM-5.3 renders thinking-off as an empty
/// `<think></think>` instead of its low-effort reasoning mode.
fn glm_empty_think_off() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_EMPTY_THINK_OFF").as_deref() == Ok("1"))
}
