// SPDX-License-Identifier: AGPL-3.0-only
//
// Chat-template rendering: build the JSON-message array, run the
// optional auto-compact, apply the Jinja template, expand image
// pad-tokens, and detect template-forced thinking.
//
// Lifted out of `chat::chat_completions_inner` (wave 4g).

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::AppState;
use crate::tokenizer::{Glm5VisionPlaceholder, QwenVisionPlaceholder};

use super::super::compact::{compact_messages, openai_error_response};
use super::msg_entry::{MsgContentPart, MsgEntry};

/// Outputs of [`render_template`]. Threaded into the streaming /
/// blocking dispatch.
pub(super) struct TemplateOut {
    pub(super) prompt_tokens: Vec<u32>,
    /// Possibly overridden by template-forced-thinking detection.
    pub(super) enable_thinking: bool,
    pub(super) thinking_budget: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
pub(super) fn render_template(
    state: &Arc<AppState>,
    tools: &[crate::tool_parser::ToolDefinition],
    messages: &[MsgEntry],
    image_pad_counts: &[usize],
    vision_placeholders: &[Glm5VisionPlaceholder],
    qwen_vision_placeholders: &[QwenVisionPlaceholder],
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    reasoning_effort: Option<crate::ir::ReasoningEffort>,
    preserve_thinking: Option<bool>,
    tools_active: bool,
) -> Result<TemplateOut, Response> {
    // Use closed thinking when client doesn't explicitly enable it.
    let template_thinking = enable_thinking;
    let reasoning_effort = reasoning_effort.map(if state.behavior.verbatim_reasoning_effort {
        crate::ir::ReasoningEffort::wire_str
    } else {
        crate::ir::ReasoningEffort::as_str
    });

    // Build JSON messages with structured tool_calls for Jinja.
    let mut json_messages =
        build_json_messages_for(messages, state.tokenizer.tool_call_ids_in_template);
    // When TSCG is enabled the parser's `system_prompt()` has already
    // placed the compact tool signatures into messages[0]; passing
    // `tools` to Jinja as well would re-render the full JSON schema and
    // defeat the compaction. Pass `None` so the template's `{% if tools
    // %}` branch falls through — the tool-call format instructions
    // still come from `system_prompt()`. A verbatim-prompt model renders
    // the request's tools whenever it has any, `tool_choice: "none"`
    // included, as the reference server does.
    let render_tools = if state.chat.verbatim_prompt {
        !tools.is_empty()
    } else {
        tools_active && (state.tokenizer.uses_deepseek_v4_encoding() || !state.chat.prompt.tscg)
    };
    let mut jinja_tools: Option<Vec<serde_json::Value>> = if render_tools {
        Some(
            tools
                .iter()
                .map(|t| serde_json::to_value(t).unwrap_or_default())
                .collect(),
        )
    } else {
        None
    };

    // GLM-5 registers its tool-envelope XML as added tokens. Escape matching
    // markup in incoming message prose before rendering so it remains text;
    // structured call envelopes are still emitted by the template as native
    // calls; literal argument values and schema descriptions remain text.
    let escaped_tool_markup = state.tokenizer.escape_glm_tool_markup(
        json_messages
            .iter_mut()
            .chain(jinja_tools.iter_mut().flatten()),
    );
    if escaped_tool_markup > 0 {
        tracing::info!("{escaped_tool_markup} GLM tool markers in message text encoded as text");
    }

    // In a request with media only the template's own vision markers may
    // become placeholders: markers written in message text encode as text
    // (`tokenizer::vision_markup`).
    if !image_pad_counts.is_empty() || !vision_placeholders.is_empty() {
        let escaped = state.tokenizer.escape_vision_markup(
            json_messages
                .iter_mut()
                .chain(jinja_tools.iter_mut().flatten()),
        );
        if escaped > 0 {
            tracing::info!("{escaped} vision markers in message text encoded as text");
        }
    }

    // Progressive auto-compact (DISABLED BY DEFAULT 2026-04-25 —
    // see project_no_auto_compaction memory feedback).
    let auto_compact_active = state
        .auto_compact_threshold
        .map(|t| t > 0.0)
        .unwrap_or(false);
    let json_messages = if auto_compact_active && json_messages.len() > 4 {
        let trial_tokens = state
            .tokenizer
            .apply_chat_template_openai_with_effort(
                &json_messages,
                jinja_tools.as_deref(),
                template_thinking,
                state.behavior.disable_tool_steering,
                reasoning_effort,
                preserve_thinking,
            )
            .map(|t| t.len())
            .unwrap_or(0);
        if trial_tokens > (state.max_seq_len as f32 * 0.70) as usize {
            compact_messages(&json_messages, trial_tokens, state.max_seq_len)
        } else {
            json_messages
        }
    } else {
        json_messages
    };

    let prompt_tokens = match state.tokenizer.apply_chat_template_openai_with_effort(
        &json_messages,
        jinja_tools.as_deref(),
        template_thinking,
        state.behavior.disable_tool_steering,
        reasoning_effort,
        preserve_thinking,
    ) {
        Ok(t) => t,
        Err(e) => {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Tokenization error: {e}"),
            ));
        }
    };

    // GLM-5's shipped template intentionally emits canonical compact marker
    // triples. Its checkpoint processor leaves the rendered text unchanged;
    // expand those triples here so every image/video encoder row has a
    // matching input token. Other model families keep the legacy pad-only
    // expansion below.
    let prompt_tokens = if !vision_placeholders.is_empty() {
        // Both placeholder families share the rendered-triple contract; the
        // flag on the first entry picks the checkpoint's expansion family
        // (homogeneous per request).
        let expanded = if vision_placeholders.iter().any(|p| p.qwen3vl) {
            state
                .tokenizer
                .expand_qwen3vl_vision_placeholders(prompt_tokens, vision_placeholders)
        } else {
            state
                .tokenizer
                .expand_glm5_vision_placeholders(prompt_tokens, vision_placeholders)
        };
        match expanded {
            Ok(tokens) => tokens,
            Err(e) => {
                return Err(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("vision placeholder error: {e}"),
                ));
            }
        }
    } else if !qwen_vision_placeholders.is_empty() {
        match state
            .tokenizer
            .expand_qwen_vision_placeholders(prompt_tokens, qwen_vision_placeholders)
        {
            Ok(tokens) => tokens,
            Err(e) => {
                return Err(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Qwen vision placeholder error: {e}"),
                ));
            }
        }
    } else if image_pad_counts.iter().any(|&c| c > 1) {
        state
            .tokenizer
            .expand_vision_pads(prompt_tokens, image_pad_counts)
    } else {
        prompt_tokens
    };

    let (enable_thinking, thinking_budget) = reconcile_prompt_thinking(
        &prompt_tokens,
        state.think_start_token_id,
        state.think_end_token_id,
        enable_thinking,
        thinking_budget,
        state.behavior.max_thinking_budget,
    );

    Ok(TemplateOut {
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    })
}
#[path = "template/reconcile.rs"]
mod reconcile;
use reconcile::reconcile_prompt_thinking;

#[cfg(test)]
#[path = "template_thinking_tests.rs"]
mod thinking_tests;

/// Build the Jinja-facing JSON message array from the processed
/// [`MsgEntry`] vec. Pure (no tokenizer/state) so it can be
/// characterization-tested directly:
///   * no media → `content` is a plain string (all text parts joined),
///   * media    → `content` is the message's parts in order: one marker
///     per media item (`{"type": "image"}` / `{"type": "video"}`) and one
///     `{"type": "text", "text": ...}` per text part, empty text omitted,
///   * `tool_calls` / `reasoning_content` attached when present.
///
/// Roles arrive canonical (`developer` → `system` normalization happens
/// at MsgEntry build time).
pub(super) fn build_json_messages(messages: &[MsgEntry]) -> Vec<serde_json::Value> {
    build_json_messages_for(messages, false)
}

fn build_json_messages_for(
    messages: &[MsgEntry],
    include_tool_call_ids: bool,
) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|m| {
            let has_media = m
                .parts
                .iter()
                .any(|p| matches!(p, MsgContentPart::Media(_)));
            let content_val = if has_media {
                let mut items: Vec<serde_json::Value> = Vec::with_capacity(m.parts.len());
                // The bundled Qwen3-VL template walks this array left to
                // right and renders `{"type": "video"}` as
                // `<|vision_start|><|video_pad|><|vision_end|>`, so this
                // order is the order the model reads. It must match the
                // encoder items and pad counts collected from the same
                // IR content sequence.
                for part in &m.parts {
                    match part {
                        MsgContentPart::Media(crate::ir::MediaKind::Image) => {
                            items.push(serde_json::json!({"type": "image"}));
                        }
                        MsgContentPart::Media(crate::ir::MediaKind::Video) => {
                            items.push(serde_json::json!({"type": "video"}));
                        }
                        MsgContentPart::Text(text) if !text.is_empty() => {
                            items.push(serde_json::json!({"type": "text", "text": text}));
                        }
                        MsgContentPart::Text(_) => {}
                    }
                }
                serde_json::Value::Array(items)
            } else {
                serde_json::Value::String(m.text())
            };
            // `developer` → `system` normalization happens upstream at
            // MsgEntry build time (msg_entry.rs), so the role arrives
            // canonical here.
            let mut msg = serde_json::json!({"role": m.role, "content": content_val});
            if let Some(ref tcs) = m.tool_calls {
                msg["tool_calls"] = serde_json::Value::Array(tcs.clone());
            }
            if include_tool_call_ids && let Some(ref id) = m.tool_call_id {
                msg["tool_call_id"] = serde_json::Value::String(id.clone());
            }
            // F1: forward historical reasoning trace to the Jinja
            // template. The template at qwen3_5_moe.jinja:90-104
            // reads `message.reasoning_content` and rehydrates the
            // `<think>` block for the historical assistant turns.
            // Without this, every historical assistant message
            // rendered an empty `<think>\n\n</think>\n\n` wrapper
            // (empty-think poisoning, → premature `<|im_end|>`).
            if let Some(ref rc) = m.reasoning_content {
                msg["reasoning_content"] = serde_json::Value::String(rc.clone());
            }
            msg
        })
        .collect()
}

#[cfg(test)]
#[path = "template/json_message_tests.rs"]
mod json_message_tests;
