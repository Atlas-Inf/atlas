// SPDX-License-Identifier: AGPL-3.0-only

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::tool_parser;

use super::types::*;

pub(super) fn anthropic_error(status: StatusCode, error_type: &str, message: String) -> Response {
    (status, Json(anthropic_error_body(error_type, &message))).into_response()
}

pub(super) fn anthropic_error_body(error_type: &str, message: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        }
    })
}

/// Anthropic's `error.type` for an HTTP status (its documented error kinds).
/// Enntity/atlas 060199eb.
pub(super) fn anthropic_error_type(status: StatusCode) -> &'static str {
    match status.as_u16() {
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        400..=499 => "invalid_request_error",
        _ => "api_error",
    }
}

// ── Conversion helpers ──

impl From<&AnthropicTool> for tool_parser::ToolDefinition {
    /// Convert an Anthropic tool to an OpenAI-compatible tool definition.
    fn from(t: &AnthropicTool) -> Self {
        tool_parser::ToolDefinition {
            tool_type: "function".to_string(),
            function: tool_parser::FunctionDefinition {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: Some(t.input_schema.clone()),
            },
        }
    }
}

impl From<&AnthropicToolChoice> for tool_parser::ToolChoice {
    /// Convert an Anthropic tool_choice to an OpenAI-compatible tool_choice.
    fn from(tc: &AnthropicToolChoice) -> Self {
        match tc.choice_type {
            ToolChoiceType::Any => tool_parser::ToolChoice::Mode("required".to_string()),
            ToolChoiceType::Auto => tool_parser::ToolChoice::Mode("auto".to_string()),
            ToolChoiceType::None => tool_parser::ToolChoice::Mode("none".to_string()),
            ToolChoiceType::Tool => {
                if let Some(ref name) = tc.name {
                    tool_parser::ToolChoice::Specific {
                        function: tool_parser::ToolChoiceFunction { name: name.clone() },
                    }
                } else {
                    tool_parser::ToolChoice::Mode("auto".to_string())
                }
            }
        }
    }
}

/// Convert an OpenAI finish reason to Anthropic's stop_reason string.
pub(super) fn convert_stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "stop" => "end_turn",
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        // Safety-filtered output maps to Anthropic's dedicated refusal
        // stop reason (2025-05 API), not a normal end_turn — clients
        // branch on this to avoid retrying verbatim.
        "content_filter" => "refusal",
        // The Anthropic vocabulary has no deadline reason. "max_tokens" is
        // its only "the output is truncated" slot — mapping a deadline cut
        // to the default "end_turn" would tell the client the turn is
        // COMPLETE, which is the exact silent-truncation bug this reason
        // exists to prevent.
        crate::ir::FINISH_REASON_TIMEOUT => "max_tokens",
        _ => "end_turn",
    }
}
