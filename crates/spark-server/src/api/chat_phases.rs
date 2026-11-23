// SPDX-License-Identifier: AGPL-3.0-only

//! Helper functions extracted from `chat::chat_completions_inner`. These
//! are the F1..F60 cross-turn agentic guards — each one inspects and
//! mutates `req.messages` (and occasionally returns a flag the caller
//! uses for log-line correlation). Keeping them in a sibling file makes
//! `chat.rs` fit the 500-LoC cap and gives each guard a clear top-level
//! seam for future testing.

use axum::http::StatusCode;
use axum::response::Response;

use crate::ir::ChatRequest;

use super::compact::{openai_error_response, openai_error_response_with_param};

/// Validate the input contract on the IR envelope: messages length,
/// max_tokens > 0, sampling ranges, tool_choice mode/name
/// compatibility. Returns `Err(Response)` for fail-fast 400 paths so
/// the caller can `?` directly into a Response.
#[allow(clippy::result_large_err)]
pub(super) fn validate_input(req: &ChatRequest) -> Result<(), Response> {
    if req.messages.is_empty() {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "messages must contain at least one message".into(),
            Some("messages"),
            None,
        ));
    }
    if req.messages.len() > 2048 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "messages array exceeds maximum length (2048)".into(),
            Some("messages"),
            None,
        ));
    }
    if let Some(t) = req.sampling.temperature
        && (!(0.0..=2.0).contains(&t))
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "temperature must be between 0 and 2".into(),
            Some("temperature"),
            None,
        ));
    }
    if let Some(p) = req.sampling.top_p
        && (p <= 0.0 || p > 1.0)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "top_p must be between 0 (exclusive) and 1".into(),
            Some("top_p"),
            None,
        ));
    }
    if let Some(p) = req.sampling.min_p
        && !(0.0..=1.0).contains(&p)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "min_p must be between 0 and 1".into(),
            Some("min_p"),
            None,
        ));
    }
    for (param, value) in [
        ("presence_penalty", req.sampling.presence_penalty),
        ("frequency_penalty", req.sampling.frequency_penalty),
    ] {
        if let Some(value) = value
            && !(-2.0..=2.0).contains(&value)
        {
            return Err(openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                format!("{param} must be between -2 and 2, got {value}"),
                Some(param),
                None,
            ));
        }
    }
    if req.max_tokens == 0 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "max_tokens must be at least 1".into(),
            Some("max_tokens"),
            None,
        ));
    }
    // OpenAI spec bounds n to 1-128; an unbounded n would drive both an
    // attacker-controlled allocation (Vec::with_capacity(n)) and an
    // unbounded sequential-inference loop in the blocking path. Fail fast
    // per spec — mirrors the /v1/completions guard.
    if req.n == 0 || req.n > 128 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            format!("n must be between 1 and 128, got {}", req.n),
            Some("n"),
            None,
        ));
    }
    if let Some(crate::tool_parser::ToolChoice::Mode(ref s)) = req.tool_choice {
        if !["auto", "none", "required"].contains(&s.as_str()) {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "Invalid tool_choice value: '{s}'. Must be 'auto', 'none', 'required', or a function object."
                ),
            ));
        }
        if s == "required" && req.tools.is_empty() {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "tool_choice is 'required' but no tools were provided".into(),
            ));
        }
    }
    if let Some(crate::tool_parser::ToolChoice::Specific { function }) = &req.tool_choice
        && !req
            .tools
            .iter()
            .any(|tool| tool.function.name == function.name)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            format!(
                "tool_choice function '{}' is not among the declared tools",
                function.name
            ),
            Some("tool_choice"),
            None,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn sampling_penalty_ranges() {
        for param in ["presence_penalty", "frequency_penalty"] {
            for (value, valid) in [(-2.1, false), (-2.0, true), (2.0, true), (2.1, false)] {
                let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(json!({
                    "model": "test",
                    "messages": [{"role": "user", "content": "hello"}],
                    (param): value,
                }))
                .unwrap();
                let result = validate_input(&wire.into());
                if valid {
                    assert!(result.is_ok(), "{param}={value} should be accepted");
                } else {
                    let response = result
                        .err()
                        .unwrap_or_else(|| panic!("{param}={value} should be rejected"));
                    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                        .await
                        .unwrap();
                    let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                    assert_eq!(error["error"]["param"], param);
                }
            }
        }
    }

    #[tokio::test]
    async fn sampling_min_p_range() {
        for (value, valid) in [(0.0, true), (1.0, true), (1.5, false), (-0.1, false)] {
            let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "min_p": value,
            }))
            .unwrap();
            let result = validate_input(&wire.into());
            if valid {
                assert!(result.is_ok(), "min_p={value} should be accepted");
            } else {
                let response = result
                    .err()
                    .unwrap_or_else(|| panic!("min_p={value} should be rejected"));
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(error["error"]["param"], "min_p");
            }
        }
    }

    fn named_tool_request(tools: Option<serde_json::Value>) -> ChatRequest {
        let mut body = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}}
        });
        if let Some(tools) = tools {
            body["tools"] = tools;
        }
        let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(body).unwrap();
        wire.into()
    }

    #[tokio::test]
    async fn named_tool_choice_rejects_undeclared_function() {
        for tools in [
            None,
            Some(json!([])),
            Some(json!([{"type": "function", "function": {"name": "other"}}])),
        ] {
            let req = named_tool_request(tools);
            let response = validate_input(&req).expect_err("undeclared tool must be rejected");
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["error"]["param"], "tool_choice");
        }
    }

    #[test]
    fn named_tool_choice_accepts_declared_function() {
        let req = named_tool_request(Some(json!([
            {"type": "function", "function": {"name": "other"}},
            {"type": "function", "function": {"name": "lookup"}}
        ])));
        assert!(validate_input(&req).is_ok());
    }
}
