// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use crate::main_modules::model_host::ProcessState;
use axum::RequestExt;
use axum::body::{Body, to_bytes};
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use super::chat_stream::run_chat_stream;
use super::responses_stream::responses_endpoint_stream;
use super::responses_translate::{
    build_responses_usage, emit, find_frame_end, translate_chat_response_to_responses,
};
use super::stored::assistant_incoming_from_ir;
use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

// Sibling-cluster items hoisted from the original `api.rs`. These uses
// give every sub-file access to helpers that the un-split file took for
// granted via single-module visibility.
use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::inference_impl::{extract_thinking, strip_stop_sequences, tokenize_stop_sequences};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};
use super::strip::strip_thinking_tags;

// Re-export sibling helpers via crate::api::* for short paths.
use super::inference_types::*;
use super::sanitizer::*;

#[derive(Default, serde::Deserialize)]
pub struct CreateConversationRequest {
    #[serde(default)]
    pub items: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

#[derive(serde::Deserialize)]
pub struct UpdateConversationRequest {
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(serde::Deserialize)]
pub struct AddItemsRequest {
    pub items: Vec<serde_json::Value>,
}

/// Build the public JSON shape for a conversation snapshot.
pub(super) fn conversation_body(
    snap: &crate::conversation_store::ConversationSnapshot,
) -> serde_json::Value {
    serde_json::json!({
        "id": snap.id,
        "object": "conversation",
        "created_at": snap.created_at,
        "metadata": snap.metadata,
    })
}

/// POST /v1/conversations — create a conversation with optional
/// initial items + metadata.
pub async fn create_conversation(
    ProcessState(process): ProcessState,
    request: Request,
) -> Response {
    // Enforce the route's body limit before checking whether the body is empty.
    let (parts, body) = request.with_limited_body().into_parts();
    let bytes = match to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Unable to read request body: {e}"),
            );
        }
    };
    let req = if bytes.is_empty() {
        CreateConversationRequest::default()
    } else {
        let request = Request::from_parts(parts, Body::from(bytes));
        match Json::<CreateConversationRequest>::from_request(request, &()).await {
            Ok(Json(req)) => req,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Invalid request JSON: {e}"),
                );
            }
        }
    };
    let items = req.items.unwrap_or_default();
    if items.len() > crate::conversation_store::MAX_ITEMS_PER_INSERT {
        return openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            format!(
                "`items` exceeds per-call cap of {} (got {}).",
                crate::conversation_store::MAX_ITEMS_PER_INSERT,
                items.len(),
            ),
            Some("items"),
            Some("items_too_many"),
        );
    }
    let id = process
        .conversation_store
        .create(items, req.metadata.unwrap_or_default());
    let snap = process.conversation_store.get(&id).expect("just created");
    Json(conversation_body(&snap)).into_response()
}

/// GET /v1/conversations/{id}
pub async fn get_conversation(
    ProcessState(process): ProcessState,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match process.conversation_store.get(&id) {
        Some(snap) => Json(conversation_body(&snap)).into_response(),
        None => openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        ),
    }
}

/// POST /v1/conversations/{id} — update metadata.
pub async fn update_conversation(
    ProcessState(process): ProcessState,
    axum::extract::Path(id): axum::extract::Path<String>,
    req: Result<Json<UpdateConversationRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    match process
        .conversation_store
        .update_metadata(&id, req.metadata)
    {
        Some(snap) => Json(conversation_body(&snap)).into_response(),
        None => openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        ),
    }
}

/// DELETE /v1/conversations/{id}
pub async fn delete_conversation(
    ProcessState(process): ProcessState,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if process.conversation_store.delete(&id) {
        Json(serde_json::json!({
            "id": id,
            "object": "conversation.deleted",
            "deleted": true,
        }))
        .into_response()
    } else {
        openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        )
    }
}

/// POST /v1/conversations/{id}/items — append items (≤20/call).
pub async fn add_conversation_items(
    ProcessState(process): ProcessState,
    axum::extract::Path(id): axum::extract::Path<String>,
    req: Result<Json<AddItemsRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    match process.conversation_store.add_items(&id, req.items) {
        Ok(items) => Json(serde_json::json!({
            "object": "list",
            "data": items,
        }))
        .into_response(),
        Err(crate::conversation_store::AddItemsError::NotFound) => {
            openai_error_response_with_param(
                StatusCode::NOT_FOUND,
                format!("Conversation '{id}' not found."),
                Some("id"),
                Some("conversation_not_found"),
            )
        }
        Err(crate::conversation_store::AddItemsError::TooMany(n)) => {
            openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                format!(
                    "`items` exceeds per-call cap of {} (got {n}).",
                    crate::conversation_store::MAX_ITEMS_PER_INSERT,
                ),
                Some("items"),
                Some("items_too_many"),
            )
        }
    }
}

/// GET /v1/conversations/{id}/items — list items with `after`, `limit` + `order`
/// query parameters (OpenAI spec: default 20, max 100, order=asc).
pub async fn list_conversation_items(
    ProcessState(process): ProcessState,
    axum::extract::Path(id): axum::extract::Path<String>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let Some(snap) = process.conversation_store.get(&id) else {
        return openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        );
    };
    let mut items = snap.items;
    let order = q.get("order").map(|s| s.as_str()).unwrap_or("asc");
    if order == "desc" {
        items.reverse();
    }
    if let Some(after) = q.get("after") {
        let Some(index) = items
            .iter()
            .position(|item| item.get("id").and_then(|id| id.as_str()) == Some(after.as_str()))
        else {
            return openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                format!("Item '{after}' not found in conversation '{id}'."),
                Some("after"),
                Some("item_not_found"),
            );
        };
        items.drain(..=index);
    }
    let limit: usize = q
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20)
        .min(100);
    let has_more = items.len() > limit;
    items.truncate(limit);
    let first_id = items
        .first()
        .and_then(|v| v.get("id").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();
    let last_id = items
        .last()
        .and_then(|v| v.get("id").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();
    Json(serde_json::json!({
        "object": "list",
        "data": items,
        "first_id": first_id,
        "last_id": last_id,
        "has_more": has_more,
    }))
    .into_response()
}

/// GET /v1/conversations/{id}/items/{item_id}
pub async fn get_conversation_item(
    ProcessState(process): ProcessState,
    axum::extract::Path((id, item_id)): axum::extract::Path<(String, String)>,
) -> Response {
    let Some(snap) = process.conversation_store.get(&id) else {
        return openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        );
    };
    for it in &snap.items {
        if it.get("id").and_then(|v| v.as_str()) == Some(item_id.as_str()) {
            return Json(it.clone()).into_response();
        }
    }
    openai_error_response_with_param(
        StatusCode::NOT_FOUND,
        format!("Item '{item_id}' not found in conversation '{id}'."),
        Some("item_id"),
        Some("item_not_found"),
    )
}

/// DELETE /v1/conversations/{id}/items/{item_id}
pub async fn delete_conversation_item(
    ProcessState(process): ProcessState,
    axum::extract::Path((id, item_id)): axum::extract::Path<(String, String)>,
) -> Response {
    if process.conversation_store.remove_item(&id, &item_id) {
        Json(serde_json::json!({
            "id": item_id,
            "object": "conversation.item.deleted",
            "deleted": true,
        }))
        .into_response()
    } else {
        openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Item '{item_id}' not found in conversation '{id}'."),
            Some("item_id"),
            Some("item_not_found"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::create_conversation;
    use crate::conversation_store::ConversationStore;
    use crate::main_modules::{model_host::ModelHost, serve_load::Carried};
    use axum::body::{Body, to_bytes};
    use axum::extract::DefaultBodyLimit;
    use axum::http::{Request, StatusCode, header::CONTENT_TYPE};
    use axum::{Router, routing::post};
    use serde_json::{Value, json};
    use std::{sync::Arc, time::Duration};
    use tower::ServiceExt;

    fn app() -> (Router, Arc<ConversationStore>) {
        let ttl = Duration::from_secs(60);
        let store = ConversationStore::with_config(10, ttl);
        let host = ModelHost::empty();
        host.set_process(Carried {
            response_store: crate::response_store::ResponseStore::with_config(10, ttl),
            rate_limiter: crate::rate_limiter::RateLimiter::from_env(),
            conversation_store: store.clone(),
        });
        let app = Router::new()
            .route("/v1/conversations", post(create_conversation))
            .layer(DefaultBodyLimit::max(128))
            .with_state(Arc::new(host));
        (app, store)
    }

    async fn create(app: Router, content_type: Option<&str>, body: Body) -> (StatusCode, Value) {
        let mut request = Request::post("/v1/conversations");
        if let Some(content_type) = content_type {
            request = request.header(CONTENT_TYPE, content_type);
        }
        let response = app.oneshot(request.body(body).unwrap()).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 8192).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    async fn assert_rejected(content_type: Option<&str>, body: Body) {
        let (app, store) = app();
        let (status, response) = create(app, content_type, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response["error"]["type"], "invalid_request_error");
        assert_eq!(store.len(), 0);
    }

    #[tokio::test]
    async fn create_rejects_invalid_json_syntax() {
        assert_rejected(Some("application/json"), Body::from(r#"{"items":"#)).await;
    }

    #[tokio::test]
    async fn create_rejects_invalid_items_type() {
        assert_rejected(Some("application/json"), Body::from(r#"{"items":"x"}"#)).await;
    }

    #[tokio::test]
    async fn create_rejects_nonempty_body_without_json_content_type() {
        for content_type in [None, Some("text/plain")] {
            assert_rejected(content_type, Body::from("{}")).await;
        }
    }

    #[tokio::test]
    async fn create_rejects_body_over_route_limit() {
        let body = format!(r#"{{"metadata":{{"key":"{}"}}}}"#, "x".repeat(128));
        assert_rejected(Some("application/json"), Body::from(body)).await;
    }

    #[tokio::test]
    async fn create_allows_empty_body_with_or_without_json_content_type() {
        let (app, store) = app();
        for content_type in [None, Some("application/json")] {
            let (status, response) = create(app.clone(), content_type, Body::empty()).await;
            assert_eq!(status, StatusCode::OK);
            let saved = store.get(response["id"].as_str().unwrap()).unwrap();
            assert!(saved.items.is_empty());
            assert!(saved.metadata.is_empty());
        }
    }

    #[tokio::test]
    async fn create_persists_valid_items_and_metadata() {
        let (app, store) = app();
        let body = Body::from(r#"{"items":[{"type":"message"}],"metadata":{"key":"value"}}"#);
        let (status, response) = create(app, Some("application/json"), body).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(response["metadata"], json!({"key": "value"}));
        let saved = store.get(response["id"].as_str().unwrap()).unwrap();
        assert_eq!(saved.items.len(), 1);
        assert_eq!(saved.items[0]["type"], "message");
    }
}
