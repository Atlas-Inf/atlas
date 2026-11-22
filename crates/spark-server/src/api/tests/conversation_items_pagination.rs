// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
    routing::get,
};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::main_modules::{model_host::ModelHost, serve_load::Carried};

fn fixture(count: usize) -> (Router, String, Vec<String>) {
    let carried = Carried::from_env();
    let id = carried.conversation_store.create(
        (0..count)
            .map(|i| json!({"type": "message", "role": "user", "content": i.to_string()}))
            .collect(),
        Default::default(),
    );
    let ids = carried
        .conversation_store
        .get(&id)
        .unwrap()
        .items
        .iter()
        .map(|item| item["id"].as_str().unwrap().to_string())
        .collect();
    let host = Arc::new(ModelHost::empty());
    host.set_process(carried);
    let app = Router::new()
        .route(
            "/v1/conversations/{id}/items",
            get(crate::api::list_conversation_items),
        )
        .with_state(host);
    (app, id, ids)
}

async fn page(app: Router, id: &str, query: &str) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/conversations/{id}/items?{query}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

#[tokio::test]
async fn limit_one_has_more() {
    let (app, id, ids) = fixture(3);
    let (status, body) = page(app, &id, "limit=1").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert_eq!(body["first_id"], ids[0]);
    assert_eq!(body["last_id"], ids[0]);
    assert_eq!(body["has_more"], true);
}

#[tokio::test]
async fn after_first_returns_second() {
    let (app, id, ids) = fixture(3);
    let (_, body) = page(app, &id, &format!("limit=1&after={}", ids[0])).await;
    assert_eq!(body["data"][0]["id"], ids[1]);
    assert_eq!(body["first_id"], ids[1]);
    assert_eq!(body["last_id"], ids[1]);
    assert_eq!(body["has_more"], true);
}

#[tokio::test]
async fn last_page_has_no_more() {
    let (app, id, ids) = fixture(3);
    let (_, body) = page(app.clone(), &id, &format!("limit=1&after={}", ids[1])).await;
    assert_eq!(body["data"][0]["id"], ids[2]);
    assert_eq!(body["has_more"], false);
    let (_, body) = page(app, &id, &format!("after={}", ids[2])).await;
    assert_eq!(body["data"], json!([]));
    assert_eq!(body["has_more"], false);
}

#[tokio::test]
async fn descending_after_follows_requested_order() {
    let (app, id, ids) = fixture(3);
    let (_, body) = page(app, &id, &format!("order=desc&limit=1&after={}", ids[2])).await;
    assert_eq!(body["data"][0]["id"], ids[1]);
    assert_eq!(body["has_more"], true);
}

#[tokio::test]
async fn unknown_after_is_invalid_request() {
    let (app, id, _) = fixture(3);
    let (status, body) = page(app, &id, "after=missing").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "after");
}

#[tokio::test]
async fn empty_conversation_has_no_more() {
    let (app, id, _) = fixture(0);
    let (_, body) = page(app, &id, "").await;
    assert_eq!(body["data"], json!([]));
    assert_eq!(body["has_more"], false);
}
