// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/score`: prefill-only option scoring.
//!
//! ```json
//! {"prefix": [ids], "branches": [[ids], ...], "options": [ids] | [[ids], ...]}
//! ```
//! returns `{"probs": [[p, ...], ...], "engine_us": n}`: per branch, the
//! softmax over its option tokens at the branch's last position. `prefix` is
//! read once and every branch continues it, all in one forward pass
//! (`spark_model::traits::Model::score_branches`). A flat `options` list
//! applies to every branch. `"attention": "prefix_bidir"` reads the prefix
//! as a bidirectional state block and each branch as a bidirectional
//! question block on it (full-attention layers only; the default is causal).

use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use super::compact::openai_error_response;
use crate::main_modules::model_host::CurrentModel;
use crate::scheduler::score::ScoreJob;

/// Upper bounds per request (branches and options per branch).
pub const MAX_BRANCHES: usize = 256;
pub const MAX_OPTIONS: usize = 256;

#[derive(Deserialize)]
#[serde(untagged)]
pub enum ScoreOptions {
    Shared(Vec<u32>),
    PerBranch(Vec<Vec<u32>>),
}

#[derive(Deserialize)]
pub struct ScoreRequest {
    #[serde(default)]
    pub prefix: Vec<u32>,
    pub branches: Vec<Vec<u32>>,
    pub options: ScoreOptions,
    /// "causal" (default) or "prefix_bidir".
    #[serde(default)]
    pub attention: Option<String>,
}

/// A validated request: prefix, branches, per-branch options, prefix_bidir.
pub type Validated = (Vec<u32>, Vec<Vec<u32>>, Vec<Vec<u32>>, bool);

#[derive(Serialize)]
struct ScoreResponse {
    probs: Vec<Vec<f32>>,
    engine_us: u64,
    queue_us: u64,
    /// Requests that shared this one's forward pass (1 = ran alone).
    batched: usize,
}

/// Validate and expand a request into (prefix, branches, per-branch options).
pub fn validate(req: ScoreRequest, vocab_size: u32) -> Result<Validated, String> {
    let ScoreRequest {
        prefix,
        branches,
        options,
        attention,
    } = req;
    let prefix_bidir = match attention.as_deref() {
        None | Some("causal") => false,
        Some("prefix_bidir") => true,
        Some(other) => {
            return Err(format!(
                "attention: {other:?} is not causal or prefix_bidir"
            ));
        }
    };
    if prefix_bidir && prefix.is_empty() {
        return Err("attention: prefix_bidir needs a non-empty prefix (the state block)".into());
    }
    if branches.is_empty() || branches.len() > MAX_BRANCHES {
        return Err(format!("branches: need 1..={MAX_BRANCHES}"));
    }
    if branches.iter().any(|b| b.is_empty()) {
        return Err("branches: every branch needs at least one token".into());
    }
    let options = match options {
        ScoreOptions::Shared(o) => vec![o; branches.len()],
        ScoreOptions::PerBranch(v) => v,
    };
    if options.len() != branches.len() {
        return Err("options: one option list per branch".into());
    }
    if options
        .iter()
        .any(|o| o.is_empty() || o.len() > MAX_OPTIONS)
    {
        return Err(format!("options: need 1..={MAX_OPTIONS} per branch"));
    }
    let all = prefix
        .iter()
        .chain(branches.iter().flatten())
        .chain(options.iter().flatten());
    if let Some(bad) = all.copied().find(|&t| t >= vocab_size) {
        return Err(format!(
            "token id {bad} out of range: vocab_size is {vocab_size}"
        ));
    }
    Ok((prefix, branches, options, prefix_bidir))
}

pub async fn score(
    CurrentModel(state): CurrentModel,
    req: Result<Json<ScoreRequest>, JsonRejection>,
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
    let (prefix, branches, options, prefix_bidir) =
        match validate(req, state.tokenizer.vocab_size()) {
            Ok(v) => v,
            Err(msg) => return openai_error_response(StatusCode::BAD_REQUEST, msg),
        };
    let Some(tx) = state.score_tx.clone() else {
        return openai_error_response(
            StatusCode::NOT_IMPLEMENTED,
            "scoring is not enabled on this server".into(),
        );
    };
    let (reply, rx) = tokio::sync::oneshot::channel();
    let job = ScoreJob {
        prefix,
        branches,
        options,
        prefix_bidir,
        enqueued: std::time::Instant::now(),
        reply,
    };
    if tx.send(job).await.is_err() {
        return openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler is not running".into(),
        );
    }
    match rx.await {
        Ok(Ok(out)) => Json(ScoreResponse {
            probs: out.probs,
            engine_us: out.engine_us,
            queue_us: out.queue_us,
            batched: out.batched,
        })
        .into_response(),
        Ok(Err(e)) => openai_error_response(StatusCode::UNPROCESSABLE_ENTITY, e),
        Err(_) => openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler dropped the request".into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(json: &str) -> ScoreRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn shared_options_expand_per_branch() {
        let (p, b, o, bidir) = validate(
            req(r#"{"prefix":[1,2],"branches":[[3],[4,5]],"options":[7,8]}"#),
            10,
        )
        .unwrap();
        assert!(!bidir);
        assert_eq!(p, vec![1, 2]);
        assert_eq!(b.len(), 2);
        assert_eq!(o, vec![vec![7, 8], vec![7, 8]]);
    }

    #[test]
    fn per_branch_options_must_match_branches() {
        let r = req(r#"{"branches":[[3],[4]],"options":[[7]]}"#);
        assert!(validate(r, 10).is_err());
        let r = req(r#"{"branches":[[3],[4]],"options":[[7],[8,9]]}"#);
        assert!(validate(r, 10).is_ok());
    }

    #[test]
    fn rejects_empty_and_out_of_vocab() {
        assert!(validate(req(r#"{"branches":[],"options":[1]}"#), 10).is_err());
        assert!(validate(req(r#"{"branches":[[]],"options":[1]}"#), 10).is_err());
        assert!(validate(req(r#"{"branches":[[1]],"options":[]}"#), 10).is_err());
        assert!(validate(req(r#"{"branches":[[1]],"options":[10]}"#), 10).is_err());
        assert!(validate(req(r#"{"prefix":[11],"branches":[[1]],"options":[2]}"#), 10).is_err());
    }

    #[test]
    fn attention_mode_is_validated() {
        let ok = req(r#"{"prefix":[1],"branches":[[3]],"options":[7],"attention":"prefix_bidir"}"#);
        assert!(validate(ok, 10).unwrap().3);
        let empty = req(r#"{"branches":[[3]],"options":[7],"attention":"prefix_bidir"}"#);
        assert!(validate(empty, 10).is_err());
        let bad = req(r#"{"prefix":[1],"branches":[[3]],"options":[7],"attention":"full"}"#);
        assert!(validate(bad, 10).is_err());
    }
}
