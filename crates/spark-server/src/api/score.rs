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
//! It is refused (422) on a model whose attention layers would read it
//! causally (MLA, sliding-window, indexer or hyper-connection attention).
//!
//! Knowledge packs (`--score-packs N`): `"pack": [ids]` puts a token block
//! (a policy, a handbook) before the prefix at positions `[0, P)`. The model
//! encodes it once and holds its K/V and SSM state (up to N packs, least
//! recently used evicted first); later requests read it instead of
//! recomputing it. Under `prefix_bidir` the pack is one bidirectional block
//! the state reads and that never reads the state. The response carries
//! `"pack_id"` (and `"pack": "built" | "held"`), which a later request can
//! send instead of the tokens; an id the model no longer holds answers 404,
//! and the client sends the tokens again.
//!
//! Scheduling: a score pass shares the prefill arena, KV pool and SSM slots
//! with chat requests, so it runs only when the scheduler is quiescent (no
//! request decoding, prefilling, admitted, swapped or preempted), one pass
//! per quiescent tick. Under sustained chat traffic score jobs wait; each
//! waits at most `--request-timeout` seconds (504, the job is then dropped).
//! A full backlog answers 503; a model that does not score, 501; a request
//! that cannot fit even one branch per pass fails fast with 422.

use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use super::compact::openai_error_response;
use crate::main_modules::model_host::CurrentModel;
use crate::scheduler::score::{PackRef, ScoreError, ScoreJob};

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
    /// A knowledge pack's tokens (before the prefix).
    #[serde(default)]
    pub pack: Option<Vec<u32>>,
    /// A pack id from an earlier response (instead of `pack`).
    #[serde(default)]
    pub pack_id: Option<String>,
}

/// A validated request: prefix, branches, per-branch options, prefix_bidir,
/// and the pack it continues.
pub type Validated = (
    Vec<u32>,
    Vec<Vec<u32>>,
    Vec<Vec<u32>>,
    bool,
    Option<PackRef>,
);

#[derive(Serialize)]
struct ScoreResponse {
    probs: Vec<Vec<f32>>,
    engine_us: u64,
    queue_us: u64,
    /// Requests that shared this one's forward pass (1 = ran alone).
    batched: usize,
    /// The pack this request ran on (16 hex digits), if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pack_id: Option<String>,
    /// "built" (this request encoded it) or "held".
    #[serde(skip_serializing_if = "Option::is_none")]
    pack: Option<&'static str>,
}

/// Validate and expand a request into (prefix, branches, per-branch options).
pub fn validate(req: ScoreRequest, vocab_size: u32) -> Result<Validated, String> {
    let ScoreRequest {
        prefix,
        branches,
        options,
        attention,
        pack,
        pack_id,
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
    let pack = match (pack, pack_id) {
        (Some(_), Some(_)) => return Err("pack: send the tokens or a pack_id, not both".into()),
        (Some(t), None) if t.is_empty() => return Err("pack: an empty pack".into()),
        (Some(t), None) => Some(PackRef::Tokens(t)),
        (None, Some(id)) => match u64::from_str_radix(&id, 16) {
            Ok(id) => Some(PackRef::Id(id)),
            Err(_) => return Err(format!("pack_id: {id:?} is not a hex pack id")),
        },
        (None, None) => None,
    };
    if prefix_bidir && prefix.is_empty() && pack.is_none() {
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
    let pack_tokens: &[u32] = match &pack {
        Some(PackRef::Tokens(t)) => t,
        _ => &[],
    };
    let all = prefix
        .iter()
        .chain(pack_tokens)
        .chain(branches.iter().flatten())
        .chain(options.iter().flatten());
    if let Some(bad) = all.copied().find(|&t| t >= vocab_size) {
        return Err(format!(
            "token id {bad} out of range: vocab_size is {vocab_size}"
        ));
    }
    Ok((prefix, branches, options, prefix_bidir, pack))
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
    let (prefix, branches, options, prefix_bidir, pack) =
        match validate(req, state.tokenizer.vocab_size()) {
            Ok(v) => v,
            Err(msg) => return openai_error_response(StatusCode::BAD_REQUEST, msg),
        };
    let Some(queue) = state.score_tx.clone() else {
        return openai_error_response(
            StatusCode::NOT_IMPLEMENTED,
            "scoring is not enabled on this server".into(),
        );
    };
    if let Err(msg) = queue.check_fits(&prefix, &branches) {
        return openai_error_response(StatusCode::UNPROCESSABLE_ENTITY, msg);
    }
    if pack.is_some() && queue.packs == 0 {
        return openai_error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "pack: knowledge packs are off on this server (start it with --score-packs N)".into(),
        );
    }
    let (reply, rx) = tokio::sync::oneshot::channel();
    let job = ScoreJob {
        prefix,
        branches,
        options,
        prefix_bidir,
        pack,
        pack_limit: queue.packs,
        enqueued: std::time::Instant::now(),
        reply,
    };
    match queue.tx.try_send(job) {
        Ok(()) => {}
        Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => return busy(),
        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
            return openai_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "scheduler is not running".into(),
            );
        }
    }
    // Dropping `rx` on timeout closes the reply, and the scheduler drops a
    // job whose reply is closed before running it.
    let reply = match state.request_deadline(None) {
        Some(at) => match tokio::time::timeout_at(at.into(), rx).await {
            Ok(r) => r,
            Err(_) => {
                return openai_error_response(
                    StatusCode::GATEWAY_TIMEOUT,
                    format!(
                        "score request timed out after {}s (--request-timeout) waiting \
                         for a quiescent scheduler",
                        state.request_timeout
                    ),
                );
            }
        },
        None => rx.await,
    };
    match reply {
        Ok(Ok(out)) => Json(ScoreResponse {
            probs: out.probs,
            engine_us: out.engine_us,
            queue_us: out.queue_us,
            batched: out.batched,
            pack_id: out.pack.map(|p| format!("{:016x}", p.id)),
            pack: out.pack.map(|p| if p.built { "built" } else { "held" }),
        })
        .into_response(),
        Ok(Err(ScoreError::Busy)) => busy(),
        Ok(Err(ScoreError::UnknownPack(id))) => openai_error_response(
            StatusCode::NOT_FOUND,
            format!("pack_id {id:016x} is not held (evicted or never built); send its tokens"),
        ),
        Ok(Err(ScoreError::Failed(e))) => {
            openai_error_response(StatusCode::UNPROCESSABLE_ENTITY, e)
        }
        Err(_) => openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler dropped the request".into(),
        ),
    }
}

fn busy() -> Response {
    openai_error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        format!(
            "score queue is full ({} jobs pending); retry later",
            crate::scheduler::score::MAX_PENDING_JOBS
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(json: &str) -> ScoreRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn shared_options_expand_per_branch() {
        let (p, b, o, bidir, pack) = validate(
            req(r#"{"prefix":[1,2],"branches":[[3],[4,5]],"options":[7,8]}"#),
            10,
        )
        .unwrap();
        assert!(!bidir);
        assert!(pack.is_none());
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

    #[test]
    fn a_pack_is_tokens_or_an_id_never_both() {
        let t = req(r#"{"prefix":[1],"branches":[[3]],"options":[7],"pack":[4,5]}"#);
        assert_eq!(
            validate(t, 10).unwrap().4,
            Some(PackRef::Tokens(vec![4, 5]))
        );
        let id =
            req(r#"{"prefix":[1],"branches":[[3]],"options":[7],"pack_id":"00000000000000ff"}"#);
        assert_eq!(validate(id, 10).unwrap().4, Some(PackRef::Id(255)));
        let both = req(r#"{"branches":[[3]],"options":[7],"pack":[4],"pack_id":"ff"}"#);
        assert!(validate(both, 10).is_err());
        let state_is_pack =
            req(r#"{"branches":[[3]],"options":[7],"pack":[4],"attention":"prefix_bidir"}"#);
        assert!(validate(state_is_pack, 10).is_ok());
        let empty = req(r#"{"branches":[[3]],"options":[7],"pack":[]}"#);
        assert!(validate(empty, 10).is_err());
        let not_hex = req(r#"{"branches":[[3]],"options":[7],"pack_id":"zz"}"#);
        assert!(validate(not_hex, 10).is_err());
        let out_of_vocab = req(r#"{"branches":[[3]],"options":[7],"pack":[10]}"#);
        assert!(validate(out_of_vocab, 10).is_err());
    }
}
