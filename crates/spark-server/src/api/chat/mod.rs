// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

//! `/v1/chat/completions` orchestrator.
//!
//! Wave-4g extraction (2026-05-03): the original 1121-LoC `chat.rs`
//! held one async fn (`chat_completions_inner`) where every phase
//! shared a function-local `MsgEntry` struct + ~25 carry-through
//! locals. This module now coordinates:
//!
//! - `msg_entry`      — `MsgEntry` + `build_msg_entries` (req →
//!                      tokenisable shape, image preprocessing,
//!                      cwd extraction)
//! - `loop_detect`    — generic loop / spinning detection +
//!                      task-pin re-anchor
//! - `thinking`       — `(enable_thinking, thinking_budget)`
//!                      resolution
//! - `template`       — JSON-message build, auto-compact,
//!                      Jinja apply, image-pad expand,
//!                      template-forced-thinking detection
//! - `sampling_setup` — preset / penalty / stop-token / grammar /
//!                      timeout / logprobs resolution

pub(crate) mod echo;
pub(crate) mod levers;
mod loop_detect;
#[cfg(test)]
#[path = "loop_detect_tests.rs"]
mod loop_detect_tests;
mod msg_entry;
pub(crate) mod prepare;
#[cfg(test)]
#[path = "reference_prompt_tests.rs"]
mod reference_prompt_tests;
pub(crate) mod remote_image;
mod sampling_setup;
mod template;
#[cfg(test)]
mod text_only_tests;
mod thinking;
pub(crate) mod vision_admit;

use crate::main_modules::model_host::CurrentModel;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;

use crate::AppState;

pub(crate) use echo::ResponseEcho;

/// Result of the shared chat pipeline. A non-streaming success carries
/// the canonical response IR for the caller's surface encoder;
/// streaming SSE and error envelopes are already-complete HTTP
/// responses (streaming moves onto the delta IR next).
pub(crate) enum ChatOutcome {
    Blocking(Box<crate::ir::ChatResponse>),
    /// Streaming success: the neutral delta stream; each surface runs
    /// its own SSE encoder over it.
    Streaming(crate::ir::DeltaStream),
    Http(Response),
}

/// Test-only accessors: cross-module tests (the Anthropic adapter's
/// rendered-prompt golden) drive the IR → MsgEntry → template-JSON
/// path without an AppState.
#[cfg(test)]
#[allow(clippy::result_large_err)]
pub(crate) fn test_build_msg_entries(
    input: &[crate::ir::Message],
    tools_active: bool,
) -> Result<Vec<msg_entry::MsgEntry>, axum::response::Response> {
    msg_entry::build_msg_entries(
        crate::api::chat::vision_admit::AdmitCtx::default(),
        None,
        None,
        &remote_image::RemoteImagePolicy::default(),
        &msg_entry::VideoDecode {
            ffmpeg: &spark_model::video_decode_ffmpeg::FfmpegPolicy {
                enabled: false,
                ..Default::default()
            },
            fps: 2.0,
        },
        input,
        tools_active,
        &levers::ChatLevers::OFF,
        false,
    )
    .map(|o| o.messages)
}

#[cfg(test)]
pub(crate) fn test_build_json_messages(entries: &[msg_entry::MsgEntry]) -> Vec<serde_json::Value> {
    template::build_json_messages(entries)
}

use super::compact::openai_error_response;

pub async fn chat_completions(
    // The HOST, not `CurrentModel`: this is the one handler that can CREATE the
    // model. Extractors run before the body is read, so `CurrentModel` would
    // hand back whatever was loaded BEFORE the request said which model it
    // wanted — exactly the thing the auto-swap has to decide on.
    axum::extract::State(host): axum::extract::State<
        std::sync::Arc<crate::main_modules::model_host::ModelHost>,
    >,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    // Parse the body ourselves (instead of using axum's `Json`
    // extractor) so the same bytes can feed both the deserialized
    // handler path and the `--dump` raw-capture path without
    // cloning the struct or cascading `Serialize` through every
    // request type.
    let req: crate::openai::ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    // Unknown reasoning_effort spellings 400 here — the raw string does
    // not survive wire→IR lowering, and a typo must never silently buy a
    // default tier (see `chat_request::validate_reasoning_effort`).
    if let Err(msg) = req.validate_reasoning_effort() {
        return openai_error_response(StatusCode::BAD_REQUEST, msg);
    }

    // Ollama-style: a request naming a different KNOWN model loads it first.
    // Off unless `--auto-swap`, and `--no-auto-swap` overrides that; every
    // other case (absent, unknown, already live) falls through untouched, which
    // is byte-identical to the behaviour before this existed.
    if host.auto_swap_enabled() {
        let live = host.live_model().unwrap_or_default();
        // Short-circuit the overwhelmingly common case — a client naming the
        // model that is already loaded — before touching the disk. Reading the
        // recipe index means an `ArtifactStore::discover` plus a JSON parse,
        // and doing that per request put blocking I/O on a runtime worker in
        // the hot path of every completion.
        if !req.model.is_empty() && req.model != live {
            let requested = req.model.clone();
            let swap_host = host.clone();
            // Both the catalogue read and the load are blocking, so BOTH belong
            // off the runtime — the decision needs the index, and the index is
            // on disk.
            let outcome = tokio::task::spawn_blocking(move || {
                let catalogue = atlas_plugin::ArtifactStore::discover()
                    .ok()
                    .map(|s| crate::recipe::fetch::cached(s.root()).recipes)
                    .unwrap_or_default();
                match crate::main_modules::auto_swap::decide(&requested, &live, &catalogue) {
                    crate::main_modules::auto_swap::Decision::SwapTo(recipe_id) => {
                        crate::main_modules::auto_swap::ensure_loaded(
                            &swap_host, &recipe_id, &requested, &catalogue,
                        )
                    }
                    // Unknown to the catalogue, or already live: serve as-is.
                    crate::main_modules::auto_swap::Decision::ServeCurrent => Ok(()),
                }
            })
            .await;
            match outcome {
                Ok(Ok(())) => {}
                // A failed load already restored the previous model where it
                // could; say so and serve on whatever is actually live rather
                // than failing a request the old model could have answered.
                Ok(Err(e)) => tracing::warn!("auto-swap to {:?} failed: {e:#}", req.model),
                Err(e) => tracing::warn!("auto-swap task failed: {e}"),
            }
        }
    }

    // Resolve AFTER any swap, so the request is served by the model it asked
    // for rather than the one that happened to be loaded when it arrived.
    let Some(state) = host.current() else {
        // `_typed`, not the plain form: the plain one derives `error.type` from
        // the status and would label this `server_error`, which is both wrong
        // (nothing failed) and inconsistent with the `CurrentModel` extractor,
        // which reports `model_not_loaded` for the identical condition.
        return crate::api::compact::openai_error_response_typed(
            StatusCode::SERVICE_UNAVAILABLE,
            "no model is loaded".to_string(),
            "model_not_loaded",
        );
    };

    // --dump: record the incoming request body verbatim.
    let dump_seq = state.dump_writer.as_ref().and_then(|d| {
        match serde_json::from_slice::<serde_json::Value>(&body) {
            Ok(v) => {
                let seq = d.next_seq();
                d.dump_request("/v1/chat/completions", seq, &v);
                Some(seq)
            }
            Err(_) => None,
        }
    });

    // Wire → IR at the edge: echo-only fields peel off beside the
    // envelope; everything downstream reads only the IR.
    let echo = ResponseEcho::from(&req);
    let ir_req =
        crate::session_key::with_override(req.into(), crate::session_key::from_headers(&headers));
    match chat_completions_inner(state.clone(), req_ctx, ir_req, dump_seq).await {
        ChatOutcome::Blocking(ir) => {
            crate::openai::encode_chat_response(&state, *ir, &echo, dump_seq)
        }
        ChatOutcome::Streaming(deltas) => {
            crate::openai::encode_sse_response(deltas, state.model_name.clone(), echo.include_usage)
        }
        ChatOutcome::Http(r) => r,
    }
}

#[path = "inner.rs"]
mod inner;
pub(crate) use inner::chat_completions_inner;
