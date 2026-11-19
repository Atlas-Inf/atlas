// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/score`: prefill-only option scoring.
//!
//! ```json
//! {"prefix": [ids], "branches": [[ids], ...], "options": [ids] | [[ids], ...]}
//! ```
//! returns `{"probs": [[p, ...], ...], "engine_us": n}`: per branch, the
//! softmax over its option tokens at the branch's last position. Optional
//! `"readouts": [[offset, ...], ...]` selects 1..=4 rows per branch (0 is the
//! last token); then each `probs` branch contains one distribution per row.
//! Without `readouts`, the response shape is unchanged. Optional `"images"`
//! supplies ordered base64 PNG/JPEG/WebP payloads for image-pad runs in the
//! prefix. `prefix` is read once and every branch continues it, all in one forward pass
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
//! and the client sends the tokens again. `"images"` with `"pack"` tokens are
//! the pack's (every image pad in the pack, none in the prefix or branches):
//! an image pack, e.g. a photo asked many questions, whose images are encoded
//! once with it. Images with `"pack_id"` are refused.
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
#[path = "score_images.rs"]
mod score_images;
use crate::main_modules::model_host::CurrentModel;
use crate::scheduler::score::{PackRef, ScoreError, ScoreJob};
use score_images::ScoreImageInput;

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
    /// Images in the order of their runs in `prefix`.
    #[serde(default)]
    images: Option<Vec<ScoreImageInput>>,
    /// Per branch, offsets back from its last token to score.
    #[serde(default)]
    pub readouts: Option<Vec<Vec<usize>>>,
}

struct Validated {
    prefix: Vec<u32>,
    branches: Vec<Vec<u32>>,
    options: Vec<Vec<u32>>,
    prefix_bidir: bool,
    pack: Option<PackRef>,
    images: Option<Vec<ScoreImageInput>>,
    readouts: Option<Vec<Vec<usize>>>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum BranchProbabilities {
    Last(Vec<f32>),
    Readouts(Vec<Vec<f32>>),
}

impl From<spark_model::traits::ScoreBranchProbs> for BranchProbabilities {
    fn from(value: spark_model::traits::ScoreBranchProbs) -> Self {
        match value {
            spark_model::traits::ScoreBranchProbs::Last(probs) => Self::Last(probs),
            spark_model::traits::ScoreBranchProbs::Readouts(probs) => Self::Readouts(probs),
        }
    }
}

#[derive(Serialize)]
struct ScoreResponse {
    probs: Vec<BranchProbabilities>,
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

/// Validate and expand the text fields of a score request.
fn validate(req: ScoreRequest, vocab_size: u32) -> Result<Validated, String> {
    let ScoreRequest {
        prefix,
        branches,
        options,
        attention,
        pack,
        pack_id,
        images,
        readouts,
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
    if let Some(readouts) = &readouts {
        if readouts.len() != branches.len() {
            return Err("readouts: one readout list per branch".into());
        }
        for (branch, offsets) in branches.iter().zip(readouts) {
            if !(1..=4).contains(&offsets.len()) {
                return Err("readouts: need 1..=4 offsets per branch".into());
            }
            if offsets.iter().any(|&offset| offset >= branch.len()) {
                return Err("readouts: every offset must be inside its branch".into());
            }
            if (1..offsets.len()).any(|i| offsets[..i].contains(&offsets[i])) {
                return Err("readouts: offsets of a branch must be distinct".into());
            }
        }
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
    Ok(Validated {
        prefix,
        branches,
        options,
        prefix_bidir,
        pack,
        images,
        readouts,
    })
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
    if score_images::images_with_pack_id(req.images.as_deref(), req.pack_id.is_some()) {
        return openai_error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "score: images cannot be combined with pack_id (send the pack's tokens with them)"
                .into(),
        );
    }
    let Validated {
        prefix,
        branches,
        options,
        prefix_bidir,
        pack,
        images,
        readouts,
    } = match validate(req, state.tokenizer.vocab_size()) {
        Ok(v) => v,
        Err(msg) => return openai_error_response(StatusCode::BAD_REQUEST, msg),
    };
    let image_count = images.as_ref().map_or(0, Vec::len);
    if score_images::vision_tower_missing(image_count, state.vision_config.is_some()) {
        return openai_error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "score: this model has no vision tower".into(),
        );
    }
    if image_count > spark_model::vision_limits::MAX_VISION_ITEMS {
        return openai_error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "score: {image_count} images, over the {} one request may carry",
                spark_model::vision_limits::MAX_VISION_ITEMS
            ),
        );
    }
    // Text-only requests skip the image-marker checks entirely (their
    // behaviour is unchanged by the image path).
    let pad_counts = if image_count == 0 {
        Vec::new()
    } else {
        let marker_ids =
            match score_images::image_token_ids(&state.tokenizer, state.vision_config.as_ref()) {
                Ok(ids) => ids,
                Err(msg) => return openai_error_response(StatusCode::UNPROCESSABLE_ENTITY, msg),
            };
        // With pack tokens the images are the pack's: their pads must all be
        // in the pack, none in the prefix or a branch.
        let checked = match &pack {
            Some(PackRef::Tokens(pack_tokens)) => {
                let rest: Vec<Vec<u32>> = std::iter::once(prefix.clone())
                    .chain(branches.iter().cloned())
                    .collect();
                score_images::validate_image_tokens(pack_tokens, &rest, image_count, marker_ids)
            }
            _ => score_images::validate_image_tokens(&prefix, &branches, image_count, marker_ids),
        };
        match checked {
            Ok(counts) => counts,
            Err(msg) => return openai_error_response(StatusCode::UNPROCESSABLE_ENTITY, msg),
        }
    };
    let vision_rows: usize = pad_counts.iter().sum();
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
    let vision_items = if let Some(images) = images.filter(|items| !items.is_empty()) {
        let config = state
            .vision_config
            .as_ref()
            .expect("images require a vision config")
            .clone();
        let max_pixels = state.vision_max_pixels;
        // Refuse before decoding when the queue is full, and bound concurrent
        // decodes: each holds up to a few images' RGB buffers on the host.
        if queue.tx.capacity() == 0 {
            return busy();
        }
        let Ok(_decode_permit) = score_images::DECODE_PERMITS.acquire().await else {
            return busy();
        };
        match tokio::task::spawn_blocking(move || {
            score_images::preprocess_images(images, &pad_counts, &config, max_pixels)
        })
        .await
        {
            Ok(Ok(items)) => items,
            Ok(Err(msg)) => {
                return openai_error_response(StatusCode::UNPROCESSABLE_ENTITY, msg);
            }
            Err(e) => {
                return openai_error_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("score: image preprocessing failed: {e}"),
                );
            }
        }
    } else {
        Vec::new()
    };
    let (reply, rx) = tokio::sync::oneshot::channel();
    let job = ScoreJob {
        prefix,
        branches,
        options,
        prefix_bidir,
        pack,
        images: vision_items,
        vision_rows,
        readouts,
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
            probs: out.probs.into_iter().map(Into::into).collect(),
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
#[path = "score_tests.rs"]
mod tests;
