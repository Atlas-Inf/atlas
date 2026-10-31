// SPDX-License-Identifier: AGPL-3.0-only

//! Admission for vision input (Atlas-Inf/atlas#180).
//!
//! A request whose images the engine cannot serve used to be admitted, fail
//! at prefill start, and reach the client as a generic error; agent clients
//! then resent the identical history forever. Each item is sized from its
//! header before it is decoded (`vision_preprocess::probe_image`), so a
//! request over its limits is refused with a 400 the client can act on before
//! any pixel is allocated, and every admitted item first takes its rows from
//! the server's in-flight budget (`vision_budget`): a spent budget is a 503
//! with `Retry-After`. Every endpoint that takes images goes through
//! `prepare_chat_prompt` (chat completions, Responses, Anthropic messages),
//! streaming or not, so this covers them all.

use std::sync::Arc;

use axum::http::{HeaderValue, StatusCode, header::RETRY_AFTER};
use axum::response::Response;
use spark_model::VisionItem;
use spark_model::vision_budget::{VisionRowBudget, VisionRowPermit};
use spark_model::vision_limits::{MAX_VISION_ITEMS, VisionRowLimits};

use super::super::compact::openai_error_response_with_param;

/// `error.code` of the refusal, for clients that branch on it.
pub(crate) const VISION_TOO_LARGE_CODE: &str = "vision_input_too_large";
/// `error.code` of the transient refusal (budget spent).
pub(crate) const VISION_BUSY_CODE: &str = "vision_capacity_busy";
/// Seconds a client is asked to wait when the budget is spent.
const RETRY_AFTER_SECS: &str = "2";

/// The server's vision bounds, as the message builder needs them.
#[derive(Clone, Copy, Default)]
pub(crate) struct AdmitCtx<'a> {
    pub(crate) limits: Option<&'a VisionRowLimits>,
    pub(crate) budget: Option<&'a Arc<VisionRowBudget>>,
}

fn too_large(message: String) -> Response {
    tracing::warn!("vision request refused at admission: {message}");
    openai_error_response_with_param(
        StatusCode::BAD_REQUEST,
        message,
        Some("messages"),
        Some(VISION_TOO_LARGE_CODE),
    )
}

fn busy(rows: usize, budget: &VisionRowBudget) -> Response {
    let message = format!(
        "vision capacity busy: an item needs {rows} vision tokens and the server holds {} \
         of its {} for requests in flight; retry shortly",
        budget.in_use(),
        budget.limit()
    );
    tracing::warn!("vision request deferred at admission: {message}");
    let mut resp = openai_error_response_with_param(
        StatusCode::SERVICE_UNAVAILABLE,
        message,
        None,
        Some(VISION_BUSY_CODE),
    );
    resp.headers_mut()
        .insert(RETRY_AFTER, HeaderValue::from_static(RETRY_AFTER_SECS));
    resp
}

/// One request's admission, item by item in prompt order.
pub(super) struct Admission<'a> {
    ctx: AdmitCtx<'a>,
    n_items: usize,
    has_video: bool,
    rows: Vec<usize>,
}

/// The rows an admitted item holds; attach them to the item once decoded.
pub(super) struct Charge(Option<VisionRowPermit>);

impl Charge {
    pub(super) fn attach(self, item: VisionItem) -> VisionItem {
        match self.0 {
            Some(permit) => item.holding(std::sync::Arc::new(permit)),
            None => item,
        }
    }
}

impl<'a> Admission<'a> {
    /// Before any item is read: the protocol's item limit, on every world size.
    #[allow(clippy::result_large_err)]
    pub(super) fn begin(
        ctx: AdmitCtx<'a>,
        n_items: usize,
        has_video: bool,
    ) -> Result<Self, Response> {
        if ctx.limits.is_some() && n_items > MAX_VISION_ITEMS {
            let noun = if has_video {
                "images or videos"
            } else {
                "images"
            };
            return Err(too_large(format!(
                "vision input too large: the request has {n_items} {noun}, more than the \
                 {MAX_VISION_ITEMS} one request may carry"
            )));
        }
        Ok(Self {
            ctx,
            n_items,
            has_video,
            rows: Vec::with_capacity(n_items.min(MAX_VISION_ITEMS)),
        })
    }

    /// Item `rows` (its vision tokens, known before it is decoded): refused
    /// past the window or the running request limit (400), or when the
    /// in-flight budget cannot hold it (503).
    #[allow(clippy::result_large_err)]
    pub(super) fn charge(&mut self, rows: usize) -> Result<Charge, Response> {
        let index = self.rows.len();
        self.rows.push(rows);
        if let Some(limits) = self.ctx.limits {
            let total = self.rows.iter().sum();
            limits
                .check_running(index, self.n_items, rows, total, self.has_video)
                .map_err(too_large)?;
        }
        match self.ctx.budget {
            Some(budget) => budget
                .try_acquire(rows)
                .map(|p| Charge(Some(p)))
                .ok_or_else(|| busy(rows, budget)),
            None => Ok(Charge(None)),
        }
    }

    /// Every item charged: the bounds over the whole request (the worst
    /// prefill chunk against the encoder window).
    #[allow(clippy::result_large_err)]
    pub(super) fn finish(self) -> Result<(), Response> {
        match (self.ctx.limits, self.rows.is_empty()) {
            (Some(limits), false) => limits.check(&self.rows, self.has_video).map_err(too_large),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
#[path = "vision_admit_tests.rs"]
mod tests;
