// SPDX-License-Identifier: AGPL-3.0-only

//! Admission for vision input (Atlas-Inf/atlas#180).
//!
//! A request whose images the engine cannot serve used to be admitted, fail
//! at prefill start, and reach the client as a generic error; agent clients
//! then resent the identical history forever. Each still is sized from its
//! header before it is decoded (`vision_preprocess::probe_image`), so a
//! request over its limits is refused with a 400 the client can act on before
//! any pixel is allocated; a request within them then takes all of its rows
//! from the server's in-flight budget (`vision_budget`) in one acquire, and a
//! spent budget is a 503 with `Retry-After`. The passes are in
//! `msg_entry::media`. Every endpoint that takes images goes through
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

fn busy(rows: usize) -> Response {
    // Counts of what other requests hold are not this client's business.
    tracing::warn!("vision request deferred at admission: {rows} rows, in-flight budget spent");
    let mut resp = openai_error_response_with_param(
        StatusCode::SERVICE_UNAVAILABLE,
        "vision capacity is busy; retry shortly".to_string(),
        None,
        Some(VISION_BUSY_CODE),
    );
    resp.headers_mut()
        .insert(RETRY_AFTER, HeaderValue::from_static(RETRY_AFTER_SECS));
    resp
}

/// One request's admission: [`Admission::begin`] before any item is read,
/// [`Admission::note`] as each is sized, [`Admission::admit`] once all are.
pub(super) struct Admission<'a> {
    ctx: AdmitCtx<'a>,
    n_items: usize,
    has_video: bool,
    rows: Vec<usize>,
}

/// An admitted request: its permit, shared by every item it attaches to.
pub(super) struct Admitted(Option<Arc<VisionRowPermit>>);

impl Admitted {
    pub(super) fn attach(&self, item: VisionItem) -> VisionItem {
        match &self.0 {
            Some(permit) => item.holding(permit.clone()),
            None => item,
        }
    }
}

impl<'a> Admission<'a> {
    /// Before any item is fetched or read: the protocol's item limit, on
    /// every world size.
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

    /// The next item carries `rows` (its vision tokens, known for a still
    /// before it is decoded): a 400 as soon as it passes the window or the
    /// running total passes the request limit.
    #[allow(clippy::result_large_err)]
    pub(super) fn note(&mut self, rows: usize) -> Result<(), Response> {
        let index = self.rows.len();
        self.rows.push(rows);
        match self.ctx.limits {
            Some(limits) => {
                let total = self.rows.iter().sum();
                limits
                    .check_running(index, self.n_items, rows, total, self.has_video)
                    .map_err(too_large)
            }
            None => Ok(()),
        }
    }

    /// Every item sized: the whole-request bounds (the worst prefill chunk
    /// against the encoder window; 400), then all of its rows from the
    /// in-flight budget in one acquire (503): a request holds all of its rows
    /// or none, so two large requests cannot each hold part and both fail.
    #[allow(clippy::result_large_err)]
    pub(super) fn admit(self) -> Result<Admitted, Response> {
        if let (Some(limits), false) = (self.ctx.limits, self.rows.is_empty()) {
            limits
                .check(&self.rows, self.has_video)
                .map_err(too_large)?;
        }
        let total: usize = self.rows.iter().sum();
        match self.ctx.budget {
            Some(budget) if total > 0 => budget
                .try_acquire(total)
                .map(|p| Admitted(Some(Arc::new(p))))
                .ok_or_else(|| busy(total)),
            _ => Ok(Admitted(None)),
        }
    }
}

#[cfg(test)]
#[path = "vision_admit_tests.rs"]
mod tests;
