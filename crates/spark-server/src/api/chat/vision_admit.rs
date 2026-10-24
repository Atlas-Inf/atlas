// SPDX-License-Identifier: AGPL-3.0-only

//! Admission for vision input (Atlas-Inf/atlas#180).
//!
//! A request whose images the engine cannot serve used to be admitted, fail
//! at prefill start, and reach the client as a generic error; agent clients
//! then resent the identical history forever. Each still is sized from its
//! header before it is decoded (`vision_preprocess::probe_image`), so a
//! request over its limits is refused with a 400 the client can act on before
//! any pixel is allocated; a request within them then takes all of its rows
//! from the server's in-flight budget (`vision_budget`) in one acquire (or a
//! conservative pre-decode reservation for opaque videos), and a spent budget
//! is a 503 with `Retry-After`. The passes are in
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
    /// A conservative Qwen-video reservation held while the decoder runs.
    /// It is resized to the actual request total in `admit`, so decoding never
    /// allocates an unaccounted clip and the final acquire cannot race after a
    /// temporary permit is dropped.
    preheld: Option<VisionRowPermit>,
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
            preheld: None,
        })
    }

    /// Hold a conservative reservation before an opaque video is decoded.
    ///
    /// Still images can be sized from their headers before their pixels are
    /// read. A video has no equivalent cheap geometry probe, so Qwen callers
    /// reserve the configured worst-case row count first. The reservation is
    /// kept in this admission until `admit` grows or shrinks it to the actual
    /// total, avoiding both an unbounded decode and a drop/reacquire race.
    #[allow(clippy::result_large_err)]
    pub(super) fn reserve_video(&mut self, rows: usize) -> Result<(), Response> {
        if rows == 0 {
            return Ok(());
        }
        if let Some(limits) = self.ctx.limits
            && limits.request_rows > 0
            && rows > limits.request_rows
        {
            return Err(too_large(format!(
                "vision input too large: a video may need {rows} vision tokens, more than the \
                 {} this server accepts per request; lower the video pixel/frame bound or raise \
                 the request limit",
                limits.request_rows
            )));
        }
        let Some(budget) = self.ctx.budget else {
            return Ok(());
        };
        if let Some(permit) = &mut self.preheld {
            if permit.try_grow(rows) {
                return Ok(());
            }
            return Err(busy(rows));
        }
        self.preheld = Some(budget.try_acquire(rows).ok_or_else(|| busy(rows))?);
        Ok(())
    }

    /// Reconcile the reservation after one opaque video has been decoded.
    /// `actual_rows` is the cumulative row count of all decoded videos in the
    /// request; still images are still header-only at this point. Keeping the
    /// permit alive while shrinking it prevents a second video from rejecting
    /// merely because the first video was short.
    #[allow(clippy::result_large_err)]
    pub(super) fn reconcile_video(&mut self, actual_rows: usize) -> Result<(), Response> {
        let Some(budget) = self.ctx.budget else {
            return Ok(());
        };
        if let Some(permit) = &mut self.preheld {
            if permit.rows() < actual_rows && !permit.try_grow(actual_rows - permit.rows()) {
                return Err(busy(actual_rows));
            }
            permit.shrink_to(actual_rows);
            return Ok(());
        }
        if actual_rows > 0 {
            self.preheld = Some(
                budget
                    .try_acquire(actual_rows)
                    .ok_or_else(|| busy(actual_rows))?,
            );
        }
        Ok(())
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
        if total == 0 {
            return Ok(Admitted(None));
        }

        if let Some(mut permit) = self.preheld {
            if permit.rows() < total && !permit.try_grow(total - permit.rows()) {
                return Err(busy(total));
            }
            permit.shrink_to(total);
            return Ok(Admitted(Some(Arc::new(permit))));
        }

        match self.ctx.budget {
            Some(budget) => budget
                .try_acquire(total)
                .map(|p| Admitted(Some(Arc::new(p))))
                .ok_or_else(|| busy(total)),
            None => Ok(Admitted(None)),
        }
    }
}

#[cfg(test)]
#[path = "vision_admit_tests.rs"]
mod tests;
