// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Error(msg)` / `StreamEvent::InvalidRequest(msg)` arms of the
// streaming `flat_map` closure.

use crate::ir::StreamDelta;

use super::ctx::StreamCtx;

type DeltaVec = Vec<StreamDelta>;

pub(super) fn handle_error(ctx: &StreamCtx, msg: String) -> DeltaVec {
    error_frame(ctx, msg, "server_error", 500)
}

/// The request itself can never be served as sent (#180: images over the
/// vision encoder's capacity): `invalid_request_error` / 400, which agent
/// clients do not retry, where `server_error` / 500 sent them round again.
pub(super) fn handle_invalid_request(ctx: &StreamCtx, msg: String) -> DeltaVec {
    error_frame(ctx, msg, "invalid_request_error", 400)
}

fn error_frame(ctx: &StreamCtx, msg: String, kind: &str, code: u16) -> DeltaVec {
    // REQUESTS_ACTIVE released by StreamCtx's ActiveRequestGuard on drop.
    // Abandoned stream — refund the full reservation.
    if let Some(ref rctx) = ctx.req_ctx {
        ctx.state
            .rate_limiter
            .refund_tokens(&rctx.identity, rctx.reserved_tokens);
    }
    // Wire-ready OpenAI error envelope; the encoder forwards it
    // verbatim as SSE data.
    let err = serde_json::json!({
        "error": {"message": msg, "type": kind, "code": code}
    });
    vec![StreamDelta::Error {
        message: err.to_string(),
    }]
}
