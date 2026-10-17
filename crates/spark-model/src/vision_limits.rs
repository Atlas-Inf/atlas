// SPDX-License-Identifier: AGPL-3.0-only

//! How many vision rows a request may carry, decided once and shared by the
//! API's admission check and the engine.
//!
//! A vision row is one merged patch: one placeholder token in the prompt and
//! one `out_hidden_size` BF16 row of encoder output. The chunked prefill
//! encodes, before each chunk's embed, only the items whose placeholders fall
//! inside that chunk (`model::vision_cache`), so what has to fit the encoder's
//! output buffer is one chunk's items, not the whole request. Three bounds
//! follow:
//!
//! - **window** — rows the encoder output buffer holds. A chunk of at most
//!   `chunk_tokens` tokens touches every item whose pads it covers, including
//!   the two it may cut through at either end, so the buffer is sized
//!   [`window_rows`] = `chunk_tokens + 2 x` the largest still image, never
//!   below the historical one-batch capacity.
//! - **request** — rows one request may carry in total. Every row is a prompt
//!   token, so the context bounds it; the operator bounds it further because
//!   rank 0 holds the request's preprocessed pixels (f32, 24 KiB per merged
//!   row on the Qwen tower) from admission until its prefill ends.
//! - **items** — the rank-to-rank protocol's item limit, applied on every
//!   world size (a single rank refuses a 1025-item request too, so a request
//!   does not pass on one deployment and fail on another).
//!
//! [`VisionRowLimits::check`] rejects a request that one of them would fail
//! after admission, so the client gets a 400 it can act on instead of a
//! prefill error it retries forever. [`VisionRowLimits::check_running`] applies
//! the per-item and running-total bounds while the items are still being
//! sized from their headers, before their pixels are decoded.
//! [`crate::vision_budget`] bounds the pixels held across requests.

/// Items one request may carry: the rank-to-rank protocol's limit.
pub const MAX_VISION_ITEMS: usize = crate::model::vision_transport::VISION_MAX_GRID_ITEMS;

/// Merged rows the encoder output buffer must hold so that any prefill chunk
/// of at most `chunk_tokens` tokens can encode every item it touches.
///
/// `p_max` is the encoder's patch capacity (one batch, and the largest still
/// image); a still image yields at most `p_max / merge^2` rows. The result is
/// never below `p_max`, the buffer's historical size, so a request that fit
/// before still fits. `chunk_tokens == 0` (unknown) keeps the historical size.
pub fn window_rows(p_max: usize, spatial_merge_size: usize, chunk_tokens: usize) -> usize {
    if chunk_tokens == 0 {
        return p_max;
    }
    let merge2 = spatial_merge_size
        .max(1)
        .saturating_mul(spatial_merge_size.max(1));
    let image_rows = p_max / merge2;
    p_max.max(chunk_tokens.saturating_add(image_rows.saturating_mul(2)))
}

/// The most rows one chunk can need for a request whose items carry
/// `item_rows` (prompt order), over every chunking with chunks of at most
/// `chunk_tokens` tokens.
///
/// A chunk that touches items `i < j` holds at least one pad of each and all
/// the pads of the items between, so it needs `r_i + r_j + middle` rows with
/// `middle <= chunk_tokens - 2` and `middle <= total - r_i - r_j`. That grows
/// with `r_i + r_j`, so the two largest items give the bound; one item alone
/// needs its own rows.
pub fn worst_chunk_rows(item_rows: &[usize], chunk_tokens: usize) -> usize {
    let total: usize = item_rows.iter().sum();
    let (mut top1, mut top2) = (0usize, 0usize);
    for &r in item_rows {
        if r > top1 {
            top2 = top1;
            top1 = r;
        } else if r > top2 {
            top2 = r;
        }
    }
    let middle = chunk_tokens.saturating_sub(2).min(total - top1 - top2);
    top1 + top2 + middle
}

/// The bounds one server applies to every vision request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionRowLimits {
    /// Encoder output rows ([`window_rows`]).
    pub window_rows: usize,
    /// The longest prefill chunk, in tokens (`max_batch_tokens`).
    pub chunk_tokens: usize,
    /// Rows one request may carry.
    pub request_rows: usize,
}

/// "1 image", "3 images", or "3 images or videos" once a video is present.
fn items_noun(n: usize, has_video: bool) -> String {
    match (n, has_video) {
        (1, false) => "1 image".into(),
        (1, true) => "1 video".into(),
        (n, false) => format!("{n} images"),
        (n, true) => format!("{n} images or videos"),
    }
}

impl VisionRowLimits {
    /// The limits a server applies: the encoder's window, its longest prefill
    /// chunk, and the rows one request may carry: the operator's per-request
    /// rows (`0` = unset), capped by `request_bound` (the engine's own bound:
    /// the context, since every row is a prompt token, or the window where the
    /// engine does not chunk vision rows across encodes) and by the in-flight
    /// budget (`0` = none), which a larger request could never acquire.
    pub fn resolve(
        window_rows: usize,
        chunk_tokens: usize,
        request_rows: usize,
        request_bound: usize,
        inflight_rows: usize,
    ) -> Self {
        let mut rows = match request_rows {
            0 => request_bound,
            n => n.min(request_bound),
        };
        if inflight_rows > 0 {
            rows = rows.min(inflight_rows);
        }
        Self {
            window_rows,
            chunk_tokens,
            request_rows: rows,
        }
    }

    /// The bounds that hold item by item, for item `index` (0-based) of
    /// `n_items` carrying `rows`, with `total` rows counted so far including
    /// it: refused as soon as one fails, before the rest are decoded.
    pub fn check_running(
        &self,
        index: usize,
        n_items: usize,
        rows: usize,
        total: usize,
        has_video: bool,
    ) -> Result<(), String> {
        if rows > self.window_rows {
            return Err(format!(
                "vision input too large: item {} of {} needs {rows} vision tokens, more than \
                 the encoder's {}-row window; send a smaller image or a shorter video",
                index + 1,
                items_noun(n_items, has_video),
                self.window_rows
            ));
        }
        if total > self.request_rows {
            return Err(format!(
                "vision input too large: the first {} of {} already need {total} vision \
                 tokens, more than the {} this server accepts per request; send fewer or \
                 smaller images",
                index + 1,
                items_noun(n_items, has_video),
                self.request_rows
            ));
        }
        Ok(())
    }

    /// Why a request whose items carry `item_rows` (prompt order) cannot be
    /// served, or `Ok`. The message names counts and limits only, never
    /// request content.
    pub fn check(&self, item_rows: &[usize], has_video: bool) -> Result<(), String> {
        let n = item_rows.len();
        let items = items_noun(n, has_video);
        if n > MAX_VISION_ITEMS {
            return Err(format!(
                "vision input too large: the request has {items}, more than the \
                 {MAX_VISION_ITEMS} one request may carry"
            ));
        }
        let total: usize = item_rows.iter().sum();
        if total > self.request_rows {
            return Err(format!(
                "vision input too large: {items} need {total} vision tokens, more than \
                 the {} this server accepts per request; send fewer or smaller images",
                self.request_rows
            ));
        }
        let worst = worst_chunk_rows(item_rows, self.chunk_tokens);
        if worst > self.window_rows {
            let largest = item_rows.iter().max().copied().unwrap_or(0);
            return Err(format!(
                "vision input too large: {items} need {total} vision tokens and a prefill \
                 chunk of up to {} tokens may need {worst} of them encoded at once (largest \
                 item {largest}), more than the encoder's {}-row window; send fewer or \
                 smaller images",
                self.chunk_tokens, self.window_rows
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "vision_limits_tests.rs"]
mod tests;
