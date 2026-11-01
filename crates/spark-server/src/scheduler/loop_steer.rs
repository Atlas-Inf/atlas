// SPDX-License-Identifier: AGPL-3.0-only

//! Content-loop steering at the pick (`ATLAS_LOOP_STEER=1`, default off).
//!
//! The content-loop watchdog ends a response whose tail repeats a short token
//! pattern (`detect_content_token_loop_with`); on the MTP/verify path it can
//! only hard-stop, since it cannot rewind the recurrent state. Greedy decoding
//! has attractors (a Go test table of `0, 0, 0, …`) that it stops mid-answer.
//! With steering armed, the pick right after such a tail never takes the token
//! that would continue the cycle, so the model steps off the attractor instead
//! of being cut off. Nothing is rolled back or held back from the stream.
//!
//! One rule for every pick, serial decode and every verify row alike
//! ([`banned_token`]): a pure function of the sequence's output so far and of
//! the live state the commit rule keeps (`<think>`, the tool body), with the
//! watchdog's own detector and thresholds. A verify span's rows see their own
//! histories through `SpanShadow`, so speculation commits exactly the tokens
//! serial decode commits. Each steer is counted at commit ([`note_commit`]);
//! after `ATLAS_LOOP_STEER_MAX` steers (default 3) a request's loops end as
//! before. Three places apply the rule, as for the min_tokens end-token ban:
//!
//! * the host pipeline (`process_position_logits`) masks the banned id;
//! * the GPU-argmax fast paths stand down for any span a steer could touch
//!   ([`span_may_steer`]) and take the host pipeline;
//! * the watchdog does not hard-stop a tail the next pick will steer
//!   ([`will_steer`]).
//!
//! Output that never loops is untouched: the rule only acts on a tail the
//! watchdog would end.

use crate::api::inference_types::RepetitionDetectionParams;
use crate::scheduler::helpers::{
    CONTENT_LOOP_MIN_REPEATS, CONTENT_LOOP_PERIOD_MAX, CONTENT_LOOP_PERIOD_MIN,
    watchdog_floor_reached,
};
use crate::scheduler::types::ActiveSeq;
use std::sync::OnceLock;

/// The startup configuration: the per-request steer budget (0 = disarmed) and
/// the operator's repeat threshold.
#[derive(Clone, Copy, Debug, Default)]
struct Config {
    max: u32,
    operator: Option<RepetitionDetectionParams>,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

/// Arm steering for this process from `ATLAS_LOOP_STEER` /
/// `ATLAS_LOOP_STEER_MAX`. It needs the content-loop watchdog: without it, or
/// with the watchdogs disabled, nothing steers. `operator` is the watchdog's
/// own threshold override (`WatchdogParams::content_loop_params(None)`).
pub fn install(
    watchdog_armed: bool,
    disable_watchdogs: bool,
    operator: Option<RepetitionDetectionParams>,
) {
    let on = std::env::var("ATLAS_LOOP_STEER").is_ok_and(|v| v == "1");
    let max = std::env::var("ATLAS_LOOP_STEER_MAX")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(3);
    let max = if on && watchdog_armed && !disable_watchdogs {
        max
    } else {
        0
    };
    if CONFIG.set(Config { max, operator }).is_ok() && max > 0 {
        tracing::info!(max, "content-loop steering armed (ATLAS_LOOP_STEER=1)");
    }
}

/// The steer budget a new sequence starts with (0 when steering is off).
pub(in crate::scheduler) fn budget_for_new_seq() -> u32 {
    CONFIG.get().map_or(0, |c| c.max)
}

/// `(period_min, period_max, min_repeats)` for this sequence: the request's
/// `repetition_detection`, else the operator's threshold, else the built-in
/// constants: the watchdog's own precedence. The period is capped at the
/// built-in `CONTENT_LOOP_PERIOD_MAX`, since the rule runs at every pick: a
/// longer loop a request asks to detect is left to the watchdog.
fn params(a: &ActiveSeq) -> (usize, usize, usize) {
    let (min, max, repeats) = match a
        .repetition_detection
        .or_else(|| CONFIG.get().and_then(|c| c.operator))
    {
        Some(p) => (
            p.min_pattern_size as usize,
            p.max_pattern_size as usize,
            p.min_count as usize,
        ),
        None => (
            CONTENT_LOOP_PERIOD_MIN,
            CONTENT_LOOP_PERIOD_MAX,
            CONTENT_LOOP_MIN_REPEATS,
        ),
    };
    (min, max.min(CONTENT_LOOP_PERIOD_MAX), repeats)
}

/// The token that would continue a loop at the end of `len` tokens read
/// through `at`: the smallest period whose last `period` tokens repeat
/// `min_repeats` times back to back (the watchdog's anchored test), and the
/// token one period back. `None` when no period anchors a repeat.
fn continuation(
    len: usize,
    at: impl Fn(usize) -> u32,
    (period_min, period_max, min_repeats): (usize, usize, usize),
) -> Option<u32> {
    if min_repeats < 2 {
        return None;
    }
    for period in period_min.max(1)..=period_max {
        if period * min_repeats > len {
            return None;
        }
        let anchored = (1..=period).all(|off| {
            let target = at(len - off);
            (1..min_repeats).all(|m| at(len - (period * m + off)) == target)
        });
        if anchored {
            return Some(at(len - period));
        }
    }
    None
}

/// The id the sequence's next pick may not take: the loop continuation, when
/// the output ends in a loop the content-loop watchdog would end, outside
/// `<think>` and tool bodies, with steers left.
pub(in crate::scheduler) fn banned_token(a: &ActiveSeq) -> Option<u32> {
    if a.loop_steers >= a.loop_steer_max
        || a.inside_thinking
        || a.inside_tool_body
        || a.strict_grammar()
        || !watchdog_floor_reached(a.output_tokens.len(), a.min_tokens)
    {
        return None;
    }
    let out = &a.output_tokens;
    continuation(out.len(), |i| out[i], params(a))
}

/// The next pick will be steered: the watchdog leaves this tail to it.
pub(in crate::scheduler) fn will_steer(a: &ActiveSeq) -> bool {
    banned_token(a).is_some()
}

/// Mask the banned id of the next pick in a host logits row (`f32`).
pub(in crate::scheduler) fn mask_row(logits: &mut [f32], a: &ActiveSeq) {
    if let Some(id) = banned_token(a)
        && let Some(v) = logits.get_mut(id as usize)
    {
        *v = f32::NEG_INFINITY;
    }
}

/// Count the steer, if any, of the pick being committed, and say whether
/// there was one. Call before the committed token is pushed and before
/// anything else the commit changes, so the rule sees the state its pick saw.
pub(in crate::scheduler) fn note_commit(a: &mut ActiveSeq) -> bool {
    let steered = will_steer(a);
    if steered {
        a.loop_steers = a.loop_steers.saturating_add(1);
    }
    steered
}

/// Whether some row of a span whose raw picks are `picks` could be steered:
/// a superset of [`banned_token`] at every row (the rows' histories end in
/// `picks[..r]`; the `<think>`, tool-body and grammar gates and the earlier
/// rows' steers are left out, since they can only narrow it). A span that may
/// steer takes the host pipeline, which applies the rule row by row.
pub(in crate::scheduler) fn span_may_steer(a: &ActiveSeq, picks: &[u32]) -> bool {
    if a.loop_steers >= a.loop_steer_max {
        return false;
    }
    let out = &a.output_tokens;
    let n = out.len();
    let p = params(a);
    (0..picks.len()).any(|r| {
        let len = n + r;
        watchdog_floor_reached(len, a.min_tokens)
            && continuation(len, |i| if i < n { out[i] } else { picks[i - n] }, p).is_some()
    })
}

#[cfg(test)]
#[path = "loop_steer_tests.rs"]
mod tests;
