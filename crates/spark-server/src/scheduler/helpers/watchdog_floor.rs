// SPDX-License-Identifier: AGPL-3.0-only

//! Request-local `min_tokens` floor for the quality watchdogs.

/// Request-local floor for semantic and token-pattern watchdogs. The
/// scheduler's hard safety stops use their own budget, deadline, cancellation,
/// and EOS checks; this predicate only keeps quality heuristics from cutting a
/// response before an explicit `min_tokens` request floor.
#[inline]
pub fn watchdog_floor_reached(output_tokens: usize, min_tokens: usize) -> bool {
    output_tokens >= min_tokens
}

#[cfg(test)]
mod tests {
    use super::watchdog_floor_reached;

    #[test]
    fn request_min_tokens_floor_is_inclusive_for_quality_watchdogs() {
        assert!(!watchdog_floor_reached(281, 400));
        assert!(watchdog_floor_reached(400, 400));
        assert!(watchdog_floor_reached(0, 0));
    }
}
