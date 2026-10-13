// SPDX-License-Identifier: AGPL-3.0-only

//! Where a new multi-chunk prefill joins `prefilling`, whose order is the
//! service order (the continue phase advances its head).

use std::time::Duration;

/// A prefill that has waited this long is never overtaken again, so a long
/// prompt cannot starve behind a stream of shorter ones.
const SRPT_MAX_WAIT: Duration = Duration::from_secs(30);

/// Index at which a prefill with `remaining` prompt tokens joins `queued`
/// (each entry: tokens left, time since its request started).
///
/// Arrival order by default. With `srpt` (`ATLAS_PREFILL_SRPT=1`) it goes
/// ahead of every prefill with more tokens left — shortest remaining first —
/// but never ahead of one that has waited [`SRPT_MAX_WAIT`]. A queued prefill
/// only shrinks while it is the head, so the order stays sorted.
pub(super) fn position(queued: &[(usize, Duration)], remaining: usize, srpt: bool) -> usize {
    if !srpt {
        return queued.len();
    }
    let floor = queued
        .iter()
        .rposition(|&(_, waited)| waited >= SRPT_MAX_WAIT)
        .map_or(0, |i| i + 1);
    queued[floor..]
        .iter()
        .position(|&(left, _)| left > remaining)
        .map_or(queued.len(), |i| floor + i)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEW: Duration = Duration::from_secs(1);

    #[test]
    fn arrival_order_by_default() {
        assert_eq!(position(&[(100_000, NEW), (5_000, NEW)], 10, false), 2);
    }

    #[test]
    fn srpt_goes_ahead_of_longer_prefills_and_behind_equal_ones() {
        let queued = [(13, NEW), (4_000, NEW), (118_000, NEW)];
        assert_eq!(position(&queued, 8, true), 0);
        assert_eq!(position(&queued, 13, true), 1);
        assert_eq!(position(&queued, 9_000, true), 2);
        assert_eq!(position(&queued, 200_000, true), 3);
        assert_eq!(position(&[], 10, true), 0);
    }

    #[test]
    fn srpt_never_overtakes_a_prefill_that_waited_too_long() {
        let queued = [(118_000, SRPT_MAX_WAIT), (50_000, NEW), (90_000, NEW)];
        assert_eq!(position(&queued, 10, true), 1);
        assert_eq!(position(&queued, 60_000, true), 2);
    }
}
