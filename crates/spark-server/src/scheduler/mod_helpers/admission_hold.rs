// SPDX-License-Identifier: AGPL-3.0-only

//! Why pending requests stay pending after an admission pass, logged on
//! change only (an otherwise silent wait: a staggered stream on a pair
//! already serving other traffic waited ~20 s for its first chunk with
//! nothing in the log; the KV gate names its own holds).

use std::sync::atomic::{AtomicUsize, Ordering};

/// What held the requests left pending.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Hold {
    /// Every free batch slot was taken.
    Slots,
    /// The scheduling policy admitted fewer than the free slots.
    Policy,
}

/// The hold, if any: `held` requests left pending after the policy took
/// `selected` of `free` batch slots.
pub(super) fn hold(held: usize, selected: usize, free: usize) -> Option<Hold> {
    (held > 0).then_some(if selected >= free {
        Hold::Slots
    } else {
        Hold::Policy
    })
}

/// The last state logged. Process-wide: one scheduler loop runs per process,
/// so a second scheduler (a test) only shares the on-change filter.
static LAST: AtomicUsize = AtomicUsize::new(0);

/// Log the pass's hold when it differs from the last one logged.
pub(super) fn log_on_change(
    held: usize,
    selected: usize,
    free: usize,
    decoding: usize,
    prefilling: usize,
    max_batch_size: usize,
) {
    let h = hold(held, selected, free);
    let key = held * 3 + h.map_or(0, |h| h as usize + 1);
    if LAST.swap(key, Ordering::Relaxed) == key {
        return;
    }
    match h {
        Some(Hold::Slots) => tracing::info!(
            "Prefill admission: {held} request(s) wait for a batch slot \
             ({decoding} decoding + {prefilling} prefilling of {max_batch_size})"
        ),
        Some(Hold::Policy) => tracing::info!(
            "Prefill admission: {held} request(s) wait; the scheduling policy admitted \
             {selected} of {free} free batch slot(s)"
        ),
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{Hold, hold};

    #[test]
    fn a_full_batch_and_a_policy_hold_are_told_apart() {
        assert_eq!(hold(0, 2, 2), None);
        assert_eq!(hold(3, 2, 2), Some(Hold::Slots));
        assert_eq!(hold(3, 0, 0), Some(Hold::Slots));
        assert_eq!(hold(3, 1, 4), Some(Hold::Policy));
    }
}
