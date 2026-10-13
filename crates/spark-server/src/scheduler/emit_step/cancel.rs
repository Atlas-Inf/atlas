// SPDX-License-Identifier: AGPL-3.0-only

//! Cooperative-cancellation retirement shared by the emit and decode paths.

use super::*;

/// Cooperative cancellation only marks retirement; lifecycle owns state cleanup.
/// An already-issued forward cannot be undone here. Preserve finish-reason and
/// hard-limit metadata rather than inventing a new cancellation reason.
pub(in crate::scheduler) fn retire_if_cancelled(a: &mut ActiveSeq) -> bool {
    if a.cancel_flag
        .as_ref()
        .is_some_and(|f| f.load(std::sync::atomic::Ordering::Acquire))
        || a.sink.receiver_closed()
    {
        a.finished = true;
        true
    } else {
        false
    }
}
