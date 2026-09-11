// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};
use std::sync::atomic::{AtomicU64, Ordering};

/// Capture-safety gate for host-side CUDA operations.
///
/// Atlas captures with `CU_STREAM_CAPTURE_MODE_RELAXED`. The driver does not
/// protect a capture for us in that mode: the capturing thread must still not
/// allocate, synchronize, or otherwise issue host-blocking driver calls, while
/// other threads are permitted to. The gate therefore tracks the *owning
/// thread* rather than the whole process — a process-global flag would reject
/// unrelated concurrent work and surface as a spurious eager fallback.
///
/// A single atomic holds the owner id (`0` = inactive). That both rejects an
/// overlapping capture from any thread and lets `finish` refuse to clear state
/// it does not own, so a stray `finish` on the wrong thread cannot silently
/// disarm a live capture.
pub(super) struct CaptureSafety {
    owner: AtomicU64,
}

impl CaptureSafety {
    pub fn new() -> Self {
        Self {
            owner: AtomicU64::new(0),
        }
    }

    pub fn begin(&self) -> Result<()> {
        self.owner
            .compare_exchange(0, current_thread_id(), Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| anyhow::anyhow!("nested CUDA graph capture is forbidden"))
    }

    pub fn finish(&self) {
        let _ = self.owner.compare_exchange(
            current_thread_id(),
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub fn ensure_allowed(&self, operation: &str) -> Result<()> {
        if self.owner.load(Ordering::Acquire) == current_thread_id() {
            bail!("{operation} is forbidden during CUDA graph capture");
        }
        Ok(())
    }
}

/// A stable, non-zero id for the calling thread.
///
/// `std::thread::ThreadId` does not expose its numeric value, so we mint our
/// own from a process-wide counter the first time a thread is seen. Zero is
/// reserved for "no capture owner", so no live thread id is ever zero.
fn current_thread_id() -> u64 {
    use std::sync::atomic::AtomicU64 as Counter;
    thread_local! {
        static ID: u64 = {
            static NEXT: Counter = Counter::new(1);
            NEXT.fetch_add(1, Ordering::Relaxed)
        };
    }
    ID.with(|id| *id)
}

#[cfg(test)]
mod tests {
    use super::CaptureSafety;

    #[test]
    fn capture_state_rejects_nested_capture_and_unsafe_operations() {
        let state = CaptureSafety::new();
        state.ensure_allowed("allocation").unwrap();
        state.begin().unwrap();
        assert!(state.begin().is_err());
        assert!(state.ensure_allowed("allocation").is_err());
        state.finish();
        state.ensure_allowed("allocation").unwrap();
    }

    #[test]
    fn other_threads_may_allocate_while_a_capture_is_in_progress() {
        let state = CaptureSafety::new();
        state.begin().unwrap();
        let allowed = std::thread::scope(|scope| {
            scope
                .spawn(|| state.ensure_allowed("allocation").is_ok())
                .join()
                .unwrap()
        });
        assert!(allowed, "a non-capturing thread must not be blocked");
        state.finish();
    }

    #[test]
    fn a_foreign_thread_cannot_disarm_a_live_capture() {
        let state = CaptureSafety::new();
        state.begin().unwrap();
        std::thread::scope(|scope| scope.spawn(|| state.finish()).join().unwrap());
        assert!(
            state.ensure_allowed("allocation").is_err(),
            "a non-owner finish must not clear the owner"
        );
        state.finish();
        state.ensure_allowed("allocation").unwrap();
    }
}
