// SPDX-License-Identifier: AGPL-3.0-only

//! The host wait of the exact-tile sizing's expert-offset reads (a D2H copy
//! that drains the stream, once per MoE layer), reported on the
//! `ATLAS_PREFILL_HOST_TIMING` line (`moe_offset_wait`).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

static READ_US: AtomicU64 = AtomicU64::new(0);
static READS: AtomicU64 = AtomicU64::new(0);

fn host_timing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_PREFILL_HOST_TIMING").as_deref() == Ok("1"))
}

/// One expert-offset read being timed (`None` inside when not timing).
pub(super) struct OffsetRead(Option<Instant>);

impl OffsetRead {
    pub(super) fn start() -> Self {
        Self(host_timing().then(Instant::now))
    }

    pub(super) fn finish(self) {
        if let Some(t) = self.0 {
            READ_US.fetch_add(t.elapsed().as_micros() as u64, Relaxed);
            READS.fetch_add(1, Relaxed);
        }
    }
}

/// `(microseconds, reads)` of the expert-offset reads since the last call.
pub fn take_offset_read_wait() -> (u64, u64) {
    (READ_US.swap(0, Relaxed), READS.swap(0, Relaxed))
}
