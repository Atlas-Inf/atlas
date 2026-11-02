// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_MOE_PREFILL_ROW_BOUND=1` (default off): bound the hottest local
//! expert's rows by the chunk's token count instead of `tokens x top_k`.
//!
//! Top-k picks each expert at most once per token, so no expert receives more
//! sorted rows than the chunk has tokens (the B-tile arm of
//! `run_routed_grouped_gemm` already sizes its grid this way). A chunk of at
//! most 64 tokens therefore has one M64 tile per expert whatever the routing,
//! and the exact-tile sizing's expert-offset read (a D2H copy that drains the
//! stream, once per MoE layer: 42 per GLM-5.3-Flash pass) can only return the
//! value this bound gives. With the switch such a chunk skips the read: same
//! grid, same kernels, same output, and the host keeps queueing ahead of the
//! device instead of waiting for it at every MoE layer. Longer chunks read
//! the offsets as before; the bound never undercuts the exact count.
//!
//! Also here: the host wait of those reads, reported on the
//! `ATLAS_PREFILL_HOST_TIMING` line (`moe_offset_wait`).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

/// `ATLAS_MOE_PREFILL_ROW_BOUND=1`. Read once per process.
pub(super) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::forward_prefill_routed::env_flag("ATLAS_MOE_PREFILL_ROW_BOUND"))
}

/// The worst-case M64 tile count of the hottest expert for `num_tokens`
/// tokens routed `top_k` ways; with `row_bound` also capped at the token
/// count.
pub(super) fn worst_case_m_tiles(num_tokens: usize, top_k: usize, row_bound: bool) -> u32 {
    let routed = (num_tokens * top_k).div_ceil(64).max(1);
    let rows = if row_bound {
        routed.min(num_tokens.div_ceil(64).max(1))
    } else {
        routed
    };
    rows as u32
}

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

#[cfg(test)]
mod tests {
    use super::worst_case_m_tiles;

    #[test]
    fn row_bound_caps_short_chunks_at_one_tile() {
        // Off: tokens x top_k, as before (GLM top-8).
        assert_eq!(worst_case_m_tiles(8, 8, false), 1);
        assert_eq!(worst_case_m_tiles(9, 8, false), 2);
        assert_eq!(worst_case_m_tiles(50, 8, false), 7);
        assert_eq!(worst_case_m_tiles(1088, 8, false), 136);
        // On: no expert takes more rows than there are tokens.
        for n in 1..=64 {
            assert_eq!(worst_case_m_tiles(n, 8, true), 1, "n={n}");
        }
        assert_eq!(worst_case_m_tiles(65, 8, true), 2);
        assert_eq!(worst_case_m_tiles(1088, 8, true), 17);
        // Never above the old bound (top-1 routing: equal).
        for n in [1, 50, 64, 65, 500, 8192] {
            for k in [1, 2, 8] {
                assert!(worst_case_m_tiles(n, k, true) <= worst_case_m_tiles(n, k, false));
            }
            assert_eq!(
                worst_case_m_tiles(n, 1, true),
                worst_case_m_tiles(n, 1, false)
            );
        }
    }
}
