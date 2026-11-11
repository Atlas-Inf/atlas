// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduling policy trait (SDD: FIFO vs SLAI).
//!
//! Controls two decisions in the scheduler loop:
//! 1. Whether to accept new prefills or prioritize decode (TBT deadline).
//! 2. Which pending requests to prefill and in what order.
//!
//! Implementations:
//! - [`FifoPolicy`]: always prefill, take first N from queue (current behavior).
//! - [`SlaiPolicy`]: skip prefills when active sequences approach TBT deadline,
//!   select the oldest pending request, then shortest prompts first from ALL
//!   pending (SLAI — arXiv:2407.08353, plus a no-starvation head slot).

use std::time::{Duration, Instant};

/// Metadata about a pending request for selection decisions.
pub struct PendingRequestInfo {
    /// Number of prompt tokens (determines prefill cost).
    pub prompt_len: usize,
    /// Index into the full pending requests vec.
    pub index: usize,
}

/// Per-sequence timing for decode urgency decisions.
pub struct ActiveSeqTiming {
    /// When the last token was emitted for this sequence.
    pub last_token_time: Instant,
}

/// Scheduling policy controlling prefill admission and ordering.
pub trait SchedulingPolicy: Send {
    /// Whether to accept new prefills this iteration.
    ///
    /// Returns `false` to skip prefill and proceed directly to decode
    /// (e.g., when active sequences approach their TBT deadline).
    fn should_prefill(&self, active_timings: &[ActiveSeqTiming]) -> bool;

    /// Select up to `capacity` requests from ALL pending, in prefill order.
    ///
    /// Returns indices into `requests` for the selected items, ordered
    /// by desired prefill execution order. FIFO takes the first N;
    /// SLAI picks the N shortest prompts.
    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize>;

    /// Number of prefill tokens to inject this iteration when fusing a
    /// prefill chunk into a decode step ("always-mixed" path).
    ///
    /// Returns a token budget in `[0, full_chunk]`:
    /// - `full_chunk` when no decode is active, or under moderate decode
    ///   pressure — fuse the WHOLE chunk (measured: shrinking the slice does
    ///   not lower decode TBT because the fused step's full-forward floor
    ///   dominates; it only slows prefill).
    /// - `0` ONLY as a hard suppress when a decode has already blown its
    ///   TBT deadline — the caller must then run decode-only this tick.
    fn prefill_slice_budget(&self, active_timings: &[ActiveSeqTiming], full_chunk: usize) -> usize {
        // Default (FIFO / unaware): inject the full chunk — same as today.
        let _ = active_timings;
        full_chunk
    }

    /// Policy name for logging.
    fn name(&self) -> &str;
}

/// FIFO scheduling: always prefill, take first N from queue.
pub struct FifoPolicy;

impl SchedulingPolicy for FifoPolicy {
    fn should_prefill(&self, _active_timings: &[ActiveSeqTiming]) -> bool {
        true
    }

    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize> {
        // First N in queue order (FIFO).
        (0..requests.len().min(capacity)).collect()
    }

    fn name(&self) -> &str {
        "fifo"
    }
}

/// SLO-aware scheduling (SLAI-inspired).
///
/// - Skips prefills when any active sequence waited > 80% of `tbt_deadline`
///   since its last token emission (decode-first priority).
/// - Selects the oldest pending request first, then the shortest prompts from
///   ALL pending (reduces median TTFT without starving long prompts).
pub struct SlaiPolicy {
    tbt_deadline: Duration,
}

impl SlaiPolicy {
    pub fn new(tbt_deadline_ms: u64) -> Self {
        Self {
            tbt_deadline: Duration::from_millis(tbt_deadline_ms),
        }
    }
}

impl SchedulingPolicy for SlaiPolicy {
    fn should_prefill(&self, active_timings: &[ActiveSeqTiming]) -> bool {
        if active_timings.is_empty() {
            return true;
        }
        let now = Instant::now();
        let margin = self.tbt_deadline.mul_f64(0.8);
        for timing in active_timings {
            if now.duration_since(timing.last_token_time) >= margin {
                return false;
            }
        }
        true
    }

    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize> {
        // Shortest prompts first, EXCEPT the oldest pending request, which
        // always takes the first slot. Pure shortest-first never admits a long
        // prompt while shorter ones keep arriving: on GB10 at 36 agentic
        // streams over 32 slots, 85k-token prompts waited ~4 h, hit the 4 h
        // turn timeout and aborted 255 follow-on turns (MLPerf C=36 point,
        // 2026-10-03). The pending queue keeps arrival order (admission pushes
        // overflow back to the FRONT), so index 0 is the oldest, and a request
        // now waits at most for the requests ahead of it.
        if capacity == 0 || requests.is_empty() {
            return Vec::new();
        }
        let oldest = requests
            .iter()
            .min_by_key(|r| r.index)
            .map(|r| r.index)
            .unwrap_or(0);
        let mut rest: Vec<usize> = (0..requests.len())
            .filter(|&i| requests[i].index != oldest)
            .collect();
        rest.sort_by_key(|&i| requests[i].prompt_len);
        let first = (0..requests.len())
            .find(|&i| requests[i].index == oldest)
            .unwrap_or(0);
        let mut indices = Vec::with_capacity(capacity.min(requests.len()));
        indices.push(first);
        indices.extend(rest);
        indices.truncate(capacity);
        indices
    }

    fn prefill_slice_budget(&self, active_timings: &[ActiveSeqTiming], full_chunk: usize) -> usize {
        // No decode active → no TBT pressure → inject the full chunk.
        if active_timings.is_empty() {
            return full_chunk;
        }

        // Hard suppress: if ANY decode has already blown its TBT deadline,
        // return 0 so the caller runs decode-only this tick — let the late
        // decode catch up (a fused step would make it wait a whole forward).
        let now = Instant::now();
        let worst = active_timings
            .iter()
            .map(|t| now.duration_since(t.last_token_time))
            .max()
            .unwrap_or_default();
        if worst >= self.tbt_deadline {
            return 0;
        }

        // Otherwise fuse the FULL chunk. Measured (varied-load burst A/B,
        // 2026-06-24): the fused step has a ~250ms full-forward floor that
        // DOMINATES, so SHRINKING the prefill slice does NOT lower decode TBT
        // — it only multiplies 250ms steps and slows prefill (slice=32 → 4.6×
        // slower prefill for no TBT gain). A full-chunk slice keeps prefill at
        // flag-off speed AND still halves the decode-freeze p99 (2529→1285ms)
        // by riding decode on every chunk. So: fuse decode into the normal
        // chunk, never shrink it. (The earlier cost-driven/EWMA shrink was the
        // wrong lever for this model and was removed.)
        full_chunk
    }

    fn name(&self) -> &str {
        "slai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_always_prefills() {
        let policy = FifoPolicy;
        let timings = vec![ActiveSeqTiming {
            last_token_time: Instant::now(),
        }];
        assert!(policy.should_prefill(&timings));
        assert!(policy.should_prefill(&[]));
    }

    #[test]
    fn fifo_selects_first_n() {
        let policy = FifoPolicy;
        let requests = vec![
            PendingRequestInfo {
                prompt_len: 100,
                index: 0,
            },
            PendingRequestInfo {
                prompt_len: 10,
                index: 1,
            },
            PendingRequestInfo {
                prompt_len: 50,
                index: 2,
            },
            PendingRequestInfo {
                prompt_len: 200,
                index: 3,
            },
        ];
        assert_eq!(policy.select_prefills(&requests, 2), vec![0, 1]);
        assert_eq!(policy.select_prefills(&requests, 10), vec![0, 1, 2, 3]);
    }

    #[test]
    fn slai_prefills_when_no_active() {
        let policy = SlaiPolicy::new(100);
        assert!(policy.should_prefill(&[]));
    }

    #[test]
    fn slai_prefills_when_fresh() {
        let policy = SlaiPolicy::new(100);
        let timings = vec![ActiveSeqTiming {
            last_token_time: Instant::now(),
        }];
        assert!(policy.should_prefill(&timings));
    }

    #[test]
    fn slai_skips_prefill_near_deadline() {
        let policy = SlaiPolicy::new(100); // 80ms margin
        let old_time = Instant::now() - Duration::from_millis(85);
        let timings = vec![ActiveSeqTiming {
            last_token_time: old_time,
        }];
        assert!(!policy.should_prefill(&timings));
    }

    #[test]
    fn slai_prefills_within_margin() {
        let policy = SlaiPolicy::new(100); // 80ms margin
        let recent = Instant::now() - Duration::from_millis(50);
        let timings = vec![ActiveSeqTiming {
            last_token_time: recent,
        }];
        assert!(policy.should_prefill(&timings));
    }

    #[test]
    fn slai_one_urgent_blocks_prefill() {
        let policy = SlaiPolicy::new(100);
        let now = Instant::now();
        let timings = vec![
            ActiveSeqTiming {
                last_token_time: now,
            },
            ActiveSeqTiming {
                last_token_time: now - Duration::from_millis(90),
            },
        ];
        assert!(!policy.should_prefill(&timings));
    }

    #[test]
    fn slai_selects_shortest_from_all() {
        let policy = SlaiPolicy::new(100);
        let requests = vec![
            PendingRequestInfo {
                prompt_len: 500,
                index: 0,
            },
            PendingRequestInfo {
                prompt_len: 10,
                index: 1,
            },
            PendingRequestInfo {
                prompt_len: 200,
                index: 2,
            },
            PendingRequestInfo {
                prompt_len: 50,
                index: 3,
            },
            PendingRequestInfo {
                prompt_len: 300,
                index: 4,
            },
        ];
        // Capacity 3: the oldest (index 0, 500 tokens) takes the head slot,
        // then the shortest of the rest → 0, 1(10), 3(50)
        assert_eq!(policy.select_prefills(&requests, 3), vec![0, 1, 3]);
    }

    #[test]
    fn slai_selects_all_when_capacity_exceeds() {
        let policy = SlaiPolicy::new(100);
        let requests = vec![
            PendingRequestInfo {
                prompt_len: 100,
                index: 0,
            },
            PendingRequestInfo {
                prompt_len: 10,
                index: 1,
            },
        ];
        // Capacity 10 > 2 requests: oldest first, then the rest by length
        assert_eq!(policy.select_prefills(&requests, 10), vec![0, 1]);
    }

    #[test]
    fn slai_stable_order_for_equal_lengths() {
        let policy = SlaiPolicy::new(100);
        let requests = vec![
            PendingRequestInfo {
                prompt_len: 50,
                index: 0,
            },
            PendingRequestInfo {
                prompt_len: 50,
                index: 1,
            },
            PendingRequestInfo {
                prompt_len: 50,
                index: 2,
            },
        ];
        assert_eq!(policy.select_prefills(&requests, 3), vec![0, 1, 2]);
    }

    /// The starvation case from the GB10 MLPerf run: a long prompt at the
    /// head of the queue while shorter ones keep arriving. Pure shortest-first
    /// never picks it at capacity 1; the head slot does, on the first tick.
    #[test]
    fn slai_never_starves_the_oldest_long_prompt() {
        let policy = SlaiPolicy::new(100);
        let mut requests = vec![PendingRequestInfo {
            prompt_len: 85_291,
            index: 0,
        }];
        for i in 1..40 {
            requests.push(PendingRequestInfo {
                prompt_len: 2_000 + i,
                index: i,
            });
        }
        assert_eq!(policy.select_prefills(&requests, 1), vec![0]);
        // With more room the remaining slots are still shortest-first.
        assert_eq!(policy.select_prefills(&requests, 3), vec![0, 1, 2]);
        assert!(policy.select_prefills(&requests, 0).is_empty());
    }

    #[test]
    fn select_prefills_empty() {
        assert!(FifoPolicy.select_prefills(&[], 5).is_empty());
        assert!(SlaiPolicy::new(100).select_prefills(&[], 5).is_empty());
    }

    #[test]
    fn fifo_slice_budget_is_full_chunk() {
        // Default trait impl: FIFO always injects the full chunk.
        let policy = FifoPolicy;
        assert_eq!(policy.prefill_slice_budget(&[], 4080), 4080);
        let timings = vec![ActiveSeqTiming {
            last_token_time: Instant::now(),
        }];
        assert_eq!(policy.prefill_slice_budget(&timings, 4080), 4080);
    }

    #[test]
    fn slai_slice_budget_full_when_no_active() {
        let policy = SlaiPolicy::new(100);
        assert_eq!(policy.prefill_slice_budget(&[], 4080), 4080);
    }

    #[test]
    fn slai_slice_budget_zero_past_deadline() {
        // worst >= tbt_deadline → hard suppress (0), decode-only this tick.
        let policy = SlaiPolicy::new(100);
        let timings = vec![ActiveSeqTiming {
            last_token_time: Instant::now() - Duration::from_millis(120),
        }];
        assert_eq!(policy.prefill_slice_budget(&timings, 4080), 0);
    }

    #[test]
    fn slai_slice_budget_bounded_and_wy4_aligned() {
        // Fresh decode, under deadline → a positive, WY4-aligned slice in
        // [min, full_chunk], never 0.
        let policy = SlaiPolicy::new(100);
        let timings = vec![ActiveSeqTiming {
            last_token_time: Instant::now(),
        }];
        let b = policy.prefill_slice_budget(&timings, 4080);
        assert!(b > 0, "non-deadline budget must be > 0");
        assert!(b <= 4080, "must never exceed full_chunk");
        assert_eq!(b % 4, 0, "must be WY4-aligned");
    }

    #[test]
    fn slai_slice_budget_never_exceeds_small_full_chunk() {
        // Small chunk cap must clamp the slice (and stay WY4-aligned).
        let policy = SlaiPolicy::new(100);
        let timings = vec![ActiveSeqTiming {
            last_token_time: Instant::now(),
        }];
        let b = policy.prefill_slice_budget(&timings, 64);
        assert!(b <= 64);
        assert_eq!(b % 4, 0);
    }
}
