// SPDX-License-Identifier: AGPL-3.0-only

//! The one-shot channel's settings.

use super::{MAX_LIMIT, MIN_MAX, RS_BIT, SPLIT_MIN};

/// The fused reduce-scatter's segment capacity on the four-rank chain: 16
/// MiB (whole 64 KiB), the four-rank profile's. Each rank pins
/// `fused::entries` of these after its command ring.
pub(in crate::nccl_backend::rdma_pair) const RS_FUSED_SEG: usize = 16 << 20;
const _: () = assert!(RS_FUSED_SEG.is_multiple_of(64 << 10));

/// One-shot settings; read from the shared profile, and checked equal on both
/// ranks at bootstrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::nccl_backend::rdma_pair) struct Config {
    /// Largest one-shot payload in bytes (`ATLAS_RDMA_ONESHOT_MAX`, 1 MiB,
    /// at least 512 KiB).
    pub(in crate::nccl_backend::rdma_pair) max: usize,
    /// Payloads from this size stripe over every rail
    /// (`ATLAS_RDMA_ONESHOT_STRIPE_MIN`, default the legacy split size).
    pub(in crate::nccl_backend::rdma_pair) stripe_min: usize,
    /// Kernel wait limit (`ATLAS_RDMA_ONESHOT_TIMEOUT_MS`, 30 s; 0 = none).
    pub(in crate::nccl_backend::rdma_pair) timeout_ns: u64,
    /// Publish `stage` with a fenced stream memop rather than from the
    /// kernel (`ATLAS_RDMA_ONESHOT_STAGE_FENCE=1`); local only, and ignored
    /// under `sm_stage`.
    pub(in crate::nccl_backend::rdma_pair) stage_fence: bool,
    /// The kernel stages the payload (`ATLAS_RDMA_ONESHOT_SM_STAGE`): 0 =
    /// copy engine (default), 1 = write-through SM stores, 2 = plain SM
    /// stores (stress-test control only).
    pub(in crate::nccl_backend::rdma_pair) sm_stage: u32,
    /// Launch the kernel with programmatic dependent launch under `sm_stage`
    /// (`ATLAS_RDMA_ONESHOT_PDL`, default 1); local only.
    pub(in crate::nccl_backend::rdma_pair) pdl: bool,
    /// The fused reduce-scatter's segment capacity in bytes ([`RS_FUSED_SEG`];
    /// 0 = off): a chain region of `fused::entries` of these. Every rank must
    /// agree (the chain head).
    pub(in crate::nccl_backend::rdma_pair) fused_seg: usize,
}

impl Config {
    /// `None` unless `ATLAS_RDMA_ONESHOT=1`.
    pub(in crate::nccl_backend::rdma_pair) fn from_env() -> Option<Self> {
        Self::parse(|key| std::env::var(key).ok())
    }

    pub(in crate::nccl_backend::rdma_pair) fn parse(
        var: impl Fn(&str) -> Option<String>,
    ) -> Option<Self> {
        if var("ATLAS_RDMA_ONESHOT").as_deref() != Some("1") {
            return None;
        }
        let num = |key: &str, default: usize| {
            var(key)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(default)
        };
        Some(Self {
            max: num("ATLAS_RDMA_ONESHOT_MAX", 1 << 20)
                .clamp(MIN_MAX, MAX_LIMIT)
                .next_multiple_of(64),
            stripe_min: num("ATLAS_RDMA_ONESHOT_STRIPE_MIN", SPLIT_MIN),
            timeout_ns: (num("ATLAS_RDMA_ONESHOT_TIMEOUT_MS", 30_000) as u64)
                .saturating_mul(1_000_000),
            stage_fence: var("ATLAS_RDMA_ONESHOT_STAGE_FENCE").as_deref() == Some("1"),
            sm_stage: match var("ATLAS_RDMA_ONESHOT_SM_STAGE").as_deref() {
                Some("1") => 1,
                Some("2") => 2,
                _ => 0,
            },
            pdl: var("ATLAS_RDMA_ONESHOT_PDL").as_deref() != Some("0"),
            fused_seg: RS_FUSED_SEG,
        })
    }

    /// These settings on a chain: its size field keeps [`RS_BIT`] for the
    /// reduce-scatter and the all-to-all, so a flood stays below it.
    pub(in crate::nccl_backend::rdma_pair) fn for_chain(self) -> Self {
        Self {
            max: self.max.min(RS_BIT - 64),
            ..self
        }
    }

    /// Whether a payload takes the one-shot path. Rank-invariant by
    /// construction: it looks at nothing but the size (never at addresses).
    pub(in crate::nccl_backend::rdma_pair) fn eligible(&self, bytes: usize) -> bool {
        bytes > 0 && bytes <= self.max && bytes.is_multiple_of(2)
    }

    /// Bootstrap form, `max`, `stripe_min` and `sm_stage`: both ranks must
    /// send the same.
    pub(in crate::nccl_backend::rdma_pair) fn wire(cfg: Option<Self>) -> [u64; 3] {
        cfg.map_or([0; 3], |c| {
            [c.max as u64, c.stripe_min as u64, u64::from(c.sm_stage)]
        })
    }
}
