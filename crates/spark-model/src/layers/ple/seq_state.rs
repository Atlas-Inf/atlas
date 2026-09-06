// SPDX-License-Identifier: AGPL-3.0-only

//! The per-sequence PLE carry ([`PleSeqState`]). Child module of `layer`
//! (the layer, its verify/aux helpers and the gather guard read the
//! fields directly); split out for the ≤500 LoC cap.

use spark_runtime::gpu::DevicePtr;

/// Per-SEQUENCE carry: the dilated conv's 9 steps and the token history the
/// id hash needs. Owned by the sequence's [`crate::layer::SsmLayerState`]
/// (Avarok #753 item B: concurrency needs one of these per in-flight
/// sequence, not a layer singleton).
pub struct PleSeqState {
    /// `[(k-1)*dilation, channels]` FP32, device.
    pub(super) conv: DevicePtr,
    /// The last `context_len` token ids, EOS-filled at a sequence start.
    pub(super) history: Vec<u32>,
    /// Set by `prestage`: the n-gram table's device VA, recorded when the
    /// step's host work (hash + fault-in + slot upload) already ran BEFORE
    /// graph replay/capture. `forward` consumes it and enqueues kernels only.
    pub(super) prestaged_va: Option<u64>,
    /// How many token rows `prestaged_va` was staged for. A verify step
    /// stages K of them; a decode step stages 1. `forward` consumes the
    /// staging only when this matches its own `num_tokens` — a K=1 staging
    /// consumed by a K=2 forward would gather one row and read the second
    /// from whatever followed it.
    pub(super) prestaged_n: usize,
    /// Per-row conv snapshots for the speculative verify in flight, slot `t`
    /// holding the state after `t` rows (slot 0 = before the window). The
    /// conv carry advances once per token and is NOT part of the SSM
    /// checkpoint set, so a partially accepted verify would otherwise leave
    /// it conditioned on rejected drafts — the n-gram injection for every
    /// later token then reads a history that never happened.
    pub(super) verify_snaps: DevicePtr,
    /// Rows with a valid snapshot. 0 = none (a prefill-width forward skips
    /// the per-row split), so a rollback then has nothing to restore.
    pub(super) verify_snap_rows: usize,
    /// `history` as it stood before the window, plus the window's ids —
    /// together these rebuild the history for any accepted prefix.
    pub(super) history_ckpt: Vec<u32>,
    pub(super) verify_tokens: Vec<u32>,
    /// The last VA `prestage` staged, never cleared. `rearm` restores it when
    /// a failed capture attempt re-runs the step eagerly: the slots are still
    /// in `slots_dev` and history has already advanced, so re-hashing would
    /// double-count the token — re-arming is the only correct recovery.
    pub(super) last_staged_va: u64,
}
