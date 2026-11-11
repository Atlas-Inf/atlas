// SPDX-License-Identifier: AGPL-3.0-only
//! Layer side of the qwen4_exp mid-chunk prefix-cache checkpoint
//! (`ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT=1`; the model side is
//! `model::trait_impl::prefill_b::qwen4exp_ckpt`).
//!
//! Without it a prefix-caching prompt's last chunk runs as two passes, cut one
//! block below the prompt end, so the first pass can end where the next turn
//! restores (the "tail split"); with prefix caching off it runs as one pass,
//! and the two shapes give different bits. With it the last chunk is one pass
//! -- the caching-off numerics -- and the checkpoint is captured INSIDE that
//! pass at `cp`, the last 64-token chunk boundary of the pass at or below the
//! cut:
//! * each GDN layer's recurrent state: the state spine's FP32 entry state of
//!   chunk `cp / 64` (`gated_delta_rule_chunk_delta_h_*_cap`), or, on the
//!   token-sequential warm-replay recurrence, the state after a split at `cp`;
//! * each GDN layer's conv window: the conv split at `cp`
//!   (`conv1d_prefill_capture`, the existing mid-chunk machinery);
//! * the PLE conv carry: the PLE conv split at `cp` (this module's PLE hook);
//! * the PLE token history and the QSA indexer keys: rebuilt at `cp` by the
//!   model from the tokens and the pooled block keys.
//!
//! Every split here is exact by construction (the conv carries and the
//! token-sequential recurrence chain their state across a split; the spine
//! twin only adds a store), so the pass's outputs are unchanged by the
//! capture. The per-thread state below lives for one forward pass on the
//! thread that runs it, set and drained by the model around `forward_layers`.

use spark_runtime::gpu::DevicePtr;
use std::cell::{Cell, RefCell};

/// `ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT=1`.
pub fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// The spine's chunk length.
pub const CHUNK: usize = 64;

/// Most checkpoints one pass captures: the `_capn` spine twins' list
/// (`CDH_CAP_MAX` in `gated_delta_rule_fla.cu`).
pub const MAX_POINTS: usize = 16;

/// One more in-pass checkpoint beyond the plan's first (dense and
/// branch-point checkpoints, `model::trait_impl::prefill_b::qwen4exp_ckpt`):
/// its row in pass coordinates and its per-GDN-layer destinations.
pub struct CapPoint {
    pub cap_local: usize,
    pub h_dsts: Vec<DevicePtr>,
    pub conv_dsts: Vec<DevicePtr>,
}

thread_local! {
    /// A checkpoint pass is running on this thread.
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    /// Armed by a GDN layer, consumed by the spine launch: (chunk, dst).
    static SPINE: RefCell<Vec<(u32, u64)>> = const { RefCell::new(Vec::new()) };
    /// GDN layers whose recurrent state was captured this pass.
    static H_DONE: Cell<usize> = const { Cell::new(0) };
    /// PLE conv captures: (row of the PLE forward, dst), and how many landed.
    static PLE: RefCell<Vec<(usize, u64)>> = const { RefCell::new(Vec::new()) };
    static PLE_DONE: Cell<usize> = const { Cell::new(0) };
}

/// Start a checkpoint pass: the PLE conv carry at forward row `ple_row` goes
/// to `ple_dst`.
pub fn begin(ple_row: usize, ple_dst: DevicePtr) {
    begin_many(&[(ple_row, ple_dst)]);
}

/// Start a checkpoint pass capturing the PLE conv carry at each
/// `(forward row, dst)`.
pub fn begin_many(ple: &[(usize, DevicePtr)]) {
    ACTIVE.with(|c| c.set(true));
    SPINE.with(|c| c.borrow_mut().clear());
    H_DONE.with(|c| c.set(0));
    PLE.with(|c| *c.borrow_mut() = ple.iter().map(|&(r, d)| (r, d.0)).collect());
    PLE_DONE.with(|c| c.set(0));
}

/// End the pass: (GDN layers captured, PLE conv carries captured).
pub fn end() -> (usize, usize) {
    ACTIVE.with(|c| c.set(false));
    SPINE.with(|c| c.borrow_mut().clear());
    PLE.with(|c| c.borrow_mut().clear());
    (H_DONE.with(Cell::get), PLE_DONE.with(Cell::get))
}

/// A checkpoint pass is running on this thread.
pub fn active() -> bool {
    ACTIVE.with(Cell::get)
}

/// Ask the next GDN state-spine launch to store chunk `chunk`'s FP32 entry
/// state at `dst` (its `_cap` twin).
pub fn arm_spine(chunk: u32, dst: DevicePtr) {
    arm_spines(&[(chunk, dst)]);
}

/// Ask the next GDN state-spine launch to store each listed chunk's FP32
/// entry state at its dst (one entry: the `_cap` twin; more: `_capn`).
pub fn arm_spines(caps: &[(u32, DevicePtr)]) {
    SPINE.with(|c| *c.borrow_mut() = caps.iter().map(|&(ch, d)| (ch, d.0)).collect());
}

/// The armed spine captures (taken: one launch serves them; empty when none).
pub fn take_spine() -> Vec<(u32, DevicePtr)> {
    SPINE
        .with(|c| std::mem::take(&mut *c.borrow_mut()))
        .into_iter()
        .map(|(c, d)| (c, DevicePtr(d)))
        .collect()
}

/// The by-value `CdhCaps` list of the `_capn` twins: 16 pointers, 16 chunk
/// indices two to a word, then the count. `None` over [`MAX_POINTS`].
pub fn capn_words(caps: &[(u32, DevicePtr)]) -> Option<[u64; 25]> {
    if caps.is_empty() || caps.len() > MAX_POINTS {
        return None;
    }
    let mut w = [0u64; 25];
    for (j, &(chunk, dst)) in caps.iter().enumerate() {
        w[j] = dst.0;
        w[16 + j / 2] |= u64::from(chunk) << (32 * (j % 2));
    }
    w[24] = caps.len() as u64;
    Some(w)
}

/// A GDN layer's recurrent state landed.
pub fn h_captured() {
    H_DONE.with(|c| c.set(c.get() + 1));
}

/// The PLE conv capture points of this pass: (forward row, dst).
pub fn ple_captures() -> Vec<(usize, DevicePtr)> {
    if !active() {
        return Vec::new();
    }
    PLE.with(|c| c.borrow().iter().map(|&(r, d)| (r, DevicePtr(d))).collect())
}

/// A PLE conv carry landed.
pub fn ple_captured() {
    PLE_DONE.with(|c| c.set(c.get() + 1));
}

/// Split points of a pass of `rows` rows: each `(row, payload)` strictly
/// inside it, ascending and without repeats.
pub fn split_points<T: Copy>(
    points: impl IntoIterator<Item = (usize, T)>,
    rows: usize,
) -> Vec<(usize, T)> {
    let mut v: Vec<(usize, T)> = points
        .into_iter()
        .filter(|&(r, _)| r > 0 && r < rows)
        .collect();
    v.sort_by_key(|&(r, _)| r);
    v.dedup_by_key(|p| p.0);
    v
}

impl crate::layer::MidchunkCapture<'_> {
    /// Every capture point of the pass for SSM layer `idx`: (row, (h dst,
    /// conv dst)), ascending; empty unless ALL of them lie strictly inside a
    /// pass of `rows` rows (a partial capture is never registered).
    pub fn points(&self, idx: usize, rows: usize) -> Vec<(usize, (DevicePtr, DevicePtr))> {
        let planned = 1 + usize::from(self.cap_local_early.is_some()) + self.extra.len();
        let early = self
            .cap_local_early
            .map(|e| (e, (self.h_dsts_early[idx], self.conv_dsts_early[idx])));
        let main = (self.cap_local, (self.h_dsts[idx], self.conv_dsts[idx]));
        let extra = self
            .extra
            .iter()
            .map(|p| (p.cap_local, (p.h_dsts[idx], p.conv_dsts[idx])));
        let points = split_points(early.into_iter().chain([main]).chain(extra), rows);
        if points.len() == planned {
            points
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pass_counts_its_captures_and_ends_clean() {
        assert!(!active() && ple_captures().is_empty());
        begin(128, DevicePtr(0x1000));
        assert!(active());
        assert_eq!(ple_captures(), vec![(128, DevicePtr(0x1000))]);
        arm_spine(2, DevicePtr(0x2000));
        assert_eq!(take_spine(), vec![(2, DevicePtr(0x2000))]);
        assert!(take_spine().is_empty(), "one launch takes the arm");
        h_captured();
        h_captured();
        ple_captured();
        assert_eq!(end(), (2, 1));
        assert!(!active() && ple_captures().is_empty() && take_spine().is_empty());
    }

    #[test]
    fn many_points_arm_together() {
        begin_many(&[(64, DevicePtr(1)), (256, DevicePtr(2))]);
        assert_eq!(ple_captures().len(), 2);
        arm_spines(&[(1, DevicePtr(0x10)), (4, DevicePtr(0x20))]);
        assert_eq!(take_spine().len(), 2);
        ple_captured();
        ple_captured();
        assert_eq!(end(), (0, 2));
    }

    #[test]
    fn the_capn_list_packs_as_the_kernel_reads_it() {
        let w = capn_words(&[
            (3, DevicePtr(0xa0)),
            (8, DevicePtr(0xb0)),
            (12, DevicePtr(0xc0)),
        ])
        .unwrap();
        assert_eq!(&w[..3], &[0xa0, 0xb0, 0xc0]);
        assert_eq!(w[16], 3 | (8 << 32));
        assert_eq!(w[17], 12);
        assert_eq!(w[24], 3);
        assert!(capn_words(&[]).is_none());
        let many: Vec<_> = (0..17).map(|i| (i, DevicePtr(1))).collect();
        assert!(capn_words(&many).is_none());
    }

    #[test]
    fn split_points_are_inside_sorted_and_unique() {
        let p = split_points(
            [(512, 'a'), (0, 'z'), (128, 'b'), (512, 'c'), (1000, 'y')],
            1000,
        );
        assert_eq!(p, vec![(128, 'b'), (512, 'a')]);
    }
}

#[cfg(test)]
#[path = "qwen4exp_ckpt_gpu_tests.rs"]
mod gpu_tests;
