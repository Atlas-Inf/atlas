// SPDX-License-Identifier: AGPL-3.0-only

//! Markov anchor slot layout: the per-sequence `[prev token, banned draft
//! depth]` words and the batched banned-depth view.

use spark_runtime::gpu::DevicePtr;

use super::BlockDiffusionDraftHead;

/// Per-sequence anchor words: `[prev token, banned draft depth]` u32.
pub(super) const MARKOV_PREV_BYTES: usize = 8;
/// Byte offset of the banned draft depth within `markov_prev_dev`.
pub(super) const MARKOV_BAN_DEPTH_OFFSET: usize = 4;

impl BlockDiffusionDraftHead {
    /// `[batch_capacity]` banned draft depths, after the batch anchors in
    /// `batch_markov_prev`.
    pub(super) fn batch_ban_depth(&self) -> DevicePtr {
        self.batch_markov_prev.offset(self.batch_capacity * 4)
    }
}
