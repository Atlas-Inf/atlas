// SPDX-License-Identifier: AGPL-3.0-only
//! Where one owner-batched verify's metadata sits in scratch.

use super::*;

/// Scratch layout of one call's metadata: the joint block for every row at
/// `META_BASE`, then one block per owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MetaLayout {
    pub(super) joint_bytes: usize,
    pub(super) owner_bytes: usize,
    pub(super) owners: usize,
}

impl MetaLayout {
    pub(super) fn new(owners: usize, rows: usize, max_blocks: usize) -> Self {
        Self {
            joint_bytes: align(META_BLOCK_TABLE + owners * rows * max_blocks * 4),
            owner_bytes: align(META_BLOCK_TABLE + rows * max_blocks * 4),
            owners,
        }
    }
    /// Scratch offset of owner `o`'s metadata block.
    pub(super) fn owner_offset(&self, o: usize) -> usize {
        META_BASE + self.joint_bytes + o * self.owner_bytes
    }
    /// Scratch bytes the layout ends at.
    pub(super) fn end(&self) -> usize {
        self.owner_offset(self.owners)
    }
}
