// SPDX-License-Identifier: AGPL-3.0-only

//! Scratch lanes: which copy of the process-global scratch (cuBLASLt and
//! CUTLASS workspaces, `OpCache::scratch`) the calling thread's launches
//! take. Lane 0 unless a [`Scope`] says otherwise. Work issued on two streams
//! at once, each in its own lane, never shares a scratch byte; work on one
//! stream needs only lane 0, as before. A lane's copy is allocated on its
//! first use.

#[cfg(any(feature = "cuda", test))]
use anyhow::{Result, bail};
use std::cell::Cell;
#[cfg(any(feature = "cuda", test))]
use std::sync::OnceLock;

/// Lanes there are.
pub const LANES: usize = 2;

thread_local! {
    static CURRENT: Cell<usize> = const { Cell::new(0) };
}

/// The calling thread's lane.
pub fn current() -> usize {
    CURRENT.with(Cell::get)
}

/// Restores the previous lane on drop.
pub struct Scope(usize);

impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(self.0));
    }
}

/// Take lane `lane` (below [`LANES`]) for this thread's launches until the
/// scope drops.
pub fn enter(lane: usize) -> Scope {
    assert!(lane < LANES, "scratch lane {lane} of {LANES}");
    Scope(CURRENT.with(|c| c.replace(lane)))
}

/// A device workspace of `size` bytes per lane, each allocated by `alloc`
/// on its lane's first use.
#[cfg(any(feature = "cuda", test))]
pub(crate) struct LaneWorkspace {
    pub(crate) size: usize,
    ptrs: [OnceLock<u64>; LANES],
    alloc: fn(usize) -> Result<u64>,
}

#[cfg(any(feature = "cuda", test))]
impl LaneWorkspace {
    /// `first` is lane 0's, allocated already.
    pub(crate) fn new(size: usize, first: u64, alloc: fn(usize) -> Result<u64>) -> Self {
        let ptrs = [OnceLock::new(), OnceLock::new()];
        let _ = ptrs[0].set(first);
        Self { size, ptrs, alloc }
    }

    /// The calling thread's lane's workspace.
    pub(crate) fn get(&self) -> Result<u64> {
        let lane = &self.ptrs[current()];
        if let Some(&p) = lane.get() {
            return Ok(p);
        }
        let p = (self.alloc)(self.size)?;
        if lane.set(p).is_err() {
            bail!("scratch lane workspace allocated twice");
        }
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanes_scope_and_allocate_once_each() {
        fn alloc(size: usize) -> Result<u64> {
            Ok(0x1000 + size as u64)
        }
        let ws = LaneWorkspace::new(64, 0x10, alloc);
        assert_eq!(ws.get().unwrap(), 0x10);
        {
            let _one = enter(1);
            assert_eq!(current(), 1);
            assert_eq!(ws.get().unwrap(), 0x1040);
            assert_eq!(ws.get().unwrap(), 0x1040);
        }
        assert_eq!(current(), 0);
        assert_eq!(ws.get().unwrap(), 0x10);
    }
}
