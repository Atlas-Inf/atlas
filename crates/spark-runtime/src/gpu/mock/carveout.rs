// SPDX-License-Identifier: AGPL-3.0-only

//! The mock's lendable carveout ([`MockGpuBackend::with_carveout`]).

use super::{MockAlloc, MockGpuBackend};
use crate::gpu::DevicePtr;
use crate::gpu::carveout::CarveoutArena;
use anyhow::Result;

/// Far above the mock's ordinary allocations.
const CARVEOUT_BASE: u64 = 1 << 44;

impl MockGpuBackend {
    /// A mock that lends a `bytes` carveout.
    pub fn with_carveout(bytes: usize) -> Self {
        let mock = Self::new();
        *mock.carveout.lock() = Some(CarveoutArena::new(CARVEOUT_BASE, bytes));
        mock
    }

    /// Carveout bytes in use.
    pub fn carveout_used(&self) -> usize {
        self.carveout.lock().as_ref().map_or(0, |a| a.used())
    }

    /// Requested sizes of the live carveout allocations, in address order.
    pub fn carveout_alloc_sizes(&self) -> Vec<usize> {
        let mut live: Vec<_> = self
            .allocs
            .lock()
            .iter()
            .filter(|(ptr, _)| **ptr >= CARVEOUT_BASE)
            .map(|(ptr, a)| (*ptr, a.bytes))
            .collect();
        live.sort_unstable();
        live.into_iter().map(|(_, bytes)| bytes).collect()
    }

    pub(super) fn carveout_bytes(&self) -> usize {
        self.carveout.lock().as_ref().map_or(0, |a| a.capacity())
    }

    pub(super) fn carveout_alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let mut arena = self.carveout.lock();
        let Some(arena) = arena.as_mut() else {
            anyhow::bail!("mock has no carveout");
        };
        let ptr = arena.alloc(bytes)?;
        let data = vec![0u8; bytes];
        self.allocs.lock().insert(ptr, MockAlloc { bytes, data });
        Ok(DevicePtr(ptr))
    }

    pub(super) fn carveout_release(&self, ptr: DevicePtr) {
        if let Some(arena) = self.carveout.lock().as_mut() {
            arena.free(ptr.0);
        }
    }
}
