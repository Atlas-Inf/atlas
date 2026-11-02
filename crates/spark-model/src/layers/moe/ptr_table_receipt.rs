// SPDX-License-Identifier: AGPL-3.0-only
//! Allocation facts can only be minted by the real pointer-table builders.
use super::ExpertPtrTable;
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub(in crate::layers::moe) struct TableAllocation {
    backend: usize,
    slots: usize,
    regions: [(DevicePtr, usize); 3],
}
impl TableAllocation {
    pub(super) fn completed(
        gpu: &dyn GpuBackend,
        slots: usize,
        regions: [(DevicePtr, usize); 3],
    ) -> Self {
        Self {
            backend: gpu as *const dyn GpuBackend as *const () as usize,
            slots,
            regions,
        }
    }
}
impl ExpertPtrTable {
    pub(in crate::layers::moe) fn owned_regions(
        &self,
        gpu: &dyn GpuBackend,
        slots: usize,
    ) -> Result<[(DevicePtr, usize); 3]> {
        let receipt = self
            .allocation
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("nonowning table"))?;
        ensure!(
            receipt.backend == gpu as *const dyn GpuBackend as *const () as usize
                && receipt.slots == slots,
            "table backend/slot authority mismatch"
        );
        ensure!(
            receipt.regions.map(|r| r.0) == [self.packed_ptrs, self.scale_ptrs, self.scale2_vals],
            "stale table allocation receipt"
        );
        Ok(receipt.regions)
    }
}
