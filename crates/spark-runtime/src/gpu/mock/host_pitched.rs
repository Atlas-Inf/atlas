// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl HostPitched for MockGpuBackend {
    fn h2d_retained(&self, src: &[u8], dst: DevicePtr, shape: Pitched, _stream: u64) -> Result<()> {
        // One tick per call, as `copy_d2d_2d_async`: the rows are emulation.
        self.host_pitched.fetch_add(1, Ordering::Relaxed);
        for r in 0..shape.height {
            let row = &src[r * shape.host_pitch..][..shape.width];
            self.copy_h2d(row, dst.offset(r * shape.dev_pitch))?;
        }
        Ok(())
    }

    fn d2h(&self, src: DevicePtr, dst: &mut [u8], shape: Pitched, _stream: u64) -> Result<()> {
        self.host_pitched.fetch_add(1, Ordering::Relaxed);
        let allocs = self.allocs.lock();
        for r in 0..shape.height {
            let from = src.offset(r * shape.dev_pitch);
            let (offset, alloc) = find_alloc(&allocs, from)
                .ok_or_else(|| anyhow::anyhow!("pitched D2H: ptr {from} not allocated"))?;
            dst[r * shape.host_pitch..][..shape.width]
                .copy_from_slice(&alloc.data[offset..offset + shape.width]);
        }
        Ok(())
    }
}
