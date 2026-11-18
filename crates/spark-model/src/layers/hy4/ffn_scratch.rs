// SPDX-License-Identifier: AGPL-3.0-only

use super::ffn::Hy4Ffn;
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Per-call scratch, never shared across concurrent streams/requests.
pub struct Hy4FfnScratch<'a> {
    gpu: &'a dyn GpuBackend,
    allocation: DevicePtr,
    pub(super) capacity: usize,
    pub(super) signature: [usize; 7],
    pub(super) ptrs: [DevicePtr; 8],
    pub(super) groups: DevicePtr,
    pub(super) partials: DevicePtr,
}
impl Hy4Ffn<'_> {
    pub(super) fn scratch_signature(&self) -> [usize; 7] {
        [
            self.h,
            self.latent,
            self.inter,
            self.shared_inter,
            self.ep.experts,
            self.routing.top_k,
            usize::from(self.dense),
        ]
    }
}
impl<'a> Hy4FfnScratch<'a> {
    pub fn new(block: &Hy4Ffn<'a>, capacity: usize) -> Result<Self> {
        ensure!(
            capacity <= 65535,
            "Hy4 scratch row count exceeds CUDA grid.y limit"
        );
        let [h, l, i, s, e, k, _] = block.scratch_signature();
        let intermediate = k
            .checked_mul(i)
            .ok_or_else(|| anyhow::anyhow!("Hy4 scratch overflow"))?
            .max(s);
        let widths = [
            e * 4,
            k * 4,
            k * 4,
            l * 2,
            intermediate * 2,
            l * 2,
            h * 2,
            h * 2,
            e * 12,
            k * h.max(l) * 2,
        ];
        let mut offsets = [0; 10];
        let mut bytes = 0usize;
        for (at, width) in widths.into_iter().enumerate() {
            offsets[at] = bytes;
            bytes = width
                .checked_mul(capacity)
                .and_then(|v| v.checked_add(7))
                .map(|v| v & !7)
                .and_then(|v| bytes.checked_add(v))
                .ok_or_else(|| anyhow::anyhow!("Hy4 scratch overflow"))?;
        }
        let allocation = if bytes == 0 {
            DevicePtr::NULL
        } else {
            block.gpu.alloc(bytes)?
        };
        Ok(Self {
            gpu: block.gpu,
            allocation,
            capacity,
            signature: block.scratch_signature(),
            ptrs: std::array::from_fn(|i| {
                if allocation.is_null() {
                    allocation
                } else {
                    allocation.offset(offsets[i])
                }
            }),
            groups: allocation.offset(offsets[8]),
            partials: allocation.offset(offsets[9]),
        })
    }
    pub fn release(&mut self) -> Result<()> {
        if !self.allocation.is_null() {
            self.gpu.free(self.allocation)?;
            self.allocation = DevicePtr::NULL;
        }
        self.capacity = 0;
        self.ptrs = [DevicePtr::NULL; 8];
        self.groups = DevicePtr::NULL;
        self.partials = DevicePtr::NULL;
        Ok(())
    }
}
impl Drop for Hy4FfnScratch<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.release() {
            tracing::error!("Hy4 scratch cleanup: {e:#}");
        }
    }
}
