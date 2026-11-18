// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub(super) struct Owned<'a> {
    pub gpu: &'a dyn GpuBackend,
    ptrs: Vec<DevicePtr>,
    live: bool,
}
impl<'a> Owned<'a> {
    pub fn new(gpu: &'a dyn GpuBackend) -> Self {
        Self {
            gpu,
            ptrs: Vec::new(),
            live: true,
        }
    }
    pub fn is_live(&self) -> bool {
        self.live
    }
    pub fn release(&mut self) -> Result<()> {
        self.live = false;
        while let Some(ptr) = self.ptrs.last() {
            self.gpu.free(*ptr)?;
            self.ptrs.pop();
        }
        Ok(())
    }
    pub fn alloc(&mut self, bytes: usize) -> Result<DevicePtr> {
        ensure!(bytes > 0, "Hy4 empty device allocation");
        ensure!(self.live, "Hy4 allocation owner already released");
        let ptr = self.gpu.alloc(bytes)?;
        self.ptrs.push(ptr);
        Ok(ptr)
    }
    pub fn upload(&mut self, bytes: &[u8]) -> Result<DevicePtr> {
        let ptr = self.alloc(bytes.len())?;
        self.gpu.copy_h2d(bytes, ptr)?;
        Ok(ptr)
    }
}
impl Drop for Owned<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.release() {
            tracing::error!("Hy4 attention cleanup: {e:#}");
        }
    }
}
