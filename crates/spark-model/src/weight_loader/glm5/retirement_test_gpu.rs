// SPDX-License-Identifier: AGPL-3.0-only
//! Recording frees, including the CUDA remove-before-error ownership contract.
use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle, mock::MockGpuBackend};
use std::collections::HashSet;

#[derive(Default)]
pub(in crate::weight_loader::glm5::retirement) struct Gpu {
    pub inner: MockGpuBackend,
    pub frees: std::sync::Arc<Mutex<Vec<DevicePtr>>>,
    pub fail: Mutex<Option<DevicePtr>>,
    live: Mutex<HashSet<u64>>,
}
impl GpuBackend for Gpu {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        let ptr = self.inner.alloc(n)?;
        self.live.lock().insert(ptr.0);
        Ok(ptr)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.alloc(n)
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        self.frees.lock().push(ptr);
        ensure!(self.live.lock().remove(&ptr.0), "duplicate/foreign free");
        self.inner.free(ptr)?;
        ensure!(
            *self.fail.lock() != Some(ptr),
            "injected free outcome unknown"
        );
        Ok(())
    }
    fn copy_h2d(&self, b: &[u8], p: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(b, p)
    }
    fn copy_d2h(&self, p: DevicePtr, b: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(p, b)
    }
    fn copy_d2d(&self, a: DevicePtr, b: DevicePtr, n: usize) -> Result<()> {
        self.inner.copy_d2d(a, b, n)
    }
    fn memset(&self, p: DevicePtr, v: u8, n: usize) -> Result<()> {
        self.inner.memset(p, v, n)
    }
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, s: u64) -> Result<()> {
        self.inner.memset_async(p, v, n, s)
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        self.inner.synchronize(s)
    }
    fn default_stream(&self) -> u64 {
        self.inner.default_stream()
    }
    fn kernel(&self, _: &str, _: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(1))
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn launch(
        &self,
        k: KernelHandle,
        g: [u32; 3],
        b: [u32; 3],
        m: u32,
        s: u64,
        a: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.inner.launch(k, g, b, m, s, a)
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
    fn sm_count(&self) -> Result<u32> {
        self.inner.sm_count()
    }
}
