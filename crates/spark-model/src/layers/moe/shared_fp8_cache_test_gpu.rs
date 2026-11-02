// SPDX-License-Identifier: AGPL-3.0-only
//! Metadata-only recording backend: no CUDA execution or numeric oracle claim.
use super::*;
use spark_runtime::gpu::KernelArg;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

#[derive(Debug, PartialEq)]
pub(super) enum Arg {
    Ptr(DevicePtr),
    Bytes(Vec<u8>),
}
#[derive(Debug)]
pub(super) struct Launch {
    pub kernel: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared: u32,
    pub stream: u64,
    pub args: Vec<Arg>,
}
pub(super) struct RecordingGpu {
    cache: spark_runtime::op_cache::OpCache,
    next: AtomicU64,
    pub effects: AtomicUsize,
    pub capturing: AtomicBool,
    pub launches: Mutex<Vec<Launch>>,
}
impl RecordingGpu {
    pub fn new() -> Self {
        Self {
            cache: spark_runtime::op_cache::OpCache::new(),
            next: AtomicU64::new(0x1000_0000),
            effects: AtomicUsize::new(0),
            capturing: AtomicBool::new(false),
            launches: Mutex::new(Vec::new()),
        }
    }
    fn effect(&self) {
        self.effects.fetch_add(1, Ordering::Relaxed);
    }
}
impl GpuBackend for RecordingGpu {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.effect();
        Ok(DevicePtr(self.next.fetch_add(
            (bytes as u64).div_ceil(256).max(1) * 256,
            Ordering::Relaxed,
        )))
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes)
    }
    fn free(&self, _: DevicePtr) -> Result<()> {
        self.effect();
        Ok(())
    }
    fn copy_h2d(&self, _: &[u8], _: DevicePtr) -> Result<()> {
        self.effect();
        Ok(())
    }
    fn copy_d2h(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
        self.effect();
        anyhow::bail!("unexpected D2H in dispatch test")
    }
    fn copy_d2d(&self, _: DevicePtr, _: DevicePtr, _: usize) -> Result<()> {
        self.effect();
        Ok(())
    }
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        anyhow::bail!("expected typed dispatch")
    }
    fn launch_typed(
        &self,
        kernel: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        self.launches.lock().unwrap().push(Launch {
            kernel: kernel.0,
            grid,
            block,
            shared,
            stream,
            args: args
                .iter()
                .map(|a| match a {
                    KernelArg::Buffer(p) => Arg::Ptr(*p),
                    KernelArg::Bytes(b) => Arg::Bytes(b.to_vec()),
                })
                .collect(),
        });
        Ok(())
    }
    fn stream_is_capturing(&self, _: u64) -> bool {
        self.capturing.load(Ordering::Relaxed)
    }
    fn synchronize(&self, _: u64) -> Result<()> {
        self.effect();
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, _: &str, _: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(1))
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        &self.cache
    }
    fn memset(&self, _: DevicePtr, _: u8, _: usize) -> Result<()> {
        self.effect();
        Ok(())
    }
    fn memset_async(&self, _: DevicePtr, _: u8, _: usize, _: u64) -> Result<()> {
        self.effect();
        Ok(())
    }
    fn total_memory(&self) -> Result<usize> {
        Ok(128usize << 30)
    }
    fn free_memory(&self) -> Result<usize> {
        Ok(120usize << 30)
    }
    fn sm_count(&self) -> Result<u32> {
        Ok(48)
    }
}
