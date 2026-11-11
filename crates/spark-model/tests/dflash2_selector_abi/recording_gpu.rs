// SPDX-License-Identifier: AGPL-3.0-only
//! Capture typed launch arguments without executing device code.
use anyhow::Result;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};
use spark_runtime::op_cache::OpCache;
use std::sync::Mutex;

#[derive(Default)]
pub struct RecordingGpu {
    pub inner: MockGpuBackend,
    pub arguments: Mutex<Vec<Vec<(bool, u64)>>>,
}

impl std::ops::Deref for RecordingGpu {
    type Target = MockGpuBackend;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

macro_rules! delegate {
    ($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        fn $name(&self, $($arg: $ty),*) -> $ret { self.inner.$name($($arg),*) }
    };
}

impl GpuBackend for RecordingGpu {
    delegate!(alloc(bytes: usize) -> Result<DevicePtr>);
    delegate!(alloc_managed(bytes: usize) -> Result<DevicePtr>);
    delegate!(free(ptr: DevicePtr) -> Result<()>);
    delegate!(copy_h2d(src: &[u8], dst: DevicePtr) -> Result<()>);
    delegate!(copy_d2h(src: DevicePtr, dst: &mut [u8]) -> Result<()>);
    delegate!(copy_d2d(src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()>);
    delegate!(synchronize(stream: u64) -> Result<()>);
    delegate!(default_stream() -> u64);
    delegate!(kernel(module: &str, name: &str) -> Result<KernelHandle>);
    delegate!(op_cache() -> &OpCache);
    delegate!(memset(ptr: DevicePtr, value: u8, bytes: usize) -> Result<()>);
    delegate!(memset_async(ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()>);
    delegate!(total_memory() -> Result<usize>);
    delegate!(free_memory() -> Result<usize>);
    delegate!(sm_count() -> Result<u32>);

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.inner
            .launch(func, grid, block, shared_mem, stream, params)
    }

    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        let values = args
            .iter()
            .map(|arg| match arg {
                KernelArg::Buffer(ptr) => (true, ptr.0),
                KernelArg::Bytes(bytes) => {
                    assert_eq!(bytes.len(), 4, "selector scalar ABI is u32");
                    (
                        false,
                        u32::from_ne_bytes((*bytes).try_into().unwrap()) as u64,
                    )
                }
            })
            .collect();
        self.arguments.lock().unwrap().push(values);
        self.inner
            .launch_typed(func, grid, block, shared_mem, stream, args)
    }
}
