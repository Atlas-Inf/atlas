// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
// The generic mock intentionally returns one dummy handle for every symbol.
// Distinguish the real TC lookup so this layer test observes dispatch order.
#[derive(Default)]
pub(super) struct TestGpu(
    pub(super) MockGpuBackend,
    pub(super) std::sync::Mutex<Vec<(u64, Vec<Vec<u8>>)>>,
);
impl std::ops::Deref for TestGpu {
    type Target = MockGpuBackend;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl GpuBackend for TestGpu {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        self.0.alloc(n)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.0.alloc_managed(n)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.0.free(p)
    }
    fn copy_h2d(&self, s: &[u8], d: DevicePtr) -> Result<()> {
        self.0.copy_h2d(s, d)
    }
    fn copy_d2h(&self, s: DevicePtr, d: &mut [u8]) -> Result<()> {
        self.0.copy_d2h(s, d)
    }
    fn copy_d2d(&self, s: DevicePtr, d: DevicePtr, n: usize) -> Result<()> {
        self.0.copy_d2d(s, d, n)
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        self.0.synchronize(s)
    }
    fn default_stream(&self) -> u64 {
        self.0.default_stream()
    }
    fn memset(&self, p: DevicePtr, value: u8, n: usize) -> Result<()> {
        self.0.memset(p, value, n)
    }
    fn memset_async(&self, p: DevicePtr, value: u8, n: usize, s: u64) -> Result<()> {
        self.0.memset_async(p, value, n, s)
    }
    fn total_memory(&self) -> Result<usize> {
        self.0.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.0.free_memory()
    }
    fn sm_count(&self) -> Result<u32> {
        self.0.sm_count()
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.0.op_cache()
    }
    fn kernel(&self, module: &str, symbol: &str) -> Result<KernelHandle> {
        if module == "glm_sparse_decode_split" && symbol == "atlas_sparse_decode_split" {
            return Ok(KernelHandle(809));
        }
        if module == "glm_sparse_decode_split_merge" && symbol == "glm_sparse_decode_split_merge" {
            return Ok(KernelHandle(810));
        }
        if module == "glm_sparse_prefill_kv_reuse"
            && symbol == "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad"
        {
            Ok(KernelHandle(805))
        } else {
            self.0.kernel(module, symbol)
        }
    }
    fn launch_typed(
        &self,
        kernel: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: u64,
        args: &[spark_runtime::gpu::KernelArg<'_>],
    ) -> Result<()> {
        self.1.lock().unwrap().push((
            kernel.0,
            args.iter()
                .map(|arg| match arg {
                    spark_runtime::gpu::KernelArg::Buffer(p) => p.0.to_ne_bytes().to_vec(),
                    spark_runtime::gpu::KernelArg::Bytes(bytes) => bytes.to_vec(),
                })
                .collect(),
        ));
        self.0
            .launch_typed(kernel, grid, block, shared, stream, args)
    }
    fn launch(
        &self,
        kernel: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: u64,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.0.launch(kernel, grid, block, shared, stream, args)
    }
}
