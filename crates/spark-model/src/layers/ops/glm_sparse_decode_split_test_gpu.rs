// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::{DevicePtr, KernelArg, KernelHandle, mock::MockGpuBackend};
use std::{ffi::c_void, sync::Mutex};

#[derive(Default)]
pub(super) struct Capture {
    pub(super) inner: MockGpuBackend,
    pub(super) arguments: Mutex<Vec<(u64, [u32; 3], u32, Vec<Vec<u8>>)>>,
    pub(super) lookups: std::sync::atomic::AtomicUsize,
    pub(super) streams: Mutex<Vec<u64>>,
}

impl GpuBackend for Capture {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc(n)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(n)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.inner.free(p)
    }
    fn copy_h2d(&self, s: &[u8], d: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(s, d)
    }
    fn copy_d2h(&self, s: DevicePtr, d: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(s, d)
    }
    fn copy_d2d(&self, s: DevicePtr, d: DevicePtr, n: usize) -> Result<()> {
        self.inner.copy_d2d(s, d, n)
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        self.streams.lock().unwrap().push(s);
        self.inner.synchronize(s)
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, module: &str, symbol: &str) -> Result<KernelHandle> {
        assert!(module == "glm_sparse_decode_split" || module == "glm_sparse_decode_split_merge");
        self.lookups
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(KernelHandle(if symbol == "atlas_sparse_decode_split" {
            0x701
        } else {
            assert_eq!(symbol, "glm_sparse_decode_split_merge");
            0x702
        }))
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut c_void],
    ) -> Result<()> {
        bail!("expected the production typed launch")
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
        assert_eq!(block, [256, 1, 1]);
        assert!(stream == 19 || stream == 0);
        self.arguments.lock().unwrap().push((
            kernel.0,
            grid,
            shared,
            args.iter()
                .map(|a| match a {
                    KernelArg::Buffer(p) => p.0.to_ne_bytes().to_vec(),
                    KernelArg::Bytes(b) => b.to_vec(),
                })
                .collect(),
        ));
        Ok(())
    }
    fn memset(&self, p: DevicePtr, v: u8, n: usize) -> Result<()> {
        self.inner.memset(p, v, n)
    }
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, s: u64) -> Result<()> {
        self.inner.memset_async(p, v, n, s)
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

pub(super) fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_topk = 2048;
    c.index_kpool = 4;
    c
}
