// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::{DevicePtr, KernelArg, mock::MockGpuBackend};
use std::{ffi::c_void, sync::Mutex};

#[derive(Default)]
struct Capture {
    inner: MockGpuBackend,
    arguments: Mutex<Vec<Vec<u8>>>,
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
        self.inner.synchronize(s)
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, module: &str, symbol: &str) -> Result<KernelHandle> {
        assert_eq!(module, "glm_hc_prefill_vec");
        assert_eq!(symbol, "glm_hc_pre_from_raw_mix_vec");
        Ok(KernelHandle(0x777))
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
        assert_eq!(kernel.0, 0x777);
        assert_eq!(grid, [2048, 1, 1]);
        assert_eq!(block, [256, 1, 1]);
        assert_eq!(shared, 0);
        assert_eq!(stream, 19);
        *self.arguments.lock().unwrap() = args
            .iter()
            .map(|a| match a {
                KernelArg::Buffer(p) => p.0.to_ne_bytes().to_vec(),
                KernelArg::Bytes(b) => b.to_vec(),
            })
            .collect();
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

#[test]
fn vector_prefill_uses_registered_kernel_and_unchanged_raw_mix_arguments() {
    let gpu = Capture::default();
    let kernel = select(&gpu, "glm5_next", KernelHandle(42), 2048, 4096, 4, true).unwrap();
    super::super::hc_pre_from_raw_mix(
        &gpu,
        kernel,
        DevicePtr(16),
        DevicePtr(32),
        DevicePtr(48),
        DevicePtr(64),
        DevicePtr(80),
        DevicePtr(96),
        DevicePtr(112),
        2048,
        4096,
        4,
        20,
        1e-5,
        1e-6,
        19,
    )
    .unwrap();
    let args = gpu.arguments.lock().unwrap();
    assert_eq!(args.len(), 12);
    for (i, p) in [16u64, 32, 48, 64, 80, 96, 112].iter().enumerate() {
        assert_eq!(args[i], p.to_ne_bytes());
    }
    assert_eq!(args[7], 4096u32.to_ne_bytes());
    assert_eq!(args[8], 4u32.to_ne_bytes());
    assert_eq!(args[9], 20u32.to_ne_bytes());
    assert_eq!(args[10], 1e-5f32.to_ne_bytes());
    assert_eq!(args[11], 1e-6f32.to_ne_bytes());
}

#[test]
fn vector_prefill_is_opt_in_shape_checked_and_preserves_small_rows() {
    let gpu = Capture::default();
    for tokens in [0, 1, 3, 5, 8] {
        assert_eq!(
            select(&gpu, "glm5_next", KernelHandle(42), tokens, 4096, 4, true)
                .unwrap()
                .0,
            42
        );
    }
    assert_eq!(
        select(&gpu, "glm5_next", KernelHandle(42), 9, 4096, 4, true)
            .unwrap()
            .0,
        0x777
    );
    assert_eq!(
        select(&gpu, "glm5_next", KernelHandle(42), 2048, 4096, 4, false)
            .unwrap()
            .0,
        42
    );
    assert_eq!(
        select(&gpu, "deepseek_v4", KernelHandle(42), 2048, 4096, 4, true)
            .unwrap()
            .0,
        42
    );
    assert!(select(&gpu, "glm5_next", KernelHandle(42), 2048, 2048, 4, true).is_err());
    assert!(select(&gpu, "glm5_next", KernelHandle(42), 2048, 4096, 2, true).is_err());
    assert!(!parse("glm5_next", None).unwrap());
    assert!(!parse("glm5_next", Some("0")).unwrap());
    assert!(parse("glm5_next", Some("1")).unwrap());
    assert!(parse("glm5_next", Some("true")).is_err());
}
