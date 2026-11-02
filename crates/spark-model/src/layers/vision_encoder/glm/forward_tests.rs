// SPDX-License-Identifier: AGPL-3.0-only

//! Launch-shape regressions for the native GLM visual tower.
//!
//! The first native tower ran scalar reference kernels (one thread per
//! (query, head) recomputing every key's norm and RoPE per output channel), so
//! a 1024x1024 image kept the head's default stream busy for hours while the
//! EP worker idled at its next command: a hang in production.

use std::ffi::c_void;
use std::sync::Mutex;

use anyhow::Result;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::GLM_MAX_OUTPUT_ROWS;
use super::test_tower::{HEADS, encoder, image};

/// Mock backend that hands out one handle per requested kernel name.
#[derive(Default)]
struct NamedGpu {
    inner: MockGpuBackend,
    names: Mutex<Vec<String>>,
}

impl NamedGpu {
    /// `(kernel name, grid, block)` for every launch so far.
    fn launches(&self) -> Vec<(String, [u32; 3], [u32; 3])> {
        let names = self.names.lock().unwrap();
        self.inner
            .launches_snapshot()
            .into_iter()
            .map(|l| (names[l.func as usize - 1].clone(), l.grid, l.block))
            .collect()
    }
}

impl GpuBackend for NamedGpu {
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
        let mut names = self.names.lock().unwrap();
        names.push(format!("{module}::{symbol}"));
        Ok(KernelHandle(names.len() as u64))
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn launch(
        &self,
        kernel: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: u64,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        self.inner
            .launch(kernel, grid, block, shared, stream, params)
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
fn every_launch_is_a_parallel_tile_and_attention_is_flash_tiled() {
    let gpu = NamedGpu::default();
    let enc = encoder(&gpu);
    let item = image(16);
    let geometry = enc.forward_items(&[&item], &gpu, 0).unwrap();
    assert_eq!(geometry, vec![(8, 8, 64)]);

    let launches = gpu.launches();
    for (name, grid, block) in &launches {
        assert!(
            block.iter().product::<u32>() >= 32,
            "{name} launched {grid:?} with a {block:?} block; single-thread \
             blocks made a 1024x1024 image take hours"
        );
    }
    let attention: Vec<_> = launches
        .iter()
        .filter(|(name, ..)| name.ends_with("glm_vision_flash_attention"))
        .collect();
    assert_eq!(attention.len(), 2, "one flash launch per block");
    for (_, grid, block) in attention {
        // 256 patches in 64-query tiles, one tile column per head.
        assert_eq!(*grid, [4, HEADS as u32, 1]);
        assert_eq!(*block, [128, 1, 1]);
    }
}

#[test]
fn over_capacity_request_fails_before_queuing_any_work() {
    // Eight 1024x1024 images (74x74 patches, 1369 merged rows each) must fit
    // the packed output a GLM request splices from.
    const _: () = assert!(8 * 37 * 37 <= GLM_MAX_OUTPUT_ROWS);

    let gpu = NamedGpu::default();
    let mut enc = encoder(&gpu);
    enc.out_rows = 64 + 16;
    let (large, small) = (image(16), image(8));
    enc.forward_items(&[&large, &small], &gpu, 0).unwrap();
    let launched = gpu.inner.launch_count();
    assert!(launched > 0);

    let err = enc
        .forward_items(&[&large, &small, &small], &gpu, 0)
        .unwrap_err();
    assert!(err.to_string().contains("merged rows"), "{err:#}");
    assert_eq!(
        gpu.inner.launch_count(),
        launched,
        "a rejected request must not leave earlier items encoding"
    );
}
