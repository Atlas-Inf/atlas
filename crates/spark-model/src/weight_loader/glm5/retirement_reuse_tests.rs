// SPDX-License-Identifier: AGPL-3.0-only
//! Recycled addresses must not turn derived buffers back into checkpoint owners.
use super::*;
use atlas_core::scope::ModelResource;
use spark_runtime::gpu::{KernelHandle, mock::MockGpuBackend};

#[derive(Default)]
struct ReusingGpu {
    state: Mutex<(u64, Vec<u64>, std::collections::HashSet<u64>)>,
    cache: MockGpuBackend,
}
impl GpuBackend for ReusingGpu {
    fn alloc(&self, _bytes: usize) -> Result<DevicePtr> {
        let mut s = self.state.lock();
        let ptr = s.1.pop().unwrap_or_else(|| {
            s.0 += 4096;
            s.0
        });
        ensure!(s.2.insert(ptr), "allocator returned a live address");
        Ok(DevicePtr(ptr))
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes)
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        let mut s = self.state.lock();
        ensure!(s.2.remove(&ptr.0), "duplicate free");
        s.1.push(ptr.0);
        Ok(())
    }
    fn copy_h2d(&self, _: &[u8], _: DevicePtr) -> Result<()> {
        anyhow::bail!("unexpected copy")
    }
    fn copy_d2h(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
        anyhow::bail!("unexpected copy")
    }
    fn copy_d2d(&self, _: DevicePtr, _: DevicePtr, _: usize) -> Result<()> {
        anyhow::bail!("unexpected copy")
    }
    fn memset(&self, _: DevicePtr, _: u8, _: usize) -> Result<()> {
        anyhow::bail!("unexpected memset")
    }
    fn memset_async(&self, _: DevicePtr, _: u8, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected memset")
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
        anyhow::bail!("unexpected launch")
    }
    fn synchronize(&self, _: u64) -> Result<()> {
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, _: &str, _: &str) -> Result<KernelHandle> {
        anyhow::bail!("unexpected kernel")
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.cache.op_cache()
    }
    fn total_memory(&self) -> Result<usize> {
        Ok(1 << 20)
    }
    fn free_memory(&self) -> Result<usize> {
        Ok(1 << 20)
    }
    fn sm_count(&self) -> Result<u32> {
        Ok(1)
    }
}

#[test]
fn production_derived_buffer_reusing_checkpoint_address_is_freed_once() {
    let gpu = ReusingGpu::default();
    let original = gpu.alloc(32).unwrap();
    let mut store = WeightStore::from_map(HashMap::from([(
        "q.weight".into(),
        WeightTensor {
            ptr: original,
            shape: vec![4, 4],
            dtype: WeightDtype::BF16,
        },
    )]));
    let source = LoadedDense::load(&store, "q.weight", &gpu, None, false).unwrap();
    let local = gpu.alloc(16).unwrap();
    let local = source.replaced(local, 16, &gpu).unwrap();
    let recycled = gpu.alloc(16).unwrap();
    assert_eq!(
        recycled, original,
        "fixture must recycle the retired source address"
    );
    let derived = local.replaced(recycled, 16, &gpu).unwrap();
    derived.release(&gpu).unwrap();
    assert!(
        gpu.state.lock().2.is_empty(),
        "derived owner must release the recycled buffer"
    );
    store.release(&gpu).unwrap();
    assert!(gpu.state.lock().2.is_empty());
}
