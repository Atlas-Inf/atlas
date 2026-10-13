// SPDX-License-Identifier: AGPL-3.0-only
//! Metadata/ABI recorder, never a CUDA or numerical permutation simulator.
use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};
use std::collections::{HashMap, HashSet};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Arg {
    Ptr(DevicePtr),
    Bytes(Vec<u8>),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Event {
    Alloc(usize),
    Free(DevicePtr),
    Scalar(DevicePtr, u64),
    Copy(DevicePtr, DevicePtr, usize, u64),
    Sync(u64),
    Launch(u64, [u32; 3], [u32; 3], u32, u64, Vec<Arg>),
}
pub(super) struct RecordingGpu {
    cache: spark_runtime::op_cache::OpCache,
    pub events: Mutex<Vec<Event>>,
    pub scalars: Mutex<HashMap<u64, u32>>,
    pub live: Mutex<HashSet<u64>>,
    pub fail_at: AtomicUsize,
    pub fail_also: AtomicUsize,
    pub capturing: AtomicBool,
    pub allocation: AtomicU64,
}
impl RecordingGpu {
    pub fn new() -> Self {
        Self {
            cache: spark_runtime::op_cache::OpCache::new(),
            events: Mutex::new(Vec::new()),
            scalars: Mutex::new(HashMap::new()),
            live: Mutex::new(HashSet::new()),
            fail_at: AtomicUsize::new(usize::MAX),
            fail_also: AtomicUsize::new(usize::MAX),
            capturing: AtomicBool::new(false),
            allocation: AtomicU64::new(0x8000_0000_0000),
        }
    }
    fn record(&self, event: Event) -> Result<()> {
        let mut events = self.events.lock().unwrap();
        events.push(event);
        ensure!(
            events.len() != self.fail_at.load(Ordering::Relaxed)
                && events.len() != self.fail_also.load(Ordering::Relaxed),
            "injected backend failure at {}",
            events.len()
        );
        Ok(())
    }
    pub fn trace(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }
    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
        self.fail_at.store(usize::MAX, Ordering::Relaxed);
        self.fail_also.store(usize::MAX, Ordering::Relaxed);
    }
}
impl GpuBackend for RecordingGpu {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.record(Event::Alloc(bytes))?;
        let ptr = self.allocation.load(Ordering::Relaxed);
        self.live.lock().unwrap().insert(ptr);
        Ok(DevicePtr(ptr))
    }
    fn alloc_managed(&self, _: usize) -> Result<DevicePtr> {
        bail!("unexpected managed allocation")
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        // Match CudaBackend: remove ledger ownership BEFORE the driver call,
        // including a failed driver free. This does not emulate GPU bytes.
        ensure!(
            self.live.lock().unwrap().remove(&ptr.0),
            "freeing non-owned allocation"
        );
        self.record(Event::Free(ptr))
    }
    fn copy_h2d(&self, _: &[u8], _: DevicePtr) -> Result<()> {
        bail!("unexpected H2D")
    }
    fn copy_d2h(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
        bail!("expected explicit stream scalar read")
    }
    fn copy_d2h_on_stream(&self, ptr: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        self.record(Event::Scalar(ptr, stream))?;
        ensure!(dst.len() == 4, "only scalar reads allowed");
        let bits = *self
            .scalars
            .lock()
            .unwrap()
            .get(&ptr.0)
            .ok_or_else(|| anyhow::anyhow!("unknown scalar"))?;
        dst.copy_from_slice(&bits.to_le_bytes());
        Ok(())
    }
    fn copy_d2d(&self, _: DevicePtr, _: DevicePtr, _: usize) -> Result<()> {
        bail!("expected explicit stream copy")
    }
    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        self.record(Event::Copy(src, dst, bytes, stream))
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
        bail!("expected typed kernel ABI")
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
        self.record(Event::Launch(
            kernel.0,
            grid,
            block,
            shared,
            stream,
            args.iter()
                .map(|a| match a {
                    KernelArg::Buffer(p) => Arg::Ptr(*p),
                    KernelArg::Bytes(b) => Arg::Bytes(b.to_vec()),
                })
                .collect(),
        ))
    }
    fn stream_is_capturing(&self, _: u64) -> bool {
        self.capturing.load(Ordering::Relaxed)
    }
    fn synchronize(&self, stream: u64) -> Result<()> {
        self.record(Event::Sync(stream))
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, _: &str, name: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(match name {
            "glm_native_to_btile_u8" => 101,
            "transpose_u8" => 102,
            _ => 103,
        }))
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        &self.cache
    }
    fn memset(&self, _: DevicePtr, _: u8, _: usize) -> Result<()> {
        bail!("unexpected memset")
    }
    fn memset_async(&self, _: DevicePtr, _: u8, _: usize, _: u64) -> Result<()> {
        bail!("unexpected memset")
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

pub(super) fn fixture(
    rank: usize,
    gpu: &RecordingGpu,
) -> (
    spark_runtime::weights::WeightStore,
    atlas_core::config::ModelConfig,
    Vec<bool>,
) {
    fixture_layer(rank, 0, gpu)
}
pub(super) fn fixture_layer(
    rank: usize,
    layer: usize,
    gpu: &RecordingGpu,
) -> (
    spark_runtime::weights::WeightStore,
    atlas_core::config::ModelConfig,
    Vec<bool>,
) {
    use spark_runtime::weights::{WeightDtype as D, WeightStore, WeightTensor};
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = 4096;
    config.moe_intermediate_size = 2048;
    config.num_experts = 288;
    config.num_experts_per_tok = 8;
    config.num_hidden_layers = 42;
    config.ep_world_size = 2;
    config.tp_world_size = 2;
    config.ep_rank = rank;
    config.tp_rank = rank;
    let local: Vec<_> = (0..288).map(|e| config.is_local_expert(e)).collect();
    let mut map = HashMap::new();
    for (expert, &owned) in local.iter().enumerate() {
        if !owned {
            continue;
        }
        for (projection, name) in ["gate_proj", "up_proj"].iter().enumerate() {
            let base = 0x1000_0000
                + (layer as u64) * 0x10_0000_0000
                + ((expert * 2 + projection) as u64) * 0x100_0000;
            let prefix = format!("{}.mlp.experts.{expert}.{name}", config.layer_prefix(layer));
            for (suffix, offset, shape, dtype) in [
                ("weight", 0, vec![2048, 2048], D::UInt8),
                ("weight_scale", 0x40_0000, vec![2048, 256], D::FP8E4M3),
                ("weight_scale_2", 0x48_0000, vec![1], D::FP32),
                ("input_scale", 0x48_0010, vec![], D::FP32),
            ] {
                map.insert(
                    format!("{prefix}.{suffix}"),
                    WeightTensor {
                        ptr: DevicePtr(base + offset),
                        shape,
                        dtype,
                    },
                );
                if dtype == D::FP32 {
                    gpu.scalars.lock().unwrap().insert(
                        base + offset,
                        if suffix == "input_scale" {
                            (-0.0f32).to_bits()
                        } else {
                            1.25f32.to_bits()
                        },
                    );
                }
            }
        }
    }
    (WeightStore::from_map(map), config, local)
}
