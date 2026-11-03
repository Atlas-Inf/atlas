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

use super::{GLM_NEIGHBOUR_ROWS, GlmCapacity, GlmVisionEncoder};
use crate::vision_preprocess::glm_max_item_tokens;

/// Mock backend that hands out one handle per requested kernel name.
/// `sized_only` records allocation sizes without backing them, for sizing
/// the shipped tower without its gigabytes of host memory.
#[derive(Default)]
struct NamedGpu {
    inner: MockGpuBackend,
    names: Mutex<Vec<String>>,
    sized_only: bool,
    allocated: Mutex<Vec<usize>>,
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
        let mut allocated = self.allocated.lock().unwrap();
        allocated.push(n);
        if self.sized_only {
            return Ok(DevicePtr(0x1000 * allocated.len() as u64));
        }
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

use super::test_tower::{HEADS, config, image};

fn encoder(gpu: &NamedGpu) -> GlmVisionEncoder {
    super::test_tower::encoder(gpu)
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

#[test]
fn a_receive_only_tower_holds_just_the_row_buffer_and_never_encodes() {
    // A multi-rank worker receives every row from rank 0, so it needs the
    // packed output buffer and nothing else: no weights, no ViT scratch.
    let gpu = NamedGpu::default();
    let enc =
        GlmVisionEncoder::new(None, &config(), GlmCapacity::budgets(&config()), &gpu).unwrap();
    assert_eq!(gpu.inner.alloc_count(), 1, "only buf_out is allocated");
    assert!(!enc.buf_out.is_null());
    assert_eq!(
        enc.out_rows,
        glm_max_item_tokens(&config()) + GLM_NEIGHBOUR_ROWS
    );
    let err = enc.forward_items(&[&image(8)], &gpu, 0).unwrap_err();
    assert!(err.to_string().contains("receives"), "{err:#}");
    assert_eq!(gpu.inner.launch_count(), 0);
}

#[test]
fn token_local_stages_run_in_row_chunks_around_whole_sequence_attention() {
    let gpu = NamedGpu::default();
    let mut enc = encoder(&gpu);
    // 256 patches in 64-row chunks: four of every token-local launch per
    // stage, while attention still sees the whole sequence.
    enc.row_chunk = 64;
    assert_eq!(
        enc.forward_items(&[&image(16)], &gpu, 0).unwrap(),
        vec![(8, 8, 64)]
    );
    let launches = gpu.launches();
    let count = |kernel: &str| {
        launches
            .iter()
            .filter(|(name, ..)| name.ends_with(kernel))
            .count()
    };
    let blocks = config().depth;
    assert_eq!(count("glm_vision_f32_to_bf16"), 4, "patch embed");
    assert_eq!(
        count("glm_vision_merge_reorder"),
        4,
        "merger, 16 blocks each"
    );
    assert_eq!(count("glm_vision_swiglu_clamp"), 4 * blocks + 4);
    assert_eq!(
        count("glm_vision_rms_norm"),
        2 * blocks + 1,
        "norms span the sequence"
    );
    for (_, grid, _) in launches
        .iter()
        .filter(|(name, ..)| name.ends_with("glm_vision_flash_attention"))
    {
        assert_eq!(*grid, [4, HEADS as u32, 1], "one 256-patch sequence");
    }
}

#[test]
fn the_shipped_tower_holds_a_full_reference_sequence_in_bounded_scratch() {
    // GLM-5.3-Flash's tower under the serving budgets: an 8000-token still is
    // 32000 patches, a two-frame 30000-token video group 120000.
    let mut config = config();
    (config.depth, config.hidden_size, config.num_heads) = (24, 1024, 16);
    (config.intermediate_size, config.out_hidden_size) = (4096, 4096);
    config.projection_intermediate_size = 10240;
    config.max_pixels = None;
    let mut weights = super::test_tower::weights();
    weights.blocks = vec![weights.blocks[0]; config.depth];
    let gpu = NamedGpu {
        sized_only: true,
        ..Default::default()
    };
    let enc =
        GlmVisionEncoder::new(Some(weights), &config, GlmCapacity::budgets(&config), &gpu).unwrap();
    assert_eq!((enc.p_max, enc.row_chunk), (120_000, 4096));
    assert_eq!(enc.out_rows, 45_000 + GLM_NEIGHBOUR_ROWS);
    let allocated = gpu.allocated.lock().unwrap();
    let rows = enc.out_rows * 4096 * 2;
    let scratch: usize = allocated.iter().sum::<usize>() - rows;
    // ~10.5 KB per patch for the residual, its norm, QKV and RoPE; the
    // token-local stages add ~0.08 GB of row-chunk scratch.
    assert_eq!(scratch, 1_338_753_024);
    assert_eq!(rows, 502_857_728);
}

#[test]
fn the_pair_keeps_its_bounded_tower_and_one_launch_per_stage() {
    let mut config = config();
    config.max_pixels = None;
    let capacity = GlmCapacity::bounded(&config);
    assert_eq!(
        capacity,
        GlmCapacity {
            p_max: 6_400,
            out_rows: 16_384,
            row_chunk: 6_400,
        }
    );
    // A 16x16 tower's whole sequence fits one chunk, so the launches are the
    // ones a tower without row chunks issues.
    let gpu = NamedGpu::default();
    let mut enc = encoder(&gpu);
    enc.row_chunk = enc.p_max;
    enc.forward_items(&[&image(16)], &gpu, 0).unwrap();
    let launches = gpu.launches();
    let count = |kernel: &str| {
        launches
            .iter()
            .filter(|(name, ..)| name.ends_with(kernel))
            .count()
    };
    assert_eq!(count("glm_vision_f32_to_bf16"), 1);
    assert_eq!(count("glm_vision_merge_reorder"), 1);
    assert_eq!(
        count("glm_vision_swiglu_clamp"),
        super::test_tower::config().depth + 1
    );
}

#[test]
fn a_multi_rank_world_holds_the_budgets() {
    use atlas_core::config::ModelConfig;
    let vision = config();
    let mut world = ModelConfig::qwen3_next_80b_nvfp4();
    (world.tp_world_size, world.ep_world_size) = (1, 1);
    assert_eq!(
        GlmCapacity::of_world(&world, &vision),
        GlmCapacity::bounded(&vision)
    );
    for ranks in [2, 4] {
        (world.tp_world_size, world.ep_world_size) = (ranks, ranks);
        assert_eq!(
            GlmCapacity::of_world(&world, &vision),
            GlmCapacity::budgets(&vision)
        );
    }
}
