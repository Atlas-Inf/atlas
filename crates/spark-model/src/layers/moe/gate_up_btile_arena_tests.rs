// SPDX-License-Identifier: AGPL-3.0-only
use super::recording::Gpu;
use super::*;
use crate::{layer::ForwardContext, layers::ops};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;

pub(super) struct ContextResources {
    dispatch: ops::GemmDispatch,
    derived: ops::DerivedWeights,
    levers: ops::ModelLevers,
    stats: ops::ModelStats,
}
impl ContextResources {
    pub(super) fn new() -> Self {
        Self {
            dispatch: ops::GemmDispatch::defaults(),
            derived: ops::DerivedWeights::new(),
            levers: ops::ModelLevers::defaults(),
            stats: ops::ModelStats::new(),
        }
    }
    pub(super) fn view<'a>(
        &'a self,
        buffers: &'a BufferArena,
        config: &'a ModelConfig,
        gpu: &'a dyn GpuBackend,
    ) -> ForwardContext<'a> {
        ForwardContext {
            buffers,
            gpu,
            config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            ssm_batch: None,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        }
    }
}

#[test]
fn actual_context_and_arena_bind_once_without_gpu_operations() {
    let gpu = Gpu::new();
    let (store, config, local, mut layer) = binding_tests::setup(&gpu, 0);
    let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
    let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let unpublished = workspace.repack(source, &family).unwrap();
    workspace.close().unwrap();
    let lease = binding::Lease::bind(&unpublished, &family, &mut layer).unwrap();
    let arena = BufferArena::new(&config, 1088, 2048, 64, 1, &gpu).unwrap();
    let resources = ContextResources::new();
    let ctx = resources.view(&arena, &config, &gpu);
    gpu.clear();
    let _checked = lease.check_arena(&ctx).unwrap();
    assert!(gpu.trace().is_empty());
}

#[test]
fn actual_arena_binding_refuses_foreign_context_capture_rank_and_weight_alias() {
    use std::sync::atomic::Ordering;
    let gpu = Gpu::new();
    let (store, config, local, mut layer) = binding_tests::setup(&gpu, 0);
    let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
    let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    let weight = source.projections()[0].packed.ptr;
    let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let unpublished = workspace.repack(source, &family).unwrap();
    workspace.close().unwrap();
    let lease = binding::Lease::bind(&unpublished, &family, &mut layer).unwrap();
    let arena = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
    let resources = ContextResources::new();
    let mut ctx = resources.view(&arena, &config, &gpu);
    gpu.clear();
    ctx.graph_capture = true;
    assert!(lease.check_arena(&ctx).is_err());
    ctx.graph_capture = false;
    gpu.capture.store(true, Ordering::Relaxed);
    assert!(lease.check_arena(&ctx).is_err());
    gpu.capture.store(false, Ordering::Relaxed);
    let foreign = Gpu::new();
    let foreign_ctx = resources.view(&arena, &config, &foreign);
    assert!(lease.check_arena(&foreign_ctx).is_err());
    let mut wrong = config.clone();
    wrong.ep_rank = 1;
    wrong.tp_rank = 1;
    let wrong_ctx = resources.view(&arena, &wrong, &gpu);
    assert!(lease.check_arena(&wrong_ctx).is_err());
    ctx.routed_lora_layers = Some(&[]);
    assert!(lease.check_arena(&ctx).is_err());
    ctx.routed_lora_layers = None;
    lease.check_arena(&ctx).unwrap();
    assert!(gpu.trace().is_empty());
    // Broken allocator returns storage inside a real borrowed weight owner.
    // The actual BufferArena constructor still supplies all pointer extents.
    gpu.next_allocation(weight);
    let aliased = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
    let ctx = resources.view(&aliased, &config, &gpu);
    gpu.clear();
    assert!(lease.check_arena(&ctx).is_err());
    assert!(gpu.trace().is_empty());
}

#[test]
fn checked_subranges_reject_alignment_capacity_and_overflow() {
    use spark_runtime::gpu::DevicePtr;
    assert!(arena::slice(DevicePtr(16), 32, 0, 32, 16).is_ok());
    for (ptr, cap, off, bytes, align) in [
        (16, 31, 0, 32, 16),
        (16, 32, 1, 16, 16),
        (0, 32, 0, 16, 16),
        (u64::MAX - 15, 32, 0, 32, 16),
        (16, usize::MAX, usize::MAX, 2, 16),
        (16, 32, 0, 0, 16),
    ] {
        assert!(arena::slice(DevicePtr(ptr), cap, off, bytes, align).is_err());
    }
}
