// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
fn fixture(run: impl FnOnce(&mut DenseFfnLayer, &ForwardContext, &MockGpuBackend)) {
    let gpu = MockGpuBackend::new();
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = 4096;
    config.intermediate_size = 12288;
    config.num_hidden_layers = 45;
    config.mlp_only_layers = vec![0, 1, 2];
    config.vocab_size = 8;
    let weights = DenseFfnWeights {
        gate_proj: QuantizedWeight::null(),
        up_proj: QuantizedWeight::null(),
        down_proj: QuantizedWeight::null(),
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    };
    let mut layer = DenseFfnLayer::new(weights, &gpu).unwrap();
    let buffers = BufferArena::new(&config, 9, 16, 16, 1, &gpu).unwrap();
    let mut dispatch = ops::GemmDispatch::defaults();
    dispatch.cublas_gemm = false;
    let derived = ops::DerivedWeights::new();
    let levers = ops::ModelLevers::defaults();
    let stats = ops::ModelStats::new();
    let ctx = ForwardContext {
        buffers: &buffers,
        gpu: &gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
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
    };
    run(&mut layer, &ctx, &gpu);
}
fn install_fake(layer: &mut DenseFfnLayer) {
    let w = |i| DenseWeight {
        weight: DevicePtr(i),
    };
    layer.prefill_bf16_weights = Some(DenseFfnWeightsBf16 {
        gate_proj: w(0x100000000),
        up_proj: w(0x200000000),
        down_proj: w(0x300000000),
    });
}
#[test]
fn dense_prefill_bf16_disabled_and_verify_rows_do_not_use_cache() {
    fixture(|layer, ctx, gpu| {
        let input = ctx.buffers.norm_output();
        let before = gpu.launch_count();
        assert!(!layer.try_glm_prefill_bf16(input, 1024, ctx, 0).unwrap());
        install_fake(layer);
        for rows in 0..=8 {
            assert!(!layer.try_glm_prefill_bf16(input, rows, ctx, 0).unwrap());
        }
        assert_eq!(before, gpu.launch_count());
        assert!(layer.bf16_weights.is_none());
        // Exercise actual K3 dispatch, comparing launches with/without cache.
        layer.forward_k3(input, ctx, 0).unwrap();
        let cached = gpu.launches_snapshot()[before..].to_vec();
        layer.prefill_bf16_weights = None;
        let before_plain = gpu.launch_count();
        layer.forward_k3(input, ctx, 0).unwrap();
        let plain = gpu.launches_snapshot()[before_plain..].to_vec();
        assert_eq!(cached.len(), plain.len());
        for (a, b) in cached.iter().zip(plain) {
            assert_eq!((a.func, a.grid, a.block), (b.func, b.grid, b.block));
        }
    });
}
#[test]
fn dense_prefill_bf16_actual_prefill_hook_requires_cublas_before_work() {
    fixture(|layer, ctx, gpu| {
        install_fake(layer);
        let before = gpu.launch_count();
        let error = layer
            .forward_prefill_inner(ctx.buffers.norm_output(), 9, ctx, 0)
            .unwrap_err();
        assert!(error.to_string().contains("cuBLAS"), "{error}");
        assert_eq!(before, gpu.launch_count());
    });
}

#[test]
fn dense_prefill_bf16_invalid_or_conflicting_source_fails_before_allocation() {
    fixture(|layer, ctx, gpu| {
        let before = (gpu.alloc_count(), gpu.launch_count());
        assert!(
            layer
                .cache_glm_prefill_bf16(ctx.config, gpu)
                .unwrap_err()
                .to_string()
                .contains("source")
        );
        assert_eq!(before, (gpu.alloc_count(), gpu.launch_count()));
        install_fake(layer);
        assert!(
            layer
                .cache_glm_prefill_bf16(ctx.config, gpu)
                .unwrap_err()
                .to_string()
                .contains("conflicts")
        );
        assert_eq!(before, (gpu.alloc_count(), gpu.launch_count()));
    });
}
