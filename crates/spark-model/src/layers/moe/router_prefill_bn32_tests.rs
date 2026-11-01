// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::layers::moe::shared_fp8_cache_test_gpu::RecordingGpu;
use spark_runtime::buffers::BufferArena;
fn fixture(run: impl FnOnce(&mut MoeLayer, &ForwardContext, &RecordingGpu)) {
    fixture_model("glm5_next", run);
}
fn fixture_model(model: &str, run: impl FnOnce(&mut MoeLayer, &ForwardContext, &RecordingGpu)) {
    let gpu = RecordingGpu::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = model.into();
    config.hidden_size = 4096;
    config.num_experts = 288;
    config.num_experts_per_tok = 8;
    config.moe_intermediate_size = 2048;
    config.intermediate_size = 12288;
    config.vocab_size = 8;
    let mut layer = MoeLayer::new(MoeWeights::empty(288), 288, None, &gpu, &config).unwrap();
    layer.weights.gate = DenseWeight {
        weight: DevicePtr(0x10000000000),
    };
    layer.dense_gemm_router = KernelHandle(111);
    let buffers = BufferArena::new(&config, 2052, 32768, 16, 1, &gpu).unwrap();
    let dispatch = ops::GemmDispatch::defaults();
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
#[test]
fn router_prefill_bn32_actual_dispatch_changes_only_enabled_long_glm_rows() {
    fixture(|layer, ctx, gpu| {
        let input = ctx.buffers.norm_output();
        let output = ctx.buffers.gate_logits();
        layer
            .router_gate_gemm_dense(input, output, 2048, 288, 4096, ctx, 19)
            .unwrap();
        assert_eq!(gpu.launches.lock().unwrap().last().unwrap().kernel, 111);
        layer.router_prefill_bn32 = KernelHandle(222);
        for m in [1, 3, 8] {
            layer
                .router_gate_gemm_dense(input, output, m, 288, 4096, ctx, 19)
                .unwrap();
            assert_eq!(gpu.launches.lock().unwrap().last().unwrap().kernel, 111);
        }
        for m in [9, 2048, 2052] {
            layer
                .router_gate_gemm_dense(input, output, m, 288, 4096, ctx, 19)
                .unwrap();
            let calls = gpu.launches.lock().unwrap();
            let call = calls.last().unwrap();
            assert_eq!(call.kernel, 222);
            assert_eq!(call.grid, [9, m.div_ceil(16), 1]);
            assert_eq!(call.block, [8, 16, 1]);
            assert_eq!(call.stream, 19);
            assert_eq!(call.shared, 0);
            assert_eq!(
                call.args[0],
                crate::layers::moe::shared_fp8_cache_test_gpu::Arg::Ptr(input)
            );
            assert_eq!(
                call.args[1],
                crate::layers::moe::shared_fp8_cache_test_gpu::Arg::Ptr(layer.weights.gate.weight)
            );
            assert_eq!(
                call.args[2],
                crate::layers::moe::shared_fp8_cache_test_gpu::Arg::Ptr(output)
            );
            assert_eq!(
                call.args[3],
                crate::layers::moe::shared_fp8_cache_test_gpu::Arg::Bytes(m.to_ne_bytes().to_vec())
            );
        }
    });
}

#[test]
fn router_prefill_bn32_non_glm_keeps_original_kernel() {
    fixture_model("qwen3_next", |layer, ctx, gpu| {
        layer.router_prefill_bn32 = KernelHandle(222);
        layer
            .router_gate_gemm_dense(
                ctx.buffers.norm_output(),
                ctx.buffers.gate_logits(),
                2048,
                288,
                4096,
                ctx,
                19,
            )
            .unwrap();
        assert_eq!(gpu.launches.lock().unwrap().last().unwrap().kernel, 111);
    });
}
#[test]
fn router_prefill_bn32_rejects_bad_shape_owner_capacity_before_work() {
    fixture(|layer, ctx, gpu| {
        layer.router_prefill_bn32 = KernelHandle(222);
        let before = gpu.launches.lock().unwrap().len();
        let input = ctx.buffers.norm_output();
        let output = ctx.buffers.gate_logits();
        for (a, c, m, n, k) in [
            (input, output, 2048, 287, 4096),
            (input, output, 2048, 288, 4095),
            (input, output, 100000, 288, 4096),
            (input, output.offset(16), 2048, 288, 4096),
            (input.offset(2), output, 2048, 288, 4096),
            (output, output, 2048, 288, 4096),
        ] {
            assert!(
                layer
                    .router_gate_gemm_dense(a, c, m, n, k, ctx, 19)
                    .is_err()
            );
        }
        assert_eq!(before, gpu.launches.lock().unwrap().len());
    });
}
#[test]
fn router_prefill_bn32_explicit_load_flag() {
    assert!(!parse(None).unwrap());
    assert!(!parse(Some("0")).unwrap());
    assert!(parse(Some("1")).unwrap());
    for value in ["", "true", "2"] {
        assert!(parse(Some(value)).is_err());
    }
}
