// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::layers::moe::shared_fp8_cache_test_gpu::{Arg, RecordingGpu};
use spark_runtime::buffers::BufferArena;

fn fixture(rows: usize, run: impl FnOnce(&mut MoeLayer, &mut ForwardContext, &RecordingGpu)) {
    fixture_capacity(rows, false, run);
}
fn fixture_capacity(
    rows: usize,
    short: bool,
    run: impl FnOnce(&mut MoeLayer, &mut ForwardContext, &RecordingGpu),
) {
    let gpu = RecordingGpu::new();
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = 4096;
    config.shared_expert_intermediate_size = 2048;
    config.moe_intermediate_size = 2048;
    config.intermediate_size = 2048;
    config.vocab_size = 8;
    config.num_experts = 8;
    config.num_experts_per_tok = 8;
    config.num_hidden_layers = 1;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 1;
    config.head_dim = 128;
    config.linear_num_key_heads = 1;
    config.linear_num_value_heads = 1;
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    if short {
        config.num_attention_heads = 1;
        config.shared_expert_intermediate_size = 512;
        config.kv_lora_rank = 0;
    }
    let mut layer = MoeLayer::new(MoeWeights::empty(8), 8, None, &gpu, &config).unwrap();
    let original = |i: u64| QuantizedWeight {
        weight: DevicePtr(0x10_0000_0000 + i * 0x100_0000),
        weight_scale: DevicePtr(0x10_0080_0000 + i * 0x100_0000),
        weight_scale_2: 0.75,
        ..QuantizedWeight::null()
    };
    layer.shared_gate_t = Some(original(0));
    layer.shared_up_t = Some(original(1));
    layer.shared_down_t = Some(original(2));
    layer.shared_gate_fp8 = Some(DevicePtr(0x20_0000_0000));
    layer.shared_up_fp8 = Some(DevicePtr(0x21_0000_0000));
    layer.shared_down_fp8 = Some(DevicePtr(0x22_0000_0000));
    layer.shared_fp8_cache.layer = Some(3);
    layer.shared_fp8_cache.verify = false;
    layer.fp8_gemm_k = KernelHandle(777);
    layer.w4a16_gemm_t = KernelHandle(888);
    let buffers = BufferArena::new(&config, rows, 2048, 64, 1, &gpu).unwrap();
    let dispatch = ops::GemmDispatch::defaults();
    let derived = ops::DerivedWeights::new();
    let levers = ops::ModelLevers::defaults();
    let stats = ops::ModelStats::new();
    let mut ctx = ForwardContext {
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
    run(&mut layer, &mut ctx, &gpu);
}

#[test]
fn actual_installed_dispatch_rejects_bad_projection_geometry_and_live_spans_before_work() {
    fixture(1025, |layer, ctx, gpu| {
        for projection_id in 0..3 {
            for rows in [1024, 1025] {
                for fault in 0..18 {
                    let (cache, owner, n, k, original) = projection(layer, ctx, projection_id);
                    let mut p = projection_id;
                    let mut input = ctx.buffers.norm_output();
                    let mut weight = cache;
                    let mut output = owner;
                    let mut n = n;
                    let mut k = k;
                    let mut changed = original;
                    match fault {
                        0 => p = 3,
                        1 => p = 8,
                        2 => p = usize::MAX,
                        3 => weight = DevicePtr(cache.0 + 16),
                        4 => output = DevicePtr(owner.0 + 2),
                        5 => n += 128,
                        6 => k += 128,
                        7 => input = owner,
                        8 => input = DevicePtr(input.0 + 2),
                        9 => changed.weight = DevicePtr::NULL,
                        10 => changed.weight_scale = DevicePtr::NULL,
                        11 => changed.weight_scale_2 = f32::NAN,
                        12 => changed.weight_scale_2_vec = DevicePtr(16),
                        13 => changed.weight = owner,
                        14 => changed.weight_scale = owner,
                        15 => changed.weight_scale = changed.weight,
                        16 => changed.weight = DevicePtr(u64::MAX - 15),
                        17 => changed.weight = DevicePtr(changed.weight.0 + 2),
                        _ => unreachable!(),
                    }
                    match projection_id {
                        0 => layer.shared_gate_t = Some(changed),
                        1 => layer.shared_up_t = Some(changed),
                        _ => layer.shared_down_t = Some(changed),
                    }
                    let before = gpu.effects.load(Ordering::Relaxed);
                    assert!(
                        layer
                            .run_shared_fp8_cache(p, input, weight, output, rows, n, k, ctx, 19)
                            .is_err(),
                        "projection={projection_id} rows={rows} fault={fault}"
                    );
                    assert_eq!(gpu.effects.load(Ordering::Relaxed), before);
                    assert!(gpu.launches.lock().unwrap().is_empty());
                    match projection_id {
                        0 => layer.shared_gate_t = Some(original),
                        1 => layer.shared_up_t = Some(original),
                        _ => layer.shared_down_t = Some(original),
                    }
                }
            }
        }
    });
}

#[test]
fn actual_installed_dispatch_requires_retained_weights_and_nonzero_handles() {
    fixture(1025, |layer, ctx, gpu| {
        for p in 0..3 {
            for fault in 0..3 {
                let (cache, output, n, k, old) = projection(layer, ctx, p);
                match fault {
                    0 => match p {
                        0 => layer.shared_gate_t = None,
                        1 => layer.shared_up_t = None,
                        _ => layer.shared_down_t = None,
                    },
                    1 => layer.w4a16_gemm_t = KernelHandle(0),
                    2 => match p {
                        0 => layer.shared_gate_fp8 = None,
                        1 => layer.shared_up_fp8 = None,
                        _ => layer.shared_down_fp8 = None,
                    },
                    _ => unreachable!(),
                }
                let before = gpu.effects.load(Ordering::Relaxed);
                for rows in [1024, 1025] {
                    assert!(
                        layer
                            .run_shared_fp8_cache(
                                p,
                                ctx.buffers.norm_output(),
                                cache,
                                output,
                                rows,
                                n,
                                k,
                                ctx,
                                19
                            )
                            .is_err()
                    );
                }
                assert_eq!(gpu.effects.load(Ordering::Relaxed), before);
                assert!(gpu.launches.lock().unwrap().is_empty());
                match p {
                    0 => {
                        layer.shared_gate_t = Some(old);
                        layer.shared_gate_fp8 = Some(cache)
                    }
                    1 => {
                        layer.shared_up_t = Some(old);
                        layer.shared_up_fp8 = Some(cache)
                    }
                    _ => {
                        layer.shared_down_t = Some(old);
                        layer.shared_down_fp8 = Some(cache)
                    }
                }
                layer.w4a16_gemm_t = KernelHandle(888);
            }
        }
    });
}

#[test]
fn actual_installed_dispatch_checks_arena_rows_and_independent_output_byte_capacity() {
    fixture(1025, |layer, ctx, gpu| {
        for p in 0..3 {
            for rows in [0, 1026, u32::MAX] {
                let (cache, output, n, k, _) = projection(layer, ctx, p);
                let before = gpu.effects.load(Ordering::Relaxed);
                assert!(
                    layer
                        .run_shared_fp8_cache(
                            p,
                            ctx.buffers.norm_output(),
                            cache,
                            output,
                            rows,
                            n,
                            k,
                            ctx,
                            19
                        )
                        .is_err()
                );
                assert_eq!(gpu.effects.load(Ordering::Relaxed), before);
                assert!(gpu.launches.lock().unwrap().is_empty());
            }
        }
    });
    fixture_capacity(1025, true, |layer, ctx, gpu| {
        for p in 0..3 {
            for rows in [1024, 1025] {
                let (cache, output, n, k, _) = projection(layer, ctx, p);
                let before = gpu.effects.load(Ordering::Relaxed);
                let error = layer
                    .run_shared_fp8_cache(
                        p,
                        ctx.buffers.norm_output(),
                        cache,
                        output,
                        rows,
                        n,
                        k,
                        ctx,
                        19,
                    )
                    .unwrap_err();
                assert!(error.to_string().contains("capacity"), "{error}");
                assert_eq!(gpu.effects.load(Ordering::Relaxed), before);
                assert!(gpu.launches.lock().unwrap().is_empty());
            }
        }
    });
}

#[test]
fn actual_verify_refuses_context_and_stream_capture_even_checked_or_fallback() {
    fixture(1025, |layer, ctx, gpu| {
        layer.shared_fp8_cache.verify = true;
        for checked in [0, 7] {
            layer
                .shared_fp8_cache
                .checked
                .store(checked, Ordering::Relaxed);
            for actual_stream in [false, true] {
                ctx.graph_capture = !actual_stream;
                gpu.capturing.store(actual_stream, Ordering::Relaxed);
                for p in 0..3 {
                    for rows in [5, 1024, 1025] {
                        let (cache, output, n, k, _) = projection(layer, ctx, p);
                        let before = gpu.effects.load(Ordering::Relaxed);
                        let error = layer
                            .run_shared_fp8_cache(
                                p,
                                ctx.buffers.norm_output(),
                                cache,
                                output,
                                rows,
                                n,
                                k,
                                ctx,
                                19,
                            )
                            .unwrap_err();
                        assert!(error.to_string().contains("eager"), "{error}");
                        assert_eq!(gpu.effects.load(Ordering::Relaxed), before);
                        assert!(gpu.launches.lock().unwrap().is_empty());
                        assert_eq!(
                            layer.shared_fp8_cache.checked.load(Ordering::Relaxed),
                            checked
                        );
                    }
                }
            }
        }
    });
}

#[test]
fn actual_verify_eager_large_fallback_does_not_fabricate_oracle_passes() {
    fixture(1025, |layer, ctx, gpu| {
        layer.shared_fp8_cache.verify = true;
        let before = gpu.effects.load(Ordering::Relaxed);
        for p in 0..3 {
            let (cache, output, n, k, _) = projection(layer, ctx, p);
            layer
                .run_shared_fp8_cache(
                    p,
                    ctx.buffers.norm_output(),
                    cache,
                    output,
                    1025,
                    n,
                    k,
                    ctx,
                    19,
                )
                .unwrap();
        }
        assert_eq!(gpu.effects.load(Ordering::Relaxed), before);
        assert_eq!(gpu.launches.lock().unwrap().len(), 3);
        assert!(gpu.launches.lock().unwrap().iter().all(|l| l.kernel == 888));
        assert_eq!(layer.shared_fp8_cache.checked.load(Ordering::Relaxed), 0);
    });
}
fn projection(
    layer: &MoeLayer,
    ctx: &ForwardContext,
    p: usize,
) -> (DevicePtr, DevicePtr, u32, u32, QuantizedWeight) {
    match p {
        0 => (
            layer.shared_gate_fp8.unwrap(),
            ctx.buffers.ssm_deinterleaved(),
            2048,
            4096,
            layer.shared_gate_t.unwrap(),
        ),
        1 => (
            layer.shared_up_fp8.unwrap(),
            ctx.buffers.ssm_qkvz(),
            2048,
            4096,
            layer.shared_up_t.unwrap(),
        ),
        2 => (
            layer.shared_down_fp8.unwrap(),
            ctx.buffers.attn_output(),
            4096,
            2048,
            layer.shared_down_t.unwrap(),
        ),
        _ => panic!("test projection"),
    }
}

#[path = "shared_fp8_cache_2048_dispatch_tests.rs"]
mod prefill_2048;
#[path = "shared_fp8_cache_4096_dispatch_tests.rs"]
mod prefill_4096;

fn u32_arg(value: u32) -> Arg {
    Arg::Bytes(value.to_le_bytes().to_vec())
}

#[test]
fn actual_fallback_uses_arena_limit_not_a_new_1025_ceiling_and_cache_requires_handle() {
    fixture(2048, |layer, ctx, gpu| {
        let before = gpu.effects.load(Ordering::Relaxed);
        for p in 0..3 {
            let (weight, output, n, k, _) = projection(layer, ctx, p);
            layer
                .run_shared_fp8_cache(
                    p,
                    ctx.buffers.norm_output(),
                    weight,
                    output,
                    2048,
                    n,
                    k,
                    ctx,
                    19,
                )
                .unwrap();
        }
        assert_eq!(gpu.effects.load(Ordering::Relaxed), before);
        assert!(
            gpu.launches
                .lock()
                .unwrap()
                .iter()
                .all(|l| l.kernel == 888 && l.grid[1] == 32)
        );
        gpu.launches.lock().unwrap().clear();
        layer.fp8_gemm_k = KernelHandle(0);
        let (weight, output, n, k, _) = projection(layer, ctx, 0);
        assert!(
            layer
                .run_shared_fp8_cache(
                    0,
                    ctx.buffers.norm_output(),
                    weight,
                    output,
                    1024,
                    n,
                    k,
                    ctx,
                    19
                )
                .is_err()
        );
        assert!(gpu.launches.lock().unwrap().is_empty());
        assert_eq!(gpu.effects.load(Ordering::Relaxed), before);
        layer.shared_fp8_cache.verify = true;
        assert!(
            layer
                .run_shared_fp8_cache(
                    0,
                    ctx.buffers.norm_output(),
                    weight,
                    output,
                    5,
                    n,
                    k,
                    ctx,
                    19
                )
                .is_err()
        );
        assert!(
            gpu.launches.lock().unwrap().is_empty(),
            "missing candidate handle must fail before reference"
        );
        assert_eq!(gpu.effects.load(Ordering::Relaxed), before);
    });
}

#[test]
fn actual_installed_dispatch_1024_cached_1025_retained_t_all_projections_no_extra_work() {
    fixture(1025, |layer, ctx, gpu| {
        let before = gpu.effects.load(Ordering::Relaxed);
        for p in 0..3 {
            for rows in [1, 4, 5, 16, 63, 64, 65, 148, 1024, 1025] {
                let (weight, output, n, k, old) = projection(layer, ctx, p);
                let input = ctx.buffers.norm_output();
                layer
                    .run_shared_fp8_cache(p, input, weight, output, rows, n, k, ctx, 19)
                    .unwrap();
                let launches = gpu.launches.lock().unwrap();
                let launch = launches.last().unwrap();
                assert_eq!(launch.kernel, if rows <= 1024 { 777 } else { 888 });
                assert_eq!(launch.grid, [n.div_ceil(128), rows.div_ceil(64), 1]);
                assert_eq!(launch.block, [128, 1, 1]);
                assert_eq!((launch.shared, launch.stream), (0, 19));
                let expected = if rows <= 1024 {
                    vec![
                        Arg::Ptr(input),
                        Arg::Ptr(weight),
                        Arg::Ptr(output),
                        u32_arg(rows),
                        u32_arg(n),
                        u32_arg(k),
                    ]
                } else {
                    vec![
                        Arg::Ptr(input),
                        Arg::Ptr(old.weight),
                        Arg::Ptr(old.weight_scale),
                        Arg::Bytes(0.75f32.to_le_bytes().to_vec()),
                        Arg::Ptr(output),
                        u32_arg(rows),
                        u32_arg(n),
                        u32_arg(k),
                        u32_arg(n),
                    ]
                };
                assert_eq!(launch.args, expected);
            }
        }
        assert_eq!(
            gpu.effects.load(Ordering::Relaxed),
            before,
            "no allocation/copy/memset/sync"
        );
        assert_eq!(gpu.launches.lock().unwrap().len(), 30);
        assert_eq!(
            layer.shared_fp8_cache.checked.load(Ordering::Relaxed),
            0,
            "no fake oracle passes"
        );
    });
}
