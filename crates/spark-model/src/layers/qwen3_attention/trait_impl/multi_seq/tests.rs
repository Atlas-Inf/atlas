// SPDX-License-Identifier: AGPL-3.0-only

//! Lever-#3 attention copy win: at K=4..8 verify the `*_os` strided-output
//! GEMVs write each Q/K/V projection straight into the interleaved
//! `qkv_buf` — removing the 3·n D2D scatter copies per attention layer that
//! `ms_qkv_batchn` used to issue through the scratch path.
//!
//! `MockGpuBackend` counts `copy_d2d*` and resolves every kernel to the
//! same 0xDEAD handle, so the d2d delta between the layer as-built and a
//! clone with the `_os` handles zeroed is exactly the scatter traffic.

use super::super::Qwen3AttentionLayer;
use super::ctx::MultiSeqCtx;
use crate::layer::ForwardContext;
use crate::layers::FfnComponent;
use crate::weight_map::{AttentionWeights, DenseWeight, QuantizedWeight};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;

fn nvfp4(gpu: &MockGpuBackend, n: usize, k: usize) -> QuantizedWeight {
    QuantizedWeight {
        weight: gpu.alloc(n * k / 2).unwrap(),
        weight_scale: gpu.alloc(n * k / 16).unwrap(),
        weight_scale_2: 1.0,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    }
}

fn attn_layer(gpu: &MockGpuBackend, config: &ModelConfig) -> Qwen3AttentionLayer {
    let h = config.hidden_size;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let q_proj_dim = q_dim * 2; // gated: [Q|gate]
    let dw = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    let attn = AttentionWeights {
        q_proj: dw(h * q_proj_dim * 2),
        k_proj: dw(h * kv_dim * 2),
        v_proj: dw(h * kv_dim * 2),
        o_proj: QuantizedWeight::null(),
        q_norm: dw(config.head_dim * 2),
        k_norm: dw(config.head_dim * 2),
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    Qwen3AttentionLayer::new(
        dw(h * 2),
        attn,
        dw(h * 2),
        FfnComponent::None,
        0,
        Some(nvfp4(gpu, q_proj_dim, h)),
        Some(nvfp4(gpu, kv_dim, h)),
        Some(nvfp4(gpu, kv_dim, h)),
        gpu,
        KvCacheDtype::Bf16,
        0,
        config,
    )
    .unwrap()
}

/// Run `ms_phase_qkv` at `n` verify rows and return how many d2d copies
/// the phase issued plus every kernel launch it recorded.
fn qkv_phase_d2d(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    layer: &Qwen3AttentionLayer,
    n: usize,
) -> (usize, Vec<spark_runtime::gpu::mock::MockLaunch>) {
    let buffers = BufferArena::new(config, 64, 4096, 16, 32, gpu).unwrap();
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let fwd = ForwardContext {
        ssm_batch: None,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        gpu,
        config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
    };
    let positions = vec![1; n];
    let c = MultiSeqCtx::new(
        layer,
        &fwd,
        buffers.hidden_states(),
        buffers.residual(),
        n,
        &positions,
        16,
        0,
    );
    let before = gpu.d2d_count();
    let launch_base = gpu.launches_snapshot().len();
    layer.ms_phase_qkv(&c).unwrap();
    let launches = gpu.launches_snapshot()[launch_base..].to_vec();
    (gpu.d2d_count() - before, launches)
}

#[test]
fn strided_qkv_os_removes_scatter_copies() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();

    // As-built: under the mock every kernel resolves, so the strided arm
    // engages — Q/K/V land in qkv_buf with zero scatter copies.
    let fused = attn_layer(&gpu, &config);
    let (os_copies, _) = qkv_phase_d2d(&gpu, &config, &fused, 4);

    // Fallback: zero the `_os` handles so the dispatch stays on the
    // scratch+scatter route — 3 copies per row at n=4.
    let mut fallback = attn_layer(&gpu, &config);
    fallback.dp4a_gemv_batch4_os_k = KernelHandle(0);
    fallback.w4a16_gemv_batch4_os_k = KernelHandle(0);
    fallback.w4a16_gemv_batch8_os_k = KernelHandle(0);
    let (scatter, _) = qkv_phase_d2d(&gpu, &config, &fallback, 4);

    assert_eq!(
        scatter - os_copies,
        3 * 4,
        "strided arm must remove exactly the 3·n scatter copies: os={os_copies} scatter={scatter}"
    );
}

/// OOB-read gate for the fused [q|k|v] verify twin: `qkv_nvfp4_t` exists
/// ONLY in the transposed tile-GEMM layout, so `ms_qkv_batchn` must dispatch
/// it there directly — it must never reach `wide_verify_gemm`'s base-weight
/// arms (DP4A / batched GEMV / base GEMM), which would run `q_proj`'s
/// non-transposed weight at the FUSED N and read past it. That is exactly
/// what happened at n = 9..16 after d6e1a9212 widened the GEMV arm from
/// m <= 8 to m <= 16 while the fused gate stayed at n > 8.
#[test]
fn fused_qkv_twin_never_reaches_the_base_weight_gemv() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let h = config.hidden_size;
    // Gated attention: q_proj_dim = 2 * (qh * hd); the fused twin stacks
    // [Q|gate|K|V] into one N.
    let fused_n = 2 * config.num_attention_heads * config.head_dim
        + 2 * config.num_key_value_heads * config.head_dim;
    // `w4a16_gemv_batchm`'s launch geometry at the fused N — the base-weight
    // GEMV signature that proves the twin reached a non-transposed arm.
    let oob = |l: &spark_runtime::gpu::mock::MockLaunch| {
        l.grid == [(fused_n.div_ceil(4)) as u32, 1, 1] && l.block == [256, 1, 1]
    };

    for n in [9usize, 12, 16] {
        let mut twin = attn_layer(&gpu, &config);
        twin.set_fused_qkv_prefill_weight(Some(nvfp4(&gpu, fused_n, h)));
        let (twin_d2d, twin_launches) = qkv_phase_d2d(&gpu, &config, &twin, n);
        assert!(
            !twin_launches.iter().any(oob),
            "n={n}: fused QKV verify ran a base-weight GEMV at the fused N \
             ({fused_n}) — that reads past the end of q_proj's weight \
             (launches: {:?})",
            twin_launches
                .iter()
                .map(|l| (l.grid, l.block))
                .collect::<Vec<_>>()
        );

        // Positive control: a plain layer (no twin) still pays the 3·n d2d
        // scatter — the delta proves the fused arm actually engaged rather
        // than the test passing vacuously on the unfused route.
        let plain = attn_layer(&gpu, &config);
        let (plain_d2d, _) = qkv_phase_d2d(&gpu, &config, &plain, n);
        assert_eq!(
            plain_d2d - twin_d2d,
            3 * n,
            "n={n}: fused twin must remove the 3·n scatter (twin={twin_d2d} plain={plain_d2d})"
        );
    }
}
