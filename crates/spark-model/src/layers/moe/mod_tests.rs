// SPDX-License-Identifier: AGPL-3.0-only

//! Tests split out of `moe/mod.rs` for the ≤500 LoC file-size cap.

use super::*;

use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn test_moe_kernel_loading() {
    let gpu = MockGpuBackend::new();
    assert!(gpu.kernel("gemv", "dense_gemv_bf16").is_ok());
    assert!(gpu.kernel("w4a16_gemv", "w4a16_gemv").is_ok());
    assert!(gpu.kernel("moe_topk", "moe_topk_softmax").is_ok());
    assert!(
        gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_gate_up")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_gate_up_2x")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_silu_down")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_silu_down_2x")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_shared_expert_fused", "moe_expert_gate_up_shared")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_shared_expert_fused", "moe_expert_silu_down_shared")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")
            .is_ok()
    );
    // K=2 batch dispatch
    assert!(gpu.kernel("moe_topk", "moe_topk_softmax_batched").is_ok());
}

#[test]
fn separate_shared_expert_requires_three_non_null_weights() {
    let gate = DenseWeight {
        weight: DevicePtr(11),
    };
    let up = DenseWeight {
        weight: DevicePtr(22),
    };
    let down = DenseWeight {
        weight: DevicePtr(33),
    };

    let shared = Bf16SharedExpert::new(gate, up, down).expect("valid BF16 shared expert");
    assert_eq!(shared.gate_proj.weight, gate.weight);
    assert_eq!(shared.up_proj.weight, up.weight);
    assert_eq!(shared.down_proj.weight, down.weight);

    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };
    assert!(Bf16SharedExpert::new(null, up, down).is_err());
    assert!(Bf16SharedExpert::new(gate, null, down).is_err());
    assert!(Bf16SharedExpert::new(gate, up, null).is_err());
}

/// The MXFP8 shared expert dispatches onto `mxfp8_proj.cu` at every row
/// band: the 1-row decode GEMV, the 2-row verify batch, and the >32-row
/// dequant+GEMM prefill arm — with `silu_mul` between up and down.
#[test]
fn mxfp8_shared_expert_runs_at_decode_verify_and_prefill_rows() {
    use crate::layer::ForwardContext;
    use crate::layers::moe::shared_fp8_cache_test_gpu::RecordingGpu;
    use atlas_core::config::ModelConfig;
    use spark_runtime::buffers::BufferArena;
    use spark_runtime::gpu::KernelHandle;

    let gpu = RecordingGpu::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 2560;
    config.num_experts = 512;
    config.num_experts_per_tok = 10;
    config.moe_intermediate_size = 640;
    config.shared_expert_intermediate_size = 640;
    config.vocab_size = 8;
    let mut layer = MoeLayer::new(MoeWeights::empty(512), 512, None, &gpu, &config).unwrap();
    let (h, inter) = (config.hidden_size, config.shared_expert_intermediate_size);
    let proj = |n: usize, k: usize, ptr: u64| Mxfp8Proj {
        w: crate::weight_map::Mxfp8Weight {
            data: DevicePtr(ptr),
            scales: DevicePtr(ptr + 0x1000),
        },
        n: n as u32,
        k: k as u32,
    };
    let kernels = ops::Mxfp8ProjKernels {
        gemv: KernelHandle(111),
        batchm: KernelHandle(222),
        dequant: KernelHandle(333),
        rows32: KernelHandle(0),
        quant: KernelHandle(0),
    };
    let scratch = ops::Mxfp8Scratch {
        ptr: DevicePtr(0x9000_0000),
        bytes: 1 << 24,
    };
    layer
        .set_mxfp8_shared_expert(
            proj(inter, h, 0x10000),
            proj(inter, h, 0x20000),
            proj(h, inter, 0x30000),
            None,
            kernels,
            scratch,
            [None, None, None],
        )
        .unwrap();
    assert!(layer.has_separate_shared_expert());

    let buffers = BufferArena::new(&config, 64, 4096, 16, 1, &gpu).unwrap();
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
    let (input, gate_out, up_out, down_out) = (
        DevicePtr(0x40000),
        DevicePtr(0x50000),
        DevicePtr(0x60000),
        DevicePtr(0x70000),
    );
    // gate GEMV, up GEMV, silu_mul, down GEMV on the band's row kernel. The
    // >32-row arm (dequant + CUTLASS GEMM) is exercised by the mx8 GPU tests —
    // its FFI cannot run on this mock.
    for (m, want) in [(1u32, 111u64), (2, 222), (32, 222)] {
        gpu.launches.lock().unwrap().clear();
        let ran = layer
            .run_separate_shared_expert(
                input,
                m,
                h as u32,
                inter as u32,
                gate_out,
                up_out,
                down_out,
                &ctx,
                true,
                19,
            )
            .unwrap();
        assert!(ran, "separate shared expert must run at m={m}");
        let kernels_run: Vec<u64> = gpu
            .launches
            .lock()
            .unwrap()
            .iter()
            .map(|l| l.kernel)
            .collect();
        // O1: batchm serves 2..=32 rows in ONE launch per projection, so the
        // shared expert is 3 launches at every row count here (m=1 takes the
        // single-row GEMV arm, also one launch each).
        let want_n = 3;
        assert_eq!(
            kernels_run.iter().filter(|&&k| k == want).count(),
            want_n,
            "m={m}: expected {want_n} launches of {want}, got {kernels_run:?}"
        );
    }
}

#[test]
fn bf16_router_keeps_dense_gemm_unless_flagged() {
    // Regression for the FP4-Local row-invariant router arm: a BF16-gate
    // MoeLayer whose builder did NOT call set_row_invariant_bf16_router()
    // must route through the scalar dense_gemm (its launch block is 2D),
    // not the qwen4exp dense_gemv_batchm tiers (1D blocks).
    use crate::layer::ForwardContext;
    use crate::layers::moe::shared_fp8_cache_test_gpu::RecordingGpu;
    use atlas_core::config::ModelConfig;
    use spark_runtime::buffers::BufferArena;

    let gpu = RecordingGpu::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 2560;
    config.num_experts = 512;
    config.num_experts_per_tok = 10;
    config.moe_intermediate_size = 640;
    config.shared_expert_intermediate_size = 640;
    config.vocab_size = 8;
    for flagged in [false, true] {
        let mut layer = MoeLayer::new(MoeWeights::empty(512), 512, None, &gpu, &config).unwrap();
        if flagged {
            layer.set_row_invariant_bf16_router();
        }
        let buffers = BufferArena::new(&config, 64, 4096, 16, 1, &gpu).unwrap();
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
        gpu.launches.lock().unwrap().clear();
        let n = 8u32;
        layer
            .batched_gate_logits(
                DevicePtr(0x40000),
                n,
                config.hidden_size as u32,
                config.num_experts as u32,
                DevicePtr(0),
                &ctx,
                19,
            )
            .unwrap();
        let launches = gpu.launches.lock().unwrap();
        // dense_gemm is a 2D-block tiled kernel; the dense_rows/dense_gemv
        // tiers all launch 1D blocks.
        let gemm_blocks = launches.iter().filter(|l| l.block[1] > 1).count();
        let rows_blocks = launches.iter().filter(|l| l.block[1] == 1).count();
        if flagged {
            assert_eq!(gemm_blocks, 0, "flagged layer must not run dense_gemm");
            assert!(rows_blocks > 0);
        } else {
            assert_eq!(gemm_blocks, 1, "unflagged BF16 router must run dense_gemm");
        }
    }
}
