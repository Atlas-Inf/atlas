// SPDX-License-Identifier: AGPL-3.0-only
//! `forward_rows_t` eligibility on the recording backend: the dedup arm is
//! eligible exactly for 2..=8 rows on a published transposed-table NVFP4
//! layer with every dedup kernel resolved and no router/norm/LoRA hook, and
//! `BISECT_MOE_T_ROWS` is its kill switch.
use super::{arena_tests::ContextResources, recording::Gpu};
use crate::layers::{moe::MoeLayer, ops};
use crate::weight_map::{DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

const H: usize = 2560;
const I: usize = 640;
const E: usize = 512;
const K: usize = 10;

fn projection(gpu: &Gpu, n: usize, k: usize) -> QuantizedWeight {
    QuantizedWeight {
        weight: gpu.alloc(n * k / 2).unwrap(),
        weight_scale: gpu.alloc(n * k / 32).unwrap(),
        weight_scale_2: 1.0,
        ..QuantizedWeight::null()
    }
}

fn expert(gpu: &Gpu) -> ExpertWeight {
    ExpertWeight {
        gate_proj: projection(gpu, I, H),
        up_proj: projection(gpu, I, H),
        down_proj: projection(gpu, H, I),
    }
}

/// A qwen4_exp-shaped MoE (mirrors `qwen4exp_fast_tests`'s fixture).
fn qwen4exp_layer(gpu: &Gpu) -> (atlas_core::config::ModelConfig, MoeLayer) {
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "qwen4_exp".into();
    config.hidden_size = H;
    config.moe_intermediate_size = I;
    config.shared_expert_intermediate_size = I;
    config.num_experts = E;
    config.num_experts_per_tok = K;
    config.scoring_func = "softmax".into();
    config.adapter_max_rank = 0;
    let mut weights = MoeWeights::empty(E);
    for w in weights.experts.iter_mut() {
        *w = expert(gpu);
    }
    weights.shared_expert = expert(gpu);
    weights.gate = DenseWeight {
        weight: gpu.alloc(E * H * 2).unwrap(),
    };
    weights.shared_expert_gate = DenseWeight {
        weight: gpu.alloc(H * 2).unwrap(),
    };
    let layer = MoeLayer::new(weights, E, None, gpu, &config).unwrap();
    (config, layer)
}

#[test]
fn t_rows_eligible_truth_table() {
    let gpu = Gpu::new();
    let (config, mut layer) = qwen4exp_layer(&gpu);
    layer.transpose_for_prefill_unified(&gpu, &config).unwrap();
    assert!(layer.use_t_layout_for_decode());
    let arena = BufferArena::new(&config, 4, 2048, 16, 4, &gpu).unwrap();
    let resources = ContextResources::new();
    let mut ctx = resources.view(&arena, &config, &gpu);

    // The dedup arm covers every verify batch the kernel's acc[8] holds.
    for rows in 2..=8 {
        assert!(layer.t_rows_eligible(rows, &ctx), "rows={rows}");
    }
    assert!(!layer.t_rows_eligible(1, &ctx));
    assert!(!layer.t_rows_eligible(9, &ctx));

    // BISECT_MOE_T_ROWS is the kill switch; other bits leave it eligible.
    let mut bisect_levers = *ctx.levers;
    bisect_levers.qwen4exp_batch_fast = true;
    bisect_levers.qwen4exp_batch_bisect = ops::BISECT_MOE_T_ROWS;
    ctx.levers = &bisect_levers;
    assert!(!layer.t_rows_eligible(4, &ctx));
    let mut other_bit = bisect_levers;
    other_bit.qwen4exp_batch_bisect = ops::BISECT_MOE_ROWS;
    ctx.levers = &other_bit;
    assert!(layer.t_rows_eligible(4, &ctx));

    // Unresolved kernels fall back to the per-row loop.
    layer.moe_expert_union_t_k = KernelHandle(0);
    assert!(!layer.t_rows_eligible(4, &ctx));
}
