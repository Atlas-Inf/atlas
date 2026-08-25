// SPDX-License-Identifier: AGPL-3.0-only

//! The two per-sublayer FFN builders: the dense SwiGLU MLP every sublayer
//! has, and the shortcut MoE that sublayer 0 computes and sublayer 1 adds.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::layers::{FfnComponent, MoeLayer};
use crate::weight_map::{
    DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight, dense, dense_f32_as_bf16,
    quantize_to_nvfp4, quantized_any,
};

pub(super) fn build_dense_ffn(
    store: &WeightStore,
    prefix: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<FfnComponent> {
    use crate::layers::dense_ffn::{DenseFfnLayer, DenseFfnWeights};
    let inter = config.intermediate_size;
    let h = config.hidden_size;
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let q = |name: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
        let w = dense(store, name)?;
        quantize_to_nvfp4(&w, n, k, gpu, absmax_k, quantize_k, stream)
    };
    let weights = DenseFfnWeights {
        gate_proj: q(&format!("{prefix}.gate_proj.weight"), inter, h)?,
        up_proj: q(&format!("{prefix}.up_proj.weight"), inter, h)?,
        down_proj: q(&format!("{prefix}.down_proj.weight"), h, inter)?,
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    };
    Ok(FfnComponent::Dense(DenseFfnLayer::new(weights, gpu)?))
}

/// The block's shortcut MoE: `mlp.router.*` + `mlp.experts.{e}.*`. LongCat has
/// NO shared expert (the zero/identity experts play that role), so the shared
/// slot is the zero-filled dummy the fused kernels still read.
pub(super) fn build_shortcut_moe(
    store: &WeightStore,
    layer_prefix: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<FfnComponent> {
    let p = format!("{layer_prefix}.mlp");
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    // Router classifier + bias ship F32 (`_keep_in_fp32_modules`); the gate
    // GEMV wants BF16 weights, the bias stays F32 for the router kernel.
    let gate_name = format!("{p}.router.classifier.weight");
    let gate = dense_f32_as_bf16(store, &gate_name, gpu)
        .or_else(|_| dense(store, &gate_name))
        .context("longcat: router classifier")?;
    let correction_bias = dense(store, &format!("{p}.router.e_score_correction_bias"))
        .context("longcat: router e_score_correction_bias")?;

    let alloc_zero = |size: usize| -> Result<DevicePtr> {
        let ptr = gpu.alloc(size)?;
        gpu.memset(ptr, 0, size)?;
        Ok(ptr)
    };
    let group = 16usize;
    let mk_zero = |packed: usize, scale: usize| -> Result<QuantizedWeight> {
        Ok(QuantizedWeight {
            weight: alloc_zero(packed)?,
            weight_scale: alloc_zero(scale)?,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        })
    };
    let shared_expert = ExpertWeight {
        gate_proj: mk_zero(inter * h / 2, inter * (h / group))?,
        up_proj: mk_zero(inter * h / 2, inter * (h / group))?,
        down_proj: mk_zero(h * inter / 2, h * (inter / group))?,
    };
    let shared_expert_gate = DenseWeight {
        weight: alloc_zero(h * 2)?,
    };

    // LongCat ships PLAIN BF16 experts (torch_dtype bfloat16, no NVFP4/FP8
    // metadata), so they are runtime-quantized at load — the Bf16Raw variant.
    let variant = crate::weight_map::detect_nvfp4_variant(store, config);
    let qctx = crate::weight_map::QuantizeCtx {
        absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quantize_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        stream: gpu.default_stream(),
    };
    let mut experts = Vec::with_capacity(config.num_experts);
    for e in 0..config.num_experts {
        let ep = format!("{p}.experts.{e}");
        experts.push(ExpertWeight {
            gate_proj: quantized_any(
                store,
                &format!("{ep}.gate_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?,
            up_proj: quantized_any(
                store,
                &format!("{ep}.up_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?,
            down_proj: quantized_any(
                store,
                &format!("{ep}.down_proj"),
                h,
                inter,
                gpu,
                variant,
                qctx,
            )?,
        });
    }

    let weights = MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: Some(correction_bias),
    };
    Ok(FfnComponent::Moe(MoeLayer::new(
        weights,
        config.num_experts,
        None,
        gpu,
        config,
    )?))
}
