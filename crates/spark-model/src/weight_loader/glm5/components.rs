// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::qwen3_attention::{HcSiteWeights, HcWeights};
use crate::layers::{DenseFfnLayer, FfnComponent, Glm5KdaWeights, Glm5Projection, MoeLayer};
use crate::tp_shard::{TpGdnDims, TpShardKind, shard_dense_bf16, shard_gdn_value_vector};
use crate::weight_map::{
    DenseWeight, ExpertWeight, MoeWeights, Nvfp4Variant, QuantizeCtx, dense_auto,
    quantize_to_nvfp4, quantized_any,
};

pub(super) fn load_hc(
    store: &WeightStore,
    lp: &str,
    layer_idx: usize,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<HcWeights> {
    let load_site = |site: &str| -> Result<HcSiteWeights> {
        let hc = config.hc_mult;
        let mix = (2 + hc) * hc;
        let hc_dim = hc * config.hidden_size;
        let load = |suffix: &str, n: usize| {
            super::super::deepseek_v4::assemble::load_hc_f32(
                store,
                &[format!("{lp}.hc_{site}_{suffix}")],
                n,
                gpu,
            )
        };
        // The checkpoint's BF16 `fn` itself (`load_hc_f32` widens it exactly
        // into `hc_fn`): a store pointer, nothing allocated.
        let fn_bf16 = store
            .get(&format!("{lp}.hc_{site}_fn"))
            .ok()
            .filter(|t| t.dtype == WeightDtype::BF16 && t.num_elements() == mix * hc_dim)
            .map_or(DevicePtr::NULL, |t| t.ptr);
        Ok(HcSiteWeights {
            hc_fn: load("fn", mix * hc_dim)?,
            hc_fn_bf16: fn_bf16,
            hc_base: load("base", mix)?,
            hc_scale: load("scale", 3)?,
            lowrank: None,
        })
    };
    Ok(HcWeights {
        attn: load_site("attn")?,
        ffn: load_site("ffn")?,
        head: None,
        hc_mult: config.hc_mult,
        sinkhorn_iters: config.hc_sinkhorn_iters,
        hc_eps: config.hc_eps,
        is_first_model_layer: layer_idx == 0,
        is_last_model_layer: layer_idx + 1 == config.num_hidden_layers,
    })
}

pub(super) fn load_ffn(
    store: &WeightStore,
    lp: &str,
    layer_idx: usize,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    allow_prefill_layout: bool,
) -> Result<FfnComponent> {
    if config.mlp_only_layers.contains(&layer_idx) {
        return load_dense_ffn(store, lp, config, gpu, variant, qctx);
    }
    load_moe(
        store,
        lp,
        layer_idx,
        config,
        gpu,
        variant,
        qctx,
        allow_prefill_layout,
    )
}

fn load_dense_ffn(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<FfnComponent> {
    let p = format!("{lp}.mlp");
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let weights = DenseFfnWeights {
        gate_proj: super::dense::load_projection(
            store,
            &format!("{p}.gate_proj"),
            inter,
            h,
            gpu,
            variant,
            qctx,
        )?,
        up_proj: super::dense::load_projection(
            store,
            &format!("{p}.up_proj"),
            inter,
            h,
            gpu,
            variant,
            qctx,
        )?,
        down_proj: super::dense::load_projection(
            store,
            &format!("{p}.down_proj"),
            h,
            inter,
            gpu,
            variant,
            qctx,
        )?,
        // Keep the memory-tight fallback for the first safe deployment.
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    };
    Ok(FfnComponent::Dense(DenseFfnLayer::new(weights, gpu)?))
}

fn load_moe(
    store: &WeightStore,
    lp: &str,
    _layer_idx: usize,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    allow_prefill_layout: bool,
) -> Result<FfnComponent> {
    let mmq_moe = env_flag("ATLAS_NVFP4_MMQ_MOE");
    let cutlass_moe = env_flag("ATLAS_MOE_GROUPED_CUTLASS");
    validate_moe_layout(config.expert_tp, mmq_moe, cutlass_moe)?;
    let p = format!("{lp}.mlp");
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let mut experts = Vec::with_capacity(config.num_experts);
    anyhow::ensure!(
        !config.expert_tp || matches!(variant, Nvfp4Variant::Standard),
        "expert TP slices ModelOpt NVFP4 experts only, got {variant:?}"
    );
    for expert in 0..config.num_experts {
        if config.expert_tp {
            let ep = format!("{p}.experts.{expert}");
            let slice = |name: &str, n: usize, k: usize, kind: TpShardKind| {
                super::expert_tp::load_expert_slice(
                    store,
                    &format!("{ep}.{name}"),
                    n,
                    k,
                    kind,
                    config,
                    gpu,
                )
                .with_context(|| format!("GLM-5 expert {expert} {name} (expert TP)"))
            };
            experts.push(ExpertWeight {
                gate_proj: slice("gate_proj", inter, h, TpShardKind::ColumnParallel)?,
                up_proj: slice("up_proj", inter, h, TpShardKind::ColumnParallel)?,
                down_proj: slice("down_proj", h, inter, TpShardKind::RowParallel)?,
            });
        } else if config.is_local_expert(expert) {
            let ep = format!("{p}.experts.{expert}");
            experts.push(ExpertWeight {
                gate_proj: quantized_any(
                    store,
                    &format!("{ep}.gate_proj"),
                    inter,
                    h,
                    gpu,
                    variant,
                    qctx,
                )
                .with_context(|| format!("GLM-5 expert {expert} gate_proj"))?,
                up_proj: quantized_any(
                    store,
                    &format!("{ep}.up_proj"),
                    inter,
                    h,
                    gpu,
                    variant,
                    qctx,
                )
                .with_context(|| format!("GLM-5 expert {expert} up_proj"))?,
                down_proj: quantized_any(
                    store,
                    &format!("{ep}.down_proj"),
                    h,
                    inter,
                    gpu,
                    variant,
                    qctx,
                )
                .with_context(|| format!("GLM-5 expert {expert} down_proj"))?,
            });
        } else {
            experts.push(ExpertWeight::null());
        }
    }
    let shared = format!("{p}.shared_experts");
    let (shared_gate, gate_origin) = crate::layers::moe::load_glm_shared_fp8_weight(
        store,
        &format!("{shared}.gate_proj"),
        config.shared_expert_intermediate_size,
        h,
        gpu,
        variant,
        qctx,
        allow_prefill_layout,
    )?;
    let (shared_up, up_origin) = crate::layers::moe::load_glm_shared_fp8_weight(
        store,
        &format!("{shared}.up_proj"),
        config.shared_expert_intermediate_size,
        h,
        gpu,
        variant,
        qctx,
        allow_prefill_layout,
    )?;
    let (shared_down, down_origin) = crate::layers::moe::load_glm_shared_fp8_weight(
        store,
        &format!("{shared}.down_proj"),
        h,
        config.shared_expert_intermediate_size,
        gpu,
        variant,
        qctx,
        allow_prefill_layout,
    )?;
    let shared_expert = ExpertWeight {
        gate_proj: shared_gate,
        up_proj: shared_up,
        down_proj: shared_down,
    };
    let bias = store.get(&format!("{p}.gate.e_score_correction_bias"))?;
    anyhow::ensure!(
        bias.num_elements() == config.num_experts,
        "GLM-5 routing bias length mismatch"
    );
    let weights = MoeWeights {
        gate: dense_auto(store, &format!("{p}.gate.weight"), gpu)?,
        shared_expert,
        // GLM-5 shared expert is always active and has no learned gate.
        shared_expert_gate: DenseWeight {
            weight: DevicePtr::NULL,
        },
        experts,
        router_pre_norm: None,
        correction_bias: Some(DenseWeight { weight: bias.ptr }),
    };
    let mut layer = MoeLayer::new(weights, config.num_experts, None, gpu, config)?;
    layer.set_shared_fp8_origins([gate_origin, up_origin, down_origin])?;
    if mmq_moe {
        layer.repack_nvfp4_mmq_unified(gpu, config, store)?;
    } else if cutlass_moe {
        // CUTLASS consumes the checkpoint-native packed weights and adds only
        // its block-scale swizzle. Avoid materializing the much larger Atlas
        // transposed expert twins in this mode.
        layer.build_cutlass_grouped_sfb(gpu, config, gpu.default_stream())?;
        layer.transpose_shared_only(gpu, config)?;
    } else if allow_prefill_layout
        && unified_moe_layout_enabled(std::env::var("ATLAS_UNIFIED_MOE_LAYOUT").ok().as_deref())
    {
        // Replace the decode-native expert storage with Atlas' transposed
        // layout one layer at a time. The transpose helper frees each source
        // phase before advancing, and EP slabs contain only locally owned
        // experts, so peak memory stays bounded on dual GB10.
        // K=4 verification batches GLM's always-on shared expert through the
        // exact-M=4 GEMV, which needs its decode-native layout. Routed experts
        // remain transposed-only to avoid hybrid layout's memory cost.
        let freed = layer.transpose_for_prefill_unified_keep_shared(gpu, config)?;
        store.mark_freed_elsewhere(&freed)?;
    }
    // The factory installs optional shared FP8 caches only after all target,
    // MTP and head loading has released replaced checkpoint allocations.
    Ok(FfnComponent::Moe(layer))
}

fn validate_moe_layout(expert_tp: bool, mmq: bool, cutlass: bool) -> Result<()> {
    anyhow::ensure!(
        !(mmq && expert_tp),
        "ATLAS_NVFP4_MMQ_MOE does not support expert TP"
    );
    anyhow::ensure!(
        !(mmq && cutlass),
        "ATLAS_NVFP4_MMQ_MOE and ATLAS_MOE_GROUPED_CUTLASS are mutually exclusive"
    );
    Ok(())
}

fn unified_moe_layout_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

pub(super) fn load_kda_weights(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    qctx: QuantizeCtx,
    retirement: Option<&super::retirement::RetirementLog>,
) -> Result<Glm5KdaWeights> {
    use super::retirement::LoadedDense;
    let p = format!("{lp}.self_attn");
    let load = |name: &str| dense_auto(store, &format!("{p}.{name}.weight"), gpu);
    let dims = TpGdnDims::from_config(config);
    let p_dim = dims.local_key_dim();
    let load_tp_dense =
        |name: &str, n: usize, k: usize, kind: TpShardKind| -> Result<LoadedDense<'_>> {
            let source =
                LoadedDense::load(store, &format!("{p}.{name}.weight"), gpu, retirement, false)?;
            let (local, local_n, local_k) = shard_dense_bf16(
                source.dense.weight,
                n,
                k,
                kind,
                dims.tp_rank,
                dims.tp_size,
                gpu,
            )?;
            source.replaced(local, local_n * local_k * 2, gpu)
        };
    let load_hot = |name: &str,
                    n: usize,
                    k: usize,
                    kind: TpShardKind,
                    transpose_prefill: bool|
     -> Result<Glm5Projection> {
        let dense = load_tp_dense(name, n, k, kind)?;
        let local_n = if kind == TpShardKind::ColumnParallel {
            n / dims.tp_size
        } else {
            n
        };
        let local_k = if kind == TpShardKind::RowParallel {
            k / dims.tp_size
        } else {
            k
        };
        let decode_nvfp4 = quantize_to_nvfp4(
            &dense.dense,
            local_n,
            local_k,
            gpu,
            qctx.absmax_k,
            qctx.quantize_k,
            qctx.stream,
        )?;
        // Quantization synchronizes internally, so the checkpoint-native BF16
        // source is no longer needed. Releasing it keeps the safe 92% memory
        // budget viable on GB10 instead of retaining two projection copies.
        dense.release(gpu)?;
        let prefill_nvfp4_t = transpose_prefill
            .then(|| decode_nvfp4.transpose_for_gemm(gpu, local_n, local_k))
            .transpose()?;
        Ok(Glm5Projection {
            nvfp4: decode_nvfp4,
            prefill_nvfp4_t,
        })
    };
    let width = config.linear_conv_kernel_dim;
    let local_conv_bytes = p_dim * width * 2;
    let conv_ptr = gpu.alloc(3 * local_conv_bytes)?;
    for (slot, name) in ["q_conv1d", "k_conv1d", "v_conv1d"].iter().enumerate() {
        let weight = load_tp_dense(
            name,
            dims.full_key_dim(),
            width,
            TpShardKind::ColumnParallel,
        )?;
        gpu.copy_d2d(
            weight.dense.weight,
            conv_ptr.offset(slot * local_conv_bytes),
            local_conv_bytes,
        )?;
        weight.release(gpu)?;
    }
    let a_log_full = LoadedDense::load(store, &format!("{p}.A_log"), gpu, retirement, true)?;
    let (a_log, a_len) = shard_gdn_value_vector(a_log_full.dense.weight, &dims, 1, 4, gpu)?;
    let a_log = a_log_full.replaced(a_log, a_len * 4, gpu)?.into_dense();
    let dt_bias_full = LoadedDense::load(store, &format!("{p}.dt_bias"), gpu, retirement, true)?;
    let (dt_bias, dt_len) =
        shard_gdn_value_vector(dt_bias_full.dense.weight, &dims, dims.kd, 4, gpu)?;
    let dt_bias = dt_bias_full
        .replaced(dt_bias, dt_len * 4, gpu)?
        .into_dense();
    Ok(Glm5KdaWeights {
        q_proj: load_hot(
            "q_proj",
            dims.full_key_dim(),
            dims.h,
            TpShardKind::ColumnParallel,
            true,
        )?,
        k_proj: load_hot(
            "k_proj",
            dims.full_key_dim(),
            dims.h,
            TpShardKind::ColumnParallel,
            true,
        )?,
        v_proj: load_hot(
            "v_proj",
            dims.full_value_dim(),
            dims.h,
            TpShardKind::ColumnParallel,
            true,
        )?,
        b_proj: load_tp_dense("b_proj", dims.full_nk, dims.h, TpShardKind::ColumnParallel)?
            .into_dense(),
        f_a_proj: load("f_a_proj")?,
        f_b_proj: load_tp_dense(
            "f_b_proj",
            dims.full_key_dim(),
            dims.kd,
            TpShardKind::ColumnParallel,
        )?
        .into_dense(),
        g_a_proj: load("g_a_proj")?,
        g_b_proj: load_tp_dense(
            "g_b_proj",
            dims.full_key_dim(),
            dims.kd,
            TpShardKind::ColumnParallel,
        )?
        .into_dense(),
        conv: DenseWeight { weight: conv_ptr },
        a_log,
        dt_bias,
        o_norm: load("o_norm")?,
        o_proj: load_hot(
            "o_proj",
            dims.h,
            dims.full_value_dim(),
            TpShardKind::RowParallel,
            true,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::{unified_moe_layout_enabled, validate_moe_layout};

    #[test]
    fn mmq_layout_rejects_unsupported_loading_combinations() {
        for expert_tp in [false, true] {
            for mmq in [false, true] {
                for cutlass in [false, true] {
                    assert_eq!(
                        validate_moe_layout(expert_tp, mmq, cutlass).is_ok(),
                        !(mmq && (expert_tp || cutlass))
                    );
                }
            }
        }
    }

    #[test]
    fn unified_moe_layout_is_explicitly_opt_in() {
        assert!(unified_moe_layout_enabled(Some("1")));
        assert!(unified_moe_layout_enabled(Some("TRUE")));
        assert!(!unified_moe_layout_enabled(None));
        assert!(!unified_moe_layout_enabled(Some("0")));
        assert!(!unified_moe_layout_enabled(Some("full")));
    }
}
