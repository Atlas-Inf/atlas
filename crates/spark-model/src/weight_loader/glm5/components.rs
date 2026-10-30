// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::qwen3_attention::{HcSiteWeights, HcWeights};
use crate::layers::{Glm5KdaWeights, Glm5Projection};
use crate::tp_shard::{TpGdnDims, TpShardKind, shard_dense_bf16, shard_gdn_value_vector};
use crate::weight_map::{DenseWeight, QuantizeCtx, dense_auto, quantize_to_nvfp4};

pub(super) use super::ffn::load_ffn;

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
    let b_proj = load_tp_dense("b_proj", dims.full_nk, dims.h, TpShardKind::ColumnParallel)?;
    let [b_proj, f_a_proj, g_a_proj] = if crate::layers::kda_side_proj_lt_enabled() {
        super::kda_side_pack::pack_side(store, &p, gpu, retirement, b_proj, &dims)?
    } else {
        [b_proj.into_dense(), load("f_a_proj")?, load("g_a_proj")?]
    };
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
        b_proj,
        f_a_proj,
        f_b_proj: load_tp_dense(
            "f_b_proj",
            dims.full_key_dim(),
            dims.kd,
            TpShardKind::ColumnParallel,
        )?
        .into_dense(),
        g_a_proj,
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
