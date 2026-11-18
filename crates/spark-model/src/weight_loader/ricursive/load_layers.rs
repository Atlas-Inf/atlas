// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layer::TransformerLayer;
use crate::layers::dense_ffn::{DenseFfnWeights, W4a4InputScales};
use crate::layers::qwen3_attention::value_residual::{ValueResidual, value_residual_blend};
use crate::layers::{DenseFfnLayer, FfnActivation, FfnComponent, Qwen3AttentionLayer};
use crate::tp_shard::{TpShardKind, shard_dense_bf16, shard_quantized_nvfp4};
use crate::weight_map::{
    AttentionWeights, DenseWeight, QuantizeCtx, QuantizedWeight, dense, detect_nvfp4_variant,
    load_kv_scales, quantized_any,
};

pub(super) fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    ensure!(
        layer_kv_dtypes.len() == config.num_hidden_layers,
        "ricursive requires one KV dtype per attention layer"
    );
    let variant = detect_nvfp4_variant(store, config);
    tracing::info!("Ricursive NVFP4 variant: {variant:?}");
    let stream = gpu.default_stream();
    let qctx = QuantizeCtx {
        absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quantize_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        stream,
    };
    let lerp_k = gpu.kernel("value_residual", "value_residual_lerp")?;
    // strix-hip ships no `bf16_gemv_m32` (cp.async m32 GEMV): handle 0, with
    // no lookup issued (the boot audit fails closed), sends the verify-row
    // projections to the BF16 GEMM fallback.
    let wide_gemv_k = if cfg!(atlas_hip) {
        spark_runtime::gpu::KernelHandle(0)
    } else {
        gpu.kernel("bf16_gemv_m32", "bf16_gemv_m32")?
    };
    let verify_attn_k = gpu.kernel("ricursive_verify_attn", "ricursive_verify_attn")?;
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        let ffn = load_ffn(store, config, gpu, &lp, variant, qctx)?;
        let mut layer = load_attention(store, config, gpu, &lp, ffn, layer_kv_dtypes[i], i)?;
        let lambda = load_bf16_scalar(store, gpu, &format!("{lp}.self_attn.value_res_lambda"))?;
        layer.set_value_residual(ValueResidual {
            x0: DevicePtr::NULL,
            blended: DevicePtr::NULL,
            max_rows: 0,
            blend: value_residual_blend(lambda),
            capture: i == 0,
            lerp_k,
            wide_gemv_k,
            verify_attn_k,
        })?;
        gpu.synchronize(stream)?;
        layers.push(Box::new(layer));
    }
    Ok(layers)
}

#[allow(clippy::too_many_arguments)]
fn load_ffn(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    variant: crate::weight_map::Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<FfnComponent> {
    let (h, inter) = (config.hidden_size, config.intermediate_size);
    let (tp_rank, tp) = (config.tp_rank, config.tp_world_size.max(1));
    ensure!(
        inter.is_multiple_of(tp),
        "ricursive intermediate_size {inter} not divisible by tp_size {tp}"
    );
    let local_inter = inter / tp;
    // Gate/up column-parallel on inter, down row-parallel on inter: each
    // rank's SwiGLU and down-GEMV run on its local slice and the partial
    // output is all-reduced by the layer's TP seam. `n`/`k` stay the FULL
    // dims — the shard helpers slice from them.
    let q = |name: &str, n: usize, k: usize, kind: TpShardKind| -> Result<QuantizedWeight> {
        let w = quantized_any(store, &format!("{lp}.mlp.{name}"), n, k, gpu, variant, qctx)?;
        if tp > 1 && kind == TpShardKind::ColumnParallel {
            ensure!(
                !w.has_per_row_scale2(),
                "{lp}.mlp.{name}: per-row scale2 does not shard column-parallel"
            );
        }
        let s = shard_quantized_nvfp4(&w, n, k, kind, tp_rank, tp, 16, gpu)?;
        // The shard copies the packed weight + per-group scales; release the
        // full tensors. Store-backed sources are reclaimed by key;
        // runtime-quantized sources (Bf16Raw, EXL3, FP8 fallback) are fresh
        // allocs and are freed directly. The per-tensor `weight_scale_2` /
        // `input_scale` stay resident — the shard replicates them.
        if tp > 1 {
            let store_backed = |ptr: DevicePtr| {
                !ptr.is_null()
                    && ["weight", "weight_packed", "weight_scale"].iter().any(|s| {
                        store
                            .get(&format!("{lp}.mlp.{name}.{s}"))
                            .map(|t| t.ptr == ptr)
                            .unwrap_or(false)
                    })
            };
            if !store_backed(w.weight) {
                gpu.free(w.weight)?;
            }
            if !store_backed(w.weight_scale) {
                gpu.free(w.weight_scale)?;
            }
            for suffix in ["weight", "weight_packed", "weight_scale", "trellis"] {
                let key = format!("{lp}.mlp.{name}.{suffix}");
                if store.contains(&key) {
                    store.reclaim(gpu, &key)?;
                }
            }
        }
        Ok(s)
    };
    let gate_proj = q("gate_proj", inter, h, TpShardKind::ColumnParallel)?;
    let up_proj = q("up_proj", inter, h, TpShardKind::ColumnParallel)?;
    let down_proj = q("down_proj", h, inter, TpShardKind::RowParallel)?;
    let input_scales = W4a4InputScales {
        gate_up: f32_scalar(gpu, gate_proj.input_scale)?.max(f32_scalar(gpu, up_proj.input_scale)?),
        down: f32_scalar(gpu, down_proj.input_scale)?,
    };
    // gfx1151 has no FP4 MMA: multi-row passes run W4A16 on transposed twins
    // (the m128 GEMM) instead of W4A4 on the native layout.
    let hip = cfg!(atlas_hip);
    let twin = |w: &QuantizedWeight, n, k| -> Result<Option<QuantizedWeight>> {
        Ok(if hip {
            Some(w.transpose_for_gemm(gpu, n, k)?)
        } else {
            None
        })
    };
    let weights = DenseFfnWeights {
        gate_proj_t: twin(&gate_proj, local_inter, h)?,
        up_proj_t: twin(&up_proj, local_inter, h)?,
        down_proj_t: twin(&down_proj, h, local_inter)?,
        gate_proj,
        up_proj,
        down_proj,
    };
    let mut layer =
        DenseFfnLayer::new_with_activation(weights, FfnActivation::SiLU, gpu, "ricursive")?;
    // Target-only clamped SwiGLU; the drafter keeps the unclamped common kernel.
    layer.set_act_mul(gpu.kernel("ricursive_swiglu", "silu_mul_clamp20")?);
    if tp > 1 {
        layer.set_tp_sharded(local_inter);
    }
    if !hip {
        let limit = atlas_core::config::RICURSIVE_FFN_CLAMP as f32;
        layer.enable_w4a4_rows(gpu, input_scales, limit, [h, local_inter])?;
    }
    Ok(FfnComponent::Dense(layer))
}

fn load_attention(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    ffn: FfnComponent,
    kv_dtype: KvCacheDtype,
    i: usize,
) -> Result<Qwen3AttentionLayer> {
    let p = format!("{lp}.self_attn");
    let (h, hd) = (config.hidden_size, config.head_dim);
    let (tp_rank, tp) = (config.tp_rank, config.tp_world_size.max(1));
    // Head counts in `config` are already TP-local (topology divided them);
    // the checkpoint dims are the full counts.
    let nq = config.num_attention_heads;
    let kv = config.num_key_value_heads;
    let (nq_full, kv_full) = (nq * tp, kv * tp);
    // Column-parallel slice of a `[rows, h]` projection: rank `tp_rank`
    // keeps its contiguous row block; the full checkpoint tensor is
    // reclaimed once the shard is copied out.
    let shard_col = |key: &str, full_rows: usize| -> Result<DevicePtr> {
        let src = bf16_matrix(store, key, full_rows, h)?;
        let (local, _, _) = shard_dense_bf16(
            src,
            full_rows,
            h,
            TpShardKind::ColumnParallel,
            tp_rank,
            tp,
            gpu,
        )?;
        if tp > 1 {
            store.reclaim(gpu, key)?;
        }
        Ok(local)
    };
    // Q and the attention gate are separate tensors (unlike Qwen3-Next's
    // fused q_proj): shard each column-parallel, then interleave this
    // rank's head block into the `[Q_i | G_i]` layout the layer expects.
    let q = shard_col(&format!("{p}.q_proj.weight"), nq_full * hd)?;
    let g = shard_col(&format!("{p}.g_proj.weight"), nq_full * hd)?;
    let qg = interleave_q_gate(gpu, q, g, nq, hd, h)?;
    if tp > 1 {
        // The interleave consumed the fresh shards; the store tensors were
        // already reclaimed inside `shard_col`.
        gpu.free(q)?;
        gpu.free(g)?;
    } else {
        store.reclaim(gpu, &format!("{p}.q_proj.weight"))?;
        store.reclaim(gpu, &format!("{p}.g_proj.weight"))?;
    }
    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let attn = AttentionWeights {
        q_proj: DenseWeight { weight: qg },
        k_proj: DenseWeight {
            weight: shard_col(&format!("{p}.k_proj.weight"), kv_full * hd)?,
        },
        v_proj: DenseWeight {
            weight: shard_col(&format!("{p}.v_proj.weight"), kv_full * hd)?,
        },
        o_proj: QuantizedWeight::null(),
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    let mut layer = Qwen3AttentionLayer::new(
        dense(store, &format!("{lp}.input_layernorm.weight"))?,
        attn,
        dense(store, &format!("{lp}.pre_feedforward_layernorm.weight"))?,
        ffn,
        i,
        None,
        None,
        None,
        gpu,
        kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;
    // O projection is row-parallel: each rank keeps the input columns of
    // its local heads, and the layer's TP all-reduce sums the partial
    // attention output across ranks.
    let o_key = format!("{p}.o_proj.weight");
    let o_src = bf16_matrix(store, &o_key, h, nq_full * hd)?;
    let (o_w, _, _) = shard_dense_bf16(
        o_src,
        h,
        nq_full * hd,
        TpShardKind::RowParallel,
        tp_rank,
        tp,
        gpu,
    )?;
    if tp > 1 {
        store.reclaim(gpu, &o_key)?;
    }
    layer.set_o_dense_bf16(DenseWeight { weight: o_w });
    layer.set_post_sublayer_norms(
        dense(store, &format!("{lp}.post_attention_layernorm.weight"))?,
        dense(store, &format!("{lp}.post_feedforward_layernorm.weight"))?,
    );
    // Sliding layers: windowed RoPE attention. Global layers: full NoPE.
    ensure!(
        config.layer_types[i] == LayerType::FullAttention,
        "ricursive layer {i} is not attention"
    );
    if config.sliding_attention_layers.contains(&i) {
        layer.set_sliding_window(Some(config.sliding_window));
        layer.set_rope_overrides(config.rope_theta as f32, hd as u32);
    } else {
        layer.set_sliding_window(None);
        layer.set_rope_disabled(true);
    }
    Ok(layer)
}

fn bf16_matrix(store: &WeightStore, key: &str, rows: usize, cols: usize) -> Result<DevicePtr> {
    let t = store.get(key)?;
    ensure!(
        t.dtype == WeightDtype::BF16 && t.shape == [rows, cols],
        "{key}: expected BF16 [{rows}, {cols}], got {:?} {:?} (Strict keeps attention BF16)",
        t.dtype,
        t.shape
    );
    Ok(t.ptr)
}

/// `[Q_0 | G_0 | Q_1 | G_1 | ...]` per head, the layout Atlas's gated
/// projections and `deinterleave_qg` expect (Qwen3-Next's fused q_proj).
fn interleave_q_gate(
    gpu: &dyn GpuBackend,
    q: DevicePtr,
    g: DevicePtr,
    heads: usize,
    hd: usize,
    h: usize,
) -> Result<DevicePtr> {
    let head_bytes = hd * h * 2;
    let qg = gpu
        .alloc(2 * heads * head_bytes)
        .context("allocate interleaved q/gate projection")?;
    let stream = gpu.default_stream();
    gpu.copy_d2d_2d_async(q, head_bytes, qg, 2 * head_bytes, head_bytes, heads, stream)?;
    let g_dst = qg.offset(head_bytes);
    gpu.copy_d2d_2d_async(
        g,
        head_bytes,
        g_dst,
        2 * head_bytes,
        head_bytes,
        heads,
        stream,
    )?;
    gpu.synchronize(stream)?;
    Ok(qg)
}

fn load_bf16_scalar(store: &WeightStore, gpu: &dyn GpuBackend, key: &str) -> Result<f32> {
    let t = store.get(key)?;
    ensure!(
        t.dtype == WeightDtype::BF16 && t.shape.iter().product::<usize>() == 1,
        "{key} must be a BF16 scalar"
    );
    let mut bytes = [0u8; 2];
    gpu.copy_d2h(t.ptr, &mut bytes)?;
    Ok(f32::from_bits((u16::from_le_bytes(bytes) as u32) << 16))
}

/// A checkpoint `input_scale` (an F32 scalar already on the device).
fn f32_scalar(gpu: &dyn GpuBackend, ptr: DevicePtr) -> Result<f32> {
    ensure!(
        !ptr.is_null(),
        "ricursive MLP projection is missing its input_scale"
    );
    let mut bytes = [0u8; 4];
    gpu.copy_d2h(ptr, &mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}
