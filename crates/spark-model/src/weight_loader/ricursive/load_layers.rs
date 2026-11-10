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
    let q = |name: &str, n, k| {
        quantized_any(store, &format!("{lp}.mlp.{name}"), n, k, gpu, variant, qctx)
    };
    let gate_proj = q("gate_proj", inter, h)?;
    let up_proj = q("up_proj", inter, h)?;
    let down_proj = q("down_proj", h, inter)?;
    let input_scales = W4a4InputScales {
        gate_up: f32_scalar(gpu, gate_proj.input_scale)?.max(f32_scalar(gpu, up_proj.input_scale)?),
        down: f32_scalar(gpu, down_proj.input_scale)?,
    };
    // gfx1151 has no FP4 MMA: multi-row passes run W4A16 on transposed twins
    // (the m128 GEMM) instead of W4A4 on the native layout.
    let hip = cfg!(atlas_hip);
    let twin = |w: &QuantizedWeight, n, k| -> Result<Option<QuantizedWeight>> {
        Ok(if hip { Some(w.transpose_for_gemm(gpu, n, k)?) } else { None })
    };
    let weights = DenseFfnWeights {
        gate_proj_t: twin(&gate_proj, inter, h)?,
        up_proj_t: twin(&up_proj, inter, h)?,
        down_proj_t: twin(&down_proj, h, inter)?,
        gate_proj,
        up_proj,
        down_proj,
    };
    let mut layer =
        DenseFfnLayer::new_with_activation(weights, FfnActivation::SiLU, gpu, "ricursive")?;
    // Target-only clamped SwiGLU; the drafter keeps the unclamped common kernel.
    layer.set_act_mul(gpu.kernel("ricursive_swiglu", "silu_mul_clamp20")?);
    if hip {
        // Verify rows (M = 17) on the row-GEMV, not m128's padded tile:
        // framework f257 0.97 vs 1.59 ms per call, fib 7.3 vs 9.6 s.
        layer.enable_rowgemv_default(gpu)?;
    } else {
        let limit = atlas_core::config::RICURSIVE_FFN_CLAMP as f32;
        layer.enable_w4a4_rows(gpu, input_scales, limit, [h, inter])?;
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
    let nq = config.num_attention_heads;
    let kv_width = config.num_key_value_heads * hd;
    let q = bf16_matrix(store, &format!("{p}.q_proj.weight"), nq * hd, h)?;
    let g = bf16_matrix(store, &format!("{p}.g_proj.weight"), nq * hd, h)?;
    let qg = interleave_q_gate(gpu, q, g, nq, hd, h)?;
    store.reclaim(gpu, &format!("{p}.q_proj.weight"))?;
    store.reclaim(gpu, &format!("{p}.g_proj.weight"))?;
    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let attn = AttentionWeights {
        q_proj: DenseWeight { weight: qg },
        k_proj: bf16_dense(store, &format!("{p}.k_proj.weight"), kv_width, h)?,
        v_proj: bf16_dense(store, &format!("{p}.v_proj.weight"), kv_width, h)?,
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
    layer.set_o_dense_bf16(bf16_dense(
        store,
        &format!("{p}.o_proj.weight"),
        h,
        nq * hd,
    )?);
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

fn bf16_dense(store: &WeightStore, key: &str, rows: usize, cols: usize) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: bf16_matrix(store, key, rows, cols)?,
    })
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
