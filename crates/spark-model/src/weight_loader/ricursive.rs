// SPDX-License-Identifier: AGPL-3.0-only

//! Ricursive Ripple weight loader (`model_type = "ricursive"`).
//!
//! Every block is `Qwen3AttentionLayer` + `DenseFfnLayer` with the Gemma-4
//! PeriNorm wiring (pre-attn, post-attn-out, pre-FFN, post-FFN-out norms),
//! loaded as HF-vanilla weights (`ships_vanilla_norm_weights`). Ripple-specific:
//!
//! * gated attention ships as a separate elementwise `g_proj` [H*D, hidden];
//!   it is folded into `q_proj` as the Qwen3.5 per-head `[q_h | gate_h]`
//!   interleave so the existing gated path (deinterleave + sigmoid gate)
//!   serves it, and the two source tensors are reclaimed;
//! * sliding layers: RoPE (theta, full head_dim) + window; global layers: NoPE;
//! * value residuals: `blend = bf16(sigmoid(value_res_lambda))` per layer;
//! * the clamped SwiGLU is selected by `DenseFfnLayer::new(.., "ricursive")`.
//!
//! The NVFP4 checkpoint quantizes the MLP only (ModelOpt ignore list covers
//! every `self_attn`, `embed_tokens` and `lm_head`), so attention is BF16.

use anyhow::{Context, Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::{DenseFfnLayer, FfnComponent, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, QuantizedWeight, dense, dense_auto,
    detect_nvfp4_variant, load_dense_ffn,
};

/// `RIPPLE_FFN_CLAMP` in `kernels/strix-hip/ripple-65b/nvfp4/ripple_ops.cu`.
const KERNEL_FFN_CLAMP: f32 = 20.0;

pub struct RicursiveWeightLoader;

impl ModelWeightLoader for RicursiveWeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.embed_tokens.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.norm.weight")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "lm_head.weight")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}

fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    ensure!(
        layer_kv_dtypes.len() == config.num_hidden_layers,
        "ricursive requires one KV dtype per attention layer"
    );
    ensure!(
        config.ffn_clamp_threshold == KERNEL_FFN_CLAMP,
        "ricursive ffn_clamp_threshold {} != the compiled ripple_silu_mul_clamp constant {}",
        config.ffn_clamp_threshold,
        KERNEL_FFN_CLAMP
    );
    ensure!(config.attn_gated, "ricursive without use_gated_attention is not supported");

    let variant = detect_nvfp4_variant(store, config);
    tracing::info!("Ripple NVFP4 variant: {variant:?}");
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        let ffn_weights =
            load_dense_ffn(store, &lp, gpu, variant, absmax_k, quantize_k, stream, config)?;
        let ffn = FfnComponent::Dense(DenseFfnLayer::new(ffn_weights, gpu, &config.model_type)?);
        let layer = load_attention(store, config, gpu, &lp, ffn, layer_kv_dtypes[i], i)?;
        layers.push(Box::new(layer));
    }
    Ok(layers)
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
    let (h, nq, nkv, hd) = (
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    );
    validate_matrix(store, &format!("{p}.q_proj.weight"), nq * hd, h)?;
    validate_matrix(store, &format!("{p}.g_proj.weight"), nq * hd, h)?;
    validate_matrix(store, &format!("{p}.k_proj.weight"), nkv * hd, h)?;
    validate_matrix(store, &format!("{p}.v_proj.weight"), nkv * hd, h)?;
    validate_matrix(store, &format!("{p}.o_proj.weight"), h, nq * hd)?;

    let q_gate = interleave_q_gate(store, gpu, &p, nq, hd, h)?;
    let attn = AttentionWeights {
        q_proj: q_gate,
        k_proj: dense_auto(store, &format!("{p}.k_proj.weight"), gpu)?,
        v_proj: dense_auto(store, &format!("{p}.v_proj.weight"), gpu)?,
        o_proj: QuantizedWeight::null(),
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    // PeriNorm: Qwen3AttentionLayer's `input_norm` is the pre-attention norm
    // and its `post_attn_norm` is the norm on the FFN INPUT (pre-FFN). The two
    // post-sublayer norms are the Gemma-4 `post_*_out_norm` slots.
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
    layer.set_dimension_overrides(hd, nq, nkv);
    layer.set_o_dense_bf16(dense_auto(store, &format!("{p}.o_proj.weight"), gpu)?);
    layer.set_post_sublayer_norms(
        dense(store, &format!("{lp}.post_attention_layernorm.weight"))?,
        dense(store, &format!("{lp}.post_feedforward_layernorm.weight"))?,
    );
    match config.layer_types[i] {
        LayerType::SlidingAttention => {
            layer.set_sliding_window(Some(config.sliding_window));
            layer.set_rope_overrides(config.rope_theta as f32, hd as u32);
        }
        LayerType::FullAttention => {
            layer.set_sliding_window(None);
            layer.set_rope_disabled(true);
        }
        other => anyhow::bail!("ricursive layer {i} is not attention: {other:?}"),
    }
    if config.value_residual {
        let lambda = load_scalar(store, gpu, &format!("{p}.value_res_lambda"))?;
        layer.set_value_residual_blend(value_residual_blend(lambda));
    }
    Ok(layer)
}

/// `torch.sigmoid(value_res_lambda[0]).to(bf16)` — the HF/SGLang blend, a
/// BF16 value carried here as the f32 it widens to.
pub(crate) fn value_residual_blend(lambda: f32) -> f32 {
    let s = 1.0 / (1.0 + (-lambda).exp());
    bf16_round(s)
}

fn bf16_round(x: f32) -> f32 {
    // Round-to-nearest-even to BF16, as `Tensor.to(torch.bfloat16)` does.
    let b = x.to_bits();
    let rounded = b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// Build the gated `[q_h | gate_h]` per-head interleave from separate
/// `q_proj` / `g_proj` [H*D, hidden] BF16 tensors, then reclaim both sources.
fn interleave_q_gate(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    p: &str,
    nq: usize,
    hd: usize,
    h: usize,
) -> Result<DenseWeight> {
    let q_name = format!("{p}.q_proj.weight");
    let g_name = format!("{p}.g_proj.weight");
    let q = store.get(&q_name)?;
    let g = store.get(&g_name)?;
    ensure!(
        q.dtype == WeightDtype::BF16 && g.dtype == WeightDtype::BF16,
        "{p}: q_proj/g_proj must be BF16 (the NVFP4 ckpt keeps attention unquantized)"
    );
    let head_bytes = hd * h * 2;
    let out = gpu
        .alloc(2 * nq * head_bytes)
        .with_context(|| format!("allocate {p} q+gate interleave"))?;
    for head in 0..nq {
        let dst = out.offset(2 * head * head_bytes);
        gpu.copy_d2d(q.ptr.offset(head * head_bytes), dst, head_bytes)?;
        gpu.copy_d2d(g.ptr.offset(head * head_bytes), dst.offset(head_bytes), head_bytes)?;
    }
    store.reclaim(gpu, &q_name)?;
    store.reclaim(gpu, &g_name)?;
    Ok(DenseWeight { weight: out })
}

fn validate_matrix(store: &WeightStore, key: &str, rows: usize, cols: usize) -> Result<()> {
    let tensor = store.get(key)?;
    ensure!(
        tensor.shape == [rows, cols],
        "{key} shape {:?}, expected [{rows}, {cols}]",
        tensor.shape
    );
    Ok(())
}

fn load_scalar(store: &WeightStore, gpu: &dyn GpuBackend, key: &str) -> Result<f32> {
    let tensor = store.get(key)?;
    ensure!(
        tensor.shape.iter().product::<usize>() == 1,
        "{key} must be scalar"
    );
    match tensor.dtype {
        WeightDtype::BF16 => {
            let mut bytes = [0u8; 2];
            gpu.copy_d2h(tensor.ptr, &mut bytes)?;
            Ok(f32::from_bits((u16::from_le_bytes(bytes) as u32) << 16))
        }
        WeightDtype::FP32 => {
            let mut bytes = [0u8; 4];
            gpu.copy_d2h(tensor.ptr, &mut bytes)?;
            Ok(f32::from_le_bytes(bytes))
        }
        dtype => anyhow::bail!("{key} must be BF16 or F32, got {dtype:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{bf16_round, value_residual_blend};

    #[test]
    fn blend_is_bf16_sigmoid() {
        assert_eq!(value_residual_blend(0.0), 0.5);
        // bf16 has 8 mantissa bits: sigmoid(1)=0.7310586 -> 0.73046875.
        assert_eq!(value_residual_blend(1.0), 0.730_468_75);
        assert_eq!(bf16_round(1.0 + 1.0 / 512.0), 1.0); // tie -> even
    }
}
