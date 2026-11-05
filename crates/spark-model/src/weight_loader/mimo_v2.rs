// SPDX-License-Identifier: AGPL-3.0-only

//! MiMo-V2 (XiaomiMiMo/MiMo-V2.6-Flash) weight loader — text backbone.
//!
//! Checkpoint layout (`model.layers.N.*`):
//!   * `self_attn.qkv_proj` FP8 E4M3 + `weight_scale_inv` (128×128 blocks).
//!     Rows are pre-sharded for TP=4: `[Q_c | K_c | V_c] × 4`, and each chunk
//!     has its own scale-block grid (full-attention chunks are 3392 rows, not
//!     a multiple of 128). Dequantized chunk by chunk to BF16, keeping the
//!     chunk row order that `mimo_qkv_rope_cache` splits.
//!   * `self_attn.o_proj` BF16 `[hidden, nq*128]`; `attention_value_scale` is
//!     applied by the attention kernels.
//!   * `self_attn.attention_sink_bias` `[nq]` on sliding-window layers.
//!   * Layer 0: dense FFN, FP8 block-scaled → NVFP4 at load.
//!   * Layers 1-47: `mlp.gate` (BF16 router), `mlp.gate.e_score_correction_bias`
//!     (F32), `mlp.experts.E.{gate,up,down}_proj` native MXFP4 (U8 nibbles +
//!     E8M0 `weight_scale`, group 32) landed transcode-free.
//!
//! Vision, audio and MTP tensors are skipped (phase 1: text, no speculation).

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::qwen3_attention::mimo::{
    MIMO_QK_DIM, MIMO_V_DIM, MimoAttn, MimoKernels, mimo_workspace_bytes,
};
use crate::layers::{DenseFfnLayer, FfnComponent, MoeLayer, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, MtpWeights, QuantizedWeight,
    WeightQuantFormat, dense, dense_auto, load_kv_scales, quantize_to_nvfp4, quantized_mxfp4_e8m0,
};

/// The published checkpoint stores fused QKV pre-sharded for TP=4.
const QKV_TP_CHUNKS: usize = 4;
/// Upper bound on concurrently decoded sequences the split-K scratch covers;
/// beyond it decode falls back to fewer splits.
const MAX_DECODE_SEQS: usize = 64;

pub struct MimoV2WeightLoader;

fn is_swa(config: &ModelConfig, layer: usize) -> bool {
    config.layer_types.get(layer) == Some(&LayerType::SlidingAttention)
}

fn layer_kv_heads(config: &ModelConfig, layer: usize) -> usize {
    if is_swa(config, layer) || config.full_num_key_value_heads == 0 {
        config.num_key_value_heads
    } else {
        config.full_num_key_value_heads
    }
}

/// Load the TP-chunked fused QKV as BF16 `[4 * chunk_rows, hidden]`.
fn load_qkv_chunked(
    store: &WeightStore,
    prefix: &str,
    nq: usize,
    nkv: usize,
    h: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DenseWeight, usize)> {
    ensure!(
        nq.is_multiple_of(QKV_TP_CHUNKS) && nkv.is_multiple_of(QKV_TP_CHUNKS),
        "{prefix}: heads ({nq} q / {nkv} kv) not divisible by the {QKV_TP_CHUNKS} checkpoint chunks"
    );
    let qk = MIMO_QK_DIM as usize;
    let vd = MIMO_V_DIM as usize;
    let chunk_rows = (nq / QKV_TP_CHUNKS) * qk + (nkv / QKV_TP_CHUNKS) * (qk + vd);
    let rows = QKV_TP_CHUNKS * chunk_rows;
    let (w_ptr, w_dtype, w_shape) = {
        let w = store.get(&format!("{prefix}.weight"))?;
        (w.ptr, w.dtype, w.shape.clone())
    };
    ensure!(
        w_shape == [rows, h],
        "{prefix}.weight: shape {w_shape:?}, expected [{rows}, {h}]"
    );
    match w_dtype {
        WeightDtype::BF16 => return Ok((DenseWeight { weight: w_ptr }, chunk_rows)),
        WeightDtype::FP8E4M3 => {}
        other => bail!("{prefix}.weight: unsupported dtype {other:?}"),
    }
    let (s_ptr, s_dtype, s_shape) = {
        let s = store.get(&format!("{prefix}.weight_scale_inv"))?;
        (s.ptr, s.dtype, s.shape.clone())
    };
    let blocks_per_chunk = chunk_rows.div_ceil(128);
    let sk = h.div_ceil(128);
    ensure!(
        s_shape == [QKV_TP_CHUNKS * blocks_per_chunk, sk],
        "{prefix}.weight_scale_inv: shape {s_shape:?}, expected [{}, {sk}] \
         ({QKV_TP_CHUNKS} chunks × {blocks_per_chunk} row blocks)",
        QKV_TP_CHUNKS * blocks_per_chunk
    );
    let scale_bytes = match s_dtype {
        WeightDtype::FP32 => 4,
        WeightDtype::BF16 => 2,
        other => bail!("{prefix}.weight_scale_inv: unsupported dtype {other:?}"),
    };
    let out = gpu.alloc(rows * h * 2)?;
    let kernel: KernelHandle = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    let stream = gpu.default_stream();
    for c in 0..QKV_TP_CHUNKS {
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(h as u32, 64), div_ceil(chunk_rows as u32, 4), 1])
            .block([64, 4, 1])
            .arg_ptr(w_ptr.offset(c * chunk_rows * h))
            .arg_ptr(s_ptr.offset(c * blocks_per_chunk * sk * scale_bytes))
            .arg_ptr(out.offset(c * chunk_rows * h * 2))
            .arg_u32(chunk_rows as u32)
            .arg_u32(h as u32)
            .arg_u32(128)
            .arg_u32(128)
            .arg_u32(sk as u32)
            .arg_u32((scale_bytes == 4) as u32)
            .launch(stream)?;
    }
    gpu.synchronize(stream)?;
    gpu.free(w_ptr)?;
    gpu.free(s_ptr)?;
    Ok((DenseWeight { weight: out }, chunk_rows))
}

#[allow(clippy::too_many_arguments)]
fn load_dense_ffn(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
) -> Result<FfnComponent> {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let quant = |name: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
        let w = dense_auto(store, &format!("{lp}.mlp.{name}.weight"), gpu)?;
        quantize_to_nvfp4(&w, n, k, gpu, absmax_k, quantize_k, stream)
    };
    let gate_q = quant("gate_proj", inter, h)?;
    let up_q = quant("up_proj", inter, h)?;
    let down_q = quant("down_proj", h, inter)?;
    let weights = DenseFfnWeights {
        gate_proj: gate_q,
        up_proj: up_q,
        down_proj: down_q,
        gate_proj_t: Some(gate_q.transpose_for_gemm(gpu, inter, h)?),
        up_proj_t: Some(up_q.transpose_for_gemm(gpu, inter, h)?),
        down_proj_t: Some(down_q.transpose_for_gemm(gpu, h, inter)?),
    };
    let mut dffn = DenseFfnLayer::new(weights, gpu, &config.model_type)?;
    dffn.finalize_q4k_load(gpu, h as u32, inter as u32, stream)?;
    Ok(FfnComponent::Dense(dffn))
}

fn load_moe(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
) -> Result<FfnComponent> {
    let p = format!("{lp}.mlp");
    let n = config.num_experts;
    let gate = dense(store, &format!("{p}.gate.weight"))?;
    let mut experts = Vec::with_capacity(n);
    for e in 0..n {
        if !config.is_local_expert(e) {
            experts.push(ExpertWeight::null());
            continue;
        }
        let ep = format!("{p}.experts.{e}");
        let proj = |name: &str| {
            quantized_mxfp4_e8m0(store, &format!("{ep}.{name}"))
                .with_context(|| format!("MiMo-V2 {ep}.{name}"))
        };
        experts.push(ExpertWeight {
            gate_proj: proj("gate_proj")?,
            up_proj: proj("up_proj")?,
            down_proj: proj("down_proj")?,
        });
    }
    let bias_key = format!("{p}.gate.e_score_correction_bias");
    let correction_bias = {
        let t = store.get(&bias_key)?;
        ensure!(
            t.dtype == WeightDtype::FP32 && t.num_elements() == n,
            "{bias_key}: expected F32 [{n}], got {:?} with {} elements",
            t.dtype,
            t.num_elements()
        );
        Some(DenseWeight { weight: t.ptr })
    };
    let weights = MoeWeights {
        gate,
        shared_expert: ExpertWeight::null(),
        shared_expert_gate: DenseWeight {
            weight: DevicePtr::NULL,
        },
        experts,
        router_pre_norm: None,
        correction_bias,
    };
    // Router stays BF16 (gate_nvfp4 = None): a 4-bit router flips low-margin
    // top-8 picks.
    let mut moe = MoeLayer::new(weights, n, None, gpu, config)?;
    moe.experts_scale_kind = WeightQuantFormat::Mxfp4E8m0;
    moe.shared_experts_scale_kind = WeightQuantFormat::Nvfp4;
    Ok(FfnComponent::Moe(moe))
}

impl ModelWeightLoader for MimoV2WeightLoader {
    fn supports_tp(&self) -> bool {
        // Expert parallelism only; attention is replicated per rank.
        false
    }

    fn dead_namespaces(&self, _config: &ModelConfig) -> Vec<String> {
        [
            "visual.",
            "audio_encoder.",
            "speech_embeddings.",
            "model.mtp.",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn kv_layer_dims(&self, config: &ModelConfig) -> Vec<(usize, usize)> {
        (0..config.num_hidden_layers)
            .map(|i| (layer_kv_heads(config, i), MIMO_QK_DIM as usize))
            .collect()
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let h = config.hidden_size;
        let nq = config.num_attention_heads;
        ensure!(
            config.head_dim == MIMO_QK_DIM as usize && config.v_head_dim == MIMO_V_DIM as usize,
            "mimo_v2: head dims {}/{} do not match the 192/128 kernels",
            config.head_dim,
            config.v_head_dim
        );
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let kernels = MimoKernels::resolve(gpu)?;
        let workspace_bytes = mimo_workspace_bytes(MAX_DECODE_SEQS, nq);
        let workspace = gpu.alloc(workspace_bytes)?;
        let (ep_lo, ep_hi) = config.local_expert_range();
        tracing::info!(
            "mimo_v2: loading {} layers, local experts [{ep_lo}, {ep_hi})",
            config.num_hidden_layers
        );

        let null = DenseWeight {
            weight: DevicePtr::NULL,
        };
        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let kv_dtype = layer_kv_dtypes[i];
            ensure!(
                matches!(kv_dtype, KvCacheDtype::Bf16),
                "mimo_v2: layer {i} KV dtype {kv_dtype:?} unsupported; the MiMo attention \
                 kernels read a BF16 paged cache (use --kv-cache-dtype bf16)"
            );
            let lp = format!("model.layers.{i}");
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
            let ffn = if config.mlp_only_layers.contains(&i) {
                load_dense_ffn(store, config, gpu, &lp, absmax_k, quantize_k, stream)?
            } else {
                load_moe(store, config, gpu, &lp)?
            };

            let p = format!("{lp}.self_attn");
            let swa = is_swa(config, i);
            let nkv = layer_kv_heads(config, i);
            let (qkv, chunk_rows) =
                load_qkv_chunked(store, &format!("{p}.qkv_proj"), nq, nkv, h, gpu)?;
            let o_key = format!("{p}.o_proj.weight");
            {
                let o = store.get(&o_key)?;
                ensure!(
                    o.shape == [h, nq * MIMO_V_DIM as usize],
                    "{o_key}: shape {:?}, expected [{h}, {}]",
                    o.shape,
                    nq * MIMO_V_DIM as usize
                );
            }
            let o_proj = dense_auto(store, &o_key, gpu)?;
            let sink_key = format!("{p}.attention_sink_bias");
            let sinks = if store.contains(&sink_key) {
                dense_auto(store, &sink_key, gpu)?.weight
            } else {
                DevicePtr::NULL
            };
            ensure!(
                !swa || !sinks.is_null(),
                "mimo_v2: sliding layer {i} has no {sink_key}"
            );
            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
            let attn = AttentionWeights {
                q_proj: null,
                k_proj: null,
                v_proj: null,
                o_proj: QuantizedWeight::null(),
                q_norm: null,
                k_norm: null,
                q_norm_full: None,
                k_norm_full: None,
                k_scale,
                v_scale,
            };
            let mut layer = Qwen3AttentionLayer::new_ungated(
                input_norm,
                attn,
                post_attn_norm,
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
            let rope_theta = if swa {
                config.swa_rope_theta
            } else {
                config.rope_theta
            };
            layer.set_mimo(MimoAttn {
                qkv,
                o_proj,
                sinks,
                nq: nq as u32,
                nkv: nkv as u32,
                tp_chunks: QKV_TP_CHUNKS as u32,
                chunk_rows: chunk_rows as u32,
                window: if swa { config.sliding_window } else { 0 },
                rope_theta: rope_theta as f32,
                rotary_dim: config.rotary_dim as u32,
                scale: 1.0 / (MIMO_QK_DIM as f32).sqrt(),
                out_scale: config.attention_value_scale as f32,
                workspace,
                workspace_bytes,
                kernels,
            });
            layers.push(Box::new(layer));
        }
        tracing::info!("mimo_v2: built {} layers", layers.len());
        Ok(layers)
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
        // model_mtp.safetensors ships 3 MTP layers; wiring them is phase 3.
        Ok(None)
    }
}
