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
//! Vision and audio tensors are skipped. `model.mtp.*` stays loadable; the
//! server-side `skip_mtp` policy drops it unless `--speculative` runs on rank 0.

use anyhow::{Context, Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::qwen3_attention::mimo::{
    MIMO_QK_DIM, MIMO_V_DIM, MimoAttn, MimoKernels, MimoQkv, mimo_workspace_bytes,
};
use crate::layers::{DenseFfnLayer, FfnComponent, MoeLayer, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, MtpWeights, QuantizedWeight,
    WeightQuantFormat, dense, dense_auto, free_loader_source, load_kv_scales, quantize_to_nvfp4,
    quantized_mxfp4_e8m0,
};

/// Upper bound on concurrently decoded sequences the split-K scratch covers;
/// beyond it decode falls back to fewer splits.
pub(super) const MAX_DECODE_SEQS: usize = 64;

pub(crate) mod fp8b;
pub(crate) mod mtp;
pub(crate) mod qkv;
pub(crate) use mtp::load_mimo_mtp_modules;
pub(super) use qkv::{QKV_TP_CHUNKS, load_qkv_chunked, qkv_chunk_rows};

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

/// Opt-in BF16 arm for the layer-0 dense FFN (`ATLAS_MIMO_DENSE_BF16=1`):
/// skips the NVFP4 requant so hardware can A/B NVFP4's quality/perf on the
/// one dense layer. Unset/0 = NVFP4 (the validated default).
fn dense_bf16_enabled() -> Result<bool> {
    static V: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    V.get_or_init(
        || match std::env::var("ATLAS_MIMO_DENSE_BF16").ok().as_deref() {
            None | Some("") | Some("0") | Some("false") | Some("off") => Ok(false),
            Some("1") | Some("true") | Some("on") => Ok(true),
            Some(other) => Err(format!(
                "ATLAS_MIMO_DENSE_BF16 must be 1 or 0, got {other:?}"
            )),
        },
    )
    .clone()
    .map_err(anyhow::Error::msg)
}

/// Reclaim a dense-FFN projection's on-disk source and scale tensors once
/// its weights have been transformed (NVFP4 requant or BF16 keep).
fn reclaim_ffn_source(store: &WeightStore, gpu: &dyn GpuBackend, wkey: &str) -> Result<()> {
    store.reclaim(gpu, wkey)?;
    let prefix = wkey.strip_suffix(".weight").expect("weight suffix");
    for scale_key in [
        format!("{prefix}.weight_scale_inv"),
        format!("{prefix}.weight_scale"),
        format!("{prefix}.scale"),
    ] {
        if store.contains(&scale_key) {
            store.reclaim(gpu, &scale_key)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_dense_ffn(
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
    if dense_bf16_enabled()? {
        // BF16 arm: keep the checkpoint's FP8->BF16 dequant (E4M3 values are
        // exact in BF16) instead of the NVFP4 requant. The FP8 source and
        // scale tensors are unused on this arm.
        let bf16 = |name: &str| -> Result<DenseWeight> {
            let wkey = format!("{lp}.mlp.{name}.weight");
            let w = dense_auto(store, &wkey, gpu)?;
            reclaim_ffn_source(store, gpu, &wkey)?;
            Ok(w)
        };
        let mut dffn = DenseFfnLayer::new(
            DenseFfnWeights {
                gate_proj: QuantizedWeight::null(),
                up_proj: QuantizedWeight::null(),
                down_proj: QuantizedWeight::null(),
                gate_proj_t: None,
                up_proj_t: None,
                down_proj_t: None,
            },
            gpu,
            &config.model_type,
        )?;
        dffn.set_bf16_weights(bf16("gate_proj")?, bf16("up_proj")?, bf16("down_proj")?);
        tracing::info!("mimo_v2: layer-0 dense FFN BF16 arm ON (ATLAS_MIMO_DENSE_BF16=1)");
        return Ok(FfnComponent::Dense(dffn));
    }
    let quant = |name: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
        let wkey = format!("{lp}.mlp.{name}.weight");
        let w = dense_auto(store, &wkey, gpu)?;
        let q = quantize_to_nvfp4(&w, n, k, gpu, absmax_k, quantize_k, stream)?;
        // Free the BF16 input (alias or dequant buffer) and the on-disk source
        // tensor; only the NVFP4 result stays resident.
        free_loader_source(store, gpu, &wkey, w)?;
        reclaim_ffn_source(store, gpu, &wkey)?;
        Ok(q)
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
    layer: usize,
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
    moe.is_dflash_capture_layer = config.dflash_capture_layers.contains(&layer);
    Ok(FfnComponent::Moe(moe))
}

impl ModelWeightLoader for MimoV2WeightLoader {
    fn supports_tp(&self) -> bool {
        // Attention shards by checkpoint chunk pairs (see `load_qkv_chunked`);
        // each rank stores only its own KV heads. The dense layer-0 FFN, the
        // MoE router and the LM head stay replicated; experts stay EP-sharded.
        // Decode/prefill/verify TP all-reduces are the generic paths in
        // `qwen3_attention` (`tp_world_size > 1` branches).
        true
    }

    fn dead_namespaces(&self, _config: &ModelConfig) -> Vec<String> {
        ["visual.", "audio_encoder.", "speech_embeddings."]
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
        let fp8_qkv = fp8b::fp8_qkv_enabled()?;
        // Attention TP: `config` head counts are already TP-local (topology
        // divides them); the checkpoint's QKV chunk layout and the o_proj /
        // sink tensors are indexed by the PRE-TP counts, recovered as
        // local × tp_size.
        let tp_rank = config.tp_rank;
        let tp_size = config.tp_world_size.max(1);
        let qkv_full = |nkv_local: usize| (nq * tp_size, nkv_local * tp_size);
        if fp8_qkv {
            let nk = crate::layers::ops::native_kernels(gpu);
            ensure!(
                nk.fp8b.complete(false) && nk.w8a16_gemm.0 != 0,
                "mimo_v2: {}=1 but the kernel target lacks the fp8b GEMV tiers or \
                 w8a16_gemm_pipelined (rebuild the kernels)",
                fp8b::FP8_QKV_ENV
            );
            tracing::info!(
                "mimo_v2: FP8 block QKV projections ON ({})",
                fp8b::FP8_QKV_ENV
            );
        }
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
                matches!(kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Fp8),
                "mimo_v2: layer {i} KV dtype {kv_dtype:?} unsupported; the MiMo attention \
                 kernels support a BF16 or FP8 paged cache"
            );
            let lp = format!("model.layers.{i}");
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
            let ffn = if config.mlp_only_layers.contains(&i) {
                load_dense_ffn(store, config, gpu, &lp, absmax_k, quantize_k, stream)?
            } else {
                load_moe(store, config, gpu, &lp, i)?
            };

            let p = format!("{lp}.self_attn");
            let swa = is_swa(config, i);
            let nkv = layer_kv_heads(config, i);
            let (nq_full, nkv_full) = qkv_full(nkv);
            let qkv_prefix = format!("{p}.qkv_proj");
            let (qkv, chunk_rows, local_chunks) = if fp8_qkv {
                let (w, rows, chunks) = fp8b::load_qkv_fp8b(
                    store,
                    &qkv_prefix,
                    nq_full,
                    nkv_full,
                    h,
                    gpu,
                    tp_rank,
                    tp_size,
                )?;
                (MimoQkv::Fp8b(w), rows, chunks)
            } else {
                let (w, rows, chunks) = load_qkv_chunked(
                    store,
                    &qkv_prefix,
                    nq_full,
                    nkv_full,
                    h,
                    gpu,
                    tp_rank,
                    tp_size,
                )?;
                (MimoQkv::Bf16(w), rows, chunks)
            };
            let o_key = format!("{p}.o_proj.weight");
            {
                let o = store.get(&o_key)?;
                ensure!(
                    o.shape == [h, nq_full * MIMO_V_DIM as usize],
                    "{o_key}: shape {:?}, expected [{h}, {}]",
                    o.shape,
                    nq_full * MIMO_V_DIM as usize
                );
            }
            // Row-parallel shard: each rank keeps the input columns of its own
            // query heads and produces a partial [n, hidden] that the generic
            // TP all-reduce (decode/prefill/multi-seq) sums after the GEMM.
            let o_full = dense_auto(store, &o_key, gpu)?;
            let o_proj = if tp_size > 1 {
                use crate::tp_shard::{TpShardKind, shard_dense_bf16};
                let (ptr, _, local_in) = shard_dense_bf16(
                    o_full.weight,
                    h,
                    nq_full * MIMO_V_DIM as usize,
                    TpShardKind::RowParallel,
                    tp_rank,
                    tp_size,
                    gpu,
                )?;
                ensure!(
                    local_in == nq * MIMO_V_DIM as usize,
                    "{o_key}: TP shard left {local_in} input columns, expected {}",
                    nq * MIMO_V_DIM as usize
                );
                free_loader_source(store, gpu, &o_key, o_full)?;
                DenseWeight { weight: ptr }
            } else {
                o_full
            };
            let sink_key = format!("{p}.attention_sink_bias");
            let sinks = if store.contains(&sink_key) {
                let s_full = dense_auto(store, &sink_key, gpu)?;
                if tp_size > 1 {
                    use crate::tp_shard::shard_dense_1d_bf16;
                    let (sp, dim) =
                        shard_dense_1d_bf16(s_full.weight, nq_full, tp_rank, tp_size, gpu)?;
                    ensure!(
                        dim == nq,
                        "{sink_key}: TP shard left {dim} sinks, expected {nq}"
                    );
                    free_loader_source(store, gpu, &sink_key, s_full)?;
                    sp
                } else {
                    s_full.weight
                }
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
                tp_chunks: local_chunks as u32,
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
        // MiMo MTP bodies are loaded by `load_mimo_mtp_modules`, not the
        // Qwen-shaped `MtpWeights` path.
        Ok(None)
    }
}
