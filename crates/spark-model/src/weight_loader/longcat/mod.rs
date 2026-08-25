// SPDX-License-Identifier: AGPL-3.0-only

//! LongCat-Flash(-Lite) weight loader — the backbone behind the n-gram
//! embeddings (`longcat_flash_ngram`).
//!
//! Architecture (HF `modeling_longcat_flash.py`), and how it maps onto Atlas:
//!
//! - Each CHECKPOINT layer is a dual-sublayer "shortcut" block: two MLA
//!   attentions, two dense SwiGLU MLPs, and ONE shortcut MoE whose output is
//!   computed on sublayer 1's post-attention normed input but added at the END
//!   of sublayer 2. Atlas serves each SUBLAYER as one `Qwen3AttentionLayer`
//!   (`num_hidden_layers` is already 2x at parse), with the shortcut carried
//!   between the pair via `set_shortcut_moe` / `set_shortcut_carry_in`.
//! - MLA is the DeepSeek-lineage q-LoRA form Atlas already serves; the two
//!   LongCat deltas (interleaved rope, sqrt LoRA scaling) fold into the
//!   WEIGHTS at load (see `prep`), so the runtime is unchanged.
//! - The MoE router is softmax + `e_score_correction_bias` over
//!   `n_routed + zero_expert_num` logits, with the zero (identity) experts
//!   folded inside the router kernel (see `moe_topk_softmax_bias.cu`).
//!
//! Tensor names are HF-standard under `model.layers.{L}.`:
//!   `self_attn.{0,1}.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,
//!                     kv_a_layernorm,kv_b_proj,o_proj}`
//!   `mlps.{0,1}.{gate,up,down}_proj`, `input_layernorm.{0,1}`,
//!   `post_attention_layernorm.{0,1}`,
//!   `mlp.router.{classifier.weight,e_score_correction_bias}`,
//!   `mlp.experts.{e}.{gate,up,down}_proj`.

mod prep;

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::qwen3_attention::MlaWeights;
use crate::layers::vision_encoder::VisionEncoder;
use crate::layers::{FfnComponent, MoeLayer, Qwen3AttentionLayer};
use crate::mistral_loader::loader_impl::{ctx as mctx, phase_block_diag, phase_per_head, phase_qk_absorbed};
use crate::weight_loader::ModelWeightLoader;
use crate::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, MtpWeights, QuantizedWeight, dense,
    dense_f32_as_bf16, quantize_to_nvfp4, quantized_any,
};

pub struct LongcatWeightLoader;

/// Tokens the shortcut-MoE carry buffer must hold: the largest prefill chunk
/// a sublayer can be handed. Sized from `max_prefill_tokens`' ceiling; the
/// producer/consumer both `ensure!` against it rather than overrunning.
const CARRY_TOKENS: usize = 8192;

impl ModelWeightLoader for LongcatWeightLoader {
    fn supports_tp(&self) -> bool {
        // MLA TP would need the same wq_b/wkv_b column sharding Mistral does,
        // plus a per-rank shortcut carry. Not validated — refuse rather than
        // serve a silently wrong shard split.
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        anyhow::ensure!(
            config.num_hidden_layers.is_multiple_of(2),
            "longcat: num_hidden_layers ({}) must be even — each checkpoint \
             layer is TWO engine sublayers",
            config.num_hidden_layers
        );
        let ckpt_layers = config.num_hidden_layers / 2;
        let h = config.hidden_size;
        let nope = config.qk_nope_head_dim;
        let rope = config.qk_rope_head_dim;
        let q_lora = config.q_lora_rank;
        let kv_lora = config.kv_lora_rank;
        let n_heads = config.num_attention_heads;
        // The reference's mla_scale_{q,kv}_lora flags (both true on Lite).
        let scale_q = (h as f32 / q_lora as f32).sqrt();
        let scale_kv = (h as f32 / kv_lora as f32).sqrt();

        tracing::info!(
            "LongCat: {ckpt_layers} checkpoint layers → {} engine sublayers \
             (MLA q_lora={q_lora} kv_lora={kv_lora} nope={nope} rope={rope}; \
             rope de-interleave + q/kv LoRA scale folded at load: \
             q×{scale_q:.4}, kv-norm×{scale_kv:.4}); {} routed + {} zero experts",
            config.num_hidden_layers,
            config.num_experts,
            config.zero_expert_num,
        );

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let mut yarn_shared = DevicePtr::NULL;
        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);

        for l in 0..ckpt_layers {
            let lp = format!("model.layers.{l}");
            // One carry buffer per checkpoint layer (producer sublayer 0 →
            // consumer sublayer 1). Allocated per block so two blocks in
            // flight (chunked prefill) cannot alias.
            let carry = gpu.alloc(CARRY_TOKENS * h * 2)?;

            for s in 0..2usize {
                let ap = format!("{lp}.self_attn.{s}");
                let global_idx = l * 2 + s;

                // ── MLA: name-bound loads + the two LongCat folds ──
                let wq_a = dense(store, &format!("{ap}.q_a_proj.weight"))?;
                let wq_b = prep::prep_q_b(
                    store,
                    &format!("{ap}.q_b_proj.weight"),
                    n_heads,
                    nope,
                    rope,
                    q_lora,
                    scale_q,
                    gpu,
                )?;
                let q_a_norm = dense(store, &format!("{ap}.q_a_layernorm.weight"))?;
                let wkv_a = prep::prep_kv_a(
                    store,
                    &format!("{ap}.kv_a_proj_with_mqa.weight"),
                    kv_lora,
                    rope,
                    h,
                    gpu,
                )?;
                let kv_a_norm = prep::prep_kv_a_norm(
                    store,
                    &format!("{ap}.kv_a_layernorm.weight"),
                    kv_lora,
                    scale_kv,
                    gpu,
                )?;
                let wkv_b = dense(store, &format!("{ap}.kv_b_proj.weight"))?;
                let wo = dense(store, &format!("{ap}.o_proj.weight"))?;

                // ── shared MLA precompute (per-head transpose → absorbed QK
                //    → block-diagonals), reusing the Mistral phases ──
                let mut c = mctx::MistralLayerCtx::new(
                    store, config, gpu, absmax_k, quantize_k, stream, global_idx,
                );
                c.wq_a_dense = Some(wq_a);
                c.wq_b = Some(wq_b);
                c.q_a_norm = Some(q_a_norm);
                c.wkv_a_dense = Some(wkv_a);
                c.wkv_a_rope_dense = Some(DenseWeight {
                    weight: wkv_a.weight.offset(kv_lora * h * 2),
                });
                c.wkv_b = Some(wkv_b);
                c.kv_a_norm = Some(kv_a_norm);
                c.wq_a_nvfp4 = Some(quantize_to_nvfp4(
                    &wq_a, q_lora, h, gpu, absmax_k, quantize_k, stream,
                )?);
                c.wq_b_nvfp4 = Some(quantize_to_nvfp4(
                    &wq_b,
                    n_heads * (nope + rope),
                    q_lora,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?);
                c.wkv_a_nvfp4 = Some(quantize_to_nvfp4(
                    &wkv_a,
                    kv_lora + rope,
                    h,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?);
                phase_per_head::build_per_head_views(&mut c)?;
                phase_qk_absorbed::build_w_qk_absorbed(&mut c)?;
                phase_block_diag::build_block_diagonals(&mut c)?;
                let o_nvfp4 = quantize_to_nvfp4(
                    &wo,
                    h,
                    n_heads * config.v_head_dim,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?;
                let yarn = mctx::ensure_yarn_inv_freq(&mut yarn_shared, config, rope, gpu)?;

                let null = DenseWeight {
                    weight: DevicePtr::NULL,
                };
                let mla = MlaWeights {
                    wq_a,
                    wq_a_fp8: None,
                    wq_a_nvfp4: c.wq_a_nvfp4,
                    wq_b,
                    wq_b_fp8: None,
                    wq_b_nvfp4: c.wq_b_nvfp4,
                    q_a_norm,
                    wkv_a,
                    wkv_a_nvfp4: c.wkv_a_nvfp4,
                    wkv_b,
                    kv_a_norm,
                    wkv_a_rope: c.wkv_a_rope_dense.expect("set above"),
                    wkv_a_merged: DenseWeight {
                        weight: wkv_a.weight,
                    },
                    wo,
                    wo_nvfp4: Some(o_nvfp4),
                    wo_a: null,
                    wo_a_nvfp4: None,
                    wo_b: null,
                    wo_b_nvfp4: None,
                    wo_b_fp8: None,
                    wo_a_fp8: None,
                    wkv_a_fp8: None,
                    wq_b_rope: c.wq_b_rope.context("longcat: wq_b_rope")?,
                    w_uk_t: c.w_uk_t.context("longcat: w_uk_t")?,
                    w_uv: c.w_uv.context("longcat: w_uv")?,
                    w_qk_absorbed: c.w_qk_absorbed.context("longcat: w_qk_absorbed")?,
                    w_uk_block_diag: c.w_uk_block_diag.context("longcat: w_uk_bd")?,
                    w_uv_block_diag: c.w_uv_block_diag.context("longcat: w_uv_bd")?,
                    yarn_inv_freq: yarn,
                    main_inv_freq: yarn,
                    q_lora_rank: q_lora,
                    kv_lora_rank: kv_lora,
                    o_lora_rank: 0,
                    nope,
                    rope,
                    v_dim: config.v_head_dim,
                    compressor: None,
                    attn_sink: DevicePtr::NULL,
                };

                // Dummy attention weights (never read on the MLA path).
                let o_dummy = QuantizedWeight {
                    weight: DevicePtr::NULL,
                    weight_scale: DevicePtr::NULL,
                    weight_scale_2: 0.0,
                    input_scale: DevicePtr::NULL,
                    weight_scale_2_vec: DevicePtr::NULL,
                };
                let attn = AttentionWeights {
                    q_proj: null,
                    k_proj: null,
                    v_proj: null,
                    o_proj: o_dummy,
                    q_norm: null,
                    k_norm: null,
                    q_norm_full: None,
                    k_norm_full: None,
                    k_scale: 1.0,
                    v_scale: 1.0,
                };

                // ── dense SwiGLU FFN for this sublayer ──
                let ffn = build_dense_ffn(store, &format!("{lp}.mlps.{s}"), config, gpu)?;
                let input_norm = dense(store, &format!("{lp}.input_layernorm.{s}.weight"))?;
                let post_norm =
                    dense(store, &format!("{lp}.post_attention_layernorm.{s}.weight"))?;
                let kv_dtype = layer_kv_dtypes
                    .get(global_idx)
                    .copied()
                    .unwrap_or(KvCacheDtype::Bf16);

                let mut layer = Qwen3AttentionLayer::new_ungated(
                    input_norm, attn, post_norm, ffn, global_idx, None, None, None, gpu,
                    kv_dtype, 0, config,
                )?;
                layer.set_mla_weights(mla);

                if s == 0 {
                    // Sublayer 0 owns the block's shortcut MoE; its output is
                    // stashed and added at the end of sublayer 1.
                    let moe = build_shortcut_moe(store, &lp, config, gpu)?;
                    layer.set_shortcut_moe(moe, carry, CARRY_TOKENS);
                } else {
                    layer.set_shortcut_carry_in(carry, CARRY_TOKENS);
                }
                layers.push(Box::new(layer));
            }

            if (l + 1) % 4 == 0 || l == ckpt_layers - 1 {
                let free = gpu.free_memory().unwrap_or(0);
                tracing::info!(
                    "LongCat L{}/{ckpt_layers} — {:.1} GB free",
                    l + 1,
                    free as f64 / 1e9
                );
            }
        }
        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.embed_tokens.weight").context("longcat: embedding")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.norm.weight").context("longcat: final norm")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")
        } else if config.tie_word_embeddings {
            dense(store, "model.embed_tokens.weight")
        } else {
            anyhow::bail!("longcat: lm_head.weight not found")
        }
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // The checkpoint ships `model.mtp.*`, but the MTP head shape is not
        // the Qwen-style one Atlas builds. Ignored (matches HF's own
        // `_keys_to_ignore_on_load_unexpected = [r"model\\.mtp.*"]`).
        Ok(None)
    }

    fn load_vision_encoder(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<VisionEncoder>> {
        Ok(None)
    }
}

/// One sublayer's dense SwiGLU FFN (`mlps.{s}`), NVFP4-quantized at load.
fn build_dense_ffn(
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
fn build_shortcut_moe(
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
                store, &format!("{ep}.gate_proj"), inter, h, gpu, variant, qctx,
            )?,
            up_proj: quantized_any(
                store, &format!("{ep}.up_proj"), inter, h, gpu, variant, qctx,
            )?,
            down_proj: quantized_any(
                store, &format!("{ep}.down_proj"), h, inter, gpu, variant, qctx,
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
