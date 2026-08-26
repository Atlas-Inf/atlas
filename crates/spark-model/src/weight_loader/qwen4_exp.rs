// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen3.8-Flash-Next` (`qwen4_exp`) weight loader. Port tracked in Avarok
//! #753.
//!
//! **This loader LOADS; it does not yet let the model SERVE.** Three pieces of
//! the forward path are unimplemented — the multi-hyperconnection residual,
//! the QSA indexer, and the PLE n-gram injection — and each refuses BY NAME
//! at the forward boundary rather than at load. The split is deliberate:
//! `hc_mult` streams touch every layer, so refusing at load would mean
//! nothing loads and we learn nothing about the real footprint; refusing at
//! first inference lets `spark serve` complete and print an honest alloc
//! ledger while making it impossible to emit fluent-but-wrong text.
//!
//! WHY THIS IS MOSTLY qwen35's LOADER. Qwen3.8-Flash-Next and Qwen3.6-35B-A3B
//! share far more than the version numbers suggest: 3:1 GDN/full-attention
//! interleave, MoE with a shared expert, gated attention, mRoPE, a ViT tower,
//! vocab 248320, rope_theta 1e7, head_dim 256, partial rotary 0.25, and the
//! same GDN key geometry. Critically, `load_ssm_qwen35` already reads
//! `in_proj_qkv` and `in_proj_z` as SEPARATE tensors and concatenates them —
//! which is exactly this model's layout, not a coincidence to be re-derived.
//! So the GDN and full-attention arms are called directly, with
//! `config.weight_prefix = "model.language_model"` making
//! `config.layer_prefix(i)` yield the real keys.
//!
//! WHAT IS GENUINELY DIFFERENT, and why each needs care:
//!
//! 1. **There are no per-layer norms.** No `input_layernorm`, no
//!    `post_attention_layernorm`, no final `model.norm`. Normalization lives
//!    inside the hyper-connection blocks as `hc_norm [hc_mult*hidden]`, and
//!    the model-level `hyper_connection_mixer` — which collapses the streams
//!    back to one before `lm_head` — carries the final norm. A loader that
//!    "helpfully" defaults these would be inventing weights.
//! 2. **mHC is 4 residual streams**, mixed low-rank (rank 320). Atlas's mHC
//!    plumbing is DeepSeek-V4's, whose mixer is Sinkhorn-normalized — same
//!    stream layout, different math.
//! 3. **A QSA indexer** on the 12 full-attention layers.
//! 4. **PLE n-gram injection** at one layer, off a ~320M-row table served
//!    from NVMe rather than resident.

use anyhow::{Context, Result};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::weight_loader::ModelWeightLoader;
use crate::weight_map::{DenseWeight, MtpWeights, dense};

mod ffn;
mod hc;
mod probe;

pub use probe::audit_namespace;

pub struct Qwen4ExpWeightLoader;

impl ModelWeightLoader for Qwen4ExpWeightLoader {
    fn supports_tp(&self) -> bool {
        // Not attempted. mHC would need the stream buffer sharded alongside
        // every projection, and the PLE row cache is a single-device arena.
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let report = audit_namespace(store, config);
        report.log();
        report.ensure_loadable()?;

        let h = config.hidden_size;
        let variant = crate::weight_map::detect_nvfp4_variant(store, config);
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();

        tracing::info!(
            "Qwen3.8-Flash-Next: {} layers ({} GDN + {} full attention), \
             {} experts top-{}, hc {} streams x rank {}, indexer budget {}, \
             PLE at {:?}; NVFP4 variant {:?}",
            config.num_hidden_layers,
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count(),
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
            config.num_experts,
            config.num_experts_per_tok,
            config.hc_mult,
            config.hc_lowrank,
            config.index_topk,
            config.ple_layer_ids,
            variant,
        );

        // The model-level mixer collapses the streams before `lm_head` and
        // carries the FINAL NORM (this checkpoint has no `model.norm.weight`).
        // Replicated onto every layer; only the last one consumes it.
        let hc_head = if config.hc_mult > 0 {
            Some(hc::load_head(store, config)?)
        } else {
            None
        };

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        for i in 0..config.num_hidden_layers {
            let lp = config.layer_prefix(i);
            let ffn = ffn::build_moe(store, &lp, config, gpu, variant)?;

            // Norm placeholders — see module docs. This model keeps its
            // normalization inside the hyper-connection blocks, so there is
            // no per-layer norm tensor to load. Ones-filled buffers keep the
            // shared arms' shape contract without inventing a scale, and they
            // are unreachable at runtime because the mHC forward refuses
            // before any layer executes.
            let input_norm = ones_norm(h, gpu)?;
            let post_attn_norm = ones_norm(h, gpu)?;

            let layer = match config.layer_types[i] {
                LayerType::LinearAttention => {
                    crate::weight_loader::qwen35::load_layers::linear_attn_arms::build_linear_attention_nvfp4(
                        store, &lp, gpu, variant, config, h, absmax_k, quantize_k, stream,
                        input_norm, post_attn_norm, ffn,
                    )
                    .with_context(|| format!("qwen4_exp: GDN layer {i}"))?
                }
                LayerType::FullAttention => {
                    let kv_dtype = layer_kv_dtypes
                        .get(attn_idx)
                        .copied()
                        .unwrap_or(KvCacheDtype::Bf16);
                    let l = crate::weight_loader::qwen35::load_layers::attention_arms::build_full_attention_nvfp4(
                        i, store, &lp, gpu, variant, config, h, absmax_k, quantize_k, stream,
                        kv_dtype, attn_idx, input_norm, post_attn_norm, ffn,
                    )
                    .with_context(|| format!("qwen4_exp: full-attention layer {i}"))?;
                    attn_idx += 1;
                    l
                }
                other => anyhow::bail!(
                    "qwen4_exp layer {i} has type {other:?}; this architecture is \
                     only linear_attention / full_attention"
                ),
            };
            // mHC: two sites per layer wrapping attention and the MoE. The
            // residual this model carries is `hc_mult * hidden` wide, so
            // without these the layer would run on a stream it never mixed.
            if config.hc_mult > 0 {
                let (attn, ffn) = hc::load_layer_sites(store, &lp, config)?;
                let mut layer = layer;
                attach_hc(&mut layer, i, attn, ffn, hc_head.clone(), config)?;
                layers.push(layer);
            } else {
                layers.push(layer);
            }
        }

        tracing::warn!(
            "Qwen3.8-Flash-Next loaded {} layers, but SERVING IS NOT WIRED: the \
             hyper-connection residual, the QSA indexer and the PLE n-gram \
             injection all refuse at the forward boundary. Weights are resident \
             so the alloc ledger below is real; inference will error by name.",
            layers.len(),
        );
        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let pfx = embed_prefix(config);
        dense(store, &format!("{pfx}.embed_tokens.weight")).context("qwen4_exp: embedding")
    }

    /// **This model has no final norm tensor.**
    ///
    /// There is no `model.norm.weight` anywhere in the checkpoint. The
    /// model-level `hyper_connection_mixer` — which collapses the `hc_mult`
    /// residual streams back to a single hidden state before `lm_head` —
    /// carries `hc_norm [hc_mult*hidden]`, and that IS the final
    /// normalization. It is the wrong width to stand in here (10240 against
    /// 2560), and applying it as though it were a plain final norm would be
    /// inventing math.
    ///
    /// A ones-filled buffer keeps the shape contract so the footprint can be
    /// measured at load. It is unreachable at inference because the mHC
    /// forward refuses first; if that ever stops being true, this is the
    /// first thing to fix.
    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let mixer = mixer_prefix(config);
        anyhow::ensure!(
            store.contains(&format!("{mixer}.hc_norm.weight")),
            "qwen4_exp: no `{mixer}.hc_norm.weight` — this architecture is \
             supposed to keep its final normalization in the hyper-connection \
             mixer, and it is not there. Refusing rather than guessing."
        );
        tracing::warn!(
            "qwen4_exp: final norm is a PLACEHOLDER. The real one is \
             `{mixer}.hc_norm` [{}], applied while collapsing the {} residual \
             streams — that is mHC work, not a final-norm substitution.",
            config.hc_mult * config.hidden_size,
            config.hc_mult,
        );
        ones_norm(config.hidden_size, gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            return dense(store, "lm_head.weight");
        }
        anyhow::ensure!(
            config.tie_word_embeddings,
            "qwen4_exp: no lm_head.weight and tie_word_embeddings is false"
        );
        let pfx = embed_prefix(config);
        dense(store, &format!("{pfx}.embed_tokens.weight")).context("qwen4_exp: tied lm_head")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // Dropped for v1 (#753 item I). The MTP block is effectively a second
        // model: its own 512-expert MoE, its own hyper-connection mixer, its
        // own QSA indexer, and `fc_embedding`/`fc_hidden` where Atlas's
        // `MtpWeights` wants a fused `eh_proj`. Wiring it before the main
        // forward path works would be building on sand.
        Ok(None)
    }
}

/// A ones-filled `[n]` BF16 norm scale.
///
/// BF16 1.0 is `0x3F80`, so the buffer cannot be produced with `memset`.
fn ones_norm(n: usize, gpu: &dyn GpuBackend) -> Result<DenseWeight> {
    let host: Vec<u8> = std::iter::repeat_n([0x80u8, 0x3Fu8], n).flatten().collect();
    let ptr = gpu.alloc(host.len())?;
    gpu.copy_h2d(&host, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// `model.language_model` for the multimodal layout, `model` otherwise.
fn embed_prefix(config: &ModelConfig) -> String {
    if config.weight_prefix.is_empty() {
        "model".to_string()
    } else {
        config.weight_prefix.clone()
    }
}

/// The model-level hyper-connection mixer that collapses the residual streams.
fn mixer_prefix(config: &ModelConfig) -> String {
    format!("{}.hyper_connection_mixer", embed_prefix(config))
}

/// Attach both hyper-connection sites to a freshly built layer.
///
/// `set_hc_weights` lives on `Qwen3AttentionLayer`, but `load_layers` hands
/// back `Box<dyn TransformerLayer>`, so the concrete type has to be recovered.
/// A failure here is a hard error rather than a skip: a layer that silently
/// keeps no mHC weights would run attention on an unmixed stream and produce
/// plausible, wrong activations.
fn attach_hc(
    layer: &mut Box<dyn TransformerLayer>,
    idx: usize,
    attn: crate::layers::qwen3_attention::HcSiteWeights,
    ffn: crate::layers::qwen3_attention::HcSiteWeights,
    head: Option<crate::layers::qwen3_attention::HcHeadWeights>,
    config: &ModelConfig,
) -> Result<()> {
    use crate::layers::qwen3_attention::HcWeights;
    // Hard error, never a skip. A layer that quietly kept no mHC weights
    // would run attention on a stream it never mixed and emit plausible,
    // wrong activations — with nothing in the log.
    let any = layer.as_any_mut().ok_or_else(|| {
        anyhow::anyhow!("qwen4_exp layer {idx}: no as_any_mut, cannot attach mHC weights")
    })?;
    // TWO concrete layer types carry mHC here: the 12 full-attention layers
    // are `Qwen3AttentionLayer`, the 36 GDN layers are `Qwen3SsmLayer`.
    // DeepSeek-V4 only ever needed the first, which is why the second had to
    // learn `set_hc_weights`.
    let w = HcWeights {
        attn,
        ffn,
        head,
        hc_mult: config.hc_mult,
        sinkhorn_iters: 0,
        hc_eps: config.rms_norm_eps as f32,
        // MODEL layer indices, not attention-layer ones. With a 3:1
        // GDN:attention interleave, model layer 0 is GDN and the last model
        // layer (47) is attention — so the layer that seeds the highway and
        // the layer that collapses it are DIFFERENT concrete types, and
        // neither is identified by `attn_layer_idx`.
        is_first_model_layer: idx == 0,
        is_last_model_layer: idx + 1 == config.num_hidden_layers,
    };
    if let Some(l) = any.downcast_mut::<crate::layers::Qwen3AttentionLayer>() {
        l.set_hc_weights(w);
        return Ok(());
    }
    if let Some(l) = any.downcast_mut::<crate::layers::Qwen3SsmLayer>() {
        l.set_hc_weights(w);
        return Ok(());
    }
    anyhow::bail!(
        "qwen4_exp layer {idx}: mHC weights have nowhere to go — the layer is \
         neither Qwen3AttentionLayer nor Qwen3SsmLayer"
    )
}
