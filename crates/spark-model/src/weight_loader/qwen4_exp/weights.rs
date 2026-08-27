// SPDX-License-Identifier: AGPL-3.0-only

//! Every `qwen4_exp` tensor, loaded into typed per-layer handles.
//!
//! Deliberately separate from the forward pass. Loading a 135 GB checkpoint and
//! running it are different problems, and the shapes here are the ones the
//! manifest already checks against two published releases
//! (`atlas_core::weight_manifest`), so a name or width that drifts fails at
//! preflight rather than in a kernel.
//!
//! Nothing here dequantizes the n-gram table: it is 51.2 B parameters and is
//! read by row at use time through [`atlas_core::ngram_table::NgramTable`].

use anyhow::{Context, Result};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{
    DenseWeight, MoeWeights, Nvfp4Variant, QuantizeCtx, dense_auto, load_moe,
};

/// Language-model weight prefix in every published `qwen4_exp` checkpoint.
pub const LM: &str = "model.language_model";

/// One hyper-connection block. Everything is `hc_count * hidden` wide — the
/// residual is that many streams concatenated, not `hidden` with a gate.
pub struct HyperConnection {
    pub hc_norm: DenseWeight,
    pub mix_down: DenseWeight,
    pub mix_up: DenseWeight,
    /// Absent on the trunk and MTP mixers, which mix without injecting.
    pub block_inject: Option<DenseWeight>,
}

impl HyperConnection {
    fn load(store: &WeightStore, prefix: &str, gpu: &dyn GpuBackend, inject: bool) -> Result<Self> {
        Ok(Self {
            hc_norm: dense_auto(store, &format!("{prefix}.hc_norm.weight"), gpu)?,
            mix_down: dense_auto(
                store,
                &format!("{prefix}.input_mix_weight_down.weight"),
                gpu,
            )?,
            mix_up: dense_auto(store, &format!("{prefix}.input_mix_weight_up.weight"), gpu)?,
            block_inject: inject
                .then(|| dense_auto(store, &format!("{prefix}.block_inject_weight.weight"), gpu))
                .transpose()?,
        })
    }
}

/// The sparse-attention indexer that gates which history a full-attention layer
/// may look at. `index_qk_proj` is FUSED: `(n_heads + kv_heads) * head_dim`.
pub struct Indexer {
    pub qk_proj: DenseWeight,
    pub q_layernorm: DenseWeight,
    pub k_layernorm: DenseWeight,
}

/// Full attention. `q_proj` is 2x `num_attention_heads * head_dim`: Q and a
/// per-head gate, interleaved. `o_proj` consumes only the Q half.
pub struct FullAttention {
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    pub indexer: Indexer,
}

impl FullAttention {
    fn load(store: &WeightStore, prefix: &str, gpu: &dyn GpuBackend) -> Result<Self> {
        let d = |suffix: &str| dense_auto(store, &format!("{prefix}.{suffix}"), gpu);
        Ok(Self {
            q_proj: d("q_proj.weight")?,
            k_proj: d("k_proj.weight")?,
            v_proj: d("v_proj.weight")?,
            o_proj: d("o_proj.weight")?,
            q_norm: d("q_norm.weight")?,
            k_norm: d("k_norm.weight")?,
            indexer: Indexer {
                qk_proj: d("indexer.index_qk_proj.weight")?,
                q_layernorm: d("indexer.q_layernorm.weight")?,
                k_layernorm: d("indexer.k_layernorm.weight")?,
            },
        })
    }
}

/// Gated-delta-net style linear attention. `in_proj_qkv` fuses q, k and v —
/// `2 * (key_heads * key_dim) + value_heads * value_dim` — and `in_proj_z` is
/// the output gate `output_gate_type` names.
pub struct LinearAttention {
    pub in_proj_qkv: DenseWeight,
    pub in_proj_a: DenseWeight,
    pub in_proj_b: DenseWeight,
    pub in_proj_z: DenseWeight,
    pub conv1d: DenseWeight,
    pub a_log: DenseWeight,
    pub dt_bias: DenseWeight,
    pub norm: DenseWeight,
    pub out_proj: DenseWeight,
}

impl LinearAttention {
    fn load(store: &WeightStore, prefix: &str, gpu: &dyn GpuBackend) -> Result<Self> {
        let d = |suffix: &str| dense_auto(store, &format!("{prefix}.{suffix}"), gpu);
        Ok(Self {
            in_proj_qkv: d("in_proj_qkv.weight")?,
            in_proj_a: d("in_proj_a.weight")?,
            in_proj_b: d("in_proj_b.weight")?,
            in_proj_z: d("in_proj_z.weight")?,
            conv1d: d("conv1d.weight")?,
            // Not `.weight`-suffixed: these are bare parameters.
            a_log: dense_auto(store, &format!("{prefix}.A_log"), gpu)?,
            dt_bias: dense_auto(store, &format!("{prefix}.dt_bias"), gpu)?,
            norm: d("norm.weight")?,
            out_proj: d("out_proj.weight")?,
        })
    }
}

/// The per-layer mixer: every layer has exactly one.
pub enum Mixer {
    Full(Box<FullAttention>),
    Linear(Box<LinearAttention>),
}

/// PLE projections. The n-gram table itself is NOT here — see the module docs.
pub struct Ple {
    pub conv1d: DenseWeight,
    pub key_proj: DenseWeight,
    pub value_proj: DenseWeight,
    pub norm_conv: DenseWeight,
    pub norm_key: DenseWeight,
    pub norm_query: DenseWeight,
}

impl Ple {
    fn load(store: &WeightStore, prefix: &str, gpu: &dyn GpuBackend) -> Result<Self> {
        let d = |suffix: &str| dense_auto(store, &format!("{prefix}.{suffix}"), gpu);
        Ok(Self {
            conv1d: d("conv1d.weight")?,
            key_proj: d("key_proj.weight")?,
            value_proj: d("value_proj.weight")?,
            norm_conv: d("norm_conv.weight")?,
            norm_key: d("norm_key.weight")?,
            norm_query: d("norm_query.weight")?,
        })
    }
}

/// One decoder layer.
pub struct Layer {
    pub attn_hc: HyperConnection,
    pub mlp_hc: HyperConnection,
    pub mixer: Mixer,
    /// Router, shared expert and all 512 routed experts.
    ///
    /// This is the SHARED `load_moe` path, not a qwen4_exp-specific one. The
    /// naming this model uses -- `{layer}.mlp.gate.weight`,
    /// `{layer}.mlp.shared_expert.{gate,up,down}_proj.weight`, and
    /// `{layer}.mlp.experts.{e}.{proj}.weight` -- is byte-for-byte what
    /// `load_moe_inner` already expects, and the target checkpoint's trunk
    /// experts are `PerExpert`, which is that function's default arm. The MoE
    /// here is reuse, the same way the linear attention turned out to be.
    ///
    /// The `Stacked` layout (HuggingFace-native, `experts.gate_up_proj` as one
    /// `[experts, 2*inter, hidden]` tensor) is NOT handled by this path. Both
    /// published releases split their trunk experts, so it is unreached today;
    /// `weight_manifest::qwen4_exp` still describes it, and preflight would
    /// name the missing tensors rather than failing in a kernel.
    pub moe: MoeWeights,
    pub ple: Option<Ple>,
}

/// The whole language model, minus experts and the n-gram table.
pub struct Qwen4ExpWeights {
    pub embed_tokens: DenseWeight,
    pub lm_head: Option<DenseWeight>,
    pub mixer_hc: HyperConnection,
    pub layers: Vec<Layer>,
}


impl Qwen4ExpWeights {
    /// Load everything the forward pass needs that is not an expert body or an
    /// n-gram row.
    pub fn load(
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        variant: Nvfp4Variant,
        qctx: QuantizeCtx,
    ) -> Result<Self> {
        anyhow::ensure!(
            config.layer_types.len() == config.num_hidden_layers,
            "layer_types must cover every layer"
        );

        let embed_tokens = dense_auto(store, &format!("{LM}.embed_tokens.weight"), gpu)
            .context("loading token embeddings")?;
        let lm_head = (!config.tie_word_embeddings)
            .then(|| dense_auto(store, "lm_head.weight", gpu))
            .transpose()
            .context("loading lm_head")?;
        let mixer_hc =
            HyperConnection::load(store, &format!("{LM}.hyper_connection_mixer"), gpu, false)
                .context("loading the trunk hyper-connection mixer")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let base = format!("{LM}.layers.{index}");
            let mixer = match config.layer_types[index] {
                LayerType::FullAttention => Mixer::Full(Box::new(
                    FullAttention::load(store, &format!("{base}.self_attn"), gpu)
                        .with_context(|| format!("layer {index}: full attention"))?,
                )),
                LayerType::LinearAttention => Mixer::Linear(Box::new(
                    LinearAttention::load(store, &format!("{base}.linear_attn"), gpu)
                        .with_context(|| format!("layer {index}: linear attention"))?,
                )),
                other => anyhow::bail!("qwen4_exp does not use layer type {other:?}"),
            };
            // ple_layer_ids is ONE-indexed; layer i hosts the tower iff i+1 is listed.
            let ple = config
                .ple_layer_ids
                .contains(&(index + 1))
                .then(|| Ple::load(store, &format!("{base}.ple"), gpu))
                .transpose()
                .with_context(|| format!("layer {index}: PLE tower"))?;

            layers.push(Layer {
                attn_hc: HyperConnection::load(
                    store,
                    &format!("{base}.attn_hyper_connection"),
                    gpu,
                    true,
                )?,
                mlp_hc: HyperConnection::load(
                    store,
                    &format!("{base}.mlp_hyper_connection"),
                    gpu,
                    true,
                )?,
                mixer,
                moe: load_moe(store, &base, config.num_experts, gpu, config, variant, qctx)
                    .with_context(|| format!("layer {index}: MoE"))?,
                ple,
            });
        }

        Ok(Self {
            embed_tokens,
            lm_head,
            mixer_hc,
            layers,
        })
    }
}
