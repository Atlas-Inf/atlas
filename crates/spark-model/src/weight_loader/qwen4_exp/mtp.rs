// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.8-Flash-Next Multi-Token-Prediction (MTP) draft-module loader.
//!
//! The checkpoint DOES ship MTP: 31 tensors, 5.21 GB of BF16 under `mtp.*`.
//! v1 declined them (`load_mtp_weights` returned `None`) on the reading that
//! the block was "effectively a second model". Measured against the tensor
//! index, that reading was too pessimistic on every count:
//!
//!   * The block is **shape-identical to a main full-attention layer** — the
//!     `q/k/v/o` projections, `q_norm`/`k_norm`, the QSA indexer, the MoE
//!     router and the shared expert all match `model.language_model.layers.11`
//!     tensor for tensor, all BF16. So [`build_full_attention_nvfp4`] builds
//!     the body unchanged; nothing about the attention arm is MTP-specific.
//!   * The mHC sites are named exactly like a main layer's
//!     (`{lp}.attn_hyper_connection`, `{lp}.mlp_hyper_connection`), so
//!     [`hc::load_layer_sites`] reads them with `lp = "mtp.layers.0"` as-is.
//!   * The "stacked expert layout" already had a zero-copy slicer
//!     ([`load_mtp_experts_stacked`]); it wants exactly the `[E, 2I, H]` /
//!     `[E, H, I]` pair this checkpoint ships.
//!   * `fc_embedding`/`fc_hidden` against a fused `eh_proj` is a concat, not a
//!     redesign: `fc(concat(e, h)) == fc_embedding·e + fc_hidden·h`.
//!
//! What is genuinely different is the MoE **storage**: the main layers ship
//! per-expert NVFP4 (packed E2M1), the MTP block ships two stacked BF16
//! tensors. [`build_mtp_moe`] slices and requantizes them so the body gets the
//! same [`MoeLayer`] every other layer gets.
//!
//! ## Forward (supplied by the proposer, not here)
//!
//! ```text
//!   n_h   = rms_norm(target_final_hc_streams, pre_fc_norm_hidden)  // [4*H]
//!   h_c   = collapse(n_h)                                          // 4*H -> H
//!   n_e   = rms_norm(embed[last_token], pre_fc_norm_embedding)     // [H]
//!   h_in  = fc_embedding·n_e + fc_hidden·h_c
//!   s     = hc_expand(h_in)                                        // -> 4 streams
//!   s     = body.decode(s, ..., mtp_kv_cache)                      // MIDDLE mHC
//!   h_out = hc_head(s)                                             // mtp.hyper_connection_mixer
//!   logits = lm_head(h_out)                                        // SHARED head
//! ```
//!
//! Two absences in the checkpoint pin that ordering down. There is no
//! `mtp.norm.weight` and no `input_layernorm`/`post_attention_layernorm` on
//! `mtp.layers.0` — because in this architecture the mixer's `hc_norm` IS the
//! final norm (the main model ships no `model.norm.weight` for the same
//! reason) and each mHC site carries its own `hc_norm`. So the mixer belongs
//! at the END, as the collapse-plus-final-norm, exactly as DeepSeek-V4's MTP
//! module does it.
//!
//! The body is assembled with `layer_idx = num_hidden_layers`, which makes its
//! mHC forward run the MIDDLE mixing only (`hc_pre` -> attn -> `hc_post` ->
//! `hc_pre` -> MoE -> `hc_post`): it calls neither `hc_expand` nor `hc_head`.
//! The proposer supplies both ends. See `weight_loader/deepseek_v4/mtp.rs`,
//! which is the same story on the V4 checkpoint.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::{attach, hc};
use crate::layer::TransformerLayer;
use crate::layers::qwen3_attention::HcHeadWeights;
use crate::layers::{FfnComponent, MoeLayer};
use crate::weight_map::{
    DenseWeight, ExpertWeight, MoeWeights, dense_auto, load_mtp_experts_stacked, quantize_to_nvfp4,
};

/// The `mtp.layers.0` prefix — this checkpoint has `mtp_num_hidden_layers = 1`.
const MTP_LAYER_PREFIX: &str = "mtp.layers.0";
/// The MTP block's own stream mixer, the twin of the model-level one.
const MTP_MIXER_PREFIX: &str = "mtp.hyper_connection_mixer";

/// A loaded Qwen3.8-Flash-Next MTP draft module.
///
/// `embed_tokens` and `lm_head` are deliberately absent: this checkpoint sets
/// `mtp_use_dedicated_embeddings = false`, so the drafter shares the target's.
/// They are handed to the proposer at build time.
// Consumed by the (forthcoming) `Qwen4ExpMtpProposer`. Phase 1 builds the
// module so the load path and its memory cost are exercised for real; the
// fields are read once the proposer lands.
#[allow(dead_code)]
pub struct Qwen4ExpMtpModule {
    /// Reused full-attention layer body (gated attn + QSA + mHC + 512-expert
    /// MoE), built from `mtp.layers.0`.
    pub body: Box<dyn TransformerLayer>,
    /// RMSNorm on the token embedding before `fc_embedding`: `[hidden]`.
    pub pre_fc_norm_embedding: DenseWeight,
    /// RMSNorm on the incoming target residual before the collapse:
    /// `[hc_mult * hidden]`. Its width is the reason the drafter needs the
    /// target's PRE-mixer streams and not the collapsed hidden state.
    pub pre_fc_norm_hidden: DenseWeight,
    /// Projects the normed embedding, `[hidden, hidden]`.
    pub fc_embedding: DenseWeight,
    /// Projects the collapsed hidden state, `[hidden, hidden]`.
    pub fc_hidden: DenseWeight,
    /// `mtp.hyper_connection_mixer` — the drafter's OWN head mixer. The body
    /// runs middle-mode mHC and never calls this; the proposer collapses
    /// `hc_streams -> h_out` with it after `body.decode`.
    pub hc_head: Option<HcHeadWeights>,
}

/// Build the MTP MoE from the stacked BF16 expert pair.
///
/// The main layers upload per-expert NVFP4 straight through and cost nothing
/// to build. This block instead ships `experts.gate_up_proj` `[E, 2I, H]` and
/// `experts.down_proj` `[E, H, I]` as BF16, so each expert is sliced out
/// (zero-copy — the slices alias the stacked allocation) and requantized to
/// NVFP4 so the body gets the same [`MoeLayer`] every other layer gets.
///
/// Requantizing is not the free choice it looks like: it leaves BOTH the 5.03
/// GB of stacked BF16 (owned by the `WeightStore`, which has no release API)
/// and the ~1.4 GB of NVFP4 resident. Keeping the experts BF16 instead is not
/// an option — [`MoeWeights`] holds [`ExpertWeight`], which is NVFP4-only.
fn build_mtp_moe(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<FfnComponent> {
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let n_experts = config.num_experts;
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    let mlp = format!("{MTP_LAYER_PREFIX}.mlp");
    let bf16_experts = load_mtp_experts_stacked(store, &mlp, n_experts)
        .with_context(|| format!("qwen4_exp MTP: stacked experts at {mlp}"))?;

    let q = |w: &DenseWeight, n: usize, k: usize| -> Result<_> {
        quantize_to_nvfp4(w, n, k, gpu, absmax_k, quantize_k, stream)
    };

    let mut experts = Vec::with_capacity(n_experts);
    for (e, x) in bf16_experts.iter().enumerate() {
        experts.push(ExpertWeight {
            gate_proj: q(&x.gate_proj, inter, h)
                .with_context(|| format!("qwen4_exp MTP: expert {e} gate_proj"))?,
            up_proj: q(&x.up_proj, inter, h)
                .with_context(|| format!("qwen4_exp MTP: expert {e} up_proj"))?,
            down_proj: q(&x.down_proj, h, inter)
                .with_context(|| format!("qwen4_exp MTP: expert {e} down_proj"))?,
        });
    }

    // The shared expert ships per-tensor BF16 like a main layer's, so it takes
    // the ordinary named path rather than the stacked slicer.
    let se = format!("{mlp}.shared_expert");
    let shared_expert = ExpertWeight {
        gate_proj: q(
            &dense_auto(store, &format!("{se}.gate_proj.weight"), gpu)?,
            inter,
            h,
        )?,
        up_proj: q(
            &dense_auto(store, &format!("{se}.up_proj.weight"), gpu)?,
            inter,
            h,
        )?,
        down_proj: q(
            &dense_auto(store, &format!("{se}.down_proj.weight"), gpu)?,
            h,
            inter,
        )?,
    };

    let gate = dense_auto(store, &format!("{mlp}.gate.weight"), gpu)?;
    let weights = MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate: dense_auto(store, &format!("{mlp}.shared_expert_gate.weight"), gpu)?,
        experts,
        router_pre_norm: None,
        correction_bias: None,
    };

    // Router precision matters more than router size here: at 512 experts the
    // top-10 weights cluster tightly. Same treatment as `ffn::build_moe`.
    let gate_nvfp4 = Some(q(&gate, n_experts, h)?);
    let moe = MoeLayer::new(weights, n_experts, gate_nvfp4, gpu, config)
        .context("qwen4_exp MTP: MoeLayer")?;
    Ok(FfnComponent::Moe(moe))
}

/// Load the Qwen3.8-Flash-Next MTP draft module.
///
/// `Ok(None)` when the checkpoint ships no `mtp.*` (so this is safe to call
/// unconditionally), or when they were skipped at upload — see `skip_mtp` in
/// `spark-server`, which must be OFF for this model when `--speculative` is
/// set or the tensors never reach the store.
pub fn load_qwen4exp_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Option<Qwen4ExpMtpModule>> {
    // `fc_embedding` is the cheapest MTP-only marker.
    if !store.contains("mtp.fc_embedding.weight") {
        tracing::info!(
            "qwen4_exp: no mtp.* tensors in the store — MTP disabled. (The checkpoint \
             DOES ship them; if this is unexpected, `skip_mtp` dropped them at upload.)"
        );
        return Ok(None);
    }

    let h = config.hidden_size;
    let variant = crate::weight_map::detect_nvfp4_variant(store, config);
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let free_before = gpu.free_memory().unwrap_or(0) as u64;

    let ffn = build_mtp_moe(store, config, gpu)?;

    // Norm placeholders, exactly as the main layers get them: this model keeps
    // its normalization inside the hyper-connection blocks, so there is no
    // per-layer norm tensor to load and the mHC forward never reads these.
    let input_norm = super::ones_norm(h, gpu)?;
    let post_attn_norm = super::ones_norm(h, gpu)?;

    // `layer_idx = num_hidden_layers` puts the body in middle-mHC mode (no
    // hc_expand, no hc_head) and falls out of every per-layer table by index.
    // `attn_idx` is one past the target's full-attention layers: the drafter
    // writes into its OWN single-layer KV cache, which the proposer supplies
    // along with MTP-specific attention metadata.
    let attn_idx = config
        .layer_types
        .iter()
        .filter(|t| matches!(t, atlas_core::config::LayerType::FullAttention))
        .count();
    let kv_dtype = layer_kv_dtypes
        .first()
        .copied()
        .unwrap_or(KvCacheDtype::Bf16);

    let mut body =
        crate::weight_loader::qwen35::load_layers::attention_arms::build_full_attention_nvfp4(
            config.num_hidden_layers,
            store,
            MTP_LAYER_PREFIX,
            gpu,
            variant,
            config,
            h,
            absmax_k,
            quantize_k,
            stream,
            kv_dtype,
            attn_idx,
            input_norm,
            post_attn_norm,
            ffn,
        )
        .context("qwen4_exp MTP: full-attention body")?;

    // mHC. The sites are named exactly like a main layer's, so the ordinary
    // loader reads them; only the head mixer lives under its own prefix.
    let hc_head = if config.hc_mult > 0 {
        let head = hc::load_head_at(store, MTP_MIXER_PREFIX, config)
            .context("qwen4_exp MTP: hyper_connection_mixer")?;
        let (attn_site, ffn_site) = hc::load_layer_sites(store, MTP_LAYER_PREFIX, config)
            .context("qwen4_exp MTP: mHC sites")?;
        attach::attach_hc(
            &mut body,
            config.num_hidden_layers,
            attn_site,
            ffn_site,
            Some(head.clone()),
            config,
        )?;
        Some(head)
    } else {
        None
    };

    // NO QSA indexer on the draft module, deliberately.
    //
    // The indexer is a SPARSE APPROXIMATION of attention: below its inert
    // bound it is inert and dense is exact, above it selection trades recall
    // for speed. Running the drafter's one attention layer dense is therefore
    // the more exact of the two, and a draft is only ever a proposal — the
    // target verifies every token, so a different draft distribution moves the
    // ACCEPTANCE RATE and nothing else. It can never make output wrong.
    //
    // What it does buy is the removal of a whole state-sync class: the
    // drafter's indexer would need its own `ingested` watermark tracking the
    // drafter's own token history across propose/rollback, and it desynced
    // immediately ("decode at pos 0 but 1 tokens ingested") because a proposal
    // is re-run at the same local position after a rejected draft. The target
    // still runs its own indexer, which is where the accuracy actually lives.
    if std::env::var("ATLAS_MTP_DRAFTER_QSA").ok().as_deref() == Some("1") {
        attach::attach_qsa(
            &mut body,
            config.num_hidden_layers,
            MTP_LAYER_PREFIX,
            store,
            config,
            gpu,
        )
        .context("qwen4_exp MTP: QSA indexer")?;
        tracing::warn!(
            "qwen4_exp MTP: QSA indexer attached to the DRAFT module \
             (ATLAS_MTP_DRAFTER_QSA=1). Its `ingested` watermark is not \
             rolled back per proposal — expect indexer desync."
        );
    }

    let module = Qwen4ExpMtpModule {
        body,
        pre_fc_norm_embedding: dense_auto(store, "mtp.pre_fc_norm_embedding.weight", gpu)?,
        pre_fc_norm_hidden: dense_auto(store, "mtp.pre_fc_norm_hidden.weight", gpu)?,
        fc_embedding: dense_auto(store, "mtp.fc_embedding.weight", gpu)?,
        fc_hidden: dense_auto(store, "mtp.fc_hidden.weight", gpu)?,
        hc_head,
    };

    let spent = free_before.saturating_sub(gpu.free_memory().unwrap_or(0) as u64);
    tracing::info!(
        "qwen4_exp MTP draft module loaded: 1 reused full-attention layer \
         (gated attn, dense — no QSA indexer + mHC + {}-expert MoE requantized from stacked BF16), \
         shared embed/lm_head, own head mixer. Construction cost {:.2} GB.",
        config.num_experts,
        spent as f64 / 1e9,
    );
    Ok(Some(module))
}
