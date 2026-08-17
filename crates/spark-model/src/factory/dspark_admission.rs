// SPDX-License-Identifier: AGPL-3.0-only

//! Pure startup admission for the official Lightning DSpark drafter.

use anyhow::{Context, Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::kv_cache::KvCacheDtype;

use super::DflashBuildArgs;
use crate::layers::dflash_head::{
    AttentionLayout, BonusLayout, CheckpointLayout, ConfidenceLayout, KvDtype, KvLayout,
    LIGHTNING_MODEL_IDENTITY, LIGHTNING_SWA_WINDOW, LightningDsparkProductPolicy,
    LightningDsparkProfile, LightningDsparkRuntimeToggles, MarkovLayout, ParallelismLayout,
};
use crate::weight_loader::dflash_loader::DflashConfig;
use crate::weight_loader::store_has_dflash_weights;

/// Runtime facts that must be checked alongside parsed Lightning metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LightningRuntimeAdmission {
    pub served_gamma: usize,
    pub num_drafts: usize,
    pub physical_kv_page_size: usize,
    pub target_kv_dtype: KvCacheDtype,
    pub tp: usize,
    pub ep: usize,
    pub fc_present: bool,
    pub markov_w1_present: bool,
    pub markov_w2_present: bool,
    pub all_required_sinks_present: bool,
}

/// Admit an exact official Lightning config, or ignore a different architecture.
///
/// This mapper has no runtime side effects. It intentionally does not infer
/// defaults for fields whose absence would change the Lightning contract.
pub(crate) fn admit_lightning_dspark(
    config: &DflashConfig,
    runtime: LightningRuntimeAdmission,
) -> Result<Option<LightningDsparkProfile>> {
    let architectures = config
        .architectures
        .as_ref()
        .context("Lightning DSpark metadata is missing `architectures`")?;
    if !architectures
        .iter()
        .any(|architecture| architecture == LIGHTNING_MODEL_IDENTITY)
    {
        return Ok(None);
    }
    if architectures.as_slice() != [LIGHTNING_MODEL_IDENTITY] {
        bail!(
            "Lightning DSpark architecture identity must be exactly [`{LIGHTNING_MODEL_IDENTITY}`], found {architectures:?}"
        );
    }

    let sub = config
        .dflash_config
        .as_ref()
        .context("Lightning DSpark metadata is missing `dflash_config`")?;
    let target = match runtime.target_kv_dtype {
        KvCacheDtype::Fp8 => KvDtype::Fp8,
        other => bail!("Lightning DSpark target KV dtype must be FP8, found {other}"),
    };
    if let Some(algo) = config
        .quantization_config
        .as_ref()
        .and_then(|quant| quant.kv_cache_quant_algo.as_deref())
    {
        bail!(
            "Lightning DSpark drafter KV quantization must be absent/null; found `quantization_config.kv_cache_quant_algo={algo}`"
        );
    }
    if !runtime.fc_present {
        bail!("Lightning DSpark required weight `fc.weight` is missing");
    }

    let root_bonus_anchor = required_bool(config.dspark_bonus_anchor, "dspark_bonus_anchor")?;
    let root_sample = required_bool(config.sample_from_anchor, "sample_from_anchor")?;
    let nested_sample = required_bool(sub.sample_from_anchor, "dflash_config.sample_from_anchor")?;
    let root_sink = required_bool(config.attention_sink_bias, "attention_sink_bias")?;
    let nested_sink = required_bool(sub.attention_sink_bias, "dflash_config.attention_sink_bias")?;
    let causal = required_bool(sub.causal, "dflash_config.causal")?;
    let use_swa = required_bool(sub.use_swa, "dflash_config.use_swa")?;
    let swa_window = required_usize(sub.swa_window_size, "dflash_config.swa_window_size")?;
    let markov_rank = config
        .markov_rank
        .context("Lightning DSpark metadata is missing `markov_rank`")?;
    let dspark_markov_rank = config
        .dspark_markov_rank
        .context("Lightning DSpark metadata is missing `dspark_markov_rank`")?;
    if dspark_markov_rank != markov_rank {
        bail!(
            "Lightning DSpark metadata disagrees: markov_rank={markov_rank}, dspark_markov_rank={dspark_markov_rank}"
        );
    }
    let root_taps = config
        .target_layer_ids
        .as_ref()
        .context("Lightning DSpark metadata is missing root `target_layer_ids`")?;
    if root_taps != &sub.target_layer_ids {
        bail!(
            "Lightning DSpark target_layer_ids disagree: root={root_taps:?}, dflash_config={:?}",
            sub.target_layer_ids
        );
    }

    let profile = LightningDsparkProfile {
        algorithm: "DSpark".to_owned(),
        model_identity: LIGHTNING_MODEL_IDENTITY.to_owned(),
        checkpoint: CheckpointLayout {
            block_size: config.block_size,
            physical_kv_page_size: runtime.physical_kv_page_size,
        },
        served_gamma: runtime.served_gamma,
        num_drafts: runtime.num_drafts,
        taps: sub.target_layer_ids.clone(),
        attention: AttentionLayout {
            causal,
            use_swa,
            swa_window,
            attention_sink_bias: root_sink && nested_sink && runtime.all_required_sinks_present,
        },
        markov: MarkovLayout {
            rank: markov_rank,
            w1_present: runtime.markov_w1_present,
            w2_present: runtime.markov_w2_present,
        },
        bonus: BonusLayout {
            bonus_anchor: root_bonus_anchor,
            sample_from_anchor: root_sample || nested_sample,
        },
        confidence: ConfidenceLayout {
            head_present: config.confidence_head == Some(true) || sub.confidence_head == Some(true),
            adaptive: config.adaptive == Some(true) || sub.adaptive == Some(true),
        },
        kv: KvLayout {
            target,
            drafter: KvDtype::Bf16,
        },
        parallelism: ParallelismLayout {
            tensor_parallel: runtime.tp,
            expert_parallel: runtime.ep,
        },
    };
    profile
        .validate()
        .context("Lightning DSpark runtime admission profile validation")?;
    Ok(Some(profile))
}

pub(crate) fn admit_lightning_dspark_build(
    args: &DflashBuildArgs<'_>,
    target: &ModelConfig,
    num_drafts: usize,
    physical_kv_page_size: usize,
    target_kv_dtype: KvCacheDtype,
) -> Result<Option<LightningDsparkProfile>> {
    let has_dspark_markers = args.drafter_config.dspark_bonus_anchor.is_some()
        || args.drafter_config.markov_rank.is_some()
        || args.drafter_config.dspark_markov_rank.is_some();
    if args.drafter_config.architectures.is_none() && !has_dspark_markers {
        return Ok(None);
    }
    let declares_lightning =
        args.drafter_config
            .architectures
            .as_ref()
            .is_some_and(|architectures| {
                architectures
                    .iter()
                    .any(|architecture| architecture == LIGHTNING_MODEL_IDENTITY)
            });
    let served_gamma = match args.gamma {
        Some(gamma) => gamma,
        None if declares_lightning => bail!("Lightning DSpark requires explicit served gamma"),
        None => args.drafter_config.block_size,
    };
    if declares_lightning {
        let window_size = args
            .window_size
            .context("Lightning DSpark requires explicit served SWA window")?;
        if window_size != LIGHTNING_SWA_WINDOW {
            bail!(
                "Lightning DSpark served SWA window must be {LIGHTNING_SWA_WINDOW}, found {window_size}"
            );
        }
    }
    let prefix = if args.drafter_store.contains("model.fc.weight") {
        "model."
    } else {
        ""
    };
    let markov_w1 = format!("{prefix}markov_head.markov_w1.weight");
    let markov_w2 = format!("{prefix}markov_head.markov_w2.weight");
    let require_sinks = args
        .drafter_config
        .dflash_config
        .as_ref()
        .and_then(|sub| sub.attention_sink_bias)
        == Some(true);
    let all_required_sinks_present = !require_sinks
        || (0..args.drafter_config.num_hidden_layers).all(|layer| {
            args.drafter_store.contains(&format!(
                "{prefix}layers.{layer}.self_attn.attention_sink_bias"
            ))
        });
    admit_lightning_dspark(
        &args.drafter_config,
        LightningRuntimeAdmission {
            served_gamma,
            num_drafts,
            physical_kv_page_size,
            target_kv_dtype,
            tp: target.tp_world_size,
            ep: target.ep_world_size,
            fc_present: store_has_dflash_weights(args.drafter_store),
            markov_w1_present: args.drafter_store.contains(&markov_w1),
            markov_w2_present: args.drafter_store.contains(&markov_w2),
            all_required_sinks_present,
        },
    )
}

pub(crate) fn admit_lightning_dspark_product_build(
    args: &DflashBuildArgs<'_>,
    target: &ModelConfig,
    num_drafts: usize,
    physical_kv_page_size: usize,
    target_kv_dtype: KvCacheDtype,
    runtime_toggles: LightningDsparkRuntimeToggles,
) -> Result<Option<LightningDsparkProductPolicy>> {
    // Product admission — and therefore strict product toggle parsing —
    // applies ONLY to a drafter that declares the exact Lightning
    // architecture. Generic DFlash builds must reach the lenient path
    // without product-only toggle errors.
    if !declares_exact_lightning(args) {
        return Ok(None);
    }
    Ok(admit_lightning_dspark_build(
        args,
        target,
        num_drafts,
        physical_kv_page_size,
        target_kv_dtype,
    )?
    .map(|profile| LightningDsparkProductPolicy::try_new(profile, runtime_toggles))
    .transpose()?)
}

/// True only when the drafter's `architectures` is exactly the official
/// Lightning identity — the single discriminator that entitles product
/// (fail-closed) toggle parsing. Anything else is generic DFlash.
pub(crate) fn declares_exact_lightning(args: &DflashBuildArgs<'_>) -> bool {
    args.drafter_config
        .architectures
        .as_ref()
        .is_some_and(|architectures| architectures.as_slice() == [LIGHTNING_MODEL_IDENTITY])
}

fn required_bool(value: Option<bool>, field: &str) -> Result<bool> {
    value.with_context(|| format!("Lightning DSpark metadata is missing `{field}`"))
}

fn required_usize(value: Option<usize>, field: &str) -> Result<usize> {
    value.with_context(|| format!("Lightning DSpark metadata is missing `{field}`"))
}
