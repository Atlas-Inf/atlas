// SPDX-License-Identifier: AGPL-3.0-only

#![allow(dead_code)]

//! Pure startup admission for the official Lightning DSpark drafter.

use anyhow::{Context, Result, bail};
use spark_runtime::kv_cache::KvCacheDtype;

use crate::layers::dflash_head::{
    AttentionLayout, BonusLayout, CheckpointLayout, ConfidenceLayout, KvDtype, KvLayout,
    LIGHTNING_MODEL_IDENTITY, LightningDsparkProfile, MarkovLayout, ParallelismLayout,
};
use crate::weight_loader::dflash_loader::DflashConfig;

/// Runtime facts that must be checked alongside parsed Lightning metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LightningRuntimeAdmission {
    pub served_gamma: usize,
    pub num_drafts: usize,
    pub physical_kv_page_size: usize,
    pub target_kv_dtype: KvCacheDtype,
    pub tp: usize,
    pub ep: usize,
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

fn required_bool(value: Option<bool>, field: &str) -> Result<bool> {
    value.with_context(|| format!("Lightning DSpark metadata is missing `{field}`"))
}

fn required_usize(value: Option<usize>, field: &str) -> Result<usize> {
    value.with_context(|| format!("Lightning DSpark metadata is missing `{field}`"))
}
