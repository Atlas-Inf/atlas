// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B audio-tower weight loader (Wave 4B).
//!
//! Maps the checkpoint's `model.audio_tower.*` + `model.embed_audio.*`
//! tensors into [`GemmaAudioWeights`] and builds the [`GemmaAudioEncoder`]
//! via its verified constructor. Tensor map (verified E2B header; all under
//! `model.audio_tower.` unless noted):
//!
//! | Checkpoint tensor | Field |
//! |---|---|
//! | `subsample_conv_projection.layer{0,1}.conv.weight` + `.{0,1}.norm.weight` | `subsample.conv{0,1}` / `ln{0,1}` |
//! | `subsample_conv_projection.input_proj_linear.weight` [1024,1024] | `subsample.input_proj_linear` |
//! | `layers.{i}.feed_forward{1,2}.ffw_layer_{1,2}.linear.weight` + 4 clip scalars each | `feed_forward{1,2}.ffw_layer_{1,2}` |
//! | `layers.{i}.feed_forward{1,2}.{pre,post}_layer_norm.weight` | the 4 FFN norms |
//! | `layers.{i}.lconv1d.linear_{start,end}.linear.weight` + 4 clip scalars each | `lconv1d.linear_{start,end}` |
//! | `layers.{i}.lconv1d.depthwise_conv1d.weight` [1024,1,5] | `lconv1d.depthwise_conv1d` |
//! | `layers.{i}.lconv1d.{pre_layer_norm,conv_norm}.weight` | the 2 lconv norms |
//! | `layers.{i}.self_attn.{q,k,v}_proj` + `post` `.linear.weight` + 4 clip scalars each | `self_attn.{q,k,v,post}_proj` |
//! | `layers.{i}.self_attn.relative_k_proj.weight` [1024,1024] | `self_attn.relative_k_proj` (plain, UNclipped) |
//! | `layers.{i}.self_attn.per_dim_scale` [128] | `self_attn.per_dim_scale` (plain parameter) |
//! | `layers.{i}.norm_{pre_attn,post_attn,out}.weight` | the 3 layer norms |
//! | `output_proj.weight` [1536,1024] + **`output_proj.bias`** [1536] | `output_proj.weight` / `output_proj.bias` |
//! | `model.embed_audio.embedding_projection.weight` [1536,1536] | `embed_audio_projection` |
//!
//! # `output_proj.bias` decision
//!
//! The audio `output_proj` is the FIRST biased linear in the gemma media
//! stack (verified: `bias` EXISTS). The encoder struct already models that
//! via a dedicated [`GemmaAudioOutputProj`] (`weight` + `bias` fields) rather
//! than stretching the bias-less vision [`ClipLinearWeights`]; the loader
//! fills the two fields from the two checkpoint tensors — no struct change.
//!
//! Clip bounds are 0-d scalars; the Gemma family stores them BF16, HF may
//! store FP32 — [`clip_scalar`] handles both dtypes, erroring on anything
//! else (PCND). The helper duplicates `loader_d`'s (that one is private with
//! "gemma vision" error text baked in; keeping the two towers' loaders
//! independent avoids parameterizing the verified vision loader).

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::{
    ClipLinearWeights, GemmaAudioAttnWeights, GemmaAudioEncoder, GemmaAudioFfnWeights,
    GemmaAudioLayerWeights, GemmaAudioLightConvWeights, GemmaAudioOutputProj,
    GemmaAudioSubsampleWeights, GemmaAudioWeights,
};
use crate::weight_map::dense;

/// Audio-tower weight prefix (top-level, NOT under `model.language_model.`).
const AUDIO_PREFIX: &str = "model.audio_tower";
const EMBED_AUDIO_PREFIX: &str = "model.embed_audio";

/// Load the Gemma-4 E2B audio tower. `Ok(None)` for text-only checkpoints
/// (`config.gemma_audio` unset — 26B/31B ship no `audio_config`); otherwise
/// loads every tensor BF16 (`dense`), parses clip bounds via [`clip_scalar`],
/// and hands [`GemmaAudioWeights`] to [`GemmaAudioEncoder::new`].
pub(super) fn load_gemma_audio_encoder_impl(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<GemmaAudioEncoder>> {
    let acfg = match &config.gemma_audio {
        Some(a) => a.clone(),
        None => return Ok(None),
    };

    let subsample = GemmaAudioSubsampleWeights {
        conv0: dense(
            store,
            &format!("{AUDIO_PREFIX}.subsample_conv_projection.layer0.conv.weight"),
        )?,
        ln0: dense(
            store,
            &format!("{AUDIO_PREFIX}.subsample_conv_projection.layer0.norm.weight"),
        )?,
        conv1: dense(
            store,
            &format!("{AUDIO_PREFIX}.subsample_conv_projection.layer1.conv.weight"),
        )?,
        ln1: dense(
            store,
            &format!("{AUDIO_PREFIX}.subsample_conv_projection.layer1.norm.weight"),
        )?,
        input_proj_linear: dense(
            store,
            &format!("{AUDIO_PREFIX}.subsample_conv_projection.input_proj_linear.weight"),
        )?,
    };

    let mut layers = Vec::with_capacity(acfg.num_hidden_layers);
    for i in 0..acfg.num_hidden_layers {
        layers.push(load_audio_layer(store, gpu, i)?);
    }

    let output_proj = GemmaAudioOutputProj {
        weight: dense(store, &format!("{AUDIO_PREFIX}.output_proj.weight"))?,
        bias: dense(store, &format!("{AUDIO_PREFIX}.output_proj.bias"))?,
    };
    let embed_audio_projection = dense(
        store,
        &format!("{EMBED_AUDIO_PREFIX}.embedding_projection.weight"),
    )?;

    let weights = GemmaAudioWeights {
        subsample,
        layers,
        output_proj,
        embed_audio_projection,
    };
    let enc = GemmaAudioEncoder::new(&weights, &acfg, gpu)?;
    tracing::info!(
        "Gemma-4 E2B: audio tower loaded — {} layers, hidden={}, heads={}, mel_bins={}",
        acfg.num_hidden_layers,
        acfg.hidden_size,
        acfg.num_attention_heads,
        acfg.mel_bins,
    );
    Ok(Some(enc))
}

/// Load one conformer layer: 2 FFNs + light conv + chunked attention + 3
/// norms (the verified `layers.{i}.*` shape).
fn load_audio_layer(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    i: usize,
) -> Result<GemmaAudioLayerWeights> {
    let lp = format!("{AUDIO_PREFIX}.layers.{i}");
    let ff1 = format!("{lp}.feed_forward1");
    let ff2 = format!("{lp}.feed_forward2");
    let lc = format!("{lp}.lconv1d");
    let sa = format!("{lp}.self_attn");
    let ffn = |prefix: &str| -> Result<GemmaAudioFfnWeights> {
        Ok(GemmaAudioFfnWeights {
            ffw_layer_1: clip_linear(store, gpu, &format!("{prefix}.ffw_layer_1"))?,
            ffw_layer_2: clip_linear(store, gpu, &format!("{prefix}.ffw_layer_2"))?,
            pre_layer_norm: dense(store, &format!("{prefix}.pre_layer_norm.weight"))?,
            post_layer_norm: dense(store, &format!("{prefix}.post_layer_norm.weight"))?,
        })
    };
    Ok(GemmaAudioLayerWeights {
        feed_forward1: ffn(&ff1)?,
        feed_forward2: ffn(&ff2)?,
        lconv1d: GemmaAudioLightConvWeights {
            linear_start: clip_linear(store, gpu, &format!("{lc}.linear_start"))?,
            linear_end: clip_linear(store, gpu, &format!("{lc}.linear_end"))?,
            depthwise_conv1d: dense(store, &format!("{lc}.depthwise_conv1d.weight"))?,
            pre_layer_norm: dense(store, &format!("{lc}.pre_layer_norm.weight"))?,
            conv_norm: dense(store, &format!("{lc}.conv_norm.weight"))?,
        },
        self_attn: GemmaAudioAttnWeights {
            q_proj: clip_linear(store, gpu, &format!("{sa}.q_proj"))?,
            k_proj: clip_linear(store, gpu, &format!("{sa}.k_proj"))?,
            v_proj: clip_linear(store, gpu, &format!("{sa}.v_proj"))?,
            post: clip_linear(store, gpu, &format!("{sa}.post"))?,
            relative_k_proj: dense(store, &format!("{sa}.relative_k_proj.weight"))?,
            per_dim_scale: dense(store, &format!("{sa}.per_dim_scale"))?,
        },
        norm_pre_attn: dense(store, &format!("{lp}.norm_pre_attn.weight"))?,
        norm_post_attn: dense(store, &format!("{lp}.norm_post_attn.weight"))?,
        norm_out: dense(store, &format!("{lp}.norm_out.weight"))?,
    })
}

/// Load one `ClippableLinear`: BF16 `linear.weight` + its 4 clip scalars.
fn clip_linear(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    prefix: &str,
) -> Result<ClipLinearWeights> {
    Ok(ClipLinearWeights {
        weight: dense(store, &format!("{prefix}.linear.weight"))?,
        input_min: clip_scalar(store, gpu, &format!("{prefix}.input_min"))?,
        input_max: clip_scalar(store, gpu, &format!("{prefix}.input_max"))?,
        output_min: clip_scalar(store, gpu, &format!("{prefix}.output_min"))?,
        output_max: clip_scalar(store, gpu, &format!("{prefix}.output_max"))?,
    })
}

/// Read a 0-d / 1-element scalar clip bound: BF16 (the Gemma family
/// convention) or FP32; anything else is a named error. Duplicate of
/// `loader_d::clip_scalar` with audio-specific error text (see module docs).
#[allow(dead_code)]
fn clip_scalar(store: &WeightStore, gpu: &dyn GpuBackend, name: &str) -> Result<f32> {
    let w = store.get(name)?;
    ensure!(
        w.num_elements() == 1,
        "gemma audio: expected scalar {name}, got shape {:?}",
        w.shape
    );
    match w.dtype {
        WeightDtype::FP32 => {
            let mut buf = [0u8; 4];
            gpu.copy_d2h(w.ptr, &mut buf)?;
            gpu.synchronize(gpu.default_stream())?;
            Ok(f32::from_le_bytes(buf))
        }
        WeightDtype::BF16 => {
            let mut buf = [0u8; 2];
            gpu.copy_d2h(w.ptr, &mut buf)?;
            gpu.synchronize(gpu.default_stream())?;
            Ok(f32::from_bits((u16::from_le_bytes(buf) as u32) << 16))
        }
        other => anyhow::bail!(
            "gemma audio: clip scalar {name} has unsupported dtype {other:?} \
             (expected BF16 or FP32)"
        ),
    }
}

#[cfg(test)]
mod loader_e_tests;
