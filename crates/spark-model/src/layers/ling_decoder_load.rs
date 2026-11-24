// SPDX-License-Identifier: AGPL-3.0-only

//! Weight construction for the native Ling decoder block.

use std::sync::{Arc, Mutex};

use anyhow::{Result, bail, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::ling_paging::LingPagingReader;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layer::TransformerLayer;
use crate::layers::FfnComponent;
use crate::layers::ling_kda::{LING_KDA_HIDDEN_SIZE, LingKdaConfig, LingKdaLayer};
use crate::layers::ling_mla::{LingMlaConfig, LingMlaKernels, LingMlaWeights};
use crate::layers::ling_moe::LingMoeFfn;
use crate::layers::ling_projection::LingProjection;

use super::*;

/// Build all normal Ling decoder layers.  The loader retains `rope_inv_freq`
/// for the model lifetime; this function stores the non-owning device pointer
/// in each MLA layer.
pub fn load_ling_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    reader: Arc<Mutex<LingPagingReader>>,
    rope_inv_freq: DevicePtr,
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    ensure!(config.model_type == "bailing_moe_v3_vl");
    ensure!(
        config.hidden_size == LING_KDA_HIDDEN_SIZE,
        "Ling decoder expects hidden size {LING_KDA_HIDDEN_SIZE}, got {}",
        config.hidden_size
    );
    ensure!(config.layer_types.len() == config.num_hidden_layers);
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);
    // PagedKvCache stores only full-attention layers for Ling.  Its layer
    // index is therefore compact (0..7), not the transformer layer number
    // (5, 11, ... 41) used by the model's interleaved KDA/MLA schedule.
    let mut attention_layer_idx = 0usize;
    for layer_idx in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer_idx}");
        let input_norm = load_norm(
            store,
            &format!("{prefix}.input_layernorm.weight"),
            config.hidden_size,
        )?;
        let post_attn_norm = load_norm(
            store,
            &format!("{prefix}.post_attention_layernorm.weight"),
            config.hidden_size,
        )?;
        let projection = LingDecoderLayer::alloc_projection_scratch(gpu, config.intermediate_size)?;
        let (attention, cache_layer_idx) = match config.layer_type(layer_idx) {
            LayerType::LinearAttention => (
                LingAttention::Kda {
                    layer: LingKdaLayer::new(
                        crate::layers::ling_kda::LingKdaWeights::load(
                            gpu,
                            store,
                            &format!("{prefix}.attention"),
                        )?,
                        LingKdaConfig {
                            lower_bound: config.kda_gate_lower_bound,
                            rms_norm_eps: config.rms_norm_eps as f32,
                        },
                        gpu,
                    )?,
                    scratch: LingDecoderLayer::alloc_kda_scratch(gpu)?,
                    projection,
                },
                None,
            ),
            LayerType::FullAttention => {
                let mla = LingMlaConfig {
                    hidden_size: config.hidden_size,
                    num_heads: config.num_attention_heads,
                    kv_lora_rank: config.kv_lora_rank,
                    qk_nope_head_dim: config.qk_nope_head_dim,
                    qk_rope_head_dim: config.qk_rope_head_dim,
                    v_head_dim: config.v_head_dim,
                    rope_theta: config.rope_theta as f32,
                    rms_eps: config.rms_norm_eps as f32,
                };
                let kernels = LingMlaKernels::load(gpu)?;
                let cache_layer_idx = attention_layer_idx;
                attention_layer_idx += 1;
                (
                    LingAttention::Mla {
                        weights: LingMlaWeights::load(
                            gpu,
                            store,
                            &format!("{prefix}.attention"),
                            mla,
                            rope_inv_freq,
                        )?,
                        scratch: LingDecoderLayer::alloc_mla_scratch(gpu, mla, projection)?,
                        config: mla,
                        kernels,
                    },
                    Some(cache_layer_idx),
                )
            }
            other => bail!("Ling layer {layer_idx} has unsupported type {other:?}"),
        };
        let mut sante_ffn = None;
        let (ffn, dense_ffn) = if config.mlp_only_layers.contains(&layer_idx) {
            (
                FfnComponent::None,
                Some(load_dense_ffn(store, &prefix, gpu, config)?),
            )
        } else if config.mtp_num_hidden_layers == 0 {
            sante_ffn = Some(sante::SanteFfn::load(
                store, gpu, &reader, layer_idx, projection,
            )?);
            (FfnComponent::None, None)
        } else {
            (
                FfnComponent::LingMoe(LingMoeFfn::load(
                    store,
                    layer_idx,
                    gpu,
                    config,
                    Arc::clone(&reader),
                )?),
                None,
            )
        };
        layers.push(Box::new(LingDecoderLayer {
            input_norm,
            post_attn_norm,
            attention,
            ffn,
            dense_ffn,
            sante_ffn,
            rms_norm_residual_k: gpu.kernel("ling_norm", "ling_rms_norm_residual")?,
            residual_add_rms_norm_k: gpu.kernel("ling_norm", "ling_residual_add_rms_norm")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            hidden_size: config.hidden_size,
            h_state_bytes: config.ssm_h_state_bytes(),
            conv_state_bytes: config.ssm_conv_state_bytes(),
            attention_layer_idx: cache_layer_idx,
        }));
    }
    Ok(layers)
}

fn load_norm(
    store: &WeightStore,
    name: &str,
    hidden_size: usize,
) -> Result<crate::weight_map::DenseWeight> {
    let tensor = store.get(name)?;
    ensure!(tensor.dtype == WeightDtype::BF16 && tensor.shape == [hidden_size]);
    Ok(crate::weight_map::DenseWeight { weight: tensor.ptr })
}

fn load_dense_ffn(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
) -> Result<LingDenseFfn> {
    let hidden = config.hidden_size;
    let gate = store.get(&format!("{prefix}.mlp.gate_proj.weight"))?;
    let up = store.get(&format!("{prefix}.mlp.up_proj.weight"))?;
    let down = store.get(&format!("{prefix}.mlp.down_proj.weight"))?;
    ensure!(
        matches!(gate.dtype, WeightDtype::BF16 | WeightDtype::FP8E4M3) && gate.shape.len() == 2,
        "Ling dense FFN gate must be BF16 or block-scaled FP8"
    );
    ensure!(up.dtype == gate.dtype && up.shape == gate.shape);
    ensure!(
        down.dtype == gate.dtype && down.shape == [hidden, gate.shape[0]],
        "Ling dense FFN down projection has shape {:?}; expected [{hidden}, {}]",
        down.shape,
        gate.shape[0]
    );
    let intermediate = gate.shape[0];
    let projection = LingDecoderLayer::alloc_projection_scratch(
        gpu,
        config.intermediate_size.max(intermediate),
    )?;
    Ok(LingDenseFfn {
        gate: LingProjection::load(
            gpu,
            store,
            &format!("{prefix}.mlp.gate_proj"),
            intermediate,
            hidden,
        )?,
        up: LingProjection::load(
            gpu,
            store,
            &format!("{prefix}.mlp.up_proj"),
            intermediate,
            hidden,
        )?,
        down: LingProjection::load(
            gpu,
            store,
            &format!("{prefix}.mlp.down_proj"),
            hidden,
            intermediate,
        )?,
        projection,
        gate_out: gpu.alloc(intermediate * 2)?,
        up_out: gpu.alloc(intermediate * 2)?,
        output: gpu.alloc(hidden * 2)?,
        intermediate,
        silu_k: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
    })
}
