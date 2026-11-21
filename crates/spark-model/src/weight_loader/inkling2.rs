// SPDX-License-Identifier: AGPL-3.0-only

//! Thinking Machines Inkling 2 Mini (`inkling2_loop`) weight loader.
//!
//! Strict: the checkpoint's formats are kept (see `plan`). The engine runs one
//! `Inkling2Layer` per visit (48 over 24 physical layers); visits of the same
//! physical layer copy one set of weight handles and own their KV pool and
//! conv state. Decode only for now (prefill falls back to it per token).

mod plan;

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::inkling2::glue::GlueKernels;
use crate::layers::inkling2::layer::{Inkling2Layer, Params};
use crate::layers::inkling2::ops::Inkling2Kernels;
use crate::layers::inkling2::schedule::{max_blocks, schedule};
use crate::layers::inkling2::shared::{Dims, Shared};
use crate::layers::inkling2::weights::{self as iw, GlobalWeights};
use crate::weight_map::{DenseWeight, MtpWeights, dense};

pub struct Inkling2WeightLoader;

/// The (dtype, shape) of every tensor the store holds.
fn header(store: &WeightStore) -> Result<plan::Header> {
    store
        .names()
        .map(|n| {
            let t = store.get(n)?;
            Ok((n.to_string(), (t.dtype, t.shape.clone())))
        })
        .collect()
}

fn dims(config: &ModelConfig, stack_blocks: usize) -> Dims {
    let ink = config.inkling2.as_ref().expect("inkling2 config");
    Dims {
        hidden: config.hidden_size,
        q_dim: config.num_attention_heads * config.head_dim,
        kv_dim: config.num_key_value_heads * config.head_dim,
        heads: config.num_attention_heads,
        rel_dim: config.num_attention_heads * ink.d_rel,
        dyn_dim: config.num_key_value_heads * ink.sconv_kernel_size,
        experts: config.num_experts,
        top_k: config.num_experts_per_tok,
        inter: config.moe_intermediate_size,
        // blocks of at least 8 tokens over the longest context
        max_blocks_per_seq: config.max_position_embeddings.div_ceil(8),
        stack_blocks,
    }
}

/// `[core, 4, H]` fp32 message queries in (q, k, v, mlp) order per loop layer:
/// the checkpoint keeps (q, mlp) and (k, v) pairs in two `[2 * core, H]` tensors.
fn merged_message_queries(
    gpu: &dyn GpuBackend,
    g: &GlobalWeights,
    core: usize,
    h: usize,
) -> Result<DevicePtr> {
    let row = h * 4;
    let dst = gpu
        .alloc(core * 4 * row)
        .context("allocate inkling2 message queries")?;
    for l in 0..core {
        let base = dst.offset(l * 4 * row);
        let src = [
            g.cross_pass_queries.offset(2 * l * row),
            g.cross_pass_kv_queries.offset(2 * l * row),
            g.cross_pass_kv_queries.offset((2 * l + 1) * row),
            g.cross_pass_queries.offset((2 * l + 1) * row),
        ];
        for (i, s) in src.into_iter().enumerate() {
            gpu.copy_d2d(s, base.offset(i * row), row)?;
        }
    }
    Ok(dst)
}

impl ModelWeightLoader for Inkling2WeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let plan = plan::plan(&header(store)?, config)?;
        let ink = config.inkling2.as_ref().context("inkling2 config")?;
        ensure!(
            layer_kv_dtypes.iter().all(|d| *d == KvCacheDtype::Bf16),
            "inkling2 serves a BF16 KV cache only (--kv-cache-dtype bf16)"
        );
        let visits = schedule(ink);
        ensure!(
            visits.len() == config.num_hidden_layers,
            "inkling2 visit count"
        );
        let physical: Vec<_> = plan
            .layers
            .iter()
            .map(|l| {
                let mx = l.experts == plan::ExpertFormat::Mxfp8;
                iw::physical(
                    store,
                    gpu,
                    l.index,
                    l.full,
                    l.looped,
                    mx,
                    config.num_experts,
                )
            })
            .collect::<Result<_>>()?;
        let g = iw::global(store)?;
        let msg_queries =
            merged_message_queries(gpu, &g, ink.core_end - ink.core_start, config.hidden_size)?;
        let (k, gk) = (Inkling2Kernels::resolve(gpu)?, GlueKernels::resolve(gpu)?);
        let p = Params {
            eps: config.rms_norm_eps as f32,
            qk_clip: ink.qk_l2_clip_threshold,
            rel_clip: ink.rel_l2_clip_threshold,
            log_n_floor: ink.log_scaling_n_floor as f32,
            log_alpha: ink.log_scaling_alpha,
            conv_k: ink.sconv_kernel_size,
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            d_rel: ink.d_rel,
            rel_extent: ink.rel_extent,
            sliding_window: config.sliding_window as usize,
            gate_rank: ink.loop_gate_rank,
            gate_softcap: ink.mlp_gate_softcap,
            up_softcap: ink.mlp_up_softcap,
            route_scale: config.routed_scaling_factor as f32,
        };
        let d = dims(config, max_blocks(&visits));
        tracing::info!(
            "Inkling2: {} physical layers, {} visits, block stack {}",
            physical.len(),
            visits.len(),
            d.stack_blocks
        );
        Ok(visits
            .into_iter()
            .enumerate()
            .map(|(i, visit)| {
                Box::new(Inkling2Layer {
                    visit_idx: i,
                    visit,
                    w: physical[visit.physical],
                    g,
                    msg_queries,
                    k,
                    gk,
                    dims: d,
                    p,
                    shared: Shared::unattached(&d),
                }) as Box<dyn TransformerLayer>
            })
            .collect())
    }

    /// One set of per-forward buffers shared by every visit.
    fn attach_layer_buffers(
        &self,
        layers: &mut [Box<dyn TransformerLayer>],
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _max_tokens: usize,
    ) -> Result<()> {
        let ink = config.inkling2.as_ref().context("inkling2 config")?;
        let shared = Shared::alloc(gpu, &dims(config, max_blocks(&schedule(ink))))?;
        for (i, layer) in layers.iter_mut().enumerate() {
            layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<Inkling2Layer>())
                .with_context(|| format!("inkling2 visit {i} is not an Inkling2Layer"))?
                .shared = shared;
        }
        Ok(())
    }

    /// The vision tower is not ported yet; text serving leaves it on disk.
    fn dead_namespaces(&self, _config: &ModelConfig) -> Vec<String> {
        vec!["model.visual.".to_string()]
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.llm.embed.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.llm.norm.weight")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.llm.unembed.weight")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}
