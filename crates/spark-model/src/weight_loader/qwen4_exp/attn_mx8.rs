// SPDX-License-Identifier: AGPL-3.0-only
//
//! Full-attention layers of nvidia/Qwen3.8-Flash-Next-FP4-Local, on their
//! MXFP8 q/k/v/o bytes.
//!
//! The NVFP4 pack ships these projections BF16, and the generic builder
//! (`build_full_attention_nvfp4`) requantizes BF16 to NVFP4 for decode. On
//! FP4-Local that would be a second, lossy quantization of weights NVIDIA's
//! QAD already trained in MXFP8, so this builder skips it: q (with its
//! interleaved gate rows), k, v and o stay exactly as shipped and every
//! forward reads them through `mxfp8_proj.cu`.
//!
//! TP: q/k/v are column-parallel (each rank a contiguous row slice — the same
//! slice the BF16 arm takes, so the gated-q layout is preserved per rank) and
//! o is row-parallel (each rank a per-row column slice of whole 32-input
//! scale groups; partial sums reconcile in the existing post-o_proj
//! all-reduce). The slices are byte-exact — packed E4M3 + E8M0 needs no
//! dequant.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::mx8_common::module;
use crate::layer::TransformerLayer;
use crate::layers::ops::{Mxfp8ProjKernels, Mxfp8Scratch};
use crate::layers::qwen3_attention::mx8::Mx8AttnRt;
use crate::layers::{FfnComponent, Qwen3AttentionLayer};
use crate::tp_shard::{TpAttentionDims, TpShardKind, shard_mxfp8_row_parallel, shard_mxfp8_rows};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Mxfp8Proj, Mxfp8Weight, QuantizedWeight, dense_auto,
    load_kv_scales,
};

/// True when layer `lp`'s attention projections are MXFP8 in the store.
pub(super) fn present(store: &WeightStore, lp: &str) -> bool {
    let q = format!("{lp}.self_attn.q_proj");
    store
        .get(&format!("{q}.weight"))
        .is_ok_and(|t| t.dtype == WeightDtype::FP8E4M3)
        && store.contains(&format!("{q}.weight_scale"))
}

/// Build full-attention layer `layer_idx` on its MXFP8 projections.
#[allow(clippy::too_many_arguments)]
pub(super) fn build(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    kv_dtype: KvCacheDtype,
    attn_idx: usize,
    (input_norm, post_attn_norm, ffn): (DenseWeight, DenseWeight, FfnComponent),
    scratch: Mxfp8Scratch,
) -> Result<Box<dyn TransformerLayer>> {
    let h = config.hidden_size;
    let dims = TpAttentionDims::from_config(config);
    let tp = dims.tp_size;
    let p = format!("{lp}.self_attn");
    let proj = |name: &str, n: usize, k: usize| -> Result<Mxfp8Proj> {
        let full = module(store, &format!("{p}.{name}"), n, k)?;
        let kind = dims
            .proj_shape(name)
            .map(|(_, _, kind)| kind)
            .unwrap_or(TpShardKind::Replicated);
        let (local, ln, lk) = match kind {
            TpShardKind::RowParallel => {
                let (l, lk) = shard_mxfp8_row_parallel(full, n, k, dims.tp_rank, tp, gpu)?;
                (l, n, lk)
            }
            _ => {
                let (l, ln) = shard_mxfp8_rows(full, &[n], k, dims.tp_rank, tp, gpu)?;
                (l, ln, k)
            }
        };
        if tp > 1 {
            // The store's full module is dead once the local slice exists.
            for t in ["weight", "weight_scale"] {
                store.reclaim(gpu, &format!("{p}.{name}.{t}"))?;
            }
        }
        Ok(Mxfp8Proj {
            w: Mxfp8Weight {
                data: local[0],
                scales: local[1],
            },
            n: ln as u32,
            k: lk as u32,
        })
    };
    let projs = [
        proj("q_proj", dims.full_q_n, h)?,
        proj("k_proj", dims.full_kv_n, h)?,
        proj("v_proj", dims.full_kv_n, h)?,
        proj("o_proj", h, dims.full_o_in)?,
    ];

    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };
    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let attn = AttentionWeights {
        q_proj: null,
        k_proj: null,
        v_proj: null,
        o_proj: QuantizedWeight::null(),
        q_norm: dense_auto(store, &format!("{p}.q_norm.weight"), gpu)?,
        k_norm: dense_auto(store, &format!("{p}.k_norm.weight"), gpu)?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    let mut layer = Qwen3AttentionLayer::new(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        attn_idx,
        None,
        None,
        None,
        gpu,
        kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;
    layer
        .set_mx8_projections(
            projs,
            Mx8AttnRt {
                kernels: Mxfp8ProjKernels::resolve(gpu)?,
                scratch,
            },
        )
        .with_context(|| format!("MXFP8 attention layer {layer_idx}"))?;
    tracing::info!(
        "Layer {layer_idx}: attention q/k/v/o native MXFP8 (FP4-Local){}",
        if tp > 1 {
            format!(", tp={tp}")
        } else {
            String::new()
        }
    );
    Ok(Box::new(layer))
}
