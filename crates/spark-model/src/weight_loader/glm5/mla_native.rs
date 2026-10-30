// SPDX-License-Identifier: AGPL-3.0-only

//! FP4-Local's FP8 block MLA projections as native twins
//! (`ATLAS_GLM_FP4L_MLA_FP8`, served by `qwen3_attention::mla_native`).
//!
//! Each target MLA layer gets this rank's slice of the checkpoint's q_a /
//! kv_a (replicated), q_b (column-parallel rows) and o (row-parallel
//! columns): E4M3 + the f32 128 x 128 grid, the checkpoint's bytes. They
//! replace the four MXFP8 twins of `ATLAS_GLM_MLA_MXFP8` (requantized from a
//! rounded BF16 copy) and, unless `ATLAS_GLM_FP4L_FP8_PREFILL=bf16`, the BF16
//! copies too: the loader frees those after the layer's other load-time
//! readers (`build_per_head_views`, the K3 O initializer) and keys each twin
//! by a never-mapped stand-in address (`mla_native::freed_key`).
//!
//! Resident per rank (11 layers, TP2): twins 54.5 M params x 1.0002 B =
//! ~600 MB, against production's BF16 (1.20 GB) + MXFP8 twins (~0.62 GB).

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::tp::MlaTpPlan;
use crate::layers::glm_native::NativePolicy;
use crate::layers::ops::Fp8bPrefill;
use crate::tp_shard::TpShardKind;
use crate::weight_map::{Fp8BlockWeight, fp8_block_twin, raw_fp8_block};

/// The four projections in key order (`freed_key`'s `proj`).
pub(super) const PROJECTIONS: [&str; 4] = ["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj"];

/// This rank's native twins of q_a, kv_a, q_b, o (with their local `[n, k]`)
/// for the MLA layer at `p` (`...self_attn`), or `None` when the switch is off,
/// for the replicated MTP body (`mtp_body`), or when the checkpoint ships q_a
/// in another format (logged once). Frees what it built on error.
pub(super) fn mla_native_twins(
    policy: NativePolicy,
    store: &WeightStore,
    p: &str,
    tp: &MlaTpPlan,
    config: &ModelConfig,
    mtp_body: bool,
    gpu: &dyn GpuBackend,
) -> Result<Option<Vec<(Fp8BlockWeight, [usize; 2])>>> {
    if !policy.mla_fp8 || mtp_body {
        return Ok(None);
    }
    if raw_fp8_block(store, &format!("{p}.q_a_proj"))?.is_none() {
        super::mode::require_format(false, &format!("{p} FP8 block weights"))?;
        static IGNORED: std::sync::Once = std::sync::Once::new();
        IGNORED.call_once(|| {
            tracing::warn!(
                "{} set, but {p}.q_a_proj is not FP8 block in this checkpoint: \
                 MLA projections keep their BF16 / MXFP8 copies",
                NativePolicy::MLA
            )
        });
        return Ok(None);
    }
    let (q_b, o) = (tp.q_b(), tp.o());
    let full = [
        (
            config.q_lora_rank,
            config.hidden_size,
            TpShardKind::Replicated,
        ),
        (
            config.kv_lora_rank + config.qk_rope_head_dim,
            config.hidden_size,
            TpShardKind::Replicated,
        ),
        q_b,
        o,
    ];
    let stream = gpu.default_stream();
    let mut twins: Vec<(Fp8BlockWeight, [usize; 2])> = Vec::with_capacity(4);
    let built = (|| -> Result<()> {
        for (name, (n, k, kind)) in PROJECTIONS.iter().zip(full) {
            let stem = format!("{p}.{name}");
            let raw = raw_fp8_block(store, &stem)?
                .ok_or_else(|| anyhow::anyhow!("{stem}: not FP8 block beside an FP8 block q_a"))?;
            ensure!(
                [raw.n, raw.k] == [n, k],
                "{stem}: checkpoint FP8 block [{}, {}] is not the config's [{n}, {k}]",
                raw.n,
                raw.k
            );
            let (twin, ln, lk) = fp8_block_twin(gpu, raw, kind, tp.tp_rank, tp.tp_size, stream)?;
            twins.push((twin, [ln, lk]));
        }
        Ok(())
    })();
    if let Err(e) = built {
        for (t, _) in twins {
            t.free(gpu)?;
        }
        return Err(e);
    }
    Ok(Some(twins))
}

/// Tile q_b's and o's twins (the large ones only `fp8b_gemv` reads) under
/// `ATLAS_GLM_MXFP8_TILED=1`, when the BF16 copy serves prefill: the prefill
/// GEMMs read the row-major bytes. Load time: synchronizes.
pub(super) fn tile_large_twins(
    twins: &mut [(Fp8BlockWeight, [usize; 2])],
    prefill: Fp8bPrefill,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    if !prefill.needs_bf16() {
        return Ok(());
    }
    let stream = gpu.default_stream();
    for (twin, [n, k]) in twins.iter_mut().skip(2) {
        crate::layers::ops::fp8b_tile_if_requested(gpu, twin, *n, *k, stream)?;
    }
    Ok(())
}

/// The lookup keys of the four twins: the BF16 copies' pointers when they
/// stay (`bf16` prefill), else each freed and replaced by its stand-in.
/// `bf16` holds the four BF16 pointers in [`PROJECTIONS`] order and is
/// rewritten to the keys.
pub(super) fn settle_bf16(
    bf16: &mut [&mut DevicePtr; 4],
    prefill: Fp8bPrefill,
    layer_idx: usize,
    gpu: &dyn GpuBackend,
) -> Result<[DevicePtr; 4]> {
    if prefill.needs_bf16() {
        return Ok([0, 1, 2, 3].map(|i| *bf16[i]));
    }
    let mut keys = [DevicePtr::NULL; 4];
    for (i, ptr) in bf16.iter_mut().enumerate() {
        gpu.free(**ptr)?;
        keys[i] = crate::layers::qwen3_attention::mla_native::freed_key(layer_idx, i);
        **ptr = keys[i];
    }
    Ok(keys)
}

#[cfg(test)]
#[path = "mla_native_tests.rs"]
mod tests;
