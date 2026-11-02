// SPDX-License-Identifier: AGPL-3.0-only

//! nvidia/GLM-5.3-Flash-FP4-Local: what loads differently, and the one place
//! its quantized sources are released.
//!
//! | tensors (decoder layer)                     | NVFP4 release (423acf37) | **FP4-Local** (76348ddb)            |
//! |---------------------------------------------|--------------------------|-------------------------------------|
//! | KDA `q/k/v/o_proj`                          | BF16                     | MXFP8 (E4M3 + UE8M0 per 32)         |
//! | MLA `q_a/q_b/kv_a/o_proj`                   | BF16                     | FP8_PB_WO (E4M3 + F32 per 128x128)  |
//! | shared experts (3..44)                      | BF16                     | FP8_PB_WO                           |
//! | dense MLP (0..2)                            | NVFP4 (+ input_scale)    | MXFP8                               |
//! | `lm_head`                                   | BF16                     | MXFP8                               |
//! | ViT `attn.qkv/proj`, MLP + merger           | BF16                     | MXFP8, W4A16 NVFP4                  |
//! | some norms, `model.norm`                    | BF16                     | F32                                 |
//!
//! Everything else (routed experts, embeddings, kv_b/indexer, KDA b/f/g,
//! hc fn) keeps its format; hc base/scale went F32 and `load_hc_f32` reads
//! F32 natively. Routed experts' CSF-compressed scales are decoded to plain
//! E4M3 by `spark_runtime`: under expert TP (production) per rank slice, from
//! the deferred components, in `expert_tp::read_slice` (`with_deferred_slice`);
//! otherwise at upload.
//!
//! Each projection loads through the same arm production runs on the NVFP4
//! release, fed an exact or nearest BF16 copy of the checkpoint's bytes:
//!
//! - MXFP8 -> BF16 is exact (`dequant_mxfp8_to_bf16`).
//! - FP8_PB_WO -> BF16 rounds: the kernel forms `e4m3 * scale` in f32 (one
//!   RNE) and stores it as BF16 (one RNE), so each element is within one BF16
//!   half-ulp (at most 2^-8 relative for normal values) of the checkpoint's
//!   value, plus the f32 rounding step. Nothing in the tree computes on
//!   FP8 x f32-block weights directly; every consumer downstream already
//!   reads BF16 or requantizes.
//! - F32 norms -> BF16 round to nearest even (`norm_bf16`), not the store's
//!   truncating `dense_f32_safe`.
//!
//! After a layer is built nothing reads its quantized sources again, so
//! [`release_expanded`] frees them there, keeping resident memory at the
//! derived copies alone.
//!
//! Teardown: the derived copies (the BF16 lm_head and ViT linears, like
//! every NVFP4 / MXFP8 twin and BF16 TP shard the GLM loader makes) are
//! loader allocations, not store tensors. `TransformerModel::release_pools`
//! frees them in its final `sweep_unreleased`, after the store. They cannot
//! join the store: loaders get `&WeightStore` (and `WeightStore::insert`
//! needs `&mut`), and the store is shared by every loader of the build.

use anyhow::{Context, Result, bail};
use atlas_core::config::LayerType;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use crate::weight_map::DenseWeight;

/// The projection stems of decoder layer `lp` that the loader only ever reads
/// through a BF16 expansion (never natively) whenever the checkpoint ships them
/// as FP8: the attention projections of its kind, and its dense MLP or shared
/// expert. Routed experts are never listed: they load natively.
pub(super) fn decoder_expanded_stems(lp: &str, kind: LayerType, dense_ffn: bool) -> Vec<String> {
    let attn: &[&str] = match kind {
        LayerType::LinearAttention => &["q_proj", "k_proj", "v_proj", "o_proj"],
        _ => &[
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
            "indexer.wq_b",
            "indexer.wk",
            "indexer.weights_proj",
        ],
    };
    let ffn = if dense_ffn {
        "mlp"
    } else {
        "mlp.shared_experts"
    };
    attn.iter()
        .map(|p| format!("{lp}.self_attn.{p}"))
        .chain(
            ["gate_proj", "up_proj", "down_proj"]
                .iter()
                .map(|p| format!("{lp}.{ffn}.{p}")),
        )
        .collect()
}

/// Free `{stem}`'s quantized checkpoint tensors once the loader has expanded
/// them: an E4M3 `.weight` (MXFP8 or FP8 block) with its `.weight_scale`, and,
/// when `nvfp4_expanded` (the caller expanded a packed NVFP4 linear to BF16,
/// as the ViT does), a U8 `.weight` with `.weight_scale` + `.weight_scale_2`.
/// BF16/F32 stems are the derived weight themselves and stay. Idempotent.
/// Returns the bytes freed.
pub(crate) fn release_expanded(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    stem: &str,
    nvfp4_expanded: bool,
) -> Result<usize> {
    let Ok(w) = store.get(&format!("{stem}.weight")) else {
        return Ok(0);
    };
    let parts: &[&str] = match w.dtype {
        WeightDtype::FP8E4M3 => &["weight", "weight_scale"],
        WeightDtype::UInt8 if nvfp4_expanded => &["weight", "weight_scale", "weight_scale_2"],
        _ => return Ok(0),
    };
    let mut bytes = 0;
    for part in parts {
        let name = format!("{stem}.{part}");
        let Ok(t) = store.get(&name) else { continue };
        if !store.was_reclaimed(&name) {
            bytes += t.byte_size();
        }
        store
            .reclaim(gpu, &name)
            .with_context(|| format!("releasing expanded source {name}"))?;
    }
    Ok(bytes)
}

/// A BF16 norm weight: the store's own BF16 tensor, or an F32 one rounded
/// to nearest even (FP4-Local ships some norms F32; the kernels read BF16).
pub(super) fn norm_bf16(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    match store.get(name)?.dtype {
        WeightDtype::BF16 => crate::weight_map::dense(store, name),
        WeightDtype::FP32 => crate::weight_map::dense_f32_as_bf16(store, name, gpu),
        other => bail!("{name}: norm weight is {other:?}, expected BF16 or F32"),
    }
}

/// The BF16 `lm_head` every GLM head path reads (`[vocab_rows, hidden]`): the
/// store's own tensor on the NVFP4 release, the exact expansion of FP4-Local's
/// MXFP8 otherwise. Anything else is refused: the head used to be read as
/// BF16 whatever its dtype. The MXFP8 source stays until the split verify
/// head has taken its shard of the checkpoint's bytes
/// (`model::prepare_glm_head_mxfp8`); the factory releases it right after.
pub(super) fn lm_head_bf16(store: &WeightStore, gpu: &dyn GpuBackend) -> Result<WeightTensor> {
    const NAME: &str = "lm_head.weight";
    let t = store.get(NAME).context("GLM-5: missing lm_head.weight")?;
    let ptr = match t.dtype {
        WeightDtype::BF16 => t.ptr,
        WeightDtype::FP8E4M3 if crate::weight_map::mxfp8_geometry(store, "lm_head")?.is_some() => {
            crate::weight_map::dense_auto(store, NAME, gpu)?.weight
        }
        other => bail!(
            "GLM-5 {NAME} is {other:?} {:?}; expected BF16 or MXFP8",
            t.shape
        ),
    };
    Ok(WeightTensor {
        ptr,
        shape: t.shape.clone(),
        dtype: WeightDtype::BF16,
    })
}

#[cfg(test)]
#[path = "fp4l_tests.rs"]
pub(super) mod tests;
