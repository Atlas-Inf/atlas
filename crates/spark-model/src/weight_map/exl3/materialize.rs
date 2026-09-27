// SPDX-License-Identifier: AGPL-3.0-only

//! Materialize the DENSE EXL3 linears to BF16 before layer loading.
//!
//! Every packed linear that is not a routed or shared expert (attention
//! q/k/v/o, GDN in/out projections, the QSA indexer, `lm_head`, MTP, vision)
//! is dequantized once on the GPU into a BF16 `[out, in]` `<stem>.weight`, the
//! four packed tensors are removed and freed, and the qwen4_exp loader then
//! reads it as a plain BF16 linear — no per-site EXL3 branch. The experts
//! keep their packing until `quantized_any` asks for them (see `requant`).

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use crate::layers::ops::{Exl3Kernels, exl3_dense_bf16_nk};
use crate::weight_map::exl3::exl3_from_store;

/// The four tensors one packed EXL3 linear arrives as.
const PACKED_SUFFIXES: [&str; 4] = ["trellis", "suh", "svh", "mul1"];

static NATIVE_DECODE: OnceLock<bool> = OnceLock::new();

/// Record `--exl3-native-decode` at serve startup, before any model loads.
/// The first call wins; returns the value in force.
pub fn set_exl3_native_decode(on: bool) -> bool {
    *NATIVE_DECODE.get_or_init(|| on)
}

/// Native EXL3 decode (`--exl3-native-decode`): GDN, attention and LM-head
/// decode run the int8 sq GEMV on the packed EXL3 weights instead of the BF16
/// materialized copies, so those tensors must survive materialize. Off unless
/// the serve flag set it (tests and tools that never call the setter get off).
pub fn exl3_native_decode() -> bool {
    NATIVE_DECODE.get().copied().unwrap_or(false)
}

/// `ATLAS_EXL3_LAZY_BF16=1` (with native decode): the GDN BF16 copies are
/// freed after load and rebuilt on demand from the packed tensors. Read once.
/// M6f-a plumbing only — no call site consumes it yet.
pub fn exl3_lazy_bf16() -> bool {
    static ONCE: OnceLock<bool> = OnceLock::new();
    *ONCE.get_or_init(|| {
        exl3_native_decode() && std::env::var("ATLAS_EXL3_LAZY_BF16").ok().as_deref() == Some("1")
    })
}

/// True iff the packed four of `stem` must survive materialize: only under
/// the native-decode gate, and only for the three GDN projections, the four
/// full-attention projections and the LM head the decode overlays consume
/// (everything else keeps the BF16-only behavior).
pub(crate) fn keeps_packed_with(stem: &str, native: bool) -> bool {
    native
        && (stem.ends_with(".linear_attn.in_proj_qkv")
            || stem.ends_with(".linear_attn.in_proj_z")
            || stem.ends_with(".linear_attn.out_proj")
            || stem.ends_with(".self_attn.q_proj")
            || stem.ends_with(".self_attn.k_proj")
            || stem.ends_with(".self_attn.v_proj")
            || stem.ends_with(".self_attn.o_proj")
            || stem == "lm_head")
}

/// [`keeps_packed_with`] with the serve flag applied.
pub(crate) fn keeps_packed(stem: &str) -> bool {
    keeps_packed_with(stem, exl3_native_decode())
}

/// Stems (name minus `.trellis`) of the DENSE packed linears in `names`,
/// sorted. Excludes routed and shared experts — those stay packed until the
/// loader requantizes them per expert — and the n-gram embedding tables,
/// which the loader streams from disk and never materializes on the GPU.
pub(crate) fn dense_stems<'a>(names: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut stems: Vec<String> = names
        .filter(|n| n.ends_with(".trellis"))
        .map(|n| n[..n.len() - ".trellis".len()].to_string())
        .filter(|s| {
            !s.contains(".mlp.experts.")
                && !s.contains(".mlp.shared_expert.")
                && !s.contains("ngram_embedding")
        })
        .collect();
    stems.sort();
    stems
}

/// Dequantize every dense EXL3 linear in `store` to a BF16 `<stem>.weight`
/// and drop the packed four. Returns `(count, BF16 bytes written)`.
pub fn exl3_materialize_dense(
    store: &mut WeightStore,
    gpu: &dyn GpuBackend,
) -> Result<(usize, u64)> {
    let stream = gpu.default_stream();
    let kernels = Exl3Kernels::resolve(gpu)?;
    let mut count = 0usize;
    let mut bytes = 0u64;
    for stem in dense_stems(store.names()) {
        let w = exl3_from_store(store, &stem, gpu)?;
        let (out, in_features) = (w.shape.out_features, w.shape.in_features);
        let buf = gpu.alloc(out * in_features * 2)?;
        exl3_dense_bf16_nk(gpu, &kernels, &w, buf, stream)?;
        gpu.synchronize(stream)?;
        // Under the native-decode gate the GDN projections keep their packing
        // (the BF16 `.weight` above still serves prefill).
        if !keeps_packed(&stem) {
            for suffix in PACKED_SUFFIXES {
                let name = format!("{stem}.{suffix}");
                // Ask before removing: a reclaimed tensor's pointer is dead and
                // must not be freed a second time.
                let reclaimed = store.was_reclaimed(&name);
                if let Some(t) = store.remove(&name)
                    && !reclaimed
                {
                    gpu.free(t.ptr)?;
                }
            }
        }
        store.insert(
            format!("{stem}.weight"),
            WeightTensor {
                ptr: buf,
                shape: vec![out, in_features],
                dtype: WeightDtype::BF16,
            },
        )?;
        count += 1;
        bytes += (out * in_features * 2) as u64;
    }
    tracing::info!(
        "EXL3: materialized {count} dense linears to BF16 ({:.2} GiB)",
        bytes as f64 / (1u64 << 30) as f64
    );
    Ok((count, bytes))
}

#[cfg(test)]
#[path = "materialize_tests.rs"]
mod tests;
