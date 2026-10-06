// SPDX-License-Identifier: AGPL-3.0-only

//! Materialize the DENSE EXL3 linears to BF16 before layer loading.
//!
//! Every packed linear that is not a routed or shared expert (attention
//! q/k/v/o, GDN in/out projections, the QSA indexer, `lm_head`, MTP, vision)
//! is dequantized once on the GPU into a BF16 `[out, in]` `<stem>.weight`, the
//! four packed tensors are removed and freed, and the qwen4_exp loader then
//! reads it as a plain BF16 linear — no per-site EXL3 branch. The experts
//! keep their packing until `quantized_any` asks for them (see `requant`).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use crate::layers::ops::{Exl3Kernels, exl3_dense_bf16_nk};
use crate::weight_map::exl3::exl3_from_store;

/// The four tensors one packed EXL3 linear arrives as.
const PACKED_SUFFIXES: [&str; 4] = ["trellis", "suh", "svh", "mul1"];

/// Stems (name minus `.trellis`) of the DENSE packed linears in `names`,
/// sorted. Excludes routed and shared experts — those stay packed until the
/// loader requantizes them per expert — and the n-gram embedding tables,
/// which the loader streams from disk and never materializes on the GPU.
/// Routed and shared experts stay packed (see `moe_pack`); they are not
/// requantized to NVFP4.
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

/// Main-model linears whose decode GEMV can run on the packed trellis
/// (`ATLAS_EXL3_DENSE_NATIVE=1`): attention q/k/v/o, GDN in_proj_qkv /
/// in_proj_z / out_proj, lm_head. The QSA indexer (cuBLASLt only), MTP and
/// vision linears stay BF16-only.
pub(crate) fn native_stem(stem: &str) -> bool {
    if stem.contains(".indexer.") {
        return false;
    }
    stem == "lm_head"
        || stem.starts_with("model.language_model.")
        || (stem.starts_with("mtp.") && native_mtp_env())
}

/// `ATLAS_EXL3_DENSE_NATIVE_MTP=1`: the MTP draft block's dense linears keep
/// their packed trellis too, so each draft step's M=1 GEMVs read the 4-bit
/// trellis rather than the BF16 copy. Off by default.
fn native_mtp_env() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_DENSE_NATIVE_MTP").as_deref() == Ok("1"))
}

// A kept packed tensor lives in the store as `<stem>.exl3n_{trellis,suh,svh}`:
// no longer ending in `.trellis`, no loader or scan mistakes the linear for an
// unmaterialized one.

/// Dequantize every dense EXL3 linear in `store` to a BF16 `<stem>.weight`
/// and drop the packed four. Returns `(count, BF16 bytes written)`.
///
/// With `ATLAS_EXL3_DENSE_NATIVE=1` the trellis/suh/svh of the
/// [`native_stem`] linears are kept (renamed `<stem>.exl3n_*`, still
/// owned by the store) and registered against the BF16 copy's pointer, so
/// `ops::dense_gemv` at M = 1 runs the packed EXL3 GEMV. The BF16 copy stays
/// for prefill and every other consumer.
pub fn exl3_materialize_dense(
    store: &mut WeightStore,
    gpu: &dyn GpuBackend,
) -> Result<(usize, u64)> {
    let stream = gpu.default_stream();
    let kernels = Exl3Kernels::resolve(gpu)?;
    let native = crate::layers::ops::exl3_dense_native_env();
    let mut count = 0usize;
    let mut bytes = 0u64;
    let (mut kept, mut kept_bytes) = (0usize, 0u64);
    for stem in dense_stems(store.names()) {
        let w = exl3_from_store(store, &stem, gpu)?;
        let (out, in_features) = (w.shape.out_features, w.shape.in_features);
        let buf = gpu.alloc(out * in_features * 2)?;
        exl3_dense_bf16_nk(gpu, &kernels, &w, buf, stream)?;
        gpu.synchronize(stream)?;
        if native && native_stem(&stem) && !store.was_reclaimed(&format!("{stem}.trellis")) {
            match crate::layers::ops::exl3_dense_register(gpu, buf, w) {
                Ok(()) => {
                    for suffix in ["trellis", "suh", "svh"] {
                        let name = format!("{stem}.{suffix}");
                        if let Some(t) = store.remove(&name) {
                            kept_bytes += t.byte_size() as u64;
                            store.insert(format!("{stem}.exl3n_{suffix}"), t)?;
                        }
                    }
                    kept += 1;
                }
                Err(e) => tracing::warn!("EXL3 dense native: {stem} stays BF16-only: {e:#}"),
            }
        }
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
    if native {
        tracing::info!(
            "EXL3 dense native (ATLAS_EXL3_DENSE_NATIVE=1): {kept} linears keep their packed trellis \
             ({:.2} GiB resident beside the BF16 copies); batch-1 decode GEMV reads the trellis",
            kept_bytes as f64 / (1u64 << 30) as f64
        );
    }
    Ok((count, bytes))
}

#[cfg(test)]
#[path = "materialize_tests.rs"]
mod tests;
