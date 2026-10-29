// SPDX-License-Identifier: AGPL-3.0-only

//! The NVFP4 n-gram table of nvidia/Qwen3.8-Flash-Next-FP4-Local.
//!
//! Its 128 shards are U8 `[R, 80]` (160 E2M1 values per row) with a sibling
//! E4M3 `shard_{i}.weight_scale` `[R, 10]`, and the table has one F32
//! `weight_scale_2`. The packed rows go through the regular row cache; this
//! module opens the identically segmented block-scale cache and reads the
//! global scale, which together are what `layers::ple::ngram_nvfp4` decodes.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::host::f32_scalar;
use crate::layers::ple::ngram_nvfp4::{NgramNvfp4, Nvfp4RowGeom};

/// The NVFP4 decoder when the table's shards are packed U8, `None` for every
/// other element type. `rows_per` and `packed_cols` come from shard 0's
/// weight; `shards` is how many weight shards were found.
#[allow(clippy::too_many_arguments)]
pub(super) fn open_if_packed(
    store: &WeightStore,
    lp: &str,
    dtype: WeightDtype,
    shards: usize,
    rows_per: usize,
    packed_cols: usize,
    gpu: &dyn GpuBackend,
) -> Result<Option<NgramNvfp4>> {
    if dtype != WeightDtype::UInt8 {
        return Ok(None);
    }
    open(store, lp, shards, rows_per, packed_cols, gpu).map(Some)
}

fn open(
    store: &WeightStore,
    lp: &str,
    shards: usize,
    rows_per: usize,
    packed_cols: usize,
    gpu: &dyn GpuBackend,
) -> Result<NgramNvfp4> {
    let table = format!("{lp}.ple_embedding.ngram_embedding");
    let mut scale_shards = Vec::with_capacity(shards);
    let mut scale_cols = 0;
    for i in 0..shards {
        let name = format!("{table}.shard_{i}.weight_scale");
        let d = store.deferred(&name).with_context(|| {
            format!("PLE nvfp4: shard {i} has packed rows but `{name}` was not deferred")
        })?;
        ensure!(
            d.dtype == WeightDtype::FP8E4M3 && d.shape.len() == 2 && d.shape[0] == rows_per,
            "PLE nvfp4: `{name}` is {:?} {:?}, expected E4M3 [{rows_per}, _]",
            d.dtype,
            d.shape
        );
        if i == 0 {
            scale_cols = d.shape[1];
        }
        ensure!(
            d.shape[1] == scale_cols,
            "PLE nvfp4: `{name}` width differs from shard 0"
        );
        scale_shards.push((d.path.clone(), d.offset));
    }
    let dim = packed_cols * 2;
    ensure!(
        scale_cols > 0 && dim.is_multiple_of(scale_cols),
        "PLE nvfp4: {dim} dims do not split into {scale_cols} block scales"
    );
    let geom = Nvfp4RowGeom {
        dim,
        group: dim / scale_cols,
        global: f32_scalar(store, &format!("{table}.weight_scale_2"), gpu)?,
    };
    Ok(NgramNvfp4 { geom, scale_shards })
}

#[cfg(feature = "cuda")]
// The NVFP4 block-scale slot width, when the table is packed U8.
pub(super) fn scale_stride(
    store: &spark_runtime::weights::WeightStore,
    lp: &str,
    dtype: spark_runtime::weights::WeightDtype,
) -> usize {
    if dtype != spark_runtime::weights::WeightDtype::UInt8 {
        return 0;
    }
    store
        .deferred(&format!(
            "{lp}.ple_embedding.ngram_embedding.shard_0.weight_scale"
        ))
        .map(|d| d.shape[1])
        .unwrap_or(0)
}
