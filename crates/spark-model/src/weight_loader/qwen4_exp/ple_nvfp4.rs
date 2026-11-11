// SPDX-License-Identifier: AGPL-3.0-only

//! The NVFP4 n-gram table of nvidia/Qwen3.8-Flash-Next-FP4-Local.
//!
//! Its 128 shards are U8 `[R, 80]` (160 E2M1 values per row) with a sibling
//! E4M3 `shard_{i}.weight_scale` `[R, 10]`, and the table has one F32
//! `weight_scale_2`. The packed rows go through the regular row cache; this
//! module opens the identically segmented block-scale cache and reads the
//! global scale, which together are what `layers::ple::ngram_nvfp4` decodes.
//!
//! A row's codes and its scales sit in two tensors at unrelated offsets, so
//! a cache miss read in place is two NVMe reads, and per-IO latency (the
//! APST wake behind `ATLAS_PLE_NVME_KEEPALIVE_MS`) is the PLE stall on
//! Spark, not bytes. `ATLAS_PLE_PACKED_DIR=<dir>` interleaves the planes once
//! (`spark_storage::ngram_pack`) into a 28.8 GB copy of 96-byte rows under
//! `<dir>`, and the cache opens that as one contiguous table: one read per
//! miss, the same bytes to the same decode. The checkpoint directory is never
//! written. Every open re-hashes a random 1.3% of the copy against its chunk
//! checksums; `ATLAS_PLE_PACKED_VERIFY=full` re-hashes all of it. Unset, the
//! table is read in place as before.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};
use spark_storage::ngram_pack::{PackSpec, Plane, Verify, open_or_pack};

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

/// `ATLAS_PLE_PACKED_DIR`: where the interleaved copy lives, `None` to read
/// the table in place.
pub(super) fn packed_dir_from_env() -> Option<PathBuf> {
    packed_dir(std::env::var_os("ATLAS_PLE_PACKED_DIR"))
}

fn packed_dir(v: Option<std::ffi::OsString>) -> Option<PathBuf> {
    v.filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// Bytes one cache slot pins, for `ATLAS_PLE_CACHE_GB`: `row_bytes` of the
/// table, plus a split NVFP4 table's scale row, or the interleaved row.
pub(super) fn slot_bytes(nv: Option<&NgramNvfp4>, packed: bool, row_bytes: usize) -> usize {
    match nv {
        Some(nv) if packed => nv.geom.interleaved_stride(),
        Some(nv) => row_bytes + nv.geom.scale_stride(),
        None => row_bytes,
    }
}

/// The row cache over the interleaved copy of the table under `dir`, packed
/// first if `dir` holds no valid one. `shards` are the code shards.
pub(super) fn open_packed(
    nv: &NgramNvfp4,
    shards: &[(PathBuf, u64)],
    rows_per: u64,
    dir: &Path,
    slots: usize,
) -> Result<spark_storage::NgramRowCache> {
    let g = &nv.geom;
    let tag = format!("nvfp4g{}", g.group);
    let spec = PackSpec {
        planes: vec![
            Plane {
                shards,
                row_bytes: g.packed_stride(),
            },
            Plane {
                shards: &nv.scale_shards,
                row_bytes: g.scale_stride(),
            },
        ],
        rows_per_shard: rows_per,
        stride: g.interleaved_stride(),
        tag: &tag,
    };
    let verify = Verify::parse(std::env::var("ATLAS_PLE_PACKED_VERIFY").ok().as_deref());
    let table = open_or_pack(&spec, dir, verify).with_context(|| {
        format!(
            "PLE: packing the NVFP4 n-gram table into {} (ATLAS_PLE_PACKED_DIR must \
             be a writable directory with ~29 GB free, or hold a valid packed copy)",
            dir.display()
        )
    })?;
    if std::env::var_os("ATLAS_PLE_SCALES_RESIDENT").is_some() {
        tracing::warn!(
            "PLE: ATLAS_PLE_SCALES_RESIDENT is set and ignored: packed rows carry their scales"
        );
    }
    tracing::info!(
        "PLE n-gram table: NVFP4 read through packed {} ({} B rows, one NVMe read per miss)",
        table.path.display(),
        table.stride
    );
    table.open_row_cache(slots)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_dir_is_off_unless_set_to_a_path() {
        assert_eq!(packed_dir(None), None);
        assert_eq!(packed_dir(Some("".into())), None);
        assert_eq!(
            packed_dir(Some("/atlas-ple".into())),
            Some(std::path::PathBuf::from("/atlas-ple"))
        );
    }

    /// The cache budget counts what a slot pins: codes plus the scale arena's
    /// row when split, the whole padded row when interleaved, the row alone
    /// for every other table.
    #[test]
    fn slot_bytes_follow_the_row_layout() {
        let nv = NgramNvfp4 {
            geom: Nvfp4RowGeom {
                dim: 160,
                group: 16,
                global: 1.0,
            },
            scale_shards: Vec::new(),
        };
        assert_eq!(slot_bytes(Some(&nv), false, 80), 90);
        assert_eq!(slot_bytes(Some(&nv), true, 80), 96);
        assert_eq!(slot_bytes(None, false, 160), 160);
    }
}
