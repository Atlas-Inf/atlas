// SPDX-License-Identifier: AGPL-3.0-only

//! Expert TP (`ModelConfig::expert_tp`): this rank's Megatron slice of each
//! routed NVFP4 expert, read straight from the checkpoint. The fast loader
//! defers the packed weights and block scales, so no rank ever holds a whole
//! expert: gate/up keep rows `[r*I/ep, (r+1)*I/ep)` (column-parallel), down
//! keeps the matching input columns (row-parallel), and the per-tensor
//! scalars are shared by both slices.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::fast_weights::with_deferred_bytes;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::tp_shard::TpShardKind;
use crate::weight_map::QuantizedWeight;

/// NVFP4 values per FP8 block scale.
const NVFP4_GROUP: usize = 16;

/// This rank's slice of the NVFP4 projection `{prefix}` of full shape `[n, k]`.
pub(super) fn load_expert_slice(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    kind: TpShardKind,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    let slice = |suffix: &str, row_bytes: usize| {
        read_slice(
            store,
            &format!("{prefix}.{suffix}"),
            n,
            row_bytes,
            kind,
            config,
            gpu,
        )
    };
    let input_scale = format!("{prefix}.input_scale");
    Ok(QuantizedWeight {
        weight: slice("weight", k / 2)?,
        weight_scale: slice("weight_scale", k / NVFP4_GROUP)?,
        weight_scale_2: crate::weight_map::scalar_f32(
            store,
            &format!("{prefix}.weight_scale_2"),
            gpu,
        )?,
        input_scale: if store.contains(&input_scale) {
            crate::weight_map::ptr(store, &input_scale)?
        } else {
            DevicePtr::NULL
        },
        weight_scale_2_vec: DevicePtr::NULL,
    })
}

fn read_slice(
    store: &WeightStore,
    name: &str,
    rows: usize,
    row_bytes: usize,
    kind: TpShardKind,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let d = store
        .deferred(name)
        .with_context(|| format!("expert TP: {name} was not deferred by the fast loader"))?;
    ensure!(
        d.shape.len() == 2 && d.shape[0] == rows && d.shape[1] * d.dtype.byte_size() == row_bytes,
        "expert TP: {name} has shape {:?}, expected [{rows}, {row_bytes} bytes]",
        d.shape
    );
    let (rank, world) = (config.ep_rank, config.ep_world_size);
    ensure!(
        rows.is_multiple_of(world) && row_bytes.is_multiple_of(world),
        "expert TP: {name} [{rows}, {row_bytes} bytes] does not split {world} ways"
    );
    match kind {
        TpShardKind::ColumnParallel => {
            let local = rows / world;
            with_deferred_bytes(d, rank * local * row_bytes, local * row_bytes, |bytes| {
                upload(gpu, bytes)
            })
        }
        TpShardKind::RowParallel => {
            let local = row_bytes / world;
            with_deferred_bytes(d, 0, rows * row_bytes, |bytes| {
                upload(gpu, &column_slice(bytes, row_bytes, rank, local))
            })
        }
        TpShardKind::Replicated => anyhow::bail!("expert TP: {name} must be sliced"),
    }
}

/// Bytes `[rank*local, (rank+1)*local)` of every `row_bytes`-wide row.
fn column_slice(bytes: &[u8], row_bytes: usize, rank: usize, local: usize) -> Vec<u8> {
    bytes
        .chunks_exact(row_bytes)
        .flat_map(|row| &row[rank * local..(rank + 1) * local])
        .copied()
        .collect()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let dst = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, dst)?;
    Ok(dst)
}

#[cfg(test)]
mod tests {
    #[test]
    fn column_slice_keeps_each_rows_rank_half() {
        let bytes: Vec<u8> = (0..12).collect(); // 3 rows x 4 bytes
        assert_eq!(super::column_slice(&bytes, 4, 0, 2), [0, 1, 4, 5, 8, 9]);
        assert_eq!(super::column_slice(&bytes, 4, 1, 2), [2, 3, 6, 7, 10, 11]);
    }
}
