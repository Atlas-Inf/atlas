// SPDX-License-Identifier: AGPL-3.0-only

//! Expert TP (`ModelConfig::expert_tp`): this rank's Megatron slice of each
//! routed NVFP4 expert, read straight from the checkpoint. The fast loader
//! defers the packed weights and block scales, so no rank ever holds a whole
//! expert: gate/up keep rows `[r*I/ep, (r+1)*I/ep)` (column-parallel), down
//! keeps the matching input columns (row-parallel), and the per-tensor
//! scalars are shared by both slices. A CSF-compressed scale is decoded per
//! slice from its deferred components (`with_deferred_slice`).

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::fast_weights::{SliceAxis, with_deferred_slice};
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

/// This rank's slice of the deferred `[rows, row_bytes]` tensor `name`, read
/// from disk — or, for a CSF-compressed scale, decoded from its deferred
/// components one matrix at a time — and uploaded.
fn read_slice(
    store: &WeightStore,
    name: &str,
    rows: usize,
    row_bytes: usize,
    kind: TpShardKind,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let axis = match kind {
        TpShardKind::ColumnParallel => SliceAxis::Rows,
        TpShardKind::RowParallel => SliceAxis::Cols,
        TpShardKind::Replicated => anyhow::bail!("expert TP: {name} must be sliced"),
    };
    let (rank, world) = (config.ep_rank, config.ep_world_size);
    with_deferred_slice(store, name, rows, row_bytes, axis, rank, world, |bytes| {
        upload(gpu, bytes)
    })
    .with_context(|| format!("expert TP: slicing {name}"))
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let dst = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, dst)?;
    Ok(dst)
}
