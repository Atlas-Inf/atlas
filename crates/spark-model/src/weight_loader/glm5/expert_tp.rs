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
use crate::weight_map::{ExpertWeight, QuantizedWeight};
use std::sync::atomic::{AtomicUsize, Ordering};

/// NVFP4 values per FP8 block scale.
const NVFP4_GROUP: usize = 16;

/// The routed experts' projections: (name, n, k, how each is sliced).
const PROJECTIONS: [(&str, Dim, Dim, TpShardKind); 3] = [
    (
        "gate_proj",
        Dim::Inter,
        Dim::Hidden,
        TpShardKind::ColumnParallel,
    ),
    (
        "up_proj",
        Dim::Inter,
        Dim::Hidden,
        TpShardKind::ColumnParallel,
    ),
    (
        "down_proj",
        Dim::Hidden,
        Dim::Inter,
        TpShardKind::RowParallel,
    ),
];

#[derive(Clone, Copy)]
enum Dim {
    Hidden,
    Inter,
}

impl Dim {
    fn of(self, config: &ModelConfig) -> usize {
        match self {
            Self::Hidden => config.hidden_size,
            Self::Inter => config.moe_intermediate_size,
        }
    }
}

/// Disk readers per layer. Each slice is one synchronous O_DIRECT read
/// (and, for a CSF-compressed scale, a CPU decode), so one reader leaves
/// the NVMe queue nearly empty: 288 experts x 3 projections x 2 tensors
/// read one at a time took ~2 s a layer at TP4.
const READERS: usize = 8;

/// One projection's packed weight and block-scale bytes for this rank.
struct HostSlice {
    weight: Vec<u8>,
    scale: Vec<u8>,
}

/// One MoE layer's routed-expert slices for this rank in host memory, read
/// ahead of their upload ([`read_layer_experts`]); ~1.9 GB at TP4.
pub(super) struct HostExperts(Vec<[HostSlice; 3]>);

/// This rank's slices of every routed expert under `{mlp}.experts`, read on
/// [`READERS`] threads.
pub(super) fn read_layer_experts(
    store: &WeightStore,
    mlp: &str,
    config: &ModelConfig,
) -> Result<HostExperts> {
    let jobs = config.num_experts * PROJECTIONS.len();
    let next = AtomicUsize::new(0);
    let read = || -> Vec<(usize, Result<HostSlice>)> {
        let mut done = Vec::new();
        loop {
            let j = next.fetch_add(1, Ordering::Relaxed);
            if j >= jobs {
                return done;
            }
            let (expert, proj) = (j / PROJECTIONS.len(), j % PROJECTIONS.len());
            let (name, n, k, kind) = PROJECTIONS[proj];
            let prefix = format!("{mlp}.experts.{expert}.{name}");
            let slice = read_projection(store, &prefix, n.of(config), k.of(config), kind, config)
                .with_context(|| format!("GLM-5 expert {expert} {name} (expert TP)"));
            done.push((j, slice));
        }
    };
    let mut slices: Vec<Option<Result<HostSlice>>> = (0..jobs).map(|_| None).collect();
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..READERS).map(|_| scope.spawn(read)).collect();
        for worker in workers {
            for (j, slice) in worker.join().expect("expert slice reader panicked") {
                slices[j] = Some(slice);
            }
        }
    });
    let mut slices = slices.into_iter().map(|s| s.expect("every job ran"));
    let experts = (0..config.num_experts)
        .map(|_| -> Result<[HostSlice; 3]> {
            Ok([
                slices.next().unwrap()?,
                slices.next().unwrap()?,
                slices.next().unwrap()?,
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(HostExperts(experts))
}

/// The routed experts under `{mlp}.experts` from their host slices, uploaded
/// in expert and projection order (the allocations and bytes of reading and
/// uploading them one by one).
pub(super) fn upload_experts(
    store: &WeightStore,
    mlp: &str,
    host: HostExperts,
    gpu: &dyn GpuBackend,
) -> Result<Vec<ExpertWeight>> {
    host.0
        .iter()
        .enumerate()
        .map(|(expert, slices)| {
            let upload = |proj: usize| {
                let (name, ..) = PROJECTIONS[proj];
                upload_projection(
                    store,
                    &format!("{mlp}.experts.{expert}.{name}"),
                    &slices[proj],
                    gpu,
                )
                .with_context(|| format!("GLM-5 expert {expert} {name} (expert TP)"))
            };
            Ok(ExpertWeight {
                gate_proj: upload(0)?,
                up_proj: upload(1)?,
                down_proj: upload(2)?,
            })
        })
        .collect()
}

/// This rank's slice of the NVFP4 projection `{prefix}` of full shape
/// `[n, k]`, as host bytes.
fn read_projection(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    kind: TpShardKind,
    config: &ModelConfig,
) -> Result<HostSlice> {
    let slice = |suffix: &str, row_bytes: usize| {
        read_slice(
            store,
            &format!("{prefix}.{suffix}"),
            n,
            row_bytes,
            kind,
            config,
        )
    };
    Ok(HostSlice {
        weight: slice("weight", k / 2)?,
        scale: slice("weight_scale", k / NVFP4_GROUP)?,
    })
}

/// The projection `{prefix}` from its host slice: the packed weight, then
/// the block scale uploaded, and the per-tensor scalars shared by every
/// rank's slice.
fn upload_projection(
    store: &WeightStore,
    prefix: &str,
    host: &HostSlice,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    let input_scale = format!("{prefix}.input_scale");
    Ok(QuantizedWeight {
        weight: upload(gpu, &host.weight)?,
        weight_scale: upload(gpu, &host.scale)?,
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
/// components one matrix at a time.
fn read_slice(
    store: &WeightStore,
    name: &str,
    rows: usize,
    row_bytes: usize,
    kind: TpShardKind,
    config: &ModelConfig,
) -> Result<Vec<u8>> {
    let axis = match kind {
        TpShardKind::ColumnParallel => SliceAxis::Rows,
        TpShardKind::RowParallel => SliceAxis::Cols,
        TpShardKind::Replicated => anyhow::bail!("expert TP: {name} must be sliced"),
    };
    let (rank, world) = (config.ep_rank, config.ep_world_size);
    with_deferred_slice(store, name, rows, row_bytes, axis, rank, world, |bytes| {
        Ok(bytes.to_vec())
    })
    .with_context(|| format!("expert TP: slicing {name}"))
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let dst = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, dst)?;
    Ok(dst)
}
