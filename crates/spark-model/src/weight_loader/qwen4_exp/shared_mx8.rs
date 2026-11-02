// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_MXFP8_SHARED=1`: FP4-Local's MXFP8 shared expert as a
//! `layers::moe::Mxfp8SharedExpert`. Under EP each rank keeps its slice of
//! the intermediate: gate and up rows `[r * I/n, (r + 1) * I/n)`, packed as
//! `[gate_r; up_r]`, and the matching input columns of down (the reference's
//! column-parallel gate/up and row-parallel down). The store's full modules
//! are reclaimed once the slices exist.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::mx8_common::{self, MX8_GROUP};
use crate::layers::moe::Mxfp8SharedExpert;
use crate::layers::ops::{Mxfp8Proj, qwen4exp_w8a8};
use crate::tp_shard::{shard_mxfp8_row_parallel, shard_mxfp8_rows};
use crate::weight_map::Mxfp8Weight;

/// Layer `lp`'s shared expert, this EP rank's slice.
pub(super) fn load(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Mxfp8SharedExpert> {
    let (h, inter) = (config.hidden_size, config.shared_expert_intermediate_size);
    let (rank, size) = (config.ep_rank, config.ep_world_size.max(1));
    let name = |proj: &str| format!("{lp}.mlp.shared_expert.{proj}");
    let [gw, gs] = mx8_common::module(store, &name("gate_proj"), inter, h)?;
    let [uw, us] = mx8_common::module(store, &name("up_proj"), inter, h)?;
    let down = mx8_common::module(store, &name("down_proj"), h, inter)?;
    // `[gate; up]` whole, then this rank's rows of each half.
    let (sk, full_w, full_s) = (
        h / MX8_GROUP,
        gpu.alloc(2 * inter * h)?,
        gpu.alloc(2 * inter * (h / MX8_GROUP))?,
    );
    gpu.copy_d2d(gw, full_w, inter * h)?;
    gpu.copy_d2d(uw, full_w.offset(inter * h), inter * h)?;
    gpu.copy_d2d(gs, full_s, inter * sk)?;
    gpu.copy_d2d(us, full_s.offset(inter * sk), inter * sk)?;
    let mut reclaim = vec!["gate_proj", "up_proj"];
    let (gu, rows) = shard_mxfp8_rows([full_w, full_s], &[inter, inter], h, rank, size, gpu)?;
    let (down, local) = shard_mxfp8_row_parallel(down, h, inter, rank, size, gpu)?;
    if size > 1 {
        // The slices are copies: the whole `[gate; up]` and the store's down
        // are dead.
        gpu.free(full_w)?;
        gpu.free(full_s)?;
        reclaim.push("down_proj");
    }
    for proj in reclaim {
        for t in ["weight", "weight_scale"] {
            store.reclaim(gpu, &format!("{}.{t}", name(proj)))?;
        }
    }
    let proj = |[data, scales]: [_; 2], n: usize, k: usize| Mxfp8Proj {
        w: Mxfp8Weight {
            data,
            scales,
            tiled: false,
        },
        n: n as u32,
        k: k as u32,
    };
    let w8a8 = qwen4exp_w8a8::enabled().then_some(config.max_batch_tokens);
    Mxfp8SharedExpert::new(gpu, proj(gu, rows, h), proj(down, h, local), size > 1, w8a8)
}
