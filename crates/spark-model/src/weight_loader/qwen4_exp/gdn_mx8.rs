// SPDX-License-Identifier: AGPL-3.0-only
//
//! GDN layers of nvidia/Qwen3.8-Flash-Next-FP4-Local, on their MXFP8 bytes.
//!
//! That checkpoint ships `in_proj_qkv`, `in_proj_z` and `out_proj` as MXFP8
//! (E4M3 + one E8M0 scale per 32 inputs), trained that way by NVIDIA's QAD,
//! and `weight_map::modelopt_local` uploads them unchanged. This builds the
//! layer around those bytes: `in_proj_qkv || in_proj_z` rows (and their
//! scale rows) become the sequential `[Q|K|V|Z]` projection, `out_proj` is
//! used in place, and the layer reads both through `mxfp8_proj.cu`. No BF16
//! copy exists; prefill expands one projection at a time into the scratch
//! `mx8_common::scratch_for` shares between every MXFP8 layer.
//!
//! `in_proj_a` / `in_proj_b` (4 MB a layer in all) stay on the BF16
//! interleaved β/α path the fused gate kernel reads.
//!
//! TP: HeadParallel, exactly the layout `build_linear_attention_dense_bf16`
//! produces — Q/K/V/Z sliced independently by the rank's contiguous head
//! range (`shard_mxfp8_rows` on `qkv_segments` + the single Z block),
//! `out_proj` row-parallel over whole 32-input scale groups
//! (`shard_mxfp8_row_parallel`), BA sliced on its group boundary, conv1d by
//! the QKV channel segments, a_log/dt_bias per value head, norm replicated.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::mx8_common::{MX8_GROUP, module};
use crate::layer::TransformerLayer;
use crate::layers::FfnComponent;
use crate::layers::Qwen3SsmLayer;
use crate::layers::ops::{Mxfp8Proj, Mxfp8ProjKernels, Mxfp8Scratch};
use crate::layers::qwen3_ssm::mx8::Mx8Gdn;
use crate::tp_shard::{
    TpGdnDims, shard_gdn_ba_rows, shard_gdn_conv_rows, shard_gdn_value_vector,
    shard_mxfp8_row_parallel, shard_mxfp8_rows,
};
use crate::weight_map::{
    DenseWeight, Mxfp8Weight, QuantizedWeight, SsmWeights, dense_auto, dense_f32_safe,
    dense_keep_f32, free_loader_source, interleave_ba,
};

/// True when layer `lp`'s GDN projections are MXFP8 in the store.
pub(super) fn present(store: &WeightStore, lp: &str) -> bool {
    let w = format!("{lp}.linear_attn.in_proj_qkv");
    store
        .get(&format!("{w}.weight"))
        .is_ok_and(|t| t.dtype == WeightDtype::FP8E4M3)
        && store.contains(&format!("{w}.weight_scale"))
}

/// Copy `a` then `b` (each `rows * row_bytes`) into one new buffer.
fn concat(
    gpu: &dyn GpuBackend,
    (a, a_rows): (DevicePtr, usize),
    (b, b_rows): (DevicePtr, usize),
    row_bytes: usize,
) -> Result<DevicePtr> {
    let buf = gpu.alloc((a_rows + b_rows) * row_bytes)?;
    gpu.copy_d2d(a, buf, a_rows * row_bytes)?;
    gpu.copy_d2d(b, buf.offset(a_rows * row_bytes), b_rows * row_bytes)?;
    Ok(buf)
}

/// Build GDN layer `layer_idx` on its MXFP8 projections.
#[allow(clippy::too_many_arguments)]
pub(super) fn build(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    scratch: Mxfp8Scratch,
) -> Result<Box<dyn TransformerLayer>> {
    let h = config.hidden_size;
    let dims = TpGdnDims::from_config(config);
    let tp = dims.tp_size;
    let (qkv_rows, z_rows) = (dims.full_conv_dim(), dims.full_value_dim());
    let (local_qkv, local_z) = (dims.local_conv_dim(), dims.local_value_dim());
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    ensure!(
        local_qkv + local_z == config.ssm_qkvz_size() && value_dim == local_z,
        "MXFP8 GDN layer {layer_idx}: local projection rows {local_qkv}+{local_z} do not match \
         the config's (TP-local) QKVZ width {}",
        config.ssm_qkvz_size()
    );
    let la = format!("{lp}.linear_attn");
    let qkv = module(store, &format!("{la}.in_proj_qkv"), qkv_rows, h)?;
    let z = module(store, &format!("{la}.in_proj_z"), z_rows, h)?;
    let out = module(store, &format!("{la}.out_proj"), h, z_rows)?;

    // Head range slice on data + scales: Q/K/V independently, Z as one block.
    let (qkv_l, qkv_lrows) = shard_mxfp8_rows(qkv, &dims.qkv_segments(), h, dims.tp_rank, tp, gpu)?;
    let (z_l, z_lrows) = shard_mxfp8_rows(z, &[z_rows], h, dims.tp_rank, tp, gpu)?;
    ensure!(
        qkv_lrows == local_qkv && z_lrows == local_z,
        "MXFP8 GDN layer {layer_idx}: local rows {qkv_lrows}+{z_lrows} != {local_qkv}+{local_z}"
    );
    let qkvz = Mxfp8Weight {
        data: concat(gpu, (qkv_l[0], local_qkv), (z_l[0], local_z), h)?,
        scales: concat(gpu, (qkv_l[1], local_qkv), (z_l[1], local_z), h / MX8_GROUP)?,
    };
    if tp > 1 {
        // `concat` built the kept [Q|K|V|Z] buffers; the fresh per-segment
        // slices are dead (~20 MB/layer). At tp=1 they alias the store
        // tensors, which `reclaim` below releases instead.
        for p in [qkv_l[0], qkv_l[1], z_l[0], z_l[1]] {
            let _ = gpu.free(p);
        }
    }
    let (out_l, out_lk) = shard_mxfp8_row_parallel(out, h, z_rows, dims.tp_rank, tp, gpu)?;
    ensure!(
        out_lk == local_z,
        "MXFP8 GDN layer {layer_idx}: out_proj local k {out_lk} != local value_dim {local_z}"
    );
    if tp > 1 {
        // The full modules are dead once the local slices exist.
        for part in ["in_proj_qkv", "in_proj_z", "out_proj"] {
            for t in ["weight", "weight_scale"] {
                store.reclaim(gpu, &format!("{la}.{part}.{t}"))?;
            }
        }
    } else {
        for part in ["in_proj_qkv", "in_proj_z"] {
            for t in ["weight", "weight_scale"] {
                store.reclaim(gpu, &format!("{la}.{part}.{t}"))?;
            }
        }
    }

    // β/α gates: BF16 (expanded at load), interleaved FULL heads then sliced
    // to the rank's contiguous key-head groups — same as the BF16 arm.
    let a = dense_auto(store, &format!("{la}.in_proj_a.weight"), gpu)?;
    let b = dense_auto(store, &format!("{la}.in_proj_b.weight"), gpu)?;
    let ba_full = interleave_ba(&a, &b, dims.full_nv, dims.full_nk, h, gpu)?;
    free_loader_source(store, gpu, &format!("{la}.in_proj_a.weight"), a)?;
    free_loader_source(store, gpu, &format!("{la}.in_proj_b.weight"), b)?;
    let (ba_ptr, _, _) = shard_gdn_ba_rows(ba_full.weight, &dims, gpu)?;
    if tp > 1 {
        let _ = gpu.free(ba_full.weight);
    }

    // conv1d follows the QKV channel segments; a_log/dt_bias slice per value
    // head; norm is the shared per-head-dim gain — REPLICATED (see the BF16
    // arm's comment on why slicing it overruns the buffer). Their full
    // sources stay store-owned as in the BF16 arm (KB-scale tensors).
    let d_conv = config.linear_conv_kernel_dim;
    let conv = dense_auto(store, &format!("{la}.conv1d.weight"), gpu)?;
    let (conv_ptr, _, _) = shard_gdn_conv_rows(conv.weight, &dims, d_conv, gpu)?;
    let a_log = dense_keep_f32(store, &format!("{la}.A_log"), gpu)?;
    let (a_log_ptr, _) = shard_gdn_value_vector(a_log.weight, &dims, 1, 4, gpu)?;
    let dt_bias = dense_keep_f32(store, &format!("{la}.dt_bias"), gpu)?;
    let (dt_bias_ptr, _) = shard_gdn_value_vector(dt_bias.weight, &dims, 1, 4, gpu)?;

    let ssm = SsmWeights {
        in_proj_qkvz: DenseWeight {
            weight: DevicePtr::NULL,
        },
        in_proj_ba: DenseWeight { weight: ba_ptr },
        conv1d: DenseWeight { weight: conv_ptr },
        // A_log and dt_bias stay FP32: the decay gate amplifies BF16 error.
        a_log: DenseWeight { weight: a_log_ptr },
        dt_bias: DenseWeight {
            weight: dt_bias_ptr,
        },
        norm: dense_f32_safe(store, &format!("{la}.norm.weight"), gpu)?,
        out_proj: QuantizedWeight::null(),
    };
    let mut layer = Qwen3SsmLayer::new_sequential(
        input_norm,
        ssm,
        post_attn_norm,
        ffn,
        None,
        None,
        None,
        config,
        gpu,
    )?;
    layer
        .set_mx8_projections(Mx8Gdn {
            qkvz: Mxfp8Proj {
                w: qkvz,
                n: config.ssm_qkvz_size() as u32,
                k: h as u32,
            },
            out: Mxfp8Proj {
                w: Mxfp8Weight {
                    data: out_l[0],
                    scales: out_l[1],
                },
                n: h as u32,
                k: local_z as u32,
            },
            kernels: Mxfp8ProjKernels::resolve(gpu)?,
            scratch,
        })
        .with_context(|| format!("MXFP8 GDN layer {layer_idx}"))?;
    Ok(Box::new(layer))
}
