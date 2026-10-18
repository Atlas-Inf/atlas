// SPDX-License-Identifier: AGPL-3.0-only

//! GDN layers of nvidia/Qwen3.8-Flash-Next-FP4-Local, on their MXFP8 bytes.
//!
//! That checkpoint ships `in_proj_qkv`, `in_proj_z` and `out_proj` as MXFP8
//! (E4M3 + one E8M0 scale per 32 inputs), trained that way by NVIDIA's QAD,
//! and `weight_map::modelopt_local` uploads them unchanged. This builds the
//! layer around those bytes: `in_proj_qkv || in_proj_z` rows (and their
//! scale rows) become the sequential `[Q|K|V|Z]` projection, `out_proj` is
//! used in place, and the layer reads both through `mxfp8_proj.cu`. No BF16
//! copy exists; prefill expands one projection at a time into a scratch
//! shared by every layer.
//!
//! `in_proj_a` / `in_proj_b` (4 MB a layer in all) stay on the BF16
//! interleaved β/α path the fused gate kernel reads.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layer::TransformerLayer;
use crate::layers::FfnComponent;
use crate::layers::Qwen3SsmLayer;
use crate::layers::ops::{Mxfp8Proj, Mxfp8ProjKernels, Mxfp8Scratch};
use crate::layers::qwen3_ssm::mx8::Mx8Gdn;
use crate::tp_shard::TpGdnDims;
use crate::weight_map::{
    DenseWeight, Mxfp8Weight, QuantizedWeight, SsmWeights, dense_auto, dense_f32_safe,
    dense_keep_f32, free_loader_source, interleave_ba,
};

const MX8_GROUP: usize = 32;

/// True when layer `lp`'s GDN projections are MXFP8 in the store.
pub(super) fn present(store: &WeightStore, lp: &str) -> bool {
    let w = format!("{lp}.linear_attn.in_proj_qkv");
    store
        .get(&format!("{w}.weight"))
        .is_ok_and(|t| t.dtype == WeightDtype::FP8E4M3)
        && store.contains(&format!("{w}.weight_scale"))
}

/// The weight and scale buffers of one MXFP8 module, shapes checked.
fn module(store: &WeightStore, name: &str, rows: usize, k: usize) -> Result<[DevicePtr; 2]> {
    let w = store.get(&format!("{name}.weight"))?;
    let s = store.get(&format!("{name}.weight_scale"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3
            && w.shape == [rows, k]
            && s.dtype == WeightDtype::UInt8
            && s.shape == [rows, k / MX8_GROUP],
        "{name}: expected MXFP8 [{rows}, {k}] + [{rows}, {}], got {:?} {:?} + {:?} {:?}",
        k / MX8_GROUP,
        w.dtype,
        w.shape,
        s.dtype,
        s.shape
    );
    Ok([w.ptr, s.ptr])
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

/// The prefill scratch every MXFP8 GDN layer shares: sized for the larger
/// projection, allocated by the first layer that needs it.
pub(super) fn scratch_for(
    shared: &mut Option<Mxfp8Scratch>,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Mxfp8Scratch> {
    let h = config.hidden_size;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let bytes = (config.ssm_qkvz_size() * h).max(h * value_dim) * 2;
    if let Some(s) = shared.filter(|s| s.bytes >= bytes) {
        return Ok(s);
    }
    ensure!(shared.is_none(), "MXFP8 GDN scratch sized twice");
    let s = Mxfp8Scratch {
        ptr: gpu.alloc(bytes)?,
        bytes,
    };
    tracing::info!(
        "MXFP8 GDN: one {:.1} MB BF16 prefill scratch shared by every GDN layer",
        bytes as f64 / 1e6
    );
    *shared = Some(s);
    Ok(s)
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
    ensure!(
        config.tp_world_size.max(1) == 1,
        "MXFP8 GDN layer {layer_idx}: tensor parallelism is not supported yet"
    );
    let h = config.hidden_size;
    let dims = TpGdnDims::from_config(config);
    let (qkv_rows, z_rows) = (dims.full_conv_dim(), dims.full_value_dim());
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    ensure!(
        qkv_rows + z_rows == config.ssm_qkvz_size() && value_dim == z_rows,
        "MXFP8 GDN layer {layer_idx}: projection rows {qkv_rows}+{z_rows} do not match \
         the config's QKVZ width {}",
        config.ssm_qkvz_size()
    );
    let la = format!("{lp}.linear_attn");
    let [qkv_w, qkv_s] = module(store, &format!("{la}.in_proj_qkv"), qkv_rows, h)?;
    let [z_w, z_s] = module(store, &format!("{la}.in_proj_z"), z_rows, h)?;
    let [out_w, out_s] = module(store, &format!("{la}.out_proj"), h, value_dim)?;

    let qkvz = Mxfp8Weight {
        data: concat(gpu, (qkv_w, qkv_rows), (z_w, z_rows), h)?,
        scales: concat(gpu, (qkv_s, qkv_rows), (z_s, z_rows), h / MX8_GROUP)?,
    };
    for part in ["in_proj_qkv", "in_proj_z"] {
        for t in ["weight", "weight_scale"] {
            store.reclaim(gpu, &format!("{la}.{part}.{t}"))?;
        }
    }

    // β/α gates: BF16 (expanded at load), interleaved per key-head group.
    let a = dense_auto(store, &format!("{la}.in_proj_a.weight"), gpu)?;
    let b = dense_auto(store, &format!("{la}.in_proj_b.weight"), gpu)?;
    let ba = interleave_ba(&a, &b, dims.full_nv, dims.full_nk, h, gpu)?;
    free_loader_source(store, gpu, &format!("{la}.in_proj_a.weight"), a)?;
    free_loader_source(store, gpu, &format!("{la}.in_proj_b.weight"), b)?;

    let ssm = SsmWeights {
        in_proj_qkvz: DenseWeight {
            weight: DevicePtr::NULL,
        },
        in_proj_ba: ba,
        conv1d: dense_auto(store, &format!("{la}.conv1d.weight"), gpu)?,
        // A_log and dt_bias stay FP32: the decay gate amplifies BF16 error.
        a_log: dense_keep_f32(store, &format!("{la}.A_log"), gpu)?,
        dt_bias: dense_keep_f32(store, &format!("{la}.dt_bias"), gpu)?,
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
                    data: out_w,
                    scales: out_s,
                },
                n: h as u32,
                k: value_dim as u32,
            },
            kernels: Mxfp8ProjKernels::resolve(gpu)?,
            scratch,
        })
        .with_context(|| format!("MXFP8 GDN layer {layer_idx}"))?;
    Ok(Box::new(layer))
}
