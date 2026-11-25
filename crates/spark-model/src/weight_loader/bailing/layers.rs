// SPDX-License-Identifier: AGPL-3.0-only

//! `BailingWeightLoader::load_layers`: build the Ling-3.2-air layer stack
//! from the validated checkpoint plan (PR #115's name mapping) plus the
//! family driver (`crate::model::bailing_c1`). Weight upload + derived-copy
//! construction lives here; the launch code lives with the layers.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::div_ceil;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layer::TransformerLayer;
use crate::model::bailing_c1::{
    FfnKind, KDA_HEAD_DIM, KDA_HEADS, LingDenseWeights, LingExpertTable, LingFfn, LingKdaLayer,
    LingKdaWeights, LingKernels, LingMlaLayer, LingMlaWeights, LingMoeWeights, LingScratch,
    LingSpec,
};
use crate::weight_map::{DenseWeight, dense, scalar_f32};

const T: usize = KDA_HEADS * KDA_HEAD_DIM;

#[allow(clippy::too_many_arguments)]
pub(super) fn load_all(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
    plan: &super::plan::CheckpointPlan,
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let spec = LingSpec::from_config(config)?;
    let _ = layer_kv_dtypes;
    let _ = plan;
    let kernels = Arc::new(LingKernels::resolve(gpu)?);
    // The block-table staging buffer must hold the longest table the house
    // pool can issue: assume block_size >= 8 (the runtime checks uploads).
    let max_blocks = LingScratch::max_blocks(config.max_position_embeddings, 8);
    let inter = spec.dense_inter.max(spec.moe_inter.max(spec.shared_inter));
    let scratch = LingScratch::arc(
        gpu,
        spec.prefill_rows,
        spec.hidden,
        spec.experts,
        spec.top_k,
        inter,
        max_blocks,
    )?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    let mut attn_idx = 0usize;
    for (idx, layer_spec) in spec.layers.iter().enumerate() {
        let lp = format!("model.layers.{idx}");
        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
        let ffn = load_ffn(store, &lp, *layer_spec, &spec, gpu, &kernels, &scratch)?;
        let layer: Box<dyn TransformerLayer> =
            if layer_spec.attention == atlas_core::config::LayerType::LinearAttention {
                let weights = load_kda_weights(store, &lp, gpu)?;
                Box::new(LingKdaLayer::new(
                    input_norm,
                    post_attn_norm,
                    weights,
                    ffn,
                    Arc::clone(&kernels),
                    Arc::clone(&scratch),
                    spec.lower_bound,
                    spec.eps,
                )?)
            } else {
                let weights = load_mla_weights(store, &lp, gpu)?;
                let layer = LingMlaLayer::new(
                    input_norm,
                    post_attn_norm,
                    weights,
                    ffn,
                    Arc::clone(&kernels),
                    Arc::clone(&scratch),
                    attn_idx,
                    config,
                )?;
                attn_idx += 1;
                Box::new(layer)
            };
        layers.push(layer);
    }
    ensure!(
        attn_idx == spec.mla_layers,
        "ling: built {attn_idx} MLA layers, the plan has {}",
        spec.mla_layers
    );
    Ok(layers)
}

fn f32_raw(store: &WeightStore, name: &str, elems: usize) -> Result<DevicePtr> {
    let t = store.get(name).with_context(|| format!("missing {name}"))?;
    ensure!(
        t.dtype == WeightDtype::FP32 && t.num_elements() == elems,
        "ling: {name} must be FP32 with {elems} elements, got {:?} x {}",
        t.dtype,
        t.num_elements()
    );
    Ok(t.ptr)
}

/// FP32 weight, promoted from BF16 storage when the checkpoint stores it
/// that way (the primitive-ai NVFP4 build ships the router gate and expert
/// bias in BF16; the config's `router_dtype: fp32` describes the routing
/// math, not the on-disk tensor). Load-time promote via the common
/// `bf16_to_f32` kernel keeps every kernel contract (FP32 router GEMV)
/// untouched.
fn f32_or_upcast(
    store: &WeightStore,
    name: &str,
    elems: usize,
    gpu: &dyn GpuBackend,
    kernels: &std::sync::Arc<LingKernels>,
) -> Result<DevicePtr> {
    let t = store.get(name).with_context(|| format!("missing {name}"))?;
    ensure!(
        t.num_elements() == elems,
        "ling: {name} must have {elems} elements, got {}",
        t.num_elements()
    );
    match t.dtype {
        WeightDtype::FP32 => Ok(t.ptr),
        WeightDtype::BF16 => {
            let dst = gpu
                .alloc(elems * 4)
                .with_context(|| format!("ling: alloc f32 promote for {name}"))?;
            spark_runtime::kernel_args::KernelLaunch::new(gpu, kernels.bf16_to_f32_k)
                .grid([div_ceil(elems as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(t.ptr)
                .arg_ptr(dst)
                .arg_u32(elems as u32)
                .launch(gpu.default_stream())
                .with_context(|| format!("ling: promote {name} bf16->f32"))?;
            gpu.synchronize(gpu.default_stream())?;
            Ok(dst)
        }
        other => anyhow::bail!("ling: {name} must be FP32 or BF16, got {other:?}"),
    }
}

fn load_kda_weights(store: &WeightStore, lp: &str, gpu: &dyn GpuBackend) -> Result<LingKdaWeights> {
    let attn = |stem: &str| format!("{lp}.attention.{stem}.weight");
    // Fused conv filter blob [3T, conv_k]: the q|k|v conv1d weights back
    // to back (each [T, 4] — the common conv kernels read one contiguous
    // [channels, d_conv] slab).
    let conv_bytes = 3 * T * 4 * 2;
    let conv_blob = gpu.alloc(conv_bytes)?;
    for (plane, stem) in ["q_conv1d", "k_conv1d", "v_conv1d"].iter().enumerate() {
        let name = attn(stem);
        let t = store.get(&name)?;
        ensure!(
            t.dtype == WeightDtype::BF16 && t.num_elements() == T * 4,
            "ling: {name} must be BF16 [{} x 4], got {:?} x {}",
            T,
            t.dtype,
            t.num_elements()
        );
        gpu.copy_d2d(t.ptr, conv_blob.offset(plane * T * 4 * 2), T * 4 * 2)?;
    }
    Ok(LingKdaWeights {
        q_proj: dense(store, &attn("q_proj"))?,
        k_proj: dense(store, &attn("k_proj"))?,
        v_proj: dense(store, &attn("v_proj"))?,
        b_proj: dense(store, &attn("b_proj"))?,
        f_proj: dense(store, &attn("f_proj"))?,
        g_proj: dense(store, &attn("g_proj"))?,
        o_proj: dense(store, &attn("o_proj"))?,
        conv: DenseWeight { weight: conv_blob },
        a_log: f32_raw(store, &format!("{lp}.attention.A_log"), KDA_HEADS)?,
        dt_bias: f32_raw(store, &format!("{lp}.attention.dt_bias"), T)?,
        o_norm: dense(store, &attn("o_norm"))?,
    })
}

fn load_mla_weights(store: &WeightStore, lp: &str, gpu: &dyn GpuBackend) -> Result<LingMlaWeights> {
    let attn = |stem: &str| format!("{lp}.attention.{stem}.weight");
    let kv_b = dense(store, &attn("kv_b_proj"))?;
    Ok(LingMlaWeights {
        q_proj: dense(store, &attn("q_proj"))?,
        kv_a: dense(store, &attn("kv_a_proj_with_mqa"))?,
        kv_a_norm: dense(store, &attn("kv_a_layernorm"))?,
        g_proj: dense(store, &attn("g_proj"))?,
        o_proj: dense(store, &attn("dense"))?,
        w_uk: build_w_uk(gpu, kv_b.weight)?,
        kv_b,
    })
}

/// The load-time `w_uk` transpose: `[heads, 512, 128]` BF16 with
/// `w_uk[h][n][j] == kv_b_proj[h * 256 + j][n]` (the absorbed-q contract
/// of `ling_mla_q_absorb_rope`). One D2H + host shuffle + H2D of 16 MiB.
fn build_w_uk(gpu: &dyn GpuBackend, kv_b: DevicePtr) -> Result<DenseWeight> {
    let nope = 128usize;
    let v = 128usize;
    let lora = 512usize;
    let heads = 32usize;
    let src = kv_b;
    let src_bytes = heads * (nope + v) * lora * 2;
    let mut buf = vec![0u8; src_bytes];
    gpu.copy_d2h(src, &mut buf)?;
    let mut out = vec![0u8; heads * lora * nope * 2];
    for h in 0..heads {
        for n in 0..lora {
            for j in 0..nope {
                let s = ((h * (nope + v) + j) * lora + n) * 2;
                let d = ((h * lora + n) * nope + j) * 2;
                out[d..d + 2].copy_from_slice(&buf[s..s + 2]);
            }
        }
    }
    let dev = gpu.alloc(out.len())?;
    gpu.copy_h2d(&out, dev)?;
    Ok(DenseWeight { weight: dev })
}

#[allow(clippy::too_many_arguments)]
fn load_ffn(
    store: &WeightStore,
    lp: &str,
    layer_spec: crate::model::bailing_c1::LayerSpec,
    spec: &LingSpec,
    gpu: &dyn GpuBackend,
    kernels: &Arc<LingKernels>,
    scratch: &Arc<LingScratch>,
) -> Result<LingFfn> {
    match layer_spec.ffn {
        FfnKind::Dense => {
            let p = format!("{lp}.mlp");
            Ok(LingFfn::dense(
                LingDenseWeights {
                    gate: dense(store, &format!("{p}.gate_proj.weight"))?,
                    up: dense(store, &format!("{p}.up_proj.weight"))?,
                    down: dense(store, &format!("{p}.down_proj.weight"))?,
                },
                spec.dense_inter,
                spec.hidden,
                Arc::clone(scratch),
                Arc::clone(kernels),
            )?)
        }
        FfnKind::Moe => {
            let weights = load_moe_weights(store, lp, layer_spec, spec, gpu, kernels)?;
            LingFfn::moe(
                weights,
                spec.experts,
                spec.top_k,
                spec.n_group,
                spec.topk_group,
                spec.routed_scaling,
                spec.moe_inter,
                spec.hidden,
                Arc::clone(scratch),
                Arc::clone(kernels),
            )
        }
    }
}

/// Host-side pointer-array builder → device copy.
fn upload_u64s(gpu: &dyn GpuBackend, values: &[u64]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

fn upload_f32s(gpu: &dyn GpuBackend, values: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

/// One routed-expert projection's device tables: the checkpoint's
/// compressed-tensors packed `[N, K/2]` / scales `[N, K/16]` bytes
/// repacked to the GLM-native NVFP4 tile layout `[K/2, N]` / `[K/16, N]`
/// (`ling_moe_tile.cuh`), one contiguous slab + per-expert pointer tables.
#[allow(clippy::too_many_arguments)]
fn load_expert_projection(
    store: &WeightStore,
    moe: &str,
    proj: &str,
    spec: &LingSpec,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    kernels: &LingKernels,
) -> Result<(LingExpertTable, Vec<String>)> {
    let experts = spec.experts;
    // gate/up: N = moe_inter, K = hidden; down: N = hidden, K = moe_inter.
    let packed_rows = n;
    let packed_cols = k / 2;
    let scale_cols = k / 16;
    let src_packed: Vec<u64> = (0..experts)
        .map(|e| {
            Ok(store
                .get(&format!("{moe}.experts.{e}.{proj}.weight_packed"))?
                .ptr
                .0)
        })
        .collect::<Result<Vec<_>>>()?;
    let src_scale: Vec<u64> = (0..experts)
        .map(|e| {
            Ok(store
                .get(&format!("{moe}.experts.{e}.{proj}.weight_scale"))?
                .ptr
                .0)
        })
        .collect::<Result<Vec<_>>>()?;
    let packed_slab = gpu.alloc(experts * packed_rows * packed_cols)?;
    let scale_slab = gpu.alloc(experts * packed_rows * scale_cols)?;
    let dst_packed: Vec<u64> = (0..experts)
        .map(|e| packed_slab.offset(e * packed_rows * packed_cols).0)
        .collect();
    let dst_scale: Vec<u64> = (0..experts)
        .map(|e| scale_slab.offset(e * packed_rows * scale_cols).0)
        .collect();
    let src_packed_dev = upload_u64s(gpu, &src_packed)?;
    let src_scale_dev = upload_u64s(gpu, &src_scale)?;
    let dst_packed_dev = upload_u64s(gpu, &dst_packed)?;
    let dst_scale_dev = upload_u64s(gpu, &dst_scale)?;
    // transpose: [rows=n, cols] -> [cols, rows] (the [K/2, N] tile layout).
    for (src, dst, rows, cols) in [
        (
            src_packed_dev,
            dst_packed_dev,
            packed_rows as u32,
            packed_cols as u32,
        ),
        (
            src_scale_dev,
            dst_scale_dev,
            packed_rows as u32,
            scale_cols as u32,
        ),
    ] {
        spark_runtime::kernel_args::KernelLaunch::new(gpu, kernels.moe_transpose_batched_k)
            .grid([div_ceil(cols, 32), div_ceil(rows, 32), experts as u32])
            .block([32, 8, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u32(rows)
            .arg_u32(cols)
            .launch(gpu.default_stream())?;
    }
    gpu.synchronize(gpu.default_stream())?;
    // The scale2 fold of this projection: input_global_scale of the
    // ACTIVATION it is about to read, divided by each expert's
    // weight_global_scale (gate/up: the route-quantized activation's
    // global; down: the gate/up output's own global).
    let mut scale2 = Vec::with_capacity(experts);
    for e in 0..experts {
        let wgs = scalar_f32(
            store,
            &format!("{moe}.experts.{e}.{proj}.weight_global_scale"),
            gpu,
        )?;
        let ig = if proj == "down_proj" {
            scalar_f32(
                store,
                &format!("{moe}.experts.{e}.{proj}.input_global_scale"),
                gpu,
            )?
        } else {
            // gate/up share the route-quantized activation scale (A).
            activation_scale(store, moe, e, proj, gpu)?
        };
        ensure!(
            wgs.is_finite() && wgs > 0.0,
            "ling: expert {e} {proj} weight_global_scale must be positive"
        );
        ensure!(
            ig.is_finite() && ig > 0.0,
            "ling: expert {e} {proj} input_global_scale must be positive"
        );
        // The compressed-tensors NVFP4 convention (weight_map::quantized_v2's
        // verified finding; magnitudes confirmed on this build: wgs ~ 1.7e4,
        // ig ~ 2.5e3): BOTH checkpoint scalars are RECIPROCALS of the dequant
        // multipliers — w = E2M1 * E4M3_group / weight_global_scale and the
        // quantized activation/input dequantizes as q * sf / input_global_scale
        // (its static recipe quantizes x * ig / sf, saturating past the
        // calibrated amax 6 * 448 / ig). The MMA already carries both e4m3
        // group scales into acc, so the epilogue's fold is (1/ig) * (1/wgs).
        scale2.push(1.0 / (ig * wgs));
    }
    // The on-disk sources are dead once the transposes retired.
    let freed = (0..experts)
        .flat_map(|e| {
            [format!("{moe}.experts.{e}.{proj}.weight_packed")]
                .into_iter()
                .chain([format!("{moe}.experts.{e}.{proj}.weight_scale")])
        })
        .collect::<Vec<_>>();
    gpu.free(src_packed_dev)?;
    gpu.free(src_scale_dev)?;
    gpu.free(dst_packed_dev)?;
    gpu.free(dst_scale_dev)?;
    Ok((
        LingExpertTable {
            packed_ptrs: upload_u64s(gpu, &dst_packed)?,
            scale_ptrs: upload_u64s(gpu, &dst_scale)?,
            scale2_vals: upload_f32s(gpu, &scale2)?,
        },
        freed,
    ))
}

/// The gate/up activation scale of expert `e` — both projections' input
/// globals must agree (the route kernel quantizes once with one scalar).
fn activation_scale(
    store: &WeightStore,
    moe: &str,
    e: usize,
    proj: &str,
    gpu: &dyn GpuBackend,
) -> Result<f32> {
    let gate = scalar_f32(
        store,
        &format!("{moe}.experts.{e}.gate_proj.input_global_scale"),
        gpu,
    )?;
    let up = scalar_f32(
        store,
        &format!("{moe}.experts.{e}.up_proj.input_global_scale"),
        gpu,
    )?;
    let mine = scalar_f32(
        store,
        &format!("{moe}.experts.{e}.{proj}.input_global_scale"),
        gpu,
    )?;
    ensure!(
        gate == up && up == mine,
        "ling MoE: expert {e} gate/up input_global_scale disagree \
         ({gate:?} / {up:?}); the route kernel quantizes the activation once, \
         so per-expert activation scales need the split-quant seam"
    );
    Ok(gate)
}

#[allow(clippy::too_many_arguments)]
fn load_moe_weights(
    store: &WeightStore,
    lp: &str,
    layer_spec: crate::model::bailing_c1::LayerSpec,
    spec: &LingSpec,
    gpu: &dyn GpuBackend,
    kernels: &Arc<LingKernels>,
) -> Result<LingMoeWeights> {
    let moe = format!("{lp}.mlp");
    let router = f32_or_upcast(
        store,
        &format!("{moe}.gate.weight"),
        spec.experts * spec.hidden,
        gpu,
        kernels,
    )?;
    let bias = f32_or_upcast(
        store,
        &format!("{moe}.gate.expert_bias"),
        spec.experts,
        gpu,
        kernels,
    )?;
    let (gate, mut freed) = load_expert_projection(
        store,
        &moe,
        "gate_proj",
        spec,
        spec.moe_inter,
        spec.hidden,
        gpu,
        kernels,
    )?;
    let (up, mut freed_up) = load_expert_projection(
        store,
        &moe,
        "up_proj",
        spec,
        spec.moe_inter,
        spec.hidden,
        gpu,
        kernels,
    )?;
    freed.append(&mut freed_up);
    let (down, mut freed_down) = load_expert_projection(
        store,
        &moe,
        "down_proj",
        spec,
        spec.hidden,
        spec.moe_inter,
        gpu,
        kernels,
    )?;
    freed.append(&mut freed_down);
    // a2_gscale[e] = input_global_scale(down): the gate/up epilogue's
    // re-quantizer (sf = e4m3(amax * g / 6), nibble = e2m1(x * g / sf))
    // encodes the checkpoint's recipe with g = ig, so the packed
    // intermediate's dequant global is its reciprocal 1 / ig (see the
    // fold comment in load_expert_projection).
    let mut a2 = Vec::with_capacity(spec.experts);
    let mut a = None;
    for e in 0..spec.experts {
        let ig_down = scalar_f32(
            store,
            &format!("{moe}.experts.{e}.down_proj.input_global_scale"),
            gpu,
        )?;
        let gate = activation_scale(store, &moe, e, "gate_proj", gpu)?;
        a = match a {
            None => Some(gate),
            Some(prev) => {
                ensure!(
                    prev == gate,
                    "ling MoE: the route kernel quantizes with ONE activation \
                     global; expert {e} disagrees"
                );
                Some(prev)
            }
        };
        ensure!(
            ig_down.is_finite() && ig_down > 0.0,
            "ling: bad down input_global_scale"
        );
        a2.push(ig_down);
    }
    // The route quantizer's dequant global of the packed input is the
    // RECIPROCAL of the checkpoint's gate/up input_global_scale (see the
    // fold comment in load_expert_projection).
    let a_scale2 = 1.0 / a.expect("experts > 0");
    for name in &freed {
        let _ = store.reclaim(gpu, name);
    }
    let shared = format!("{moe}.shared_experts");
    let swiglu_routed = if layer_spec.routed_swiglu_limit > 0 {
        layer_spec.routed_swiglu_limit as f32
    } else {
        0.0
    };
    let swiglu_shared = if layer_spec.shared_swiglu_limit > 0 {
        layer_spec.shared_swiglu_limit as f32
    } else {
        0.0
    };
    Ok(LingMoeWeights {
        router,
        bias,
        gate,
        up,
        down,
        a2_gscale: upload_f32s(gpu, &a2)?,
        a_scale2,
        shared_gate: dense(store, &format!("{shared}.gate_proj.weight"))?,
        shared_up: dense(store, &format!("{shared}.up_proj.weight"))?,
        shared_down: dense(store, &format!("{shared}.down_proj.weight"))?,
        swiglu_routed,
        swiglu_shared,
    })
}

#[cfg(test)]
#[path = "layers_moe_parity_gpu.rs"]
mod layers_moe_parity_gpu;

#[cfg(test)]
#[path = "layers_dense_parity_gpu.rs"]
mod layers_dense_parity_gpu;
