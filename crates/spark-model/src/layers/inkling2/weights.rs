// SPDX-License-Identifier: AGPL-3.0-only

//! Device handles for one physical Inkling 2 layer and for the model-level
//! block-residual and loop-message queries. Handles are plain device
//! pointers into the weight store (BF16/F32/U8/F8 as the checkpoint stores
//! them), so every visit of a physical layer copies the same struct. Derived at
//! load: the experts' per-expert global scales (`amax /
//! outer_scale_denominator`, F32 `[E]`).

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::weight_map::QuantizedWeight;

use super::ops::StackedExperts;
use spark_runtime::kernel_args::KernelLaunch;

const PREFIX: &str = "model.llm";
/// The Spark overlay's module names (transformers layout).
const SPARK: &str = "model.language_model";

fn ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    Ok(store
        .get(name)
        .with_context(|| format!("inkling2 tensor {name}"))?
        .ptr)
}

/// A dense projection: the checkpoint's BF16 `[N, K]`, or the Spark overlay's
/// NVFP4 (ModelOpt codes, E4M3 scales, per-tensor global) for decode and verify
/// rows with the checkpoint's BF16 kept for prefill rows.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Linear {
    Bf16(DevicePtr),
    Nvfp4 {
        q: QuantizedWeight,
        prefill: DevicePtr,
        /// AMD only: `q.weight`/`q.weight_scale` repacked into the WMMA tile
        /// replay `w4a16_wmma` reads (same bytes, permuted); NULL elsewhere or
        /// when the kernel is absent.
        wmma_packed: DevicePtr,
        wmma_scales: DevicePtr,
    },
}

/// Static (SWA) or dynamic (full layer) k/v short convolution.
#[derive(Clone, Copy, Debug)]
pub(crate) enum KvConv {
    /// `[C, 1, k]`
    Static { weight: DevicePtr },
    /// static taps `[k, C]` and the per-token tap projection `[kv_heads * k, H]`
    Dynamic {
        static_w: DevicePtr,
        proj: DevicePtr,
    },
}

/// The loop's message gate for one of the q, k, v or MLP inputs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MessageGate {
    pub down: DevicePtr,
    pub forget: DevicePtr,
    pub gain: DevicePtr,
}

/// Input norms: one for SWA layers, separate q/k/v norms for full layers.
#[derive(Clone, Copy, Debug)]
pub(crate) enum InputNorms {
    Shared(DevicePtr),
    Separate {
        q: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PhysicalWeights {
    pub input_norms: InputNorms,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wr: Linear,
    pub rel_proj: DevicePtr,
    /// `rel_proj` by distance for the prefill attention: on AMD (`atlas_scale`) F32
    /// `[rel_extent, d_rel]`, read as 16-byte loads; on gb10 the BF16 lanes of
    /// [`bias_lanes`], one 8-byte load per bias MMA fragment
    pub rel_proj_t: DevicePtr,
    pub q_scale: DevicePtr,
    pub rel_scale: DevicePtr,
    pub gate: Linear,
    pub wo: Linear,
    pub sink: DevicePtr,
    pub k_conv: KvConv,
    pub v_conv: KvConv,
    pub attn_conv: DevicePtr,
    pub mlp_conv: DevicePtr,
    pub mlp_norm: DevicePtr,
    pub router: DevicePtr,
    pub router_bias: DevicePtr,
    /// The routed experts decode and verify rows read: the Spark overlay's where it
    /// has them, else the checkpoint's
    pub experts: StackedExperts,
    /// The checkpoint's routed experts, which prefill rows read in both modes (at
    /// its declared numerics)
    pub prefill_experts: StackedExperts,
    /// q, k, v, mlp message gates (loop core only)
    pub messages: Option<[MessageGate; 4]>,
    /// Block-residual query for the attention input: `[H]` (SWA) or `[3, H]`
    /// (full: q, k, v inputs); NULL for layer 0.
    pub attn_query: DevicePtr,
    pub mlp_query: DevicePtr,
}

/// Model-level queries.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GlobalWeights {
    pub output_query: DevicePtr,
    /// `[2 * core, H]`: per loop layer the (q, mlp) message queries
    pub cross_pass_queries: DevicePtr,
    /// `[2 * core, H]`: per loop layer the (k, v) message queries
    pub cross_pass_kv_queries: DevicePtr,
}

pub(crate) fn global(store: &WeightStore) -> Result<GlobalWeights> {
    Ok(GlobalWeights {
        output_query: ptr(store, &format!("{PREFIX}.attn_res.output_query"))?,
        cross_pass_queries: ptr(store, &format!("{PREFIX}.middle_cycle_cross_pass_queries"))?,
        cross_pass_kv_queries: ptr(
            store,
            &format!("{PREFIX}.middle_cycle_cross_pass_kv_queries"),
        )?,
    })
}

/// A BF16 `[rows, cols]` tensor as F32 `[cols, rows]` in a new device array.
fn transposed_f32(store: &WeightStore, gpu: &dyn GpuBackend, name: &str) -> Result<DevicePtr> {
    let t = store
        .get(name)
        .with_context(|| format!("inkling2 tensor {name}"))?;
    let [rows, cols] = t.shape[..] else {
        anyhow::bail!("inkling2 {name}: shape {:?}", t.shape);
    };
    let mut raw = vec![0u8; rows * cols * 2];
    gpu.copy_d2h(t.ptr, &mut raw)?;
    let mut out = Vec::with_capacity(rows * cols * 4);
    for c in 0..cols {
        for r in 0..rows {
            let o = (r * cols + c) * 2;
            let bits = u32::from(u16::from_le_bytes([raw[o], raw[o + 1]])) << 16;
            out.extend_from_slice(&f32::from_bits(bits).to_le_bytes());
        }
    }
    let dev = gpu
        .alloc(out.len())
        .with_context(|| format!("allocate {name} transposed"))?;
    gpu.copy_h2d(&out, dev)?;
    Ok(dev)
}

/// `inkling2_flash_prefill`'s bias weights: BF16 `proj [16, extent]` regrouped by
/// distance, row `d` holding lane quad `q`'s m16n8k16 B fragment (`proj[2q][d]`,
/// `proj[2q + 1][d]`, `proj[2q + 8][d]`, `proj[2q + 9][d]`) for q = 0..4, then a zero
/// row at `extent` for the distances past the bias's reach.
pub(crate) fn bias_lanes(raw: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>> {
    ensure!(
        rows == 16 && raw.len() == rows * cols * 2,
        "inkling2 bias lanes: proj [{rows}, {cols}]"
    );
    let mut out = vec![0u8; (cols + 1) * rows * 2];
    for d in 0..cols {
        for q in 0..4 {
            for (i, k) in [2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9]
                .into_iter()
                .enumerate()
            {
                let (src, dst) = ((k * cols + d) * 2, (d * rows + q * 4 + i) * 2);
                out[dst..dst + 2].copy_from_slice(&raw[src..src + 2]);
            }
        }
    }
    Ok(out)
}

fn lanes_bf16(store: &WeightStore, gpu: &dyn GpuBackend, name: &str) -> Result<DevicePtr> {
    let t = store
        .get(name)
        .with_context(|| format!("inkling2 tensor {name}"))?;
    let [rows, cols] = t.shape[..] else {
        anyhow::bail!("inkling2 {name}: shape {:?}", t.shape);
    };
    let mut raw = vec![0u8; rows * cols * 2];
    gpu.copy_d2h(t.ptr, &mut raw)?;
    let out = bias_lanes(&raw, rows, cols).with_context(|| name.to_string())?;
    let dev = gpu
        .alloc(out.len())
        .with_context(|| format!("allocate {name} lanes"))?;
    gpu.copy_h2d(&out, dev)?;
    Ok(dev)
}

/// `amax[e] / denominator` as an F32 `[E]` device array.
fn expert_globals(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    name: &str,
    experts: usize,
) -> Result<DevicePtr> {
    let mut amax = vec![0u8; experts * 4];
    gpu.copy_d2h(ptr(store, &format!("{name}_amax"))?, &mut amax)?;
    let mut den = [0u8; 4];
    gpu.copy_d2h(
        ptr(store, &format!("{name}_outer_scale_denominator"))?,
        &mut den,
    )?;
    let den = f32::from_le_bytes(den);
    ensure!(
        den.is_finite() && den > 0.0,
        "inkling2 {name}: outer scale denominator {den}"
    );
    let globals: Vec<u8> = amax
        .chunks_exact(4)
        .flat_map(|c| (f32::from_le_bytes([c[0], c[1], c[2], c[3]]) / den).to_le_bytes())
        .collect();
    let dev = gpu
        .alloc(globals.len())
        .with_context(|| format!("allocate {name} globals"))?;
    gpu.copy_h2d(&globals, dev)?;
    Ok(dev)
}

/// Physical layer `lp`'s routed experts as the checkpoint ships them.
fn checkpoint_experts(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    lp: &str,
    mxfp8: bool,
    experts: usize,
) -> Result<StackedExperts> {
    let p = |name: &str| ptr(store, name);
    let ex = format!("{lp}.mlp.experts");
    Ok(StackedExperts {
        mxfp8,
        w13: p(&format!("{ex}.w13_weight"))?,
        w13_scale: p(&format!("{ex}.w13_weight_scale"))?,
        w13_global: if mxfp8 {
            DevicePtr::NULL
        } else {
            expert_globals(store, gpu, &format!("{ex}.w13_weight"), experts)?
        },
        w2: p(&format!("{ex}.w2_weight"))?,
        w2_scale: p(&format!("{ex}.w2_weight_scale"))?,
        w2_global: if mxfp8 {
            DevicePtr::NULL
        } else {
            expert_globals(store, gpu, &format!("{ex}.w2_weight"), experts)?
        },
    })
}

/// Spark overlay experts for physical layer `l` (`{SPARK}.layers.{l}.mlp.experts`,
/// ModelOpt NVFP4 with a per-expert `weight_scale_2`), or `None`. The overlay
/// keeps transformers' gate/up layout (all gate rows, then all up rows), so the
/// codes and scales are re-interleaved into the checkpoint's row order once.
fn spark_experts(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    l: usize,
) -> Result<Option<StackedExperts>> {
    let ex = format!("{SPARK}.layers.{l}.mlp.experts");
    if !store.contains(&format!("{ex}.gate_up_proj.weight")) {
        return Ok(None);
    }
    let t = |name: String| {
        store
            .get(&name)
            .with_context(|| format!("inkling2 tensor {name}"))
    };
    let interleave = |name: String| -> Result<DevicePtr> {
        let w = t(name)?;
        let [e, n, row_bytes] = w.shape[..] else {
            anyhow::bail!("inkling2 overlay gate_up shape {:?}", w.shape);
        };
        let out = gpu
            .alloc(e * n * row_bytes)
            .context("allocate inkling2 interleaved experts")?;
        let kernel = gpu.kernel("inkling2_moe_prefill", "inkling2_interleave_rows")?;
        KernelLaunch::new(gpu, kernel)
            .grid([u32::try_from(e * n)?, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(w.ptr)
            .arg_ptr(out)
            .arg_u32(u32::try_from(n / 2)?)
            .arg_u32(u32::try_from(row_bytes)?)
            .launch(0)?;
        gpu.synchronize(0)?;
        Ok(out)
    };
    Ok(Some(StackedExperts {
        mxfp8: false,
        w13: interleave(format!("{ex}.gate_up_proj.weight"))?,
        w13_scale: interleave(format!("{ex}.gate_up_proj.weight_scale"))?,
        w13_global: t(format!("{ex}.gate_up_proj.weight_scale_2"))?.ptr,
        w2: t(format!("{ex}.down_proj.weight"))?.ptr,
        w2_scale: t(format!("{ex}.down_proj.weight_scale"))?.ptr,
        w2_global: t(format!("{ex}.down_proj.weight_scale_2"))?.ptr,
    }))
}

/// AMD only: repack an NVFP4 projection into the `w4a16_wmma` tile replay
/// (N%128 and K%64 required; NULL,NULL otherwise or when the kernel is
/// absent). The pretile is the same bytes permuted, so it adds
/// `N*K/2 + N*K/16` device bytes beside `q`.
fn wmma_pretile(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    q: &str,
) -> Result<(DevicePtr, DevicePtr)> {
    if !cfg!(atlas_scale) {
        return Ok((DevicePtr::NULL, DevicePtr::NULL));
    }
    let w = store
        .get(&format!("{q}.weight"))
        .with_context(|| format!("inkling2 tensor {q}.weight"))?;
    let s = store
        .get(&format!("{q}.weight_scale"))
        .with_context(|| format!("inkling2 tensor {q}.weight_scale"))?;
    let n = w.shape[0];
    let k = w.shape[1] * 2;
    let kernel = crate::layers::try_kernel(gpu, "inkling2_w4a16_wmma", "w4a16_wmma_pretile");
    if kernel.0 == 0 || !n.is_multiple_of(128) || !k.is_multiple_of(64) {
        return Ok((DevicePtr::NULL, DevicePtr::NULL));
    }
    let pk = gpu
        .alloc(n * k / 2)
        .with_context(|| format!("allocate {q} wmma packed"))?;
    let sc = gpu
        .alloc(n * k / 16)
        .with_context(|| format!("allocate {q} wmma scales"))?;
    KernelLaunch::new(gpu, kernel)
        .grid([u32::try_from(n / 128 * (k / 64))?, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(w.ptr)
        .arg_ptr(s.ptr)
        .arg_ptr(pk)
        .arg_ptr(sc)
        .arg_u32(u32::try_from(n)?)
        .arg_u32(u32::try_from(k)?)
        .launch(0)?;
    gpu.synchronize(0)?;
    Ok((pk, sc))
}

/// Handles for physical layer `l` (layout already checked by the plan).
pub(crate) fn physical(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    l: usize,
    full: bool,
    looped: bool,
    mxfp8: bool,
    experts: usize,
) -> Result<PhysicalWeights> {
    let lp = format!("{PREFIX}.layers.{l}");
    let a = format!("{lp}.attn");
    let p = |name: &str| ptr(store, name);
    let conv = |kv: &str| -> Result<KvConv> {
        Ok(if full {
            KvConv::Dynamic {
                static_w: p(&format!("{a}.{kv}_sconv.static_w"))?,
                proj: p(&format!("{a}.{kv}_sconv.weight_proj"))?,
            }
        } else {
            KvConv::Static {
                weight: p(&format!("{a}.{kv}_sconv.weight"))?,
            }
        })
    };
    // Spark overlay: an attention projection quantized with the transformers
    // module names (`{SPARK}.layers.{l}.self_attn.*`, ModelOpt NVFP4) replaces
    // the checkpoint's BF16 tensor
    let linear = |ckpt: &str, spark: &str| -> Result<Linear> {
        let q = format!("{SPARK}.layers.{l}.self_attn.{spark}");
        let bf16 = p(&format!("{a}.{ckpt}.weight"))?;
        Ok(if store.contains(&format!("{q}.weight")) {
            let (wmma_packed, wmma_scales) = wmma_pretile(store, gpu, &q)?;
            Linear::Nvfp4 {
                q: crate::weight_map::quantized(store, &q, gpu)?,
                prefill: bf16,
                wmma_packed,
                wmma_scales,
            }
        } else {
            Linear::Bf16(bf16)
        })
    };
    let prefill_experts = checkpoint_experts(store, gpu, &lp, mxfp8, experts)?;
    let stacked = spark_experts(store, gpu, l)?.unwrap_or(prefill_experts);
    let messages = if looped {
        let gate = |c: &str| -> Result<MessageGate> {
            let cp = format!("{lp}.controllers.{c}");
            Ok(MessageGate {
                down: p(&format!("{cp}.down_weight"))?,
                forget: p(&format!("{cp}.forget_weight"))?,
                gain: p(&format!("{cp}.gain_weight"))?,
            })
        };
        Some([gate("attn")?, gate("k")?, gate("v")?, gate("mlp")?])
    } else {
        None
    };
    Ok(PhysicalWeights {
        input_norms: if full {
            InputNorms::Separate {
                q: p(&format!("{lp}.q_input_norm.weight"))?,
                k: p(&format!("{lp}.k_input_norm.weight"))?,
                v: p(&format!("{lp}.v_input_norm.weight"))?,
            }
        } else {
            InputNorms::Shared(p(&format!("{lp}.attn_norm.weight"))?)
        },
        wq: linear("wq_du", "q_proj.linear")?,
        wk: linear("wk_dv", "k_proj.linear")?,
        wv: linear("wv_dv", "v_proj.linear")?,
        wr: linear("wr_du", "relative_position_bias.linear")?,
        rel_proj: p(&format!("{a}.rel_logits_proj.proj"))?,
        rel_proj_t: if cfg!(atlas_scale) {
            transposed_f32(store, gpu, &format!("{a}.rel_logits_proj.proj"))?
        } else {
            lanes_bf16(store, gpu, &format!("{a}.rel_logits_proj.proj"))?
        },
        q_scale: p(&format!("{a}.q_l2_clip.scale"))?,
        rel_scale: p(&format!("{a}.rel_l2_clip.scale"))?,
        gate: linear("gate_attn", "o_proj.gate_proj")?,
        wo: linear("wo_ud", "o_proj.linear")?,
        sink: p(&format!("{a}.attn_sink_proj.weight"))?,
        k_conv: conv("k")?,
        v_conv: conv("v")?,
        attn_conv: p(&format!("{lp}.attn_sconv.weight"))?,
        mlp_conv: p(&format!("{lp}.mlp_sconv.weight"))?,
        mlp_norm: p(&format!("{lp}.mlp_norm.weight"))?,
        router: p(&format!("{lp}.mlp.gate.weight"))?,
        router_bias: p(&format!("{lp}.mlp.gate.bias"))?,
        experts: stacked,
        prefill_experts,
        messages,
        // checkpoint index N feeds layer N + 1's attention input
        attn_query: if l == 0 {
            DevicePtr::NULL
        } else {
            p(&format!("{PREFIX}.attn_res.attn_queries.{}", l - 1))?
        },
        mlp_query: p(&format!("{PREFIX}.attn_res.mlp_queries.{l}"))?,
    })
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
