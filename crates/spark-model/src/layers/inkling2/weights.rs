// SPDX-License-Identifier: AGPL-3.0-only

//! Device handles for one physical Inkling 2 layer and for the model-level
//! block-residual and loop-message queries. Handles are plain device
//! pointers into the weight store (BF16/F32/U8/F8 as the checkpoint stores
//! them), so every visit of a physical layer copies the same struct. Derived at
//! load: the experts' per-expert global scales (`amax /
//! outer_scale_denominator`, F32 `[E]`), the per-expert pointer tables the
//! grouped prefill GEMMs take, and a BF16 copy of each MXFP8 layer's experts
//! (exact; no grouped MXFP8 GEMM exists).

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightStore, WeightTensor};

use super::ops::{GroupedExperts, StackedExperts};
use spark_runtime::kernel_args::KernelLaunch;

const PREFIX: &str = "model.llm";

fn ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    Ok(store
        .get(name)
        .with_context(|| format!("inkling2 tensor {name}"))?
        .ptr)
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
    pub wq: DevicePtr,
    pub wk: DevicePtr,
    pub wv: DevicePtr,
    pub wr: DevicePtr,
    pub rel_proj: DevicePtr,
    pub q_scale: DevicePtr,
    pub rel_scale: DevicePtr,
    pub gate: DevicePtr,
    pub wo: DevicePtr,
    pub sink: DevicePtr,
    pub k_conv: KvConv,
    pub v_conv: KvConv,
    pub attn_conv: DevicePtr,
    pub mlp_conv: DevicePtr,
    pub mlp_norm: DevicePtr,
    pub router: DevicePtr,
    pub router_bias: DevicePtr,
    pub experts: StackedExperts,
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
    pub embed_norm: DevicePtr,
    pub output_query: DevicePtr,
    /// `[2 * core, H]`: per loop layer the (q, mlp) message queries
    pub cross_pass_queries: DevicePtr,
    /// `[2 * core, H]`: per loop layer the (k, v) message queries
    pub cross_pass_kv_queries: DevicePtr,
}

pub(crate) fn global(store: &WeightStore) -> Result<GlobalWeights> {
    Ok(GlobalWeights {
        embed_norm: ptr(store, &format!("{PREFIX}.embed_norm.weight"))?,
        output_query: ptr(store, &format!("{PREFIX}.attn_res.output_query"))?,
        cross_pass_queries: ptr(store, &format!("{PREFIX}.middle_cycle_cross_pass_queries"))?,
        cross_pass_kv_queries: ptr(
            store,
            &format!("{PREFIX}.middle_cycle_cross_pass_kv_queries"),
        )?,
    })
}

/// `[E]` device table of `base + e * stride` (no copy).
fn ptr_table(
    gpu: &dyn GpuBackend,
    base: DevicePtr,
    experts: usize,
    stride: usize,
) -> Result<DevicePtr> {
    let ptrs: Vec<u8> = (0..experts)
        .flat_map(|e| (base.0 + (e * stride) as u64).to_le_bytes())
        .collect();
    let dev = gpu
        .alloc(ptrs.len())
        .context("allocate expert pointer table")?;
    gpu.copy_h2d(&ptrs, dev)?;
    Ok(dev)
}

/// One stacked expert tensor `[E, N, K]` (MXFP8, with E8M0 scales `[E, N, K/32]`)
/// as BF16 `[E, N, K]`.
fn mxfp8_as_bf16(
    gpu: &dyn GpuBackend,
    codes: &WeightTensor,
    scales: DevicePtr,
) -> Result<DevicePtr> {
    let [e, n, k] = codes.shape[..] else {
        anyhow::bail!("inkling2 MXFP8 experts shape {:?}", codes.shape);
    };
    let out = gpu
        .alloc(e * n * k * 2)
        .context("allocate inkling2 BF16 experts")?;
    let kernel = gpu.kernel("inkling2_moe_prefill", "inkling2_mxfp8_to_bf16")?;
    KernelLaunch::new(gpu, kernel)
        .grid([u32::try_from(e * n)?, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(codes.ptr)
        .arg_ptr(scales)
        .arg_ptr(out)
        .arg_u32(u32::try_from(k)?)
        .launch(0)?;
    gpu.synchronize(0)?;
    Ok(out)
}

/// Per-expert tables over `{ex}.w13_weight` / `{ex}.w2_weight` for the grouped GEMMs.
fn grouped_experts(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    ex: &str,
    mxfp8: bool,
) -> Result<GroupedExperts> {
    let t = |name: String| {
        store
            .get(&name)
            .with_context(|| format!("inkling2 tensor {name}"))
    };
    let one = |w: &str| -> Result<(DevicePtr, DevicePtr)> {
        let codes = t(format!("{ex}.{w}"))?;
        let scales = t(format!("{ex}.{w}_scale"))?;
        let e = codes.shape[0];
        let per = |x: &WeightTensor| x.num_elements() / e;
        Ok(if mxfp8 {
            let bf16 = mxfp8_as_bf16(gpu, codes, scales.ptr)?;
            (ptr_table(gpu, bf16, e, per(codes) * 2)?, DevicePtr::NULL)
        } else {
            (
                ptr_table(gpu, codes.ptr, e, per(codes))?,
                ptr_table(gpu, scales.ptr, e, per(scales))?,
            )
        })
    };
    let (w13, w13_scale) = one("w13_weight")?;
    let (w2, w2_scale) = one("w2_weight")?;
    Ok(GroupedExperts {
        w13,
        w13_scale,
        w2,
        w2_scale,
    })
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
    let ex = format!("{lp}.mlp.experts");
    let grouped = grouped_experts(store, gpu, &ex, mxfp8)?;
    let stacked = StackedExperts {
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
        grouped,
    };
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
        wq: p(&format!("{a}.wq_du.weight"))?,
        wk: p(&format!("{a}.wk_dv.weight"))?,
        wv: p(&format!("{a}.wv_dv.weight"))?,
        wr: p(&format!("{a}.wr_du.weight"))?,
        rel_proj: p(&format!("{a}.rel_logits_proj.proj"))?,
        q_scale: p(&format!("{a}.q_l2_clip.scale"))?,
        rel_scale: p(&format!("{a}.rel_l2_clip.scale"))?,
        gate: p(&format!("{a}.gate_attn.weight"))?,
        wo: p(&format!("{a}.wo_ud.weight"))?,
        sink: p(&format!("{a}.attn_sink_proj.weight"))?,
        k_conv: conv("k")?,
        v_conv: conv("v")?,
        attn_conv: p(&format!("{lp}.attn_sconv.weight"))?,
        mlp_conv: p(&format!("{lp}.mlp_sconv.weight"))?,
        mlp_norm: p(&format!("{lp}.mlp_norm.weight"))?,
        router: p(&format!("{lp}.mlp.gate.weight"))?,
        router_bias: p(&format!("{lp}.mlp.gate.bias"))?,
        experts: stacked,
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
