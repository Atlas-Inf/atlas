// SPDX-License-Identifier: AGPL-3.0-only

//! Device handles for one physical Inkling 2 layer and for the model-level
//! block-residual and loop-message queries. Handles are plain device
//! pointers into the weight store (BF16/F32/U8/F8 as the checkpoint stores
//! them), so every visit of a physical layer copies the same struct. The only
//! derived tensors are the experts' per-expert global scales
//! (`amax / outer_scale_denominator`, F32 `[E]`), computed once at load.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::ops::StackedExperts;

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
    pub index: usize,
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
        index: l,
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
