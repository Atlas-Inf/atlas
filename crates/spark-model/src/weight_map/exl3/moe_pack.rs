// SPDX-License-Identifier: AGPL-3.0-only

//! Packed EXL3 MoE experts. Routed and shared experts stay `Exl3Weight`.
//! This loader does not dequantize them and does not call `quantize_to_nvfp4`.
//!
//! The pack is registered by the BF16 router gate pointer so `MoeLayer` can
//! find it without a new field on `MoeWeights` (that struct is built in a
//! dozen loaders). Scratch is one reconstruct tile per layer, not a BF16
//! copy of every expert.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::layers::ops::{Exl3CoopMk, Exl3Kernels, coopmk_requested};
use crate::weight_map::{ExpertWeight, MoeWeights, dense_auto};

use super::{Exl3Weight, exl3_from_store};

/// One expert's three packed projections.
pub(crate) struct Exl3Expert {
    pub gate: Exl3Weight,
    pub up: Exl3Weight,
    pub down: Exl3Weight,
}

impl Exl3Expert {
    fn null() -> Self {
        Self {
            gate: Exl3Weight::null(),
            up: Exl3Weight::null(),
            down: Exl3Weight::null(),
        }
    }

    pub(crate) fn is_null(&self) -> bool {
        self.gate.is_null()
    }
}

/// One reconstruct tile plus the M=1 activation scratches, reused across
/// every projection of this layer. Not a resident dequant of the experts.
pub(crate) struct Exl3MoeScratch {
    pub w_inner: DevicePtr,
    pub xh: DevicePtr,
    pub y: DevicePtr,
    pub silu: DevicePtr,
    pub shared_out: DevicePtr,
}

pub(crate) struct Exl3MoePack {
    /// Checkpoint prefix of the MoE block (for logs).
    pub name: String,
    pub experts: Vec<Exl3Expert>,
    pub shared: Exl3Expert,
    pub kernels: Exl3Kernels,
    pub scratch: Exl3MoeScratch,
    /// CoopMK tables (pointer tables into the packed trellis + bitrate table +
    /// scratch). Built only when `ATLAS_EXL3_COOPMK=1` or a toggle file is
    /// set; `None` keeps the interim path.
    pub coopmk: Option<Exl3CoopMk>,
}

static REG: LazyLock<Mutex<HashMap<u64, Exl3MoePack>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn reg() -> std::sync::MutexGuard<'static, HashMap<u64, Exl3MoePack>> {
    REG.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Number of MoE blocks registered as packed EXL3.
pub(crate) fn exl3_moe_pack_count() -> usize {
    reg().len()
}

pub(crate) fn exl3_moe_registered(gate_ptr: u64) -> bool {
    gate_ptr != 0 && reg().contains_key(&gate_ptr)
}

pub(crate) fn with_exl3_moe<R>(gate_ptr: u64, f: impl FnOnce(&Exl3MoePack) -> R) -> Option<R> {
    reg().get(&gate_ptr).map(f)
}

fn load_proj(
    store: &WeightStore,
    prefix: &str,
    out: usize,
    inn: usize,
    gpu: &dyn GpuBackend,
) -> Result<Exl3Weight> {
    let w = exl3_from_store(store, prefix, gpu)
        .with_context(|| format!("EXL3 packed expert {prefix}"))?;
    if w.shape.out_features != out || w.shape.in_features != inn {
        bail!(
            "EXL3: {prefix} is {}x{} (out x in), the loader asked for {out}x{inn}",
            w.shape.out_features,
            w.shape.in_features
        );
    }
    Ok(w)
}

fn alloc_scratch(
    gpu: &dyn GpuBackend,
    hidden: usize,
    inter: usize,
    shared_inter: usize,
) -> Result<Exl3MoeScratch> {
    let widest = hidden.max(inter).max(shared_inter);
    let w_elems = hidden * inter.max(shared_inter);
    let silu_elems = inter.max(shared_inter);
    let mut owned: Vec<DevicePtr> = Vec::with_capacity(5);
    let alloc = |owned: &mut Vec<DevicePtr>, elems: usize| -> Result<DevicePtr> {
        match gpu.alloc(elems * 2) {
            Ok(p) => {
                owned.push(p);
                Ok(p)
            }
            Err(e) => {
                for p in owned.drain(..) {
                    let _ = gpu.free(p);
                }
                Err(e)
            }
        }
    };
    let w_inner = alloc(&mut owned, w_elems)?;
    let xh = alloc(&mut owned, widest)?;
    let y = alloc(&mut owned, widest)?;
    let silu = alloc(&mut owned, silu_elems)?;
    let shared_out = alloc(&mut owned, hidden)?;
    Ok(Exl3MoeScratch {
        w_inner,
        xh,
        y,
        silu,
        shared_out,
    })
}

/// Load routed + shared experts as packed EXL3. The returned `MoeWeights`
/// carries null NVFP4 expert slots so `MoeLayer::new` can still build its
/// pointer tables; decode must not launch those kernels.
pub(crate) fn load_moe_exl3_packed(
    store: &WeightStore,
    mlp: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    skip_routed_experts: bool,
) -> Result<MoeWeights> {
    let hidden = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let shared_inter = config.shared_expert_intermediate_size;
    let gate = dense_auto(store, &format!("{mlp}.gate.weight"), gpu)
        .with_context(|| format!("EXL3 MoE router at {mlp}.gate"))?;
    let shared_expert_gate = dense_auto(store, &format!("{mlp}.shared_expert_gate.weight"), gpu)
        .with_context(|| format!("EXL3 MoE shared gate at {mlp}"))?;
    if gate.weight.0 == 0 {
        bail!("EXL3 MoE router at {mlp}.gate has a null pointer");
    }
    // The registration key must be a pointer the LAYER still holds.
    // `build_moe` requantizes `weights.gate` to NVFP4 afterwards and the
    // layer routes through that copy, so keying on the BF16 router gate
    // makes every later lookup miss. `shared_expert_gate` is left as the
    // BF16 tensor both sides share.
    if shared_expert_gate.weight.0 == 0 {
        bail!("EXL3 MoE shared gate at {mlp} has a null pointer");
    }

    let mut experts = Vec::with_capacity(num_experts);
    let mut n_local = 0usize;
    for e in 0..num_experts {
        if skip_routed_experts || !config.is_local_expert(e) {
            experts.push(Exl3Expert::null());
            continue;
        }
        let p = format!("{mlp}.experts.{e}");
        experts.push(Exl3Expert {
            gate: load_proj(store, &format!("{p}.gate_proj"), inter, hidden, gpu)?,
            up: load_proj(store, &format!("{p}.up_proj"), inter, hidden, gpu)?,
            down: load_proj(store, &format!("{p}.down_proj"), hidden, inter, gpu)?,
        });
        n_local += 1;
    }
    let se = format!("{mlp}.shared_expert");
    let shared = Exl3Expert {
        gate: load_proj(store, &format!("{se}.gate_proj"), shared_inter, hidden, gpu)?,
        up: load_proj(store, &format!("{se}.up_proj"), shared_inter, hidden, gpu)?,
        down: load_proj(store, &format!("{se}.down_proj"), hidden, shared_inter, gpu)?,
    };

    let kernels = Exl3Kernels::resolve(gpu)?;
    let scratch = alloc_scratch(gpu, hidden, inter, shared_inter)?;
    let coopmk = if coopmk_requested() && !skip_routed_experts && n_local == num_experts {
        if shared_inter != inter {
            // The shared expert rides as one more slot of the routed launch,
            // which needs the same intermediate width.
            tracing::warn!(
                "EXL3 CoopMK not bound at {mlp}: shared intermediate {shared_inter} != routed {inter}"
            );
            None
        } else {
            let routed: Vec<(&Exl3Weight, &Exl3Weight, &Exl3Weight)> =
                experts.iter().map(|e| (&e.gate, &e.up, &e.down)).collect();
            let c = Exl3CoopMk::build(
                gpu,
                &routed,
                (&shared.gate, &shared.up, &shared.down),
                hidden,
                inter,
                config.num_experts_per_tok,
            )
            .with_context(|| format!("EXL3 CoopMK tables at {mlp}"))?;
            tracing::info!("EXL3 CoopMK bound at {mlp}: {}", c.describe());
            Some(c)
        }
    } else {
        None
    };
    tracing::info!(
        "EXL3 MoE packed at {mlp}: {n_local}/{num_experts} routed experts + shared expert stay trellis (no NVFP4 requant), key {:#x}",
        shared_expert_gate.weight.0
    );
    reg().insert(
        shared_expert_gate.weight.0,
        Exl3MoePack {
            name: mlp.to_string(),
            experts,
            shared,
            kernels,
            scratch,
            coopmk,
        },
    );

    let null_expert = ExpertWeight::null();
    Ok(MoeWeights {
        gate,
        shared_expert: null_expert,
        shared_expert_gate,
        experts: vec![null_expert; num_experts],
        router_pre_norm: None,
        correction_bias: None,
    })
}
