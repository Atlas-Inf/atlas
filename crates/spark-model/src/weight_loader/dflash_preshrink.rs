// SPDX-License-Identifier: AGPL-3.0-only
//! Drafter BF16 preshrink (`ATLAS_DFLASH_DROP_BF16=1`).
//!
//! The KV pool is sized before the drafter head is built, so BF16 drafter
//! projections that serving never reads still cost KV blocks. Under the NVFP4
//! tensor-core tiers (`ATLAS_DFLASH_NVFP4_TC=1`) every drafter row block reads
//! the NVFP4 twins, and so do the batched-GEMV tiers when every pass is at
//! most 16 rows (a target's MODEL.toml `[dflash].drop_bf16`, which the server
//! applies only at max_batch_size x γ <= 16). Right after the drafter store
//! loads this quantizes
//! q/o/gate/up/down, frees their BF16 tensors and parks the twins for
//! `try_install_nvfp4`. k/v stay BF16: the context precompute reads them
//! through `fused_kv_weight` (as do `fc`, the norms, the DFlash2 convs and
//! the candidate selector). A dropped projection reaching a BF16 GEMM fails
//! loudly (`small_m_gemm::ensure_bf16_present`).

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::weight_map::{DenseWeight, QuantizedWeight, quantize_to_nvfp4};

/// Projections whose BF16 copies are dropped, as `(checkpoint path, field)`.
const DROPPED: [(&str, &str); 5] = [
    ("self_attn.q_proj", "q_proj"),
    ("self_attn.o_proj", "o_proj"),
    ("mlp.gate_proj", "gate_proj"),
    ("mlp.up_proj", "up_proj"),
    ("mlp.down_proj", "down_proj"),
];

static TWINS: Mutex<Option<HashMap<(usize, &'static str), QuantizedWeight>>> = Mutex::new(None);

static TARGET_DEFAULT: AtomicBool = AtomicBool::new(false);

/// The served target's MODEL.toml asks for the preshrink (`[dflash].drop_bf16`).
pub fn set_target_default() {
    TARGET_DEFAULT.store(true, Ordering::Relaxed);
}

/// Whether `ATLAS_DFLASH_DROP_BF16=1`, or it is unset and the target asks.
pub fn requested() -> bool {
    match std::env::var("ATLAS_DFLASH_DROP_BF16") {
        Ok(v) => v == "1",
        Err(_) => TARGET_DEFAULT.load(Ordering::Relaxed),
    }
}

/// Whether the preshrink parked NVFP4 twins for `try_install_nvfp4`.
pub(crate) fn parked() -> bool {
    TWINS.lock().expect("drafter twins lock").is_some()
}

fn env_is(name: &str, value: &str) -> bool {
    std::env::var(name).as_deref() == Ok(value)
}

/// Quantize and free the dropped BF16 projections of `store`. Returns the
/// BF16 bytes released.
pub fn preshrink(store: &mut WeightStore, layers: usize, gpu: &dyn GpuBackend) -> Result<usize> {
    // Tensor-core tiers both on, or neither (the batched-GEMV tiers).
    let tc = env_is("ATLAS_DFLASH_NVFP4_TC", "1");
    ensure!(
        tc == env_is("ATLAS_W4A16_TC", "1")
            && !env_is("ATLAS_DFLASH_DRAFTER_FP8", "1")
            && std::env::var("ATLAS_NO_DFLASH_DRAFTER_NVFP4").is_err()
            // The legacy contiguous path reads the BF16 projections.
            && !env_is("ATLAS_DFLASH_OPTION_B", "0"),
        "ATLAS_DFLASH_DROP_BF16=1 needs an NVFP4 drafter (ATLAS_DFLASH_NVFP4_TC and \
         ATLAS_W4A16_TC both 1 or both unset, no drafter FP8, Option B)"
    );
    let absmax = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quant = gpu.kernel(
        "quantize_nvfp4",
        crate::weight_loader::glm5_mode::weight_quantizer()?,
    )?;
    let prefix = if store.contains("model.fc.weight") {
        "model."
    } else {
        ""
    };
    let mut twins = HashMap::new();
    let mut dropped = Vec::new();
    for layer in 0..layers {
        for (path, field) in DROPPED {
            let name = format!("{prefix}layers.{layer}.{path}.weight");
            let w = store.get(&name)?;
            ensure!(
                w.dtype == spark_runtime::weights::WeightDtype::BF16 && w.shape.len() == 2,
                "{name}: expected a 2-D BF16 tensor"
            );
            let (n, k) = (w.shape[0], w.shape[1]);
            let twin =
                quantize_to_nvfp4(&DenseWeight { weight: w.ptr }, n, k, gpu, absmax, quant, 0)
                    .with_context(|| format!("NVFP4 twin of {name}"))?;
            twins.insert((layer, field), twin);
            dropped.push(name);
        }
    }
    gpu.synchronize(0)?;
    let mut bytes = 0;
    for name in dropped {
        let w = store.remove(&name).expect("dropped tensor was just read");
        bytes += w.byte_size();
        gpu.free(w.ptr)?;
    }
    *TWINS.lock().expect("drafter twins lock") = Some(twins);
    tracing::info!(
        "DFlash preshrink: {layers} layers x {} projections now NVFP4-only ({:.2} GiB BF16 freed before KV sizing)",
        DROPPED.len(),
        bytes as f64 / (1u64 << 30) as f64
    );
    Ok(bytes)
}

/// The placeholder for a dropped BF16 projection, if `name` was dropped.
pub(crate) fn dropped_weight(store: &WeightStore, name: &str) -> Option<DenseWeight> {
    let dropped = !store.contains(name)
        && TWINS.lock().expect("drafter twins lock").is_some()
        && DROPPED
            .iter()
            .any(|(path, _)| name.ends_with(&format!("{path}.weight")));
    dropped.then_some(DenseWeight {
        weight: DevicePtr(0),
    })
}

/// Take the parked NVFP4 twin of `field` in `layer`.
pub(crate) fn take_twin(layer: usize, field: &str) -> Option<QuantizedWeight> {
    let mut twins = TWINS.lock().expect("drafter twins lock");
    let key = DROPPED.iter().find(|(_, f)| *f == field)?.1;
    twins.as_mut()?.remove(&(layer, key))
}
