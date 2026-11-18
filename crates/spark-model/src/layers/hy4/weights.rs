// SPDX-License-Identifier: AGPL-3.0-only

use crate::weight_map::{ExpertWeight, QuantizedWeight, quantized};
use anyhow::{Result, ensure};
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend},
    weights::{WeightDtype, WeightStore},
};

pub(super) fn tensor(
    store: &WeightStore,
    name: &str,
    shape: &[usize],
    dtype: WeightDtype,
) -> Result<DevicePtr> {
    let t = store.get(name)?;
    ensure!(
        t.shape == shape && t.dtype == dtype,
        "Hy4 {name}: expected {dtype:?} {shape:?}, got {:?} {:?}",
        t.dtype,
        t.shape
    );
    Ok(t.ptr)
}

fn projection(
    store: &WeightStore,
    name: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(16),
        "Hy4 NVFP4 dimensions"
    );
    tensor(
        store,
        &format!("{name}.weight"),
        &[n, k / 2],
        WeightDtype::UInt8,
    )?;
    tensor(
        store,
        &format!("{name}.weight_scale"),
        &[n, k / 16],
        WeightDtype::FP8E4M3,
    )?;
    for suffix in ["input_scale", "weight_scale_2"] {
        let t = store.get(&format!("{name}.{suffix}"))?;
        ensure!(
            t.dtype == WeightDtype::FP32 && t.num_elements() == 1,
            "Hy4 {name}.{suffix}: expected scalar FP32"
        );
    }
    // Existing ModelOpt loader: borrowed packed bytes, scalar extracted once.
    let w = quantized(store, name, gpu)?;
    ensure!(
        w.weight_scale_2.is_finite() && w.weight_scale_2 > 0.0,
        "Hy4 {name}: invalid global weight scale"
    );
    Ok(w)
}

pub(super) fn mlp(
    store: &WeightStore,
    prefix: &str,
    h: usize,
    i: usize,
    gpu: &dyn GpuBackend,
) -> Result<ExpertWeight> {
    Ok(ExpertWeight {
        gate_proj: projection(store, &format!("{prefix}.gate_proj"), i, h, gpu)?,
        up_proj: projection(store, &format!("{prefix}.up_proj"), i, h, gpu)?,
        down_proj: projection(store, &format!("{prefix}.down_proj"), h, i, gpu)?,
    })
}

/// Explicit little-endian 24-byte Hy4Q4 ABI (two pointers, scalar, padding).
pub(super) fn descriptors(experts: &[ExpertWeight]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(experts.len() * 72);
    for e in experts {
        for w in [e.gate_proj, e.up_proj, e.down_proj] {
            bytes.extend(w.weight.0.to_le_bytes());
            bytes.extend(w.weight_scale.0.to_le_bytes());
            bytes.extend(w.weight_scale_2.to_le_bytes());
            bytes.extend(0u32.to_le_bytes());
        }
    }
    bytes
}
