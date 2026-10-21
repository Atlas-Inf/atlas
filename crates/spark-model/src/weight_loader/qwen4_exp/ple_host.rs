// SPDX-License-Identifier: AGPL-3.0-only

//! Small resident PLE tensors read back to the host: the id-hash tables, the
//! per-tensor dequant scalars, and EXL3's FP16 head bias.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

/// Read a small I64 device tensor back to the host.
///
/// `layer_multipliers` and the two per-head tables are 3 and 16 elements —
/// they are uploaded like any other weight, and the id hash needs them on the
/// host. Reading them back beats adding a host-side path to `WeightStore` for
/// 280 bytes.
pub(super) fn i64_host(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<Vec<u64>> {
    let t = store.get(name).with_context(|| format!("PLE: {name}"))?;
    let n = t.num_elements();
    let mut raw = vec![0u8; n * 8];
    gpu.copy_d2h(t.ptr, &mut raw)
        .with_context(|| format!("PLE: reading {name} back to host"))?;
    Ok(raw
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

/// A one-element BF16 tensor, read back to the host as f32.
///
/// The FP8 n-gram table's dequant scale is stored this way — `shape: [1]`,
/// dtype BF16 — rather than as a per-row scale file.
pub(super) fn bf16_scalar(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<f32> {
    let t = store.get(name).with_context(|| format!("PLE: {name}"))?;
    anyhow::ensure!(
        t.num_elements() == 1,
        "PLE: {name} has {} elements, expected 1 (a per-tensor scale)",
        t.num_elements()
    );
    let mut raw = [0u8; 2];
    gpu.copy_d2h(t.ptr, &mut raw)
        .with_context(|| format!("PLE: reading {name} back to host"))?;
    let bits = u16::from_le_bytes(raw);
    let v = f32::from_bits((bits as u32) << 16);
    anyhow::ensure!(
        v.is_finite() && v > 0.0,
        "PLE: {name} is {v}, which cannot be a dequant scale"
    );
    Ok(v)
}

/// A resident FP16 vector, read back as raw fp16 bits.
///
/// EXL3 `head_bias` stays FP16 (`keeps_raw_f16`) because the n-gram dequant
/// adds it in the same units as `decode_mul1` * scale.
pub(super) fn fp16_host(
    store: &WeightStore,
    name: &str,
    n: usize,
    gpu: &dyn GpuBackend,
) -> Result<Vec<u16>> {
    let t = store.get(name).with_context(|| format!("PLE: {name}"))?;
    anyhow::ensure!(
        t.num_elements() == n,
        "PLE: {name} has {} elements, expected {n}",
        t.num_elements()
    );
    anyhow::ensure!(
        matches!(t.dtype, spark_runtime::weights::WeightDtype::FP16),
        "PLE: {name} is {:?}, expected FP16 (raw head_bias)",
        t.dtype
    );
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(t.ptr, &mut raw)
        .with_context(|| format!("PLE: reading {name} back to host"))?;
    Ok(raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect())
}

/// A resident one-element F32 tensor (`weight_scale_2`), read to the host.
pub(super) fn f32_scalar(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<f32> {
    let t = store
        .get(name)
        .with_context(|| format!("PLE nvfp4: {name}"))?;
    ensure!(
        t.dtype == WeightDtype::FP32 && t.num_elements() == 1,
        "PLE nvfp4: {name} is {:?} {:?}, expected one F32",
        t.dtype,
        t.shape
    );
    let mut raw = [0u8; 4];
    gpu.copy_d2h(t.ptr, &mut raw)?;
    let v = f32::from_le_bytes(raw);
    ensure!(
        v.is_finite() && v > 0.0,
        "PLE nvfp4: {name} is {v}, not a dequant scale"
    );
    Ok(v)
}
