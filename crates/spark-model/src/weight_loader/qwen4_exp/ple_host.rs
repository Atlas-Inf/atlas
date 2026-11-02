// SPDX-License-Identifier: AGPL-3.0-only

//! Small resident PLE tensors read back to the host: the id-hash tables, the
//! per-tensor dequant scalars, and EXL3's FP16 head bias.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

/// The host readback size and decode are chosen from the EXPECTED dtype, not
/// from the tensor — a mis-typed checkpoint tensor (e.g. BF16 where I64 is
/// needed) would otherwise be read `n*8` bytes off a `n*2` allocation
/// (CWE-125). One check, shared by every PLE host read.
fn expect_dtype(
    t: &spark_runtime::weights::WeightTensor,
    name: &str,
    dt: WeightDtype,
) -> Result<()> {
    ensure!(
        t.dtype == dt,
        "PLE: {name} is {:?} {:?}, expected {:?} — refusing to read {dt:?}-sized bytes off it",
        t.dtype,
        t.shape,
        dt,
    );
    Ok(())
}

/// Read a small I64 device tensor back to the host.
///
/// `layer_multipliers` and the two per-head tables are 3 and 16 elements —
/// they are uploaded like any other weight, and the id hash needs them on the
/// host. Reading them back beats adding a host-side path to `WeightStore` for
/// 280 bytes.
pub(super) fn i64_host(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<Vec<u64>> {
    let t = store.get(name).with_context(|| format!("PLE: {name}"))?;
    expect_dtype(t, name, WeightDtype::Int64)?;
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
    expect_dtype(t, name, WeightDtype::BF16)?;
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
    expect_dtype(t, name, WeightDtype::FP16)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::WeightTensor;
    use std::collections::HashMap;

    fn store(name: &str, shape: &[usize], dtype: WeightDtype) -> (WeightStore, MockGpuBackend) {
        let gpu = MockGpuBackend::new();
        let n: usize = shape.iter().product();
        let bytes = n * dtype.byte_size().max(2);
        let mut map = HashMap::new();
        map.insert(
            name.to_string(),
            WeightTensor {
                ptr: gpu.alloc(bytes).unwrap(),
                shape: shape.to_vec(),
                dtype,
            },
        );
        (WeightStore::from_map(map), gpu)
    }

    /// A BF16 `layer_multipliers` (the Qwen4 synth bug) must fail before the
    /// DtoH read, naming the expected vs actual dtype — not as
    /// `cuMemcpyDtoHAsync: invalid value` on a 6 B alloc.
    #[test]
    fn i64_host_rejects_a_bf16_multipliers_tensor() {
        let name = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers";
        let (store, gpu) = store(name, &[3], WeightDtype::BF16);
        let err = i64_host(&store, name, &gpu).unwrap_err().to_string();
        assert!(
            err.contains("BF16") && err.contains("Int64") && err.contains(name),
            "unexpected message: {err}"
        );
    }
}
