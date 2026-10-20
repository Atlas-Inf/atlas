// SPDX-License-Identifier: AGPL-3.0-only
//! Atlas-owned kernel resolution and sparse prefill launch sequence.
use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use spark_runtime::kernel_args::KernelLaunch;

use super::plan::{Metadata, Plan};

const FLAG: &str = "ATLAS_GLM_SPARSE_NATIVE";
const SYMBOLS: [(&str, &str); 5] = [
    ("glm_sparse_native_prep", "glm_sparse_native_prep_q"),
    ("glm_sparse_native_prep", "glm_sparse_native_prep_kv"),
    ("glm_sparse_native_split", "glm_sparse_native_split"),
    ("glm_sparse_native_attention", "glm_sparse_native_attention"),
    ("glm_sparse_native_merge", "glm_sparse_native_merge"),
];

pub(super) fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => bail!("{FLAG} must be 0 or 1"),
    }
}

pub(super) fn enabled() -> Result<bool> {
    match std::env::var(FLAG) {
        Ok(value) => parse(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse(None),
        Err(error) => Err(error.into()),
    }
}

pub(super) fn resolve(gpu: &dyn GpuBackend) -> Result<[KernelHandle; 5]> {
    let mut handles = [KernelHandle(0); 5];
    for (handle, (module, symbol)) in handles.iter_mut().zip(SYMBOLS) {
        *handle = gpu.kernel(module, symbol)?;
        ensure!(
            handle.0 != 0,
            "Atlas GLM sparse kernel is unavailable: {symbol}"
        );
    }
    Ok(handles)
}

pub(super) fn execute(plan: &Plan, gpu: &dyn GpuBackend) -> Result<()> {
    // Resolve every symbol and derive every subspan before the first write.
    // A missing capability must never leave half-prepared scratch contents.
    let [q, kv, split, attention, merge] = resolve(gpu)?;
    let a = &plan.operands;
    let metadata = Metadata::new(a.rows as usize).map_err(anyhow::Error::msg)?;
    let m = metadata
        .offsets
        .map(|offset| DevicePtr(a.metadata + offset as u64));
    let rows = a.rows;
    let end = a.seq_start + rows;
    KernelLaunch::new(gpu, q)
        .grid([(rows * 32 * 576).div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(DevicePtr(a.q))
        .arg_ptr(DevicePtr(a.qpad))
        .arg_u32(rows)
        .launch(a.stream)?;
    KernelLaunch::new(gpu, kv)
        .grid([end.div_ceil(64) * 64, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(DevicePtr(a.kv))
        .arg_ptr(DevicePtr(a.table))
        .arg_ptr(DevicePtr(a.packed_kv))
        .arg_u32(end)
        .launch(a.stream)?;
    KernelLaunch::new(gpu, split)
        .grid([rows, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(DevicePtr(a.selected))
        .arg_ptr(m[0])
        .arg_ptr(m[1])
        .arg_ptr(m[2])
        .arg_ptr(m[3])
        .arg_ptr(m[4])
        .arg_u32(rows)
        .arg_u32(a.seq_start)
        .launch(a.stream)?;
    for (indices, output, lse, lengths) in [
        (m[0], a.main_out, m[5], m[2]),
        (m[1], a.tail_out, m[6], m[3]),
    ] {
        KernelLaunch::new(gpu, attention)
            .grid([rows, 1, 1])
            .block([384, 1, 1])
            .shared_mem(91920)
            .arg_ptr(DevicePtr(a.qpad))
            .arg_ptr(DevicePtr(a.packed_kv))
            .arg_ptr(indices)
            .arg_ptr(DevicePtr(output))
            .arg_ptr(lse)
            .arg_ptr(lengths)
            .arg_u32(rows)
            .launch(a.stream)?;
    }
    KernelLaunch::new(gpu, merge)
        .grid([rows, 32, 1])
        .block([256, 1, 1])
        .arg_ptr(DevicePtr(a.main_out))
        .arg_ptr(DevicePtr(a.tail_out))
        .arg_ptr(m[5])
        .arg_ptr(m[6])
        .arg_ptr(m[4])
        .arg_ptr(DevicePtr(a.out))
        .arg_ptr(m[7])
        .arg_u32(rows)
        .arg_u32(32)
        .launch(a.stream)
}
