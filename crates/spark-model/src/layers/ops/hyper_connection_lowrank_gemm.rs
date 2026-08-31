// SPDX-License-Identifier: AGPL-3.0-only

//! Raw-GEMM plumbing split out of `hyper_connection_lowrank.rs` to keep
//! that file under the 500-LoC cap; same module surface via re-import.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use spark_runtime::kernel_args::KernelLaunch;

pub(crate) fn gemm_raw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(128), m.div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Block width for stage 3 (`ATLAS_HC_FIN_BLOCK`, default 128). Kept as a knob
/// because it is pure launch geometry — it cannot change the arithmetic, only
/// how much of the machine runs it.
pub(crate) fn hc_finish_block() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_HC_FIN_BLOCK")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|n| (32..=1024).contains(n) && n % 32 == 0)
            .unwrap_or(128)
    })
}
