// SPDX-License-Identifier: AGPL-3.0-only
//! Staged load-time byte ABI; no serving caller until the reader family is ready.
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub(crate) fn glm_native_to_btile(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    source: DevicePtr,
    destination: DevicePtr,
    stream: u64,
) -> Result<()> {
    const BYTES: u64 = 4_194_304;
    ensure!(kernel.0 != 0, "missing native-to-B-tile handle");
    let end = |p: DevicePtr| -> Result<u64> {
        ensure!(p.0 != 0 && p.0.is_multiple_of(16), "invalid packed span");
        p.0.checked_add(BYTES)
            .ok_or_else(|| anyhow::anyhow!("packed span overflow"))
    };
    let a = end(source)?;
    let b = end(destination)?;
    ensure!(
        a <= destination.0 || b <= source.0,
        "packed permutation alias"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([16384, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(source)
        .arg_ptr(destination)
        .arg_u32(2048)
        .arg_u32(4096)
        .launch(stream)
}
