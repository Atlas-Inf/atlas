// SPDX-License-Identifier: AGPL-3.0-only

//! Eager-only single-output numerical transaction; no GPU allocations.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_output(
    gpu: &dyn GpuBackend,
    stream: u64,
    output: DevicePtr,
    bytes: usize,
    capturing: bool,
    overlap: bool,
    old: KernelHandle,
    new: KernelHandle,
    launch: impl Fn(KernelHandle) -> Result<()>,
) -> Result<()> {
    anyhow::ensure!(
        !capturing && !overlap && !gpu.stream_is_capturing(stream),
        "GLM M5 projection VERIFY requires eager, non-overlapped execution"
    );
    anyhow::ensure!(
        bytes > 0
            && bytes <= 40960
            && bytes.is_multiple_of(2)
            && !output.is_null()
            && output.0.is_multiple_of(2)
            && output.0.checked_add(bytes as u64).is_some(),
        "GLM M5 oracle output span"
    );
    let snapshot = |kernel| -> Result<Vec<u8>> {
        gpu.memset_async(output, 0xff, bytes, stream)?;
        launch(kernel)?;
        let mut values = vec![0; bytes];
        gpu.copy_d2h_on_stream(output, &mut values, stream)?;
        Ok(values)
    };
    let expected = snapshot(old)?;
    anyhow::ensure!(
        expected
            .chunks_exact(2)
            .all(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 != 0x7f80),
        "GLM M5 oracle reference contains nonfinite/unwritten BF16"
    );
    let actual = snapshot(new)?;
    if let Some(i) = expected.iter().zip(&actual).position(|(a, b)| a != b) {
        anyhow::bail!(
            "GLM M5 projection native oracle mismatch byte={i}: old={} new={}",
            expected[i],
            actual[i]
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "m5_projection_oracle_tests.rs"]
mod tests;
