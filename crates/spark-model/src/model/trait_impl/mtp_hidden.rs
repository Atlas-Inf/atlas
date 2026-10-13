// SPDX-License-Identifier: AGPL-3.0-only

//! Copy the target representation required by each MTP model. The caller owns
//! live BF16 rows until this same-stream copy has consumed them.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[allow(clippy::too_many_arguments)]
pub(super) fn copy_target_hidden_row(
    gpu: &dyn GpuBackend,
    model_type: &str,
    raw: DevicePtr,
    normalized: DevicePtr,
    destination: DevicePtr,
    hidden_size: usize,
    row: usize,
    stream: u64,
) -> Result<()> {
    let bytes = hidden_size
        .checked_mul(2)
        .context("MTP hidden row size overflow")?;
    let offset = row
        .checked_mul(bytes)
        .context("MTP hidden row offset overflow")?;
    // GLM's target returns final-normalized rows to its MTP layer, whose
    // separate learned hnorm still applies. Match its prompt capture and
    // plain generate path. Other families retain their raw-hidden contract.
    // Decode/verify already produced these rows; do not normalize twice.
    let source = if model_type == "glm5_next" {
        normalized
    } else {
        raw
    };
    let address = source
        .0
        .checked_add(offset as u64)
        .context("MTP hidden address overflow")?;
    gpu.copy_d2d_async(DevicePtr(address), destination, bytes, stream)
}

#[cfg(test)]
#[path = "mtp_hidden_tests.rs"]
mod tests;
