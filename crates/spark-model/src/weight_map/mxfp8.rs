// SPDX-License-Identifier: AGPL-3.0-only

//! MXFP8 weight twins (OCP microscaling FP8): E4M3 values plus one E8M0
//! scale per 32 input columns.

use spark_runtime::gpu::DevicePtr;

/// MXFP8 twin of a projection: E4M3 `[n, k]` + E8M0 `[n, k/32]`.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp8Weight {
    pub data: DevicePtr,
    pub scales: DevicePtr,
}
