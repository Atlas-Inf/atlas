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

/// An MXFP8 projection with its geometry: `out = act @ W^T`, `W` `[n, k]`.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp8Proj {
    pub w: Mxfp8Weight,
    /// Output features (weight rows).
    pub n: u32,
    /// Input features; a multiple of 32.
    pub k: u32,
}

impl Mxfp8Proj {
    /// Bytes of the exact BF16 expansion a prefill GEMM multiplies by.
    pub fn bf16_bytes(&self) -> usize {
        self.n as usize * self.k as usize * 2
    }
}
