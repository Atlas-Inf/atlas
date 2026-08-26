// SPDX-License-Identifier: AGPL-3.0-only

//! QK-Norm: per-head RMSNorm over head_dim, applied to q and k AFTER the
//! Q/K projections and BEFORE the rotary + attention (Gemma's attention
//! scale is 1.0 — QK-norm replaces 1/√head_dim).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::GemmaVisionEncoder;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

impl GemmaVisionEncoder {
    /// Normalize one projection slice in place with its per-head QK-Norm
    /// weight (`q_norm`/`k_norm`, [head_dim]).
    ///
    /// Wave 2: `k_qk_norm` resolves to the generic `norm::rms_norm`
    /// (RMSNorm over dim 64 — shape-compatible per head, launched with the
    /// `[p×heads, head_dim]` view). Wave 3 points the handle at a dedicated
    /// `gemma_vision_qk_norm` kernel that consumes the interleaved
    /// `[p, heads×head_dim]` layout directly; the launch convention below
    /// (input, weight, output, rows, dim, eps) is shared by both.
    pub(super) fn qk_norm_inplace(
        &self,
        buf: DevicePtr,
        weight: &DenseWeight,
        p: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ops::rms_norm(
            gpu,
            self.k_qk_norm,
            buf,
            weight,
            buf,
            p * self.num_heads as u32,
            self.head_dim as u32,
            self.norm_eps,
            stream,
        )
    }
}
