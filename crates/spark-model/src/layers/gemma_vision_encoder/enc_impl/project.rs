// SPDX-License-Identifier: AGPL-3.0-only

//! `embed_vision` projection (INSIDE the encoder, so the downstream splice
//! is a straight copy): RMSNorm(hidden, `with_scale=False`) → Linear
//! hidden → OUT_HIDDEN_SIZE, no bias.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::{GemmaVisionEncoder, OUT_HIDDEN_SIZE};
use crate::layers::ops;
use crate::weight_map::DenseWeight;

impl GemmaVisionEncoder {
    /// Project `soft` pooled soft-token rows from `src` to `[soft,
    /// OUT_HIDDEN_SIZE]` BF16 at `dst` (a packed `buf_out` slice): unweighted
    /// RMSNorm (`norm_unit_w` — ones, `with_scale=False`) then the 768→1536
    /// `embedding_projection` GEMM.
    pub(super) fn embed_vision_project(
        &self,
        soft: usize,
        src: DevicePtr,
        dst: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let s = soft as u32;
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            src,
            &DenseWeight {
                weight: self.norm_unit_w,
            },
            src,
            s,
            self.hidden_size as u32,
            self.norm_eps,
            stream,
        )?;
        ops::dense_gemm(
            gpu,
            self.k_gemm,
            src,
            &DenseWeight {
                weight: self.embed_vision_proj_w,
            },
            dst,
            s,
            OUT_HIDDEN_SIZE as u32,
            self.hidden_size as u32,
            stream,
        )
    }
}
