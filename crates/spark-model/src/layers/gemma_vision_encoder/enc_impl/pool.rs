// SPDX-License-Identifier: AGPL-3.0-only

//! Pooler: average-pool `pooling_kernel_size²` patch groups into soft
//! tokens, scaled by √hidden_size in f32, padding stripped.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::GemmaVisionEncoder;
use super::launch_optional;
use crate::media::gemma_vision::GemmaImageInput;

impl GemmaVisionEncoder {
    /// Average-pool one image's encoded `[P, hidden]` states into
    /// `pooling_kernel_size²`-sized patch groups → `[soft, hidden]` soft
    /// tokens at `dst`, each pooled row scaled by `√hidden_size` (f32).
    /// `img` supplies the grid geometry; `src` is the image's slice of the
    /// packed hidden buffer.
    ///
    /// Wave 2: dispatched through the Wave-3 `gemma_vision_pool` stub
    /// (null handle → no-op). The documented kernel arg layout:
    /// `(src, dst, grid_w, pks, hidden, seq, soft, scale)` — the host
    /// passes the row counts so the kernel can strip padding from
    /// non-divisible grids (the preprocessor's grids always divide).
    pub(super) fn pool_stage(
        &self,
        img: &GemmaImageInput,
        src: DevicePtr,
        dst: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let p = (img.grid_h * img.grid_w) as u32;
        let soft = img.soft_token_count as u32;
        let scale = (self.hidden_size as f32).sqrt();
        launch_optional(gpu, self.k_pool, "gemma_vision_pool", stream, |k| {
            k.grid([soft, 1, 1])
                .block([self.hidden_size.min(256) as u32, 1, 1])
                .arg_ptr(src)
                .arg_ptr(dst)
                .arg_u32(img.grid_w as u32)
                .arg_u32(self.pooling_kernel_size as u32)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(p)
                .arg_u32(soft)
                .arg_f32(scale)
        })
    }
}
